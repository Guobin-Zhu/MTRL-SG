import torch
import time
import os
import numpy as np
from pathlib import Path
from datetime import datetime
from tensorboardX import SummaryWriter

import gym
from gym.wrappers import AdversarialSwarmWrapper
from cfg import get_adversarial_args
from algorithm.algorithms import MADDPG
from algorithm.utils import ReplayBuffer

USE_CUDA = False 

def run(cfg):
    """
    Main training function for adversarial swarm multi-agent reinforcement learning
    
    Args:
        cfg: Configuration object containing training hyperparameters
    """

    ## ======================================= record =======================================
    # Setup directories for model saving and logging
    model_dir = Path('./models') / cfg.env_name 
    curr_run = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")  # Timestamp for unique run ID
    run_dir = model_dir / curr_run   
    log_dir = run_dir / 'logs'    
    os.makedirs(log_dir)    
    logger = SummaryWriter(str(log_dir))  # TensorBoard logger for training metrics

    ## ======================================= Initialize =======================================
    # Set random seeds for reproducible training
    torch.manual_seed(cfg.seed)  
    np.random.seed(cfg.seed)   
    if not USE_CUDA:
        torch.set_num_threads(cfg.n_training_threads) 

    # Initialize environment
    scenario_name = 'AdversarialSwarm-v0'
    base_env = gym.make(scenario_name).unwrapped
    env = AdversarialSwarmWrapper(base_env, args)

    # Define agent group slices: left agents vs right agents
    start_stop_num = [slice(0, env.num_l), slice(env.num_l, env.num_l + env.num_r)]   

    # Initialize MADDPG algorithm with specified hyperparameters
    maddpg = MADDPG.init_from_env(env, agent_alg=cfg.agent_alg, adversary_alg=cfg.adversary_alg, 
                                  tau=cfg.tau, lr_actor=cfg.lr_actor, lr_critic=cfg.lr_critic, 
                                  epsilon=cfg.epsilon, noise=cfg.noise, hidden_dim=cfg.hidden_dim)  
    
    # Alternative: Load from previous checkpoint
    # last_run = '2024-07-27-18-52-55'
    # last_run_dir = model_dir / last_run / 'model.pt'
    # maddpg = MADDPG.init_from_save_with_id(last_run_dir, [0, 1])         

    # Initialize separate replay buffers for left and right agent groups
    left_agents_buffer = ReplayBuffer(cfg.buffer_length, env.num_l, 
                                    state_dim=env.observation_space.shape[0], 
                                    action_dim=env.action_space.shape[0], 
                                    start_stop_index=start_stop_num[0])    
    right_agents_buffer = ReplayBuffer(cfg.buffer_length, env.num_r, 
                                     state_dim=env.observation_space.shape[0], 
                                     action_dim=env.action_space.shape[0], 
                                     start_stop_index=start_stop_num[1])    
    buffer_total = [left_agents_buffer, right_agents_buffer]  

    # Determine which agent groups to train based on strategy configuration
    if cfg.l_strategy == 'input' and cfg.r_strategy != 'input':
        training_index = range(1)        # Train only left agents
    elif cfg.l_strategy != 'input' and cfg.r_strategy == 'input':
        training_index = range(1, 2)     # Train only right agents
    elif cfg.l_strategy == 'input' and cfg.r_strategy == 'input':
        training_index = range(2)        # Train both agent groups

    ## ======================================= Training =======================================
    print('Training Starts...')
    
    # Main training loop over episodes
    for ep_index in range(0, cfg.n_episodes, cfg.n_rollout_threads):

        # Initialize episode statistics
        episode_reward_mean_l = 0  # Cumulative mean reward for left agents
        episode_reward_mean_r = 0  # Cumulative mean reward for right agents
        episode_reward_std_l = 0   # Cumulative std reward for left agents
        episode_reward_std_r = 0   # Cumulative std reward for right agents
        
        # Reset environment and update agent group indices
        obs = env.reset()  
        start_stop_num = [slice(0, env.n_l_last), slice(env.n_l_last, env.n_lr_last)]   
        # maddpg.prep_rollouts(device='cpu') 
      
        # Configure exploration noise for current episode
        maddpg.scale_noise(maddpg.noise, maddpg.epsilon)
        maddpg.reset_noise()
        
        ########################### step one episode ###########################
        start_time_1 = time.time()
        et_i = 0
        
        # Episode loop: continue until max steps or termination (no agents left)
        while et_i in range(cfg.episode_length) and env.n_l > 0 and env.n_r > 0:
            # Render environment periodically for visualization
            if ep_index % 500 == 0:
                env.render()

            # Generate actions using MADDPG policy with exploration noise
            torch_obs = torch.Tensor(obs).requires_grad_(False)  
            torch_agent_actions = maddpg.step(torch_obs, start_stop_num, explore=True) 
            agent_actions = np.column_stack([ac.data.numpy() for ac in torch_agent_actions])
  
            # Execute actions and collect environment feedback
            next_obs, rewards, dones, _ = env.step(agent_actions)    
            
            # Update agent group indices based on surviving agents
            start_stop_num = [slice(0, env.n_l_last), slice(env.n_l_last, env.n_lr_last)]
            
            # Store experience in replay buffers
            if obs.shape[1] == next_obs.shape[1]:
                left_agents_buffer.push(obs, agent_actions, rewards, next_obs, dones, start_stop_num[0])  
                right_agents_buffer.push(obs, agent_actions, rewards, next_obs, dones, start_stop_num[1])
            obs = next_obs   

            # Accumulate episode statistics
            episode_reward_mean_l += np.mean(rewards[:, start_stop_num[0]])
            episode_reward_mean_r += np.mean(rewards[:, start_stop_num[1]])
            episode_reward_std_l += np.std(rewards[:, start_stop_num[0]]) 
            episode_reward_std_r += np.std(rewards[:, start_stop_num[1]]) 

            et_i += 1

        end_time_1 = time.time()
        
        ########################### train ###########################
        start_time_2 = time.time()
        
        # Prepare networks for training mode
        maddpg.prep_training(device=cfg.device)
        
        # Perform multiple training updates per episode
        for _ in range(20):    
            for a_i in training_index:
                # Only update if buffer has enough samples
                # if len(buffer_total[a_i]) >= buffer_total[a_i].total_length:
                if len(buffer_total[a_i]) >= cfg.batch_size:
                    # Sample batch from replay buffer
                    sample = buffer_total[a_i].sample(cfg.batch_size, to_gpu=True if cfg.device == 'gpu' else False, env_name=cfg.env_name)  
                    obs_sample, acs_sample, rews_sample, next_obs_sample, dones_sample = sample  
                    
                    # Update actor and critic networks
                    maddpg.update(obs_sample, acs_sample, rews_sample, next_obs_sample, dones_sample, a_i, logger=logger)     # parameter update 
            
            # Update target networks with soft updates
            maddpg.update_all_targets()

        # Switch back to rollout mode for next episode
        maddpg.prep_rollouts(device='cpu')  
        
        # Optional: Decay exploration noise over time
        # maddpg.noise = max(0.05, maddpg.noise-5e-5)
        # maddpg.epsilon = max(0.05, maddpg.epsilon-5e-5)
        
        end_time_2 = time.time()  

        ########################### process data ###########################
        # Print training progress every 10 episodes
        if ep_index % 10 == 0:
            avg_reward_l = episode_reward_mean_l / cfg.episode_length
            avg_reward_r = episode_reward_mean_r / cfg.episode_length
            step_time = end_time_1 - start_time_1
            train_time = end_time_2 - start_time_2
            
            print("Episodes %i of %i, left's reward: %f, right's reward: %f, step time: %f, training time: %f, number: (%f, %f)" % 
                  (ep_index, cfg.n_episodes, avg_reward_l, avg_reward_r, step_time, train_time, env.n_l, env.n_r))

        # Log metrics to TensorBoard at specified intervals
        if ep_index % cfg.save_interval == 0:
            logger.add_scalars('data', {
                'episode_reward_mean_l': episode_reward_mean_l / cfg.episode_length, 
                'episode_reward_mean_r': episode_reward_mean_r / cfg.episode_length, 
                'episode_reward_std_l': episode_reward_std_l / cfg.episode_length, 
                'episode_reward_std_r': episode_reward_std_r / cfg.episode_length
            }, ep_index)

        # Save incremental model checkpoints periodically
        if ep_index % (4 * cfg.save_interval) < cfg.n_rollout_threads:   
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_index + 1)))

    # Save final trained model
    maddpg.save(run_dir / 'model.pt')
     
    # Export training logs and close logger
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()

if __name__ == '__main__':
    # Get training configuration and start training
    args = get_adversarial_args()
    run(args)