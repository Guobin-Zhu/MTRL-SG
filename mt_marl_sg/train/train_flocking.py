import torch
import time
import os
import numpy as np
from pathlib import Path
from tensorboardX import SummaryWriter
from datetime import datetime

import gym
from gym.wrappers import FlockingSwarmWrapper
from cfg import get_flocking_args
from algorithm.algorithms import MADDPG
from algorithm.utils import ReplayBuffer

def run(cfg):
    """
    Main training function for flocking swarm multi-agent reinforcement learning
    
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
    if cfg.device == 'cpu':
        torch.set_num_threads(cfg.n_training_threads) 

    # Initialize flocking swarm environment
    scenario_name = 'FlockingSwarm-v0'
    base_env = gym.make(scenario_name).unwrapped
    env = FlockingSwarmWrapper(base_env, args)

    # Define agent group indices (all agents in one cooperative group)
    start_stop_num = [slice(0, env.num_agents)]  

    # Initialize MADDPG algorithm with flocking-specific parameters
    maddpg = MADDPG.init_from_env(env, agent_alg=cfg.agent_alg, tau=cfg.tau, 
                                  lr_actor=cfg.lr_actor, lr_critic=cfg.lr_critic, 
                                  lambda_s=cfg.lambda_s, epsilon_p=cfg.epsilon_p, 
                                  hidden_dim=cfg.hidden_dim, device=cfg.device, 
                                  epsilon=cfg.epsilon, noise=cfg.noise_scale)
    
    # Alternative: Load from previous checkpoint
    # last_run = '2024-07-27-16-33-51'
    # last_run_dir = model_dir / last_run / 'model.pt'
    # maddpg = MADDPG.init_from_save(last_run_dir)

    # Initialize replay buffer for all agents
    agent_buffer = [ReplayBuffer(cfg.buffer_length, env.num_agents, 
                                state_dim=env.observation_space.shape[0], 
                                action_dim=env.action_space.shape[0], 
                                start_stop_index=start_stop_num[0])]    

    ## ======================================= Training =======================================
    print('Training Starts...')
    
    # Main training loop over episodes
    for ep_index in range(0, cfg.n_episodes, cfg.n_rollout_threads):

        # Initialize episode statistics
        episode_reward_mean = 0  # Cumulative mean reward for all agents
        episode_reward_std = 0   # Cumulative reward standard deviation
        
        # Reset environment for new episode
        obs = env.reset()     
        maddpg.prep_rollouts(device='cpu')  # Set networks to evaluation mode
      
        # Configure exploration noise for current episode
        maddpg.scale_noise(maddpg.noise, maddpg.epsilon)
        maddpg.reset_noise()
        
        ########################### step one episode ###########################
        start_time_1 = time.time()
        
        # Execute one complete episode
        for _ in range(cfg.episode_length):
            # Render environment periodically for visualization
            if ep_index % 500 == 0:
                env.render()

            # Generate actions using MADDPG policy with exploration noise
            torch_obs = torch.Tensor(obs).requires_grad_(False)  
            torch_agent_actions = maddpg.step(torch_obs, start_stop_num, explore=True) 
            agent_actions = np.column_stack([ac.data.numpy() for ac in torch_agent_actions])

            # Execute actions and collect environment feedback
            next_obs, rewards, dones, _ = env.step(agent_actions)    
            
            # Store experience in replay buffer
            agent_buffer[0].push(obs, agent_actions, rewards, next_obs, dones, start_stop_num[0])
            obs = next_obs  

            # Accumulate episode statistics
            episode_reward_mean += np.mean(rewards)  # Mean reward across all agents
            episode_reward_std += np.std(rewards)    # Reward standard deviation

        end_time_1 = time.time()
        
        ########################### train ###########################
        start_time_2 = time.time()
        
        # Prepare networks for training mode
        maddpg.prep_training(device=cfg.device)
        
        # Perform multiple training updates per episode
        for _ in range(20):      
            # Update each agent's policy
            for a_i in range(maddpg.nagents):
                # Only update if buffer has enough samples
                if len(agent_buffer[a_i]) >= cfg.batch_size:
                    # Sample batch from replay buffer
                    sample = agent_buffer[a_i].sample(cfg.batch_size, to_gpu=True if cfg.device == 'gpu' else False)  
                    obs_sample, acs_sample, rews_sample, next_obs_sample, dones_sample = sample  
                    
                    # Update actor and critic networks
                    maddpg.update(obs_sample, acs_sample, rews_sample, next_obs_sample, dones_sample, a_i, logger=logger)     # parameter update 
            
            # Update target networks with soft updates
            maddpg.update_all_targets()
            
        # Switch back to rollout mode for next episode
        maddpg.prep_rollouts(device='cpu')    
        
        # Optional: Decay exploration noise over time
        # maddpg.noise = max(0.1, maddpg.noise - cfg.noise_scale/cfg.n_episodes)
        # maddpg.epsilon = max(0.1, maddpg.epsilon - cfg.epsilon/cfg.n_episodes)
        
        end_time_2 = time.time()

        ########################### process data ###########################
        # Print training progress every 10 episodes
        if ep_index % 10 == 0:
            avg_reward = episode_reward_mean / cfg.episode_length
            step_time = end_time_1 - start_time_1
            train_time = end_time_2 - start_time_2
            
            print("Episodes %i of %i, episode reward: %f, step time: %f, training time: %f" % 
                  (ep_index, cfg.n_episodes, avg_reward, step_time, train_time))
        
        # Log metrics to TensorBoard at specified intervals
        if ep_index % cfg.save_interval == 0:
            ALIGN_epi = 0 
            logger.add_scalars('agent/data', {
                'episode_reward_mean': episode_reward_mean / cfg.episode_length, 
                'episode_reward_std': episode_reward_std / cfg.episode_length, 
                'ALIGN_epi': ALIGN_epi
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
    # Get flocking configuration and start training
    args = get_flocking_args()
    run(args)