import torch
import time
import os
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import gym
from gym.wrappers import AdversarialSwarmWrapper
from cfg import get_adversarial_args
from algorithm.algorithms import MADDPG

USE_CUDA = False 

def run(cfg, sta, t_id, test_num):
    """
    Main evaluation function for adversarial swarm environment
    
    Args:
        cfg: Configuration object containing hyperparameters
        sta: Statistics array to store results
        t_id: Test ID for current run
        test_num: Total number of tests
    """
 
    ## ======================================= Initialize =======================================
    # Set random seeds for reproducibility
    torch.manual_seed(cfg.seed)  
    np.random.seed(cfg.seed)   
    if not USE_CUDA:
        torch.set_num_threads(cfg.n_training_threads) 

    # Setup model and results directories
    model_dir = Path('./models') / cfg.env_name 
    curr_run = 'your_experiment_name'  # Specific run timestamp

    results_dir = os.path.join(model_dir, curr_run, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Setup video recording if enabled
    if cfg.video:
        cfg.video_path = results_dir + '/video.mp4'

    # Initialize environment
    scenario_name = 'AdversarialSwarm-v0'
    base_env = gym.make(scenario_name).unwrapped
    env = AdversarialSwarmWrapper(base_env, cfg)
    
    # Define agent group indices: left agents (0 to num_l), right agents (num_l to num_l+num_r)
    start_stop_num = [slice(0, env.num_l), slice(env.num_l, env.num_l + env.num_r)]   

    # Load pre-trained MADDPG model
    run_dir = model_dir / curr_run / 'model.pt'
    # run_dir = model_dir / curr_run / 'incremental' / 'model_ep801.pt'  # Alternative checkpoint
    # maddpg = MADDPG.init_from_save(run_dir)
    maddpg = MADDPG.init_from_save_with_id(run_dir, [0, 1])  # Load specific policies: 0(left), 1(right), 2(non-trained)

    # Initialize storage arrays
    p_store = []
    dp_store = []
    torch_agent_actions = []

    ## ======================================= Evaluation =======================================
    # Run evaluation episodes
    for _ in range(0, 1, cfg.n_rollout_threads):

        episode_length = 200
        episode_reward_mean_l = 0  # Left agents cumulative reward
        episode_reward_mean_r = 0  # Right agents cumulative reward
        obs = env.reset()     
        # maddpg.prep_rollouts(device='cpu') 

        # Configure noise for exploration (set to 0 for deterministic evaluation)
        maddpg.scale_noise(maddpg.noise, maddpg.epsilon)
        maddpg.reset_noise()

        # Initialize data storage arrays
        M_l, N_l = np.shape(env.p)     # Position dimensions
        M_v, N_v = np.shape(env.dp)    # Velocity dimensions
        p_store = np.zeros((M_l, N_l, episode_length))       # Position history
        dp_store = np.zeros((M_v, N_v, episode_length))      # Velocity history
        index_store = -np.ones((M_l, N_l, episode_length), dtype=np.int32)  # Agent index history
        
        ########################### step one episode ###########################
        start_time_1 = time.time()
        et_index = 0
        while et_index in range(episode_length):
            # Render environment if this is a visual test
            if test_num == 1:
                env.render()

            # Store current state information
            p_store[:, :, et_index] = env.p             # Store positions
            dp_store[:, :, et_index] = env.dp           # Store velocities
            index_store[0, env.index_l_last, et_index] = env.index_l_last  # Store left agent indices
            index_store[1, env.index_r_last - env.n_l_init, et_index] = env.index_r_last  # Store right agent indices

            # Get actions from MADDPG policy
            torch_obs = torch.Tensor(obs).requires_grad_(False)  # Convert to tensor
            torch_agent_actions = maddpg.step(torch_obs, start_stop_num, explore=False)  # Get deterministic actions
            agent_actions = np.column_stack([ac.data.numpy() for ac in torch_agent_actions])

            # Execute actions and get environment response
            next_obs, rewards, _, _ = env.step(agent_actions)  
            
            # Update agent group indices based on current alive agents
            start_stop_num = [slice(0, env.n_l_last), slice(env.n_l_last, env.n_lr_last)]  
            obs = next_obs    

            # Accumulate rewards for both agent groups
            episode_reward_mean_l += np.mean(rewards[:, start_stop_num[0]])
            episode_reward_mean_r += np.mean(rewards[:, start_stop_num[1]]) 
            
            et_index += 1

        end_time_1 = time.time()
        env.close()

        ########################### process data ###########################
        # Print episode statistics
        avg_reward_l = episode_reward_mean_l / episode_length
        avg_reward_r = episode_reward_mean_r / episode_length
        episode_time = end_time_1 - start_time_1
        
        print("left's reward: %f, right's reward: %f, step time: %f" % (avg_reward_l, avg_reward_r, episode_time))

        # Check win condition (left agents win if right agents are eliminated)
        if env.n_l > 0 and env.n_r == 0:
            # Store statistics: [win_flag, self_damage_rate, simulation_time]
            sta[t_id] = np.array([1, (env.n_l_init - env.n_l) / env.n_l_init, env.simulation_time])
            print('Attack successfully, seed:', cfg.seed)

        # Save episode data
        np.savez(os.path.join(results_dir, 'state_data.npz'), 
                 pos=p_store, vel=dp_store, index_agent=index_store, t_step=et_index)

        ########################### plot ###########################
        # Generate training curve plot for the first test
        if test_num == 1:
            log_dir = model_dir / curr_run / 'logs'
            
            # Load training log data
            with open(log_dir / 'summary.json', 'r') as f:
                data = json.load(f)

            # Extract episode reward data for both agent groups
            episode_rewards_mean_l = data[str(log_dir) + '/data/episode_reward_mean_l']
            episode_rewards_mean_r = data[str(log_dir) + '/data/episode_reward_mean_r']
            episode_rewards_std_l = data[str(log_dir) + '/data/episode_reward_std_l']
            episode_rewards_std_r = data[str(log_dir) + '/data/episode_reward_std_r']

            # Extract timestamps and reward values
            timestamps = np.array([entry[1] for entry in episode_rewards_mean_l])
            rewards_mean_l = np.array([entry[2] for entry in episode_rewards_mean_l])
            rewards_std_l = np.array([entry[2] for entry in episode_rewards_std_l])
            rewards_mean_r = np.array([entry[2] for entry in episode_rewards_mean_r])
            rewards_std_r = np.array([entry[2] for entry in episode_rewards_std_r])

            # Create and save training curve plot
            plt.figure(figsize=(18, 12))
            plt.plot(timestamps, rewards_mean_l, c=(0.0, 0.392, 0.0), label="Blue agents Reward")
            plt.plot(timestamps, rewards_mean_r, c=(1, 0.388, 0.278), label="Orange agents Reward")
            
            # Add confidence intervals
            plt.fill_between(timestamps, rewards_mean_l - rewards_std_l, rewards_mean_l + rewards_std_l, 
                           color=(0.0, 0.392, 0.0), alpha=0.2)
            plt.fill_between(timestamps, rewards_mean_r - rewards_std_r, rewards_mean_r + rewards_std_r, 
                           color=(1, 0.388, 0.278), alpha=0.2)
            
            # Format plot
            plt.xlabel('Episode', fontsize=25)
            plt.ylabel('Reward', fontsize=25)
            plt.xticks(fontsize=25)
            plt.yticks(fontsize=25)
            plt.grid(True)
            plt.title('Episode Reward Curve', fontsize=25)
            plt.legend(loc='lower right', fontsize=25)
            
            # Save and display plot
            plt.savefig(os.path.join(results_dir, 'reward_curve.png'), format='png')
            plt.show()


if __name__ == '__main__':
    # Get configuration arguments
    args = get_adversarial_args()
    test_num = 1
    
    # Initialize statistics array: [win_flag, self_damage_rate, simulation_time]
    statistics = np.zeros((test_num, 3))
    
    # Run evaluation tests
    for test_id in range(test_num):
        args.seed = test_id + 200  # Set different seed for each test
        run(args, statistics, test_id, test_num)  
    
    # Calculate and print final statistics
    num_win = np.count_nonzero(statistics[:, 0])  # Count successful attacks
    win_rate = num_win / test_num
    
    # Calculate self-damage statistics (only for winning episodes)
    if num_win > 0:
        self_damage_rate = np.sum(statistics[:, 1]) / num_win
        self_damage_rate_std = np.std(statistics[:, 1][statistics[:, 1] != 0])
        mean_time = np.sum(statistics[:, 2]) / num_win
        mean_time_std = np.std(statistics[:, 2][statistics[:, 2] != 0])
    else:
        self_damage_rate = self_damage_rate_std = mean_time = mean_time_std = 0
    
    # Print final evaluation results
    print('win rate: {:.1%}, self-damage rate: {:.1%}±{:.1%} mean time: {:.2f}±{:.2f} s'.format(
        win_rate, self_damage_rate, self_damage_rate_std, mean_time, mean_time_std))