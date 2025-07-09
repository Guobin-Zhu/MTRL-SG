import torch
import time
import os
import numpy as np
import gym
from gym.wrappers import FlockingSwarmWrapper
from cfg import get_flocking_args
from pathlib import Path
from algorithm.algorithms import MADDPG
import json
import matplotlib.pyplot as plt

# Device configuration
USE_CUDA = False  # Flag for CUDA usage

def run(cfg, sta, t_id, test_num):
    """Main execution function for evaluating the flocking model.
    """

    ## ======================================= Initialize =======================================
    # Set seeds for reproducibility
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if not USE_CUDA:
        torch.set_num_threads(cfg.n_training_threads)  # Configure CPU threads

    # Model and result directories setup
    model_dir = Path('./models') / cfg.env_name
    curr_run = 'your_experiment_name'  # Timestamp-based run identifier
    results_dir = os.path.join(model_dir, curr_run, 'results')
    os.makedirs(results_dir, exist_ok=True)  # Create results directory if needed
    
    if args.video:
        args.video_path = os.path.join(results_dir, 'video.mp4')

    # Environment setup
    scenario_name = 'FlockingSwarm-v0'
    base_env = gym.make(scenario_name).unwrapped
    env = FlockingSwarmWrapper(base_env, args)  # Wrap environment
    start_stop_num = [slice(0, env.num_agents)]  # Agent indexing

    # Algorithm setup
    run_dir = model_dir / curr_run / 'model.pt'
    # run_dir = model_dir / curr_run / 'incremental' / 'model_ep801.pt'  # Alternative checkpoint
    maddpg = MADDPG.init_from_save(run_dir)  # Load pretrained model

    ## ======================================= Evaluation =======================================
    print('Evaluation started...')
    for ep_index in range(0, 1, cfg.n_rollout_threads):  # Single episode loop
        episode_length = 500
        episode_reward = 0
        obs = env.reset()  # Initialize environment
        
        # Prepare for evaluation
        maddpg.scale_noise(0, 0)  # Disable exploration noise
        maddpg.reset_noise()

        # Initialize state storage
        p_store = np.zeros((2, env.num_agents, episode_length))  # Position history
        dp_store = np.zeros((2, env.num_agents, episode_length))  # Velocity history
        
        ########################### Execute one episode ###########################
        start_time_1 = time.time()
        for et_index in range(episode_length):
            if test_num == 1:  # Render only for single test
                env.render()
            
            # Record current states
            p_store[:, :, et_index] = env.p
            dp_store[:, :, et_index] = env.dp

            # Get actions from policy
            torch_obs = torch.Tensor(obs).requires_grad_(False)
            torch_agent_actions = maddpg.step(torch_obs, start_stop_num, explore=False)
            agent_actions = np.column_stack([ac.detach().numpy() for ac in torch_agent_actions])

            # Environment step
            next_obs, rewards, _, _ = env.step(agent_actions)
            obs = next_obs
            episode_reward += np.mean(rewards)  # Accumulate reward

        end_time_1 = time.time()
        duration = end_time_1 - start_time_1

        ########################### Process results ###########################
        print(f"Episode {ep_index+1}, Reward/step: {episode_reward/episode_length:.4f}, "
              f"Time: {duration:.2f}s")
        
        # Calculate metrics
        dist_epi = env.dist_metric(p_store, episode_length, env.num_agents)
        order_epi = env.order_metric(p_store, dp_store, episode_length, env.num_agents)
        
        # Store statistics
        sta[t_id, 0] = dist_epi
        sta[t_id, 1] = order_epi

        # Save trajectory data
        np.savez(os.path.join(results_dir, 'state_data.npz'), 
                 pos=p_store, vel=dp_store, t_step=et_index)

        ########################### Generate plots ###########################
        if test_num == 1:  # Only for single test run
            log_dir = model_dir / curr_run / 'logs'
            with open(log_dir / 'summary.json', 'r') as f:
                data = json.load(f)

            # Extract training rewards
            mean_key = next(k for k in data if 'episode_reward_mean' in k)
            std_key = next(k for k in data if 'episode_reward_std' in k)
            
            rewards_mean = np.array([entry[2] for entry in data[mean_key]])
            rewards_std = np.array([entry[2] for entry in data[std_key]])
            timestamps = np.array([entry[1] for entry in data[mean_key]])

            # Plot learning curve
            plt.figure(figsize=(8, 6))
            plt.plot(timestamps, rewards_mean, label='Mean Reward')
            plt.fill_between(timestamps, 
                            rewards_mean - rewards_std, 
                            rewards_mean + rewards_std,
                            alpha=0.3, label='Std Dev')
            
            plt.xlabel('Episode', fontsize=12)
            plt.ylabel('Reward', fontsize=12)
            plt.xticks(fontsize=11)
            plt.yticks(fontsize=11)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.title('Training Performance', fontsize=14)
            plt.legend(fontsize=11)
            plt.tight_layout()
            
            # Save and display
            plt.savefig(os.path.join(results_dir, 'reward_curve.pdf'), dpi=300)
            plt.show()


if __name__ == '__main__':
    # Entry point for execution
    args = get_flocking_args()
    test_num = 1
    statistics = np.zeros((test_num, 3))  # Store metrics [distance, order]
    
    # Execute multiple test runs
    for test_id in range(test_num):
        args.seed = test_id + 200  # Unique seed per test
        run(args, statistics, test_id, test_num)

    # Calculate and print summary statistics
    mean_dist = statistics[:, 0].mean()
    mean_dist_std = statistics[:, 0].std()
    mean_order = statistics[:, 1].mean()
    mean_order_std = statistics[:, 1].std()
    
    print(f'\n{"-"*50}\nSummary over {test_num} runs:')
    print(f'Distance Metric: {mean_dist:.1%} ± {mean_dist_std:.1%}')
    print(f'Order Metric:    {mean_order:.1%} ± {mean_order_std:.1%}')
    print('-'*50)