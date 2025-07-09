"""
Configuration file for Adversarial Swarm Environment
"""

from typing import Union
import numpy as np
import argparse

# Parse all arguments
def get_adversarial_args() -> argparse.Namespace:
    """
    Parse command line arguments for the adversarial swarm environment.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """

    # Initialize argument parser for command line configuration
    parser = argparse.ArgumentParser("Gym-AdversarialSwarm Arguments")

    ## ==================== Environment Parameters ====================

    # Agent population settings
    parser.add_argument("--n-l", type=int, default=50, help='Number of left agents')
    parser.add_argument("--n-r", type=int, default=50, help='Number of right agents')

    # Environment physics and boundaries
    parser.add_argument("--is_boundary", type=bool, default=True, help='Set whether has wall or periodic boundaries')
    parser.add_argument("--is_con_self_state", type=bool, default=True, help="Whether contain myself state in the observation")

    # Movement and dynamics configuration
    parser.add_argument("--dynamics-mode", type=str, default='Cartesian', help="Select one from ['Cartesian', 'Polar']")

    # Agent behavior strategies
    parser.add_argument("--l-strategy", type=str, default='input', help="Select one from ['input', 'static', 'random', 'nearest']")
    parser.add_argument("--r-strategy", type=str, default='nearest', help="Select one from ['input', 'static', 'random', 'nearest']")

    # Visualization settings
    parser.add_argument("--render-traj", type=bool, default=True, help="Whether render trajectories of agents")
    parser.add_argument("--traj_len", type=int, default=10, help="Length of the trajectory")

    # Special modes
    parser.add_argument("--billiards-mode", type=bool, default=False, help="Billiards mode")
    parser.add_argument("--video", type=bool, default=False, help="Record video")

    ## ==================== Training Parameters ====================

    # Basic training configuration
    parser.add_argument("--env_name", default="adversarial", type=str)
    parser.add_argument("--seed", default=226, type=int, help="Random seed")

    # Threading and parallelization
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=5, type=int)

    # Training schedule and memory
    parser.add_argument("--buffer_length", default=int(1e4), type=int)
    parser.add_argument("--n_episodes", default=1200, type=int)
    parser.add_argument("--episode_length", default=200, type=int)
    parser.add_argument("--batch_size", default=512, type=int, help="Batch size for model training")

    # Neural network architecture
    parser.add_argument("--hidden_dim", default=256, type=int)

    # Learning rates and optimization
    parser.add_argument("--lr_actor", default=1e-4, type=float)
    parser.add_argument("--lr_critic", default=1e-3, type=float)

    # Training hyperparameters
    parser.add_argument("--epsilon", default=0.1, type=float)  # Exploration parameter
    parser.add_argument("--noise", default=0.9, type=float)    # Action noise
    parser.add_argument("--tau", default=0.01, type=float)     # Target network update rate

    # Algorithm selection
    parser.add_argument("--agent_alg", default="MADDPG", type=str, choices=['MADDPG', 'DDPG'])
    parser.add_argument("--adversary_alg", default="MADDPG", type=str, choices=['MADDPG', 'DDPG'])

    # Hardware and logging
    parser.add_argument("--device", default="cpu", type=str, choices=['cpu', 'gpu'])
    parser.add_argument("--save_interval", default=50, type=int, help="Save data for every 'save_interval' episodes")

    ## ==================== Training Parameters ====================

    return parser.parse_args()