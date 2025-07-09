import gym
import numpy as np

class Agent:
    """Represents a single agent in the environment"""
    def __init__(self, adversary=False):
        self.adversary = adversary  # Determines agent role (adversary or regular)
        
class EnvHighSwarmWrapper(gym.Wrapper):
    """Wrapper for high-level control in swarm environments"""

    def __init__(self, env, args):
        super(EnvHighSwarmWrapper, self).__init__(env)
        
        # Configure environment parameters from arguments
        self.env.n_l = args.n_l        # Number of leader agents
        self.env.n_r = args.n_r        # Number of regular agents
        self.env.l_strategy = args.l_strategy  # Strategy for leader agents
        self.env.r_strategy = args.r_strategy  # Strategy for regular agents
        self.env.dynamics_mode = args.dynamics_mode  # Physics model
        self.env.render_traj = args.render_traj  # Trajectory rendering flag
        self.env.traj_len = args.traj_len  # Trajectory history length
        self.env.billiards_mode = args.billiards_mode  # Billiards mode flag

        # Track agent counts
        self.num_r = args.n_r  # Regular agent count
        self.num_l = args.n_l  # Leader agent count

        # Agent type classification
        self.agent_types = ['adversary', 'agent']  # Leader/Adversary vs Follower/Regular
        
        # Reinitialize environment with updated parameters
        env.__reinit__(args)
        
        # Configure action spaces
        self.action_space = self.env.action_space  # Standard action space
        self.action_space_high = self.env.action_space_high  # High-level action space
        
        # Configure observation spaces
        self.observation_space_flocking = self.env.observation_space_flocking
        self.observation_space_adversarial = self.env.observation_space_adversarial
        self.observation_space_high = self.env.observation_space_high