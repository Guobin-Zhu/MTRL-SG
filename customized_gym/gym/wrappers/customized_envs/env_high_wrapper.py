import gym
import numpy as np

class Agent:
    def __init__(self, adversary=False):
        self.adversary = adversary
        
class EnvHighSwarmWrapper(gym.Wrapper):

    def __init__(self, env, args):
        super(EnvHighSwarmWrapper, self).__init__(env)
        self.env.n_l = args.n_l
        self.env.n_r = args.n_r
        self.env.l_strategy = args.l_strategy
        self.env.r_strategy = args.r_strategy
        self.env.dynamics_mode = args.dynamics_mode
        self.env.render_traj = args.render_traj
        self.env.traj_len = args.traj_len
        self.env.billiards_mode = args.billiards_mode

        self.num_r = args.n_r
        self.num_l = args.n_l

        self.agent_types = ['adversary', 'agent']
        env.__reinit__(args)
        self.action_space = self.env.action_space
        self.action_space_high = self.env.action_space_high
        self.observation_space_flocking = self.env.observation_space_flocking
        self.observation_space_adversarial = self.env.observation_space_adversarial
        self.observation_space_high = self.env.observation_space_high
        print('Comprehensive environment initialized successfully.')
