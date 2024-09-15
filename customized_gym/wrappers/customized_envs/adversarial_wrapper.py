import gym
import numpy as np

class Agent:
    def __init__(self, adversary=False):
        self.adversary = adversary
        
class AdversarialSwarmWrapper(gym.Wrapper):

    def __init__(self, env, args):
        super(AdversarialSwarmWrapper, self).__init__(env)
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

        self.agents = [Agent() for _ in range(self.num_r)] + [Agent(adversary=True) for _ in range(self.num_l)]
        self.agent_types = ['adversary', 'agent']
        env.__reinit__(args)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        print('Adversarial environment initialized successfully.')
