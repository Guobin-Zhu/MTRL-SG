from torch import Tensor
from torch.optim import Adam
from .networks import MLPNetwork        
from .misc import hard_update, gumbel_softmax, onehot_from_logits
from .noise import GaussianNoise
import numpy as np

class DDPGAgent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, dim_input_policy, dim_output_policy, dim_input_critic,
                 lr_actor, lr_critic, hidden_dim=64, discrete_action=False, epsilon=0.1, noise=0.1):
        """
        Initialize DDPG agent with policy and critic networks
        
        Inputs:
            dim_input_policy (int): number of dimensions for policy input
            dim_output_policy (int): number of dimensions for policy output
            dim_input_critic (int): number of dimensions for critic input
            lr_actor (float): learning rate for actor/policy network
            lr_critic (float): learning rate for critic network
            hidden_dim (int): hidden layer dimension for networks
            discrete_action (bool): whether actions are discrete or continuous
            epsilon (float): probability for random action in exploration
            noise (float): noise scale for continuous action exploration
        """
        # Main policy network (actor)
        self.policy = MLPNetwork(dim_input_policy, dim_output_policy, 
                                 hidden_dim=hidden_dim, 
                                 constrain_out=True,
                                 discrete_action=discrete_action)
        
        # Target policy network for stable training
        self.target_policy = MLPNetwork(dim_input_policy, dim_output_policy,
                                        hidden_dim=hidden_dim,
                                        constrain_out=True,
                                        discrete_action=discrete_action)
        
        # Main critic network (value function)
        self.critic = MLPNetwork(dim_input_critic, 1,
                                 hidden_dim=hidden_dim,
                                 constrain_out=False)
        
        # Target critic network for stable training
        self.target_critic = MLPNetwork(dim_input_critic, 1,
                                        hidden_dim=hidden_dim,
                                        constrain_out=False)
        
        # Initialize target networks with same weights as main networks
        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        
        # Optimizers for training
        self.policy_optimizer = Adam(self.policy.parameters(), lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr_critic)
        
        # Exploration parameters
        self.epsilon = epsilon
        self.noise = noise
        
        # Set up exploration strategy based on action type
        if not discrete_action:
            # Gaussian noise for continuous actions
            self.exploration = GaussianNoise(dim_output_policy, noise)   
        else:
            # Epsilon-greedy for discrete actions
            self.exploration = 0.3  # epsilon for eps-greedy
            
        self.discrete_action = discrete_action

    def reset_noise(self):
        """Reset exploration noise (for continuous actions)"""
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        """Scale exploration noise/epsilon"""
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        # Get action from policy network
        action = self.policy(obs)             
        
        if self.discrete_action: # discrete action
            if explore:
                # Use Gumbel-Softmax for differentiable discrete actions
                action = gumbel_softmax(action, hard=True)
            else:
                # Convert logits to one-hot for deterministic action
                action = onehot_from_logits(action)
        else:  # continuous action
            if explore:
                if np.random.rand() < self.epsilon:
                    # Random action with epsilon probability
                    action = Tensor(np.random.uniform(-1, 1, size=action.shape)).requires_grad_(False)
                else:
                    # Add exploration noise to policy output
                    action += Tensor(self.exploration.noise(action.shape[0])).requires_grad_(False)
                    # Clamp actions to valid range
                    action = action.clamp(-1, 1) 
    
        return action.t()                        

    def get_params(self):
        """Get all network parameters and optimizer states for saving"""
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        """Load all network parameters and optimizer states from checkpoint"""
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])