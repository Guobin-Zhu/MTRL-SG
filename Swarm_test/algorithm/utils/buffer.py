import numpy as np
from torch import Tensor

class ReplayBuffer(object):
    """
    Replay Buffer for multi-agent RL with parallel rollouts
    """
    def __init__(self, max_steps, num_agents, start_stop_index, state_dim, action_dim):
        """
        Initialize replay buffer for storing multi-agent experience
        
        Inputs:
            max_steps (int): Maximum number of timepoints to store in buffer
            num_agents (int): Number of agents in environment
            start_stop_index: Index range for selecting specific agents
            state_dim (int): Dimension of state/observation space
            action_dim (int): Dimension of action space
        """
        self.max_steps = max_steps
        self.num_agents = num_agents
        
        # Initialize empty buffer lists (unused in current implementation)
        self.obs_buffs = []
        self.ac_buffs = []
        self.rew_buffs = []
        self.next_obs_buffs = []
        self.done_buffs = []
        
        # Total buffer capacity
        self.total_length = self.max_steps * self.num_agents

        # Create buffer arrays with total capacity
        self.obs_buffs = np.zeros((self.total_length, state_dim)) 
        self.ac_buffs = np.zeros((self.total_length, action_dim))
        self.rew_buffs = np.zeros((self.total_length, 1))
        self.next_obs_buffs = np.zeros((self.total_length, state_dim))
        self.done_buffs = np.zeros((self.total_length, 1))

        # Buffer management indices
        self.filled_i = 0  # index of first empty location in buffer (last index when full)
        self.curr_i = 0  # current index to write to (overwrite oldest data)

        # Agent selection index for filtering specific agents
        self.agent_index = start_stop_index

    def __len__(self):
        """Return current number of stored experiences"""
        return self.filled_i
                                            
    def push(self, observations_original, actions_original, rewards_original, next_observations_original, dones_original, index):
        """
        Add new experiences to the buffer with flexible indexing
        
        Inputs:
            observations_original: Original observations from all agents
            actions_original: Original actions from all agents  
            rewards_original: Original rewards from all agents
            next_observations_original: Original next observations from all agents
            dones_original: Original done flags from all agents
            index: Slice object defining which agents to store
        """
        # Extract agent range information
        start = index.start
        stop = index.stop
        span = range(start, stop)
        data_length = len(span)

        # Extract data for specific agents using index slice
        observations = observations_original[:, index].T   
        actions = actions_original[:,index].T
        rewards = rewards_original[:, index].T                  
        next_observations = next_observations_original[:, index].T
        dones = dones_original[:, index].T          
   
        # Handle buffer overflow with simple wraparound (rolling disabled)
        if self.curr_i + data_length > self.total_length:   
            rollover = data_length - (self.total_length - self.curr_i) # num of indices to roll over
            self.curr_i -= rollover

        # Store data_length transitions at current position
        self.obs_buffs[self.curr_i:self.curr_i + data_length, :] = observations             
        self.ac_buffs[self.curr_i:self.curr_i + data_length, :] = actions # actions are already batched by agent, so they are indexed differently
        self.rew_buffs[self.curr_i:self.curr_i + data_length, :] = rewards
        self.next_obs_buffs[self.curr_i:self.curr_i + data_length, :] = next_observations     
        self.done_buffs[self.curr_i:self.curr_i + data_length, :] = dones         

        # Update current index
        self.curr_i += data_length

        # Update filled counter with buffer capacity limit
        if self.filled_i < self.total_length:
            self.filled_i += data_length  
        else:
            self.filled_i = self.total_length         
        
        # Reset current index when buffer is full
        if self.curr_i >= self.total_length: 
            self.curr_i = 0  

    def sample(self, N, to_gpu=False, env_name='flocking'):
        """
        Sample a batch of experiences from the buffer
        
        Inputs:
            N (int): Number of experiences to sample
            to_gpu (bool): Whether to move tensors to GPU
            env_name (str): Name of the environment for specific sampling logic
        
        Returns:
            tuple: (observations, actions, rewards, next_observations, dones)
        """
        # Initialize arrays for sampled data
        obs_inds = np.zeros((N, self.obs_buffs.shape[1]))
        act_inds = np.zeros((N, self.ac_buffs.shape[1]))
        rew_inds = np.zeros((N, 1))
        next_obs_inds = np.zeros((N, self.next_obs_buffs.shape[1]))
        done_inds = np.zeros((N, 1))

        # Generate random indices for sampling (fixed range approach)
        begin_index_range = 2e5 
        if env_name == 'adversarial':
            begin_index_range = 4e5 # total length = 5e5
        elif env_name == 'flocking':
            begin_index_range = 2e5 # total length = 2.5e5

        begin_index = np.random.randint(0, begin_index_range)
        inds = np.random.choice(np.arange(begin_index, self.total_length - begin_index_range + begin_index, dtype=np.int32), size=N, replace=False)

        # Extract sampled experiences
        obs_inds = self.obs_buffs[inds, :]
        act_inds = self.ac_buffs[inds, :]
        rew_inds = self.rew_buffs[inds, :]
        next_obs_inds = self.next_obs_buffs[inds, :]
        done_inds = self.done_buffs[inds, :]
        
        # Convert to tensors with optional GPU transfer
        if to_gpu:
            cast = lambda x: Tensor(x).requires_grad_(False).cuda()
        else:
            cast = lambda x: Tensor(x).requires_grad_(False)

        return (cast(obs_inds), cast(act_inds), cast(rew_inds), cast(next_obs_inds), cast(done_inds))

    def get_average_rewards(self, N):
        """
        Get average rewards for the last N experiences
        
        Inputs:
            N (int): Number of recent experiences to average over
            
        Returns:
            list: Average rewards for each agent
        """
        # Select indices for last N experiences
        if self.filled_i == self.max_steps:
            inds = np.arange(self.curr_i - N, self.curr_i)  # allow for negative indexing
        else:
            inds = np.arange(max(0, self.curr_i - N), self.curr_i)   
        
        # Return mean rewards for each agent
        return [self.rew_buffs[i][inds].mean() for i in range(self.num_agents)]