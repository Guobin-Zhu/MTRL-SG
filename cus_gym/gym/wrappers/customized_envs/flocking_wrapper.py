import gym
import numpy as np
from scipy.spatial.distance import pdist, squareform

class Agent:
    """Represents a single agent in the flocking environment"""
    def __init__(self, adversary=False):
        self.adversary = adversary  # Determines agent role
        
class FlockingSwarmWrapper(gym.Wrapper):
    """Wrapper for flocking behavior metrics and environment setup"""

    def __init__(self, env, args):
        super(FlockingSwarmWrapper, self).__init__(env)
        env.__reinit__(args)
        self.num_agents = self.env.n_a  # Total agent count
        self.agents = [Agent() for _ in range(self.num_agents)]
        self.agent_types = ['agent']  # Consistent agent typing
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        print('Flocking environment initialized.')

    def order_metric(self, pos, vel, T, N):    
        """
        Measures velocity alignment between neighbors.
        
        Args:
            pos: Position tensor (2 x N x T)
            vel: Velocity tensor (2 x N x T)
            T: Time steps
            N: Agent count
        Returns:
            Normalized velocity alignment metric
        """
        cos_theta_sum = 0
        for t in range(T):  # Iterate through time steps
            dist = pdist(pos[:, :, t].T)
            dist_mat = squareform(dist)
            sorted_indices = np.argsort(dist_mat, axis=0)
            neighbor_dist = np.take_along_axis(dist_mat, sorted_indices, axis=0)
            
            for j in range(N):  # Process each agent
                # Identify neighbors within sensing range
                neighbors_mask = neighbor_dist[:, j] <= self.env.d_sen
                neighbors = sorted_indices[neighbors_mask, j][1:]  # Exclude self
                
                if len(neighbors) > self.env.topo_nei_max:
                    neighbors = neighbors[:self.env.topo_nei_max]  # Limit neighbor count
                
                if neighbors.size > 0:
                    # Compute velocity alignment with neighbors
                    v_j = vel[:, j, t]
                    neighbor_vels = vel[:, neighbors, t]
                    norms_j = np.linalg.norm(v_j)
                    
                    for idx in range(neighbor_vels.shape[1]):
                        v_neighbor = neighbor_vels[:, idx]
                        norms_neighbor = np.linalg.norm(v_neighbor)
                        cos_theta_sum += np.dot(v_j, v_neighbor) / (norms_j * norms_neighbor + 1e-6)
                        
        return cos_theta_sum / (T * N)  # Temporal and agent-wise average

    def dist_metric(self, pos, T, N):
        """
        Measures distance maintenance relative to reference distance.
        
        Args:
            pos: Position tensor (2 x N x T)
            T: Time steps
            N: Agent count
        Returns:
            Normalized distance maintenance metric
        """
        dist_count = 0
        for t in range(T):  # Process each timestep
            dist = pdist(pos[:, :, t].T)
            dist_mat = squareform(dist)
            sorted_indices = np.argsort(dist_mat, axis=0)
            neighbor_dist = np.take_along_axis(dist_mat, sorted_indices, axis=0)
            
            for j in range(N):  # Evaluate each agent
                # Find valid neighbors
                neighbors_mask = neighbor_dist[:, j] <= self.env.d_sen
                neighbors = sorted_indices[neighbors_mask, j][1:]  # Skip self
                
                if len(neighbors) > self.env.topo_nei_max:
                    neighbors = neighbors[:self.env.topo_nei_max]
                
                if neighbors.size > 0:
                    # Calculate distances to neighbors
                    agent_pos = pos[:, j, t]
                    neighbor_positions = pos[:, neighbors, t]
                    distances = np.linalg.norm(neighbor_positions - agent_pos[:, None], axis=0)
                    
                    # Count neighbors near reference distance
                    valid_neighbors = np.sum(
                        (distances > self.env.d_ref - 0.05) & 
                        (distances < self.env.d_ref + 0.05)
                    )
                    dist_count += valid_neighbors / neighbors.size

        return dist_count / (T * N)  # Temporal and agent-wise average