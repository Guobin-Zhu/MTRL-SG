import gym
import numpy as np
from scipy.spatial.distance import pdist, squareform

class Agent:
    def __init__(self, adversary=False):
        self.adversary = adversary
        
class FlockingSwarmWrapper(gym.Wrapper):

    def __init__(self, env, args):
        super(FlockingSwarmWrapper, self).__init__(env)
        env.__reinit__(args)
        self.num_agents = self.env.n_a
        self.agents = [Agent() for _ in range(self.num_agents)]
        self.agent_types = ['agent']
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        print('Flocking environment initialized successfully.')


    def order_metric(self, pos, vel, T, N):    
        """
        Args:
            x: agent 的位置列表。
            T: 时间步数。
            N: 智能体总数。
        """
        cos_theta_agent = 0
        for t in range(pos.shape[2]): # traverse time step
            dist = pdist(pos[:, :, t].T)
            dist_mat = squareform(dist)
            sorted_indices = np.argsort(dist_mat, axis=0)
            B = np.take_along_axis(dist_mat, sorted_indices, axis=0)
            I = sorted_indices
            for j in range(N): # traverse agent
                list_nei_indices = B[:, j] <= self.env.d_sen
                list_nei = I[list_nei_indices, j]
                list_nei = list_nei[1:]
                if len(list_nei) > self.env.topo_nei_max:
                    list_nei = list_nei[:self.env.topo_nei_max]

                cos_theta = 0
                if len(list_nei) > 0:
                    for agent2 in list_nei: # traverse neighbors
                        cos_theta += np.dot(vel[:,j,t], vel[:,agent2,t]) / (np.linalg.norm(vel[:,j,t]) * np.linalg.norm(vel[:,agent2,t]) + 1e-6)
                cos_theta_agent += cos_theta / (len(list_nei) + 1e-6)

        order_met = cos_theta_agent / (T * N)

        return order_met

    def dist_metric(self, pos, T, N):
        
        dist_agent = 0
        for t in range(pos.shape[2]): # traverse time step
            dist = pdist(pos[:, :, t].T)
            dist_mat = squareform(dist)
            sorted_indices = np.argsort(dist_mat, axis=0)
            B = np.take_along_axis(dist_mat, sorted_indices, axis=0)
            I = sorted_indices
            for j in range(N): # traverse agent
                list_nei_indices = B[:, j] <= self.env.d_sen
                list_nei = I[list_nei_indices, j]
                list_nei = list_nei[1:]
                if len(list_nei) > self.env.topo_nei_max:
                    list_nei = list_nei[:self.env.topo_nei_max]
                
                count_j = 0
                if len(list_nei) > 0:
                    dist = np.linalg.norm(pos[:,list_nei,t] - pos[:,[j],t], axis=0)
                    count_j = np.sum((dist > self.env.d_ref - 0.05) & (dist < self.env.d_ref + 0.05))
                dist_agent += count_j / (len(list_nei) + 1e-6)

        dist_met = dist_agent / (T * N)

        return dist_met