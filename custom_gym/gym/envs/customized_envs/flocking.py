__credits__ = ["zhugb@buaa.edu.cn"]

import gym
from gym import spaces
from scipy.spatial.distance import pdist, squareform
import numpy as np
import matplotlib.pyplot as plt
from .VideoWriter import VideoWriter
import ctypes
from .envs_cplus.c_lib import as_double_c_array, as_bool_c_array, as_int32_c_array, _load_lib

_LIB = _load_lib(env_name='Flocking')

class FlockingSwarmEnv(gym.Env):
    """Gym environment for flocking/swarm behavior simulation."""

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 45}
    
    def __init__(self):
        """Initialize environment parameters and default settings."""
        # Reward settings
        self.reward_sharing_mode = 'individual'  # ['sharing_mean', 'sharing_max', 'individual']
        
        # Penalty configuration flags
        self.penalize_control_effort = False
        self.penalize_inter_agent_distance = True
        self.penalize_collide_agents = False
        self.penalize_collide_obstacles = False
        self.penalize_collide_walls = False

        # Simulation dimensionality
        self.dim = 2

        # Agent and obstacle counts
        self.n_a = 10  # Number of agents
        self.n_o = 0   # Number of obstacles

        # Observation parameters
        self.topo_nei_max = 6  # Max neighbors per agent in observation
        
        # Action dimensionality
        self.act_dim_agent = self.dim
        
        # Physical properties
        self.m_a = 1     # Agent mass
        self.m_o = 10    # Obstacle mass
        
        # Size properties  
        self.size_a = 0.035  # Agent size
        self.size_o = 0.2    # Obstacle size
        
        # Interaction distances
        self.d_sen = 3  # Sensing distance
        self.d_ref = 0.6 # Reference distance

        # Physical constraints
        self.Vel_max = 1   # Max velocity
        self.Vel_min = 0.0 # Min velocity
        self.Acc_max = 1   # Max acceleration

        # Virtual leader configuration
        self.center_leader_state = np.zeros((2*self.dim, 1))
        self.way_count = 1
        self.leader_v = 0.3  # Leader velocity

        # Obstacle properties
        self.obstacles_cannot_move = True 
        self.obstacles_is_constant = False
        if self.obstacles_is_constant:  # Fixed obstacle positions
            self.p_o = np.array([[-0.5, 0.5],[0, 0]])

        # Boundary configuration
        self.boundary_L_half = 3  # Half-length of boundary
        self.bound_center = np.zeros(2)  # Center position

        # Physics parameters
        self.L = self.boundary_L_half
        self.k_ball = 30   # Agent-agent contact stiffness 
        self.k_wall = 100   # Agent-wall contact stiffness
        self.c_wall = 5     # Agent-wall damping
        self.c_aero = 1.2   # Aerodynamic drag

        # Simulation timing
        self.simulation_time = 0
        self.dt = 0.1       # Time step
        self.n_frames = 1   # Physics steps per env step
        self.sensitivity = 1  # Action scaling

        # Rendering settings
        self.traj_len = 15  # Trajectory history length
        self.plot_initialized = 0  # Plot initialization flag
        self.center_view_on_swarm = False  # Dynamic view centering
        self.fontsize = 20
        width = 13
        height = 13
        self.figure_handle = plt.figure(figsize=(width, height))

    def __reinit__(self, args):
        """Reinitialize environment with new parameters."""
        # Set agent counts and labels
        self.n_a = args.n_a
        self.n_l = args.n_l
        self.s_text = np.char.mod('%d', np.arange(self.n_a))
        
        # Rendering settings
        self.render_traj = args.render_traj
        self.traj_len = args.traj_len
        self.video = args.video

        # Boundary configuration
        self.is_boundary = args.is_boundary
        self.is_periodic = False if self.is_boundary else True

        # Leader configuration
        self.is_leader = args.is_leader[0]
        self.is_remarkable_leader = args.is_leader[1] if self.is_leader else False
        
        # Dynamics configuration
        self.dynamics_mode = args.dynamics_mode
        self.agent_strategy = args.agent_strategy
        self.is_augmented = args.augmented
        self.is_con_self_state = args.is_con_self_state

        # Waypoint configuration
        self.leader_waypoint_origin = args.leader_waypoint[:self.dim]
        self.leader_waypoint = args.leader_waypoint[:self.dim]

        # Preallocate state arrays
        self.n_ao = self.n_a + self.n_o
        self.B = np.zeros((self.n_ao, self.n_ao))
        self.I = np.zeros((self.n_ao, self.n_ao))
        self.is_collide_b2b = np.random.choice([True, False], size=(self.n_a, self.n_a))
        self.is_collide_b2w = np.zeros((4, self.n_a), dtype=bool)
        self.d_b2w = np.ones((4, self.n_ao))

        # Initialize spaces and properties
        self.observation_space = self._get_observation_space()  
        self.action_space = self._get_action_space()   
        self.m = self._get_mass()  
        self.size, self.sizes = self._get_size()  
        
        # Validate dynamics settings
        if self.dynamics_mode == 'Cartesian':
            self.is_Cartesian = True
            self.Acc_min = -1    
            assert (self.Acc_min, self.Acc_max) == (-1, 1)
        elif self.dynamics_mode == 'Polar':
            self.is_Cartesian = False
            self.Acc_min = 0    
            self.angle_max = 0.5   
            assert self.Acc_min >= 0
        else:
            raise ValueError('Invalid dynamics mode')
        
        # Set agent colors (blue for leaders, orange for followers)
        self.color = np.tile(np.array([1, 0.5, 0]), (self.n_a, 1))
        if self.is_leader:
            self.color[:self.n_l] = np.tile(np.array([0, 0, 1]), (self.n_l, 1))
        
        # Initialize video recording
        if self.video:
            self.video = VideoWriter(output_rate=self.dt, fps=40)
            self.video.video.setup(self.figure_handle, args.video_path)

    def reset(self):
        """Reset environment to initial state."""
        self.simulation_time = 0
        self.d_sen = 6  # Reset sensing distance
        self.heading = np.zeros((self.dim, self.n_a))  # Initial heading
        self.leader_state = np.zeros((2 * self.dim, self.n_l))  # Leader state

        # Set boundary positions
        self.bound_center = np.array([0, 0])
        self.boundary_pos = np.array([
            self.bound_center[0] - self.boundary_L_half,  # x_min
            self.bound_center[1] + self.boundary_L_half,  # y_max
            self.bound_center[0] + self.boundary_L_half,  # x_max
            self.bound_center[1] - self.boundary_L_half   # y_min
        ], dtype=np.float64)

        # Initialize agent positions
        random_int = 3
        if self.is_leader:
            self.p = np.concatenate((
                np.random.uniform(-random_int, random_int, (2, self.n_l)) / 2, 
                np.random.uniform(-random_int, random_int, (2, self.n_ao - self.n_l))
            ), axis=1)
        else:
            self.p = np.random.uniform(-random_int, random_int, (2, self.n_ao))

        # Initialize trajectory history
        if self.render_traj:
            self.p_traj = np.zeros((self.traj_len, 2, self.n_ao))
            self.p_traj[-1, :, :] = self.p
         
        # Initialize velocities
        if self.dynamics_mode == 'Cartesian':
            self.dp = np.random.uniform(-0.5, 0.5, (self.dim, self.n_a))
        else:
            self.dp = np.zeros((self.dim, self.n_ao))

        # Fixed obstacles have zero velocity
        if self.obstacles_cannot_move:
            self.dp[:, self.n_a:self.n_ao] = 0

        # Initialize acceleration
        self.ddp = np.zeros((2, self.n_ao))

        # Configure leader behavior
        if self.is_boundary:
            # Set leader positions to waypoints
            self.leader_state[:self.dim] = self.leader_waypoint[:, [0]] + self.p[:, :self.n_l]
            self.center_leader_state[:self.dim] = self.leader_waypoint[:, [0]]
            self.initial_p = self.p.copy()
        else:
            # Random leader velocity in open space
            self.leader_state[:self.dim] = np.zeros((2, self.n_l))
            random_integer = np.random.randint(0, 16, size=1) * 2 * np.pi / 15 * np.ones(self.n_l)
            self.leader_state[self.dim:] = self.leader_v * np.array([np.cos(random_integer), np.sin(random_integer)])

        # Initialize heading for polar dynamics
        if self.dynamics_mode == 'Polar': 
            self.theta = np.pi * np.random.uniform(-1, 1, (1, self.n_ao))
            self.heading = np.concatenate((np.cos(self.theta), np.sin(self.theta)), axis=0)

        # Get initial observations
        obs = self._get_obs()
        return obs

    def _get_obs(self):
        """Compute observation vector for each agent."""
        # Initialize observation array
        self.obs = np.zeros(self.observation_space.shape) 
        
        # Preallocate neighbor tracking arrays
        self.neighbor_index = -1 * np.ones((self.n_a, self.topo_nei_max), dtype=np.int32)
        self.nearest_leader_index = -1 * np.ones((self.n_a), dtype=np.int32)
        
        # Create permutation for observation ordering
        random_permutation = np.random.permutation(self.topo_nei_max).astype(np.int32)
        
        # Pack environment conditions into array
        conditions = np.array([
            self.is_periodic,
            self.is_leader,
            self.is_remarkable_leader,
            self.is_Cartesian,
            self.is_augmented,
            self.is_con_self_state
        ])

        # Call C++ observation function
        _LIB._get_observation(as_double_c_array(self.p), 
                              as_double_c_array(self.dp), 
                              as_double_c_array(self.heading),
                              as_double_c_array(self.obs),
                              as_double_c_array(self.leader_state), 
                              as_int32_c_array(self.neighbor_index),
                              as_int32_c_array(self.nearest_leader_index),
                              as_int32_c_array(random_permutation),
                              as_double_c_array(self.boundary_pos),
                              ctypes.c_double(self.d_sen), 
                              ctypes.c_int(self.topo_nei_max), 
                              ctypes.c_int(self.n_a),
                              ctypes.c_int(self.n_l), 
                              ctypes.c_int(self.obs_dim_agent), 
                              ctypes.c_int(self.dim), 
                              as_bool_c_array(conditions))

        # self.obs = np.zeros(self.observation_space.shape)

        # # Process each agent to construct their observation
        # for i in range(self.n_a):
        #     # Compute relative position and velocity to other agents
        #     relPos_a2a = self.p[:, :self.n_a] - self.p[:, [i]]
            
        #     # Handle periodic boundary conditions
        #     if self.is_periodic:
        #         relPos_a2a = self._make_periodic(relPos_a2a, is_rel=True)
            
        #     # Compute relative velocities based on dynamics mode
        #     if self.dynamics_mode == 'Cartesian':
        #         relVel_a2a = self.dp[:, :self.n_a] - self.dp[:, [i]]
        #     else:  # Polar dynamics
        #         relVel_a2a = self.heading[:, :self.n_a] - self.heading[:, [i]]
            
        #     # Identify neighbors within sensing range
        #     relPos_a2a, relVel_a2a, neigh_index = self._get_focused(
        #         relPos_a2a, relVel_a2a, self.d_sen, self.topo_nei_max, True
        #     )
        #     nei_num = len(neigh_index)
            
        #     # Track neighbor indices for current agent
        #     if nei_num > 0:
        #         self.neighbor_index[i, :nei_num] = neigh_index
            
        #     # Optionally permute neighbor order (data augmentation)
        #     if self.is_augmented:
        #         relPos_a2a = relPos_a2a[:, random_permutation]
        #         relVel_a2a = relVel_a2a[:, random_permutation]
            
        #     # Construct agent's observation space
        #     if self.is_con_self_state:
        #         # Include own position and velocity in observation
        #         obs_agent_pos = np.concatenate((self.p[:, [i]], relPos_a2a), axis=1)
        #         obs_agent_vel = np.concatenate((self.dp[:, [i]], relVel_a2a), axis=1)
        #         obs_agent = np.concatenate((obs_agent_pos, obs_agent_vel), axis=0)
        #     else:
        #         # Observation contains only relative neighbor states
        #         obs_agent = np.concatenate((relPos_a2a, relVel_a2a), axis=0)
            
        #     # Leader-aware agent processing
        #     if self.is_leader and self.is_remarkable_leader:
        #         # Compute leader-relative positions
        #         leader_pos_rel_mat = self.leader_state[:self.dim, :] - self.p[:, [i]]
        #         if self.is_periodic:
        #             leader_pos_rel_mat = self._make_periodic(leader_pos_rel_mat, is_rel=True)
                
        #         # Identify nearest leader
        #         if self.n_l > 1:
        #             leader_agent_norms = np.linalg.norm(leader_pos_rel_mat, axis=0)
        #             min_index = np.argmin(leader_agent_norms)
        #         else:
        #             min_index = 0
                
        #         # Initialize leader-relative state vectors
        #         leader_pos_rel = np.zeros(self.dim)
        #         leader_vel_rel = np.zeros(self.dim)
        #         leader_heading_rel = np.zeros(self.dim)
                
        #         # Only track leaders within sensing range
        #         if np.linalg.norm(leader_pos_rel_mat[:, min_index]) < self.d_sen:
        #             self.nearest_leader_index[i] = min_index
        #             leader_pos_rel = leader_pos_rel_mat[:, min_index]
        #             leader_vel_rel = self.leader_state[self.dim:, min_index] - self.dp[:, i]
                    
        #             # Compute relative heading
        #             norm_1 = np.linalg.norm(self.leader_state[self.dim:, min_index]) + 1e-8
        #             leader_heading_rel = self.leader_state[self.dim:, min_index] / norm_1 - self.heading[:, i]
                
        #         # Format observation based on dynamics mode
        #         if self.dynamics_mode == 'Cartesian':
        #             # Cartesian: position + velocity + leader state
        #             self.obs[:self.obs_dim_agent - 2*self.dim, i] = obs_agent.T.reshape(-1)       
        #             self.obs[self.obs_dim_agent - 2*self.dim:self.obs_dim_agent - self.dim, i] = leader_pos_rel
        #             self.obs[self.obs_dim_agent - self.dim:, i] = leader_vel_rel
        #         elif self.dynamics_mode == 'Polar':
        #             # Polar: position + heading + leader state
        #             self.obs[:self.obs_dim_agent - 3*self.dim, i] = obs_agent.T.reshape(-1)
        #             self.obs[self.obs_dim_agent - 3*self.dim:self.obs_dim_agent - 2*self.dim, i] = leader_pos_rel
        #             self.obs[self.obs_dim_agent - 2*self.dim:self.obs_dim_agent - self.dim, i] = leader_heading_rel
        #             self.obs[self.obs_dim_agent - self.dim:, i] = self.heading[:, i]
        #     else:
        #         # Non-leader or non-remarkable leader cases
        #         if self.dynamics_mode == 'Cartesian':
        #             self.obs[:, i] = obs_agent.T.reshape(-1)
        #         elif self.dynamics_mode == 'Polar':
        #             self.obs[:self.obs_dim_agent - self.dim, i] = obs_agent.T.reshape(-1)
        #             self.obs[self.obs_dim_agent - self.dim:, i] = self.heading[:, i]

        return self.obs
      
    def _get_reward(self, a):
        """
        Calculate reward for each agent based on multiple behavioral components.
        """

        # Initialize reward array
        reward_a = np.zeros((1, self.n_ao))

        # repulsion, cohesion, alignment
        coefficients = np.array([15, 6, 2] + [1.4, 6] + [4] + [1, 1, 1] + [5] + [12], dtype=np.float64) # 15, 2.5, 2
        conditions = np.array([self.is_periodic, self.is_Cartesian, self.penalize_inter_agent_distance, self.is_remarkable_leader, self.penalize_collide_agents, 
                               self.penalize_control_effort, self.penalize_collide_obstacles, self.penalize_collide_walls], dtype=bool)

        _LIB._get_reward(as_double_c_array(self.p), 
                         as_double_c_array(self.dp), 
                         as_double_c_array(self.heading),
                         as_double_c_array(a.astype(np.float64)), 
                         as_double_c_array(reward_a), 
                         as_double_c_array(self.leader_state),
                         as_int32_c_array(self.neighbor_index),
                         as_int32_c_array(self.nearest_leader_index),
                         as_double_c_array(self.boundary_pos),
                         ctypes.c_double(self.d_sen), 
                         ctypes.c_double(self.d_ref),
                         ctypes.c_int(self.topo_nei_max), 
                         ctypes.c_int(self.n_a), 
                         ctypes.c_int(self.n_l),
                         ctypes.c_int(self.dim), 
                         as_bool_c_array(conditions),
                         as_bool_c_array(self.is_collide_b2b),
                         as_bool_c_array(self.is_collide_b2w),
                         as_double_c_array(coefficients)) 

        # reward_a = np.zeros((1, self.n_ao))

        # # Calculate inter-agent alignment and cohesion rewards
        # if self.penalize_inter_agent_distance:
        #     for agent in range(self.n_a):
        #         # Retrieve valid neighbor indices
        #         neighbors = self.neighbor_index[agent]
        #         valid_neighbors = neighbors[neighbors != -1]  # Filter placeholder values
                
        #         if valid_neighbors.size > 0:
        #             # Inter-agent distance penalties (repulsion and cohesion)
        #             for neighbor in valid_neighbors:
        #                 pos_diff = self.p[:, neighbor] - self.p[:, agent]
                        
        #                 # Apply periodic boundary conditions
        #                 if self.is_periodic:
        #                     pos_diff = self._make_periodic(pos_diff, is_rel=True)
                        
        #                 distance = np.linalg.norm(pos_diff)
                        
        #                 # Penalize deviations from reference distance (d_ref)
        #                 penalty = max(
        #                     coefficients[0] * (self.d_ref - distance - 0.05),
        #                     coefficients[1] * (distance - self.d_ref - 0.05)
        #                 )
        #                 reward_a[0, agent] -= penalty
                    
        #             # Velocity alignment penalty
        #             neighbor_vels = self.dp[:, valid_neighbors]
        #             vel_norms = np.linalg.norm(neighbor_vels, axis=0) + 1e-8  # Avoid division by zero
        #             avg_neighbor_vel = np.mean(neighbor_vels / vel_norms, axis=1)
                    
        #             agent_vel = self.dp[:, agent]
        #             agent_vel_norm = np.linalg.norm(agent_vel) + 1e-8
        #             reward_a[0, agent] -= coefficients[2] * np.linalg.norm(
        #                 avg_neighbor_vel - agent_vel / agent_vel_norm
        #             )

        # # Tracking virtual leader rewards
        # if self.is_remarkable_leader:
        #     for agent in range(self.n_a):
        #         leader_idx = self.nearest_leader_index[agent]
        #         if leader_idx >= 0:  # Valid leader exists
        #             # Position tracking penalty
        #             leader_pos_rel = self.leader_state[:self.dim, leader_idx] - self.p[:, agent]
        #             if self.is_periodic:
        #                 leader_pos_rel = self._make_periodic(leader_pos_rel, is_rel=True)
        #             reward_a[0, agent] -= coefficients[3] * np.linalg.norm(leader_pos_rel)
                    
        #             # Velocity tracking penalty
        #             leader_vel = self.leader_state[self.dim:2*self.dim, leader_idx]
        #             agent_vel = self.dp[:, agent]
        #             reward_a[0, agent] -= coefficients[4] * np.linalg.norm(leader_vel - agent_vel)

        # # Agent-agent collision penalties
        # if self.penalize_collide_agents:
        #     # Sum collision flags across all agents
        #     agent_collisions = self.is_collide_b2b[:self.n_a, :self.n_a].sum(axis=0)
        #     reward_a -= coefficients[5] * agent_collisions[None, :]  # None for dimension alignment

        # # Control effort penalties
        # if self.penalize_control_effort:
        #     if self.dynamics_mode == 'Cartesian':
        #         # Penalize overall control magnitude
        #         reward_a -= coefficients[6] * np.linalg.norm(a, axis=0, keepdims=True)
        #     elif self.dynamics_mode == 'Polar':
        #         # Separate penalties for linear and angular control
        #         linear_ctrl = np.abs(a[[0], :])
        #         angular_ctrl = np.abs(a[[1], :])
        #         reward_a -= coefficients[7] * linear_ctrl + coefficients[8] * angular_ctrl

        # # Agent-obstacle collision penalties
        # if self.penalize_collide_obstacles:
        #     obstacle_collisions = self.is_collide_b2b[self.n_a:self.n_ao, :self.n_a].sum(axis=0)
        #     reward_a -= coefficients[9] * obstacle_collisions[None, :]

        # # Wall collision penalties
        # if self.penalize_collide_walls:
        #     wall_collisions = self.is_collide_b2w[:, :self.n_a].sum(axis=0)
        #     reward_a -= coefficients[10] * wall_collisions[None, :]         

        return reward_a

    def _get_dist_b2b(self):
        """
        Calculate distances and collision states between all agents and obstacles.
        """
        # Compute relative positions between all entities
        all_pos = np.tile(self.p, (self.n_ao, 1))   
        my_pos = self.p.T.reshape(2 * self.n_ao, 1) 
        my_pos = np.tile(my_pos, (1, self.n_ao))   
        relative_p_2n_n =  all_pos - my_pos
        
        # Apply periodic boundary correction if needed
        if self.is_periodic:
            relative_p_2n_n = self._make_periodic(relative_p_2n_n, is_rel=True)
        
        # Calculate Euclidean distances between entity centers
        d_b2b_center = np.sqrt(relative_p_2n_n[::2,:]**2 + relative_p_2n_n[1::2,:]**2)  
        
        # Calculate edge-to-edge distances (accounting for sizes)
        d_b2b_edge = d_b2b_center - self.sizes
        
        # Determine collision states
        isCollision = (d_b2b_edge < 0)
        
        return d_b2b_center, np.abs(d_b2b_edge), isCollision
    
    def _get_dist_b2w(self):
        _LIB._get_dist_b2w(as_double_c_array(self.p), 
                           as_double_c_array(self.size), 
                           as_double_c_array(self.d_b2w), 
                           as_bool_c_array(self.is_collide_b2w),
                           ctypes.c_int(self.dim), 
                           ctypes.c_int(self.n_a), 
                           as_double_c_array(self.boundary_pos))

        # p = self.p
        # r = self.size
        # d_b2w = np.zeros((4, self.n_ao))
        # # isCollision = np.zeros((4,self.n_ao))
        # for i in range(self.n_ao):
        #     d_b2w[:,i] = np.array([ p[0,i] - r[i] - self.boundary_pos[0], 
        #                             self.boundary_pos[1] - (p[1,i] + r[i]),
        #                             self.boundary_pos[2] - (p[0,i] + r[i]),
        #                             p[1,i] - r[i] - self.boundary_pos[3]])  
        # self.is_collide_b2w = d_b2w < 0
        # self.d_b2w = np.abs(d_b2w) 

    def _get_done(self):
        all_done = np.zeros( (1, self.n_a) ).astype(bool)
        return all_done

    def _get_info(self):
        return np.array( [None, None, None] ).reshape(3,1)

    def _get_done(self):
        """Determine termination status for each agent (dummy implementation)."""
        all_done = np.zeros((1, self.n_a)).astype(bool)
        return all_done

    def _get_info(self):
        """Placeholder for additional info (compatible with Gym interface)."""
        return np.array([None, None, None]).reshape(3, 1)

    def step(self, a): 
        """
        Perform a simulation step.
        """
        # Update simulation time
        self.simulation_time += self.dt 
        
        # Perform physics calculations for multiple frames
        for _ in range(self.n_frames): 
            # Scale actions for polar dynamics
            if self.dynamics_mode == 'Polar':  
                a[0, :self.n_a] *= self.angle_max  # Scale angle change
                a[1, :self.n_a] = (self.Acc_max - self.Acc_min)/2 * a[1,:self.n_a] + (self.Acc_max + self.Acc_min)/2  # Scale acceleration

            # Compute agent-agent distances and collisions
            self.d_b2b_center, self.d_b2b_edge, self.is_collide_b2b = self._get_dist_b2b()

            # sf_b2b_all = np.zeros((2*self.n_ao, self.n_ao)) 
            sf_b2b = np.zeros((2, self.n_ao))

            _LIB._sf_b2b_all(as_double_c_array(self.p), 
                             as_double_c_array(sf_b2b), 
                             as_double_c_array(self.d_b2b_edge), 
                             as_bool_c_array(self.is_collide_b2b),
                             as_double_c_array(self.boundary_pos),
                             as_double_c_array(self.d_b2b_center),
                             ctypes.c_int(self.n_a), 
                             ctypes.c_int(self.dim), 
                             ctypes.c_double(self.k_ball),
                             ctypes.c_bool(self.is_periodic))
            # for i in range(self.n_ao):
            #     for j in range(i):
            #         delta = self.p[:,j]-self.p[:,i]
            #         if self.is_periodic:
            #             delta = self._make_periodic(delta, is_rel=True)
            #         dir = delta / self.d_b2b_center[i,j]
            #         sf_b2b_all[2*i:2*(i+1),j] = self.is_collide_b2b[i,j] * self.d_b2b_edge[i,j] * self.k_ball * (-dir)
            #         sf_b2b_all[2*j:2*(j+1),i] = - sf_b2b_all[2*i:2*(i+1),j]  
                   
            # sf_b2b_1 = np.sum(sf_b2b_all, axis=1, keepdims=True).reshape(self.n_ao,2).T 

            # Calculate wall interactions for bounded environments
            if self.is_boundary:
                self._get_dist_b2w()
                # Spring force from walls
                sf_b2w = np.array([[1, 0, -1, 0], [0, -1, 0, 1]]).dot(self.is_collide_b2w * self.d_b2w) * self.k_wall 
                df_b2w = np.array([[-1, 0, -1, 0], [0, -1, 0, -1]]).dot(self.is_collide_b2w * np.concatenate((self.dp, self.dp), axis=0)) * self.c_wall

            # Apply agent strategy for action selection
            if self.agent_strategy == 'input':
                # Use provided actions unchanged
                pass               
            elif self.agent_strategy == 'random':
                # Generate random actions
                a = np.random.uniform(-1, 1, (self.act_dim_agent, self.n_a)) 
                if self.dynamics_mode == 'Polar': 
                    a[0, :self.n_a] *= self.angle_max
                    a[1, :self.n_a] = (self.Acc_max - self.Acc_min)/2 * a[1,:self.n_a] + (self.Acc_max + self.Acc_min)/2
            elif self.agent_strategy == 'rule':
                # Rule-based flocking behavior
                a = np.zeros((2, self.n_ao))
                dist = pdist(self.p.T)
                dist_mat = squareform(dist)
                sorted_indices = np.argsort(dist_mat, axis=0)
                self.B = np.take_along_axis(dist_mat, sorted_indices, axis=0)
                self.I = sorted_indices
                for agent in range(self.n_a):
                    # Find neighbors within sensing distance
                    list_nei_indices = self.B[:, agent] <= self.d_sen
                    list_nei = self.I[list_nei_indices, agent]
                    list_nei = list_nei[1:]  # Exclude self
                    
                    # Limit to max observable neighbors
                    if len(list_nei) > self.topo_nei_max:
                        list_nei = list_nei[:self.topo_nei_max]

                    if len(list_nei) > 0:
                        # Compute relative positions to neighbors
                        pos_rel = self.p - self.p[:, [agent]]
                        for agent2 in list_nei:
                            # Alignment behavior: match velocity with neighbors
                            vel_heading = self.dp[:,agent2] / (np.linalg.norm(self.dp[:,agent2]) + 1e-6)
                            a[:, agent] += 0.4 * pos_rel[:,agent2] + vel_heading
            else:
                print('Error: Unknown agent strategy in step function')

            # Convert actions to forces
            if self.dynamics_mode == 'Cartesian':
                u = a   # Cartesian actions are already forces
            elif self.dynamics_mode == 'Polar':      
                # Update heading direction
                self.theta += a[[0],:]
                self.theta = self._normalize_angle(self.theta)
                self.heading = np.concatenate((np.cos(self.theta), np.sin(self.theta)), axis=0) 
                u = a[[1], :] * self.heading 
            else:
                print('Error: Unknown dynamics mode')

            # Sum all forces acting on agents
            if self.is_boundary:
                F = self.sensitivity * u + sf_b2b + sf_b2w + df_b2w
            else: 
                F = self.sensitivity * u + sf_b2b

            # Update dynamics
            self.ddp = F / self.m  # Acceleration
            
            # Update velocity
            self.dp += self.ddp * self.dt
            
            # Apply constraints to leader agents
            if self.is_leader:
                # Force leaders to maintain assigned velocity
                self.dp[:, :self.n_l] = self.leader_state[self.dim:, :self.n_l]
                self.heading[:, :self.n_l] = self.dp[:, :self.n_l] / (
                    np.linalg.norm(self.dp[:, :self.n_l], axis=0) + 1e-6)
            
            # Apply minimum velocity constraints
            if self.is_boundary and not self.is_leader:
                self._regularize_min_velocity()
                
            # Clamp velocity to physical limits
            self.dp = np.clip(self.dp, -self.Vel_max, self.Vel_max)

            # Keep obstacles stationary
            if self.obstacles_cannot_move:
                self.dp[:, self.n_a:self.n_ao] = 0
        
            # Update position
            self.p += self.dp * self.dt
            
            # Constraints for leader positions
            if self.is_leader:
                if self.is_periodic:
                    self.leader_state[:self.dim, :self.n_l] = self.p[:, :self.n_l]
                else:
                    # Fixed leader positions in bounded environments
                    self.p[:, :self.n_l] = self.leader_state[:self.dim, :self.n_l]
            
            # Maintain constant obstacle positions
            if self.obstacles_is_constant:
                self.p[:, self.n_a:self.n_ao] = self.p_o
            
            # Apply periodic boundary conditions
            if self.is_periodic:
                self.p = self._make_periodic(self.p, is_rel=False)

            # Update leader states in bounded environments
            if self.is_leader and self.is_boundary:
                self._get_way_count()
                self._update_leader_state()
                # Update leader positions and velocities
                self.leader_state[:self.dim] = self.center_leader_state[:self.dim, [0]] + self.initial_p[:, :self.n_l]
                self.leader_state[self.dim:] = np.tile(self.center_leader_state[self.dim:, [0]], (1, self.n_l))

            # Update trajectory history for rendering
            if self.render_traj:
                self.p_traj = np.concatenate(
                    (self.p_traj[1:,:,:], self.p.reshape(1, 2, self.n_ao)), axis=0
                )

        # Prepare return values
        obs = self._get_obs()
        rew = self._get_reward(a)
        done = self._get_done()
        info = self._get_info()
        
        return obs, rew, done, info

    def render(self, mode="human"): 

        size_agents = 700

        if self.plot_initialized == 0:

            plt.ion()

            left, bottom, width, height = 0.09, 0.06, 0.9, 0.9
            ax = self.figure_handle.add_axes([left, bottom, width, height],projection = None)
                
            # Observation range
            if self.center_view_on_swarm == False:
                axes_lim = self.axis_lim_view_static()
            else:
                axes_lim = self.axis_lim_view_dynamic()
            
            ax.set_xlim(axes_lim[0],axes_lim[1])
            ax.set_ylim(axes_lim[2],axes_lim[3])
            ax.set_xlabel('X position [m]')
            ax.set_ylabel('Y position [m]')
            ax.set_title('Simulation time: %.2f seconds' % self.simulation_time)
            ax.grid(True)

            plt.ioff()
            plt.pause(0.01)

            self.plot_initialized = 1
        else:
            self.figure_handle.axes[0].cla()
            ax = self.figure_handle.axes[0]

            plt.ion()

            # Plot agents position
            ax.scatter(self.p[0,:], self.p[1,:], s = size_agents, c = self.color, marker = ".", alpha = 1)

            # for agent_index in range(self.n_a):
            #     ax.text(self.p[0,agent_index], self.p[1,agent_index], self.s_text[agent_index], fontsize=self.fontsize)

            if self.simulation_time / self.dt > self.traj_len:
                for agent_index in range(self.n_a):
                    distance_index = self._calculate_distances(agent_index)
                    if distance_index:
                        ax.plot(self.p_traj[distance_index:,0,agent_index], self.p_traj[distance_index:,1,agent_index], linestyle='-', color=self.color[agent_index], alpha=0.4)
                    else:
                        ax.plot(self.p_traj[:,0,agent_index], self.p_traj[:,1,agent_index], linestyle='-', color=self.color[agent_index], alpha=0.4)

            if self.dynamics_mode == 'Polar':
                ax.quiver(self.p[0,:], self.p[1,:], self.heading[0,:], self.heading[1,:], scale=30, color=self.color, width = 0.002)
            
            ax.plot(np.array([self.boundary_pos[0], self.boundary_pos[0], self.boundary_pos[2], self.boundary_pos[2], self.boundary_pos[0]]), 
                    np.array([self.boundary_pos[3], self.boundary_pos[1], self.boundary_pos[1], self.boundary_pos[3], self.boundary_pos[3]]))

            # for index_i in range(self.n_a):
            #     for index_j in range(index_i):
            #         dist_ij = np.linalg.norm(self.p[:, index_i] - self.p[:, index_j])
            #         if (dist_ij > self.d_ref - 0.05) and (dist_ij < self.d_ref + 0.05):
            #             ax.plot(np.array([self.p[0][index_i], self.p[0][index_j]]), np.array([self.p[1][index_i], self.p[1][index_j]]), linestyle='-', color=(0.1, 0.2, 0.3))
            
            # Observation range
            if self.center_view_on_swarm == False:
                axes_lim = self.axis_lim_view_static()
            else:
                axes_lim = self.axis_lim_view_dynamic()

            ax.set_xlim(axes_lim[0],axes_lim[1])
            ax.set_ylim(axes_lim[2],axes_lim[3])
            ax.set_xlabel('X position [m]', fontsize=self.fontsize)
            ax.set_ylabel('Y position [m]', fontsize=self.fontsize)
            ax.set_title('Simulation time: %.2f seconds' % self.simulation_time, fontsize=self.fontsize)
            ax.tick_params(axis='both', labelsize=self.fontsize)
            ax.grid(True)

            plt.ioff()
            plt.pause(0.01)
        
            if self.video:
                self.video.update()

    def axis_lim_view_static(self):
        """
        Return view boundaries for static camera mode.
        """
        indent = 0.1
        x_min = self.boundary_pos[0] - indent  # Left boundary
        x_max = self.boundary_pos[2] + indent  # Right boundary
        y_min = self.boundary_pos[3] - indent  # Bottom boundary
        y_max = self.boundary_pos[1] + indent  # Top boundary
        return [x_min, x_max, y_min, y_max]
    
    def axis_lim_view_dynamic(self):
        """
        Return view boundaries for dynamic camera mode.
        """
        indent = 0.5  # Larger padding for dynamic view
        # Get agent position extents
        x_min = np.min(self.p[0]) - indent
        x_max = np.max(self.p[0]) + indent
        y_min = np.min(self.p[1]) - indent
        y_max = np.max(self.p[1]) + indent

        return [x_min, x_max, y_min, y_max]

    def _make_periodic(self, x, is_rel):
        """
        Apply periodic boundary conditions to positions or vectors.
        """
        if is_rel:
            # Relative vectors: wrap values that exceed boundary half-length
            x[x > self.L] -= 2 * self.L 
            x[x < -self.L] += 2 * self.L
        else:
            # Absolute positions: wrap agents crossing the boundary
            # Left boundary wrapping
            x[0, x[0, :] < self.boundary_pos[0]] += 2 * self.L
            # Right boundary wrapping
            x[0, x[0, :] > self.boundary_pos[2]] -= 2 * self.L
            # Bottom boundary wrapping
            x[1, x[1, :] < self.boundary_pos[3]] += 2 * self.L
            # Top boundary wrapping
            x[1, x[1, :] > self.boundary_pos[1]] -= 2 * self.L
        return x
    
    def _normalize_angle(self, x):
        """
        Normalize angles to the range [-π, π].
        """
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    def _get_size(self):
        """
        Get agent sizes and pairwise size sums.
        """
        # Create size array for agents and obstacles
        size = np.concatenate((
            np.array([self.size_a] * self.n_a),   # Agent sizes
            np.array([self.size_o] * self.n_o)    # Obstacle sizes
        ))  
        
        # Create matrix of pairwise size sums (for collision detection)
        sizes = np.tile(size.reshape(self.n_ao, 1), (1, self.n_ao))
        sizes = sizes + sizes.T  # SizeA + SizeB for each pair
        
        # Set self-interactions to zero (no collision with self)
        np.fill_diagonal(sizes, 0)
        return size, sizes
    
    def _get_mass(self):
        """
        Get masses of all entities (agents and obstacles).
        """
        m = np.concatenate((
            np.array([self.m_a] * self.n_a),    # Agent masses
            np.array([self.m_o] * self.n_o)     # Obstacle masses
        )) 
        return m

    def _get_observation_space(self):
        """
        Configure observation space based on environment settings.
        """
        # Check if agent's own state is included in observation
        self_flag = 1 if self.is_con_self_state else 0

        # Handle different observation configurations
        if self.is_leader and self.is_remarkable_leader:
            # Observe neighbors + leader + optionally self
            self.obs_dim_agent = 2 * self.dim * (self.topo_nei_max + 1 + self_flag)
        else:
            # Observe neighbors + optionally self
            self.obs_dim_agent = 2 * self.dim * (self.topo_nei_max + self_flag)   
        
        # Add extra dimension for heading in Polar dynamics
        if self.dynamics_mode == 'Polar':
            self.obs_dim_agent += self.dim

        # Create Box observation space
        observation_space = spaces.Box(
            low=-np.inf, high=+np.inf, 
            shape=(self.obs_dim_agent, self.n_a), 
            dtype=np.float32
        )
        return observation_space

    def _get_action_space(self):
        """
        Configure action space based on environment settings.
        """
        action_space = spaces.Box(
            low=-np.inf, high=+np.inf, 
            shape=(self.act_dim_agent, self.n_a), 
            dtype=np.float32
        )
        return action_space

    def _get_focused(self, Pos, Vel, norm_threshold, width, remove_self):
        """
        Find nearest neighbors within observation range.
        """
        # Calculate neighbor distances
        norms = np.linalg.norm(Pos, axis=0)
        # Sort neighbors by distance
        sorted_seq = np.argsort(norms)    
        # Reorder positions by distance
        Pos = Pos[:, sorted_seq]   
        norms = norms[sorted_seq] 
        
        # Filter neighbors beyond sensing distance
        Pos = Pos[:, norms < norm_threshold] 
        sorted_seq = sorted_seq[norms < norm_threshold]   
        
        # Remove self from neighbors
        if remove_self:
            Pos = Pos[:, 1:]  
            sorted_seq = sorted_seq[1:]                    
        
        # Update velocities to match sorted sequence
        Vel = Vel[:, sorted_seq]
        
        # Prepare output arrays
        target_Pos = np.zeros((2, width))
        target_Vel = np.zeros((2, width))
        
        # Determine how many neighbors to include
        until_idx = min(Pos.shape[1], width)
        
        # Copy neighbor data to output
        target_Pos[:, :until_idx] = Pos[:, :until_idx] 
        target_Vel[:, :until_idx] = Vel[:, :until_idx]
        
        # Get neighbor indices
        target_Nei = sorted_seq[:until_idx]
        return target_Pos, target_Vel, target_Nei

    def _regularize_min_velocity(self):
        """
        Ensure all agents maintain minimum velocity.
        """
        # Calculate current speeds
        norms = np.linalg.norm(self.dp, axis=0)
        # Identify agents below minimum velocity
        mask = norms < self.Vel_min
        # Scale velocities to meet minimum while preserving direction
        self.dp[:, mask] *= self.Vel_min / (norms[mask] + 1e-5)

    def _calculate_distances(self, id_self):
        """
        Detect when an agent's trajectory crosses environment boundaries.
        """
        # Extract x and y coordinates from trajectory
        x_coords = self.p_traj[:, 0, id_self]
        y_coords = self.p_traj[:, 1, id_self]
        
        # Calculate segment lengths between consecutive points
        distances = np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2)
        
        # Find segments where boundary was crossed
        points_greater = np.where(distances > self.boundary_L_half)[0]
        
        # Return last segment index if crossing found
        if len(points_greater) > 0:
            return points_greater[-1] + 1
        else:
            return False

    def _get_way_count(self):
        """
        Update waypoint index when current waypoint is reached.
        """
        # Calculate distance to current waypoint
        dist = np.linalg.norm(
            self.center_leader_state[:self.dim] - 
            self.leader_waypoint[:, [self.way_count]]
        )
        
        # Advance to next waypoint when sufficiently close
        if dist < 0.1 and self.way_count < self.leader_waypoint.shape[1] - 1:
            self.way_count += 1

    def _update_leader_state(self):
        """Update virtual leader position and velocity based on current waypoint."""
        # Calculate distance to target waypoint
        d_shill_target = np.linalg.norm(
            self.leader_waypoint[:, self.way_count].reshape(self.dim, 1) - 
            self.center_leader_state[0:self.dim]
        )
        
        if d_shill_target > 0.1:
            # Move toward waypoint at constant speed
            self.center_leader_state[self.dim:2*self.dim] = (
                self.leader_v * 
                (self.leader_waypoint[:, self.way_count].reshape(self.dim, 1) - 
                 self.center_leader_state[0:self.dim]) / 
                d_shill_target
            )
        else:
            # Slow down when near waypoint
            self.center_leader_state[self.dim:2*self.dim] *= 0.001
        
        # Update position based on velocity
        self.center_leader_state[:self.dim] += (
            self.center_leader_state[self.dim:2*self.dim] * 
            self.dt
        )
            


