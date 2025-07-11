__credits__ = ["zhugb@buaa.edu.cn", "lijianan@westlake.edu.cn"]

import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from .VideoWriter import VideoWriter
from .envs_cplus.c_lib import as_double_c_array, as_bool_c_array, as_int32_c_array, _load_lib
import ctypes

_LIB = _load_lib(env_name='Adversarial')

class AdversarialSwarmEnv(gym.Env):
    """
    Adversarial multi-agent swarm environment for reinforcement learning.
    Two teams of agents compete against each other in a bounded or periodic space.
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}
    
    def __init__(self):
        # Reward configuration
        self.reward_sharing_mode = 'individual'   # select one from ['sharing_mean', 'sharing_max', 'individual'] 
        
        # Penalty flags for different collision types
        self.penalize_control_effort = False      
        self.penalize_collide_agents = False 
        self.penalize_collide_opponents = False 
        self.penalize_collide_walls = False   
        self.penalize_collide_obstacles = False  

        # Environment dimensions
        self.dim = 2 

        # Agent counts
        self.n_o = 0     # number of obstacles

        # Observation topology - neighbor counts for each team interaction
        self.topo_nei_l2l = 3     # left agents to left agents 
        self.topo_nei_l2r = 3     # left agents to right agents 
        self.topo_nei_r2l = 3     # right agents to left agents 
        self.topo_nei_r2r = 3     # right agents to right agents  
        
        # Action space dimensions
        self.act_dim_l = 2  # left team action dimension
        self.act_dim_r = 2  # right team action dimension
        
        # Physical properties - mass
        self.m_p = 1    
        self.m_e = 1    
        self.m_o = 10   # mass of obstacles
        
        # Physical properties - size/radius
        self.size_p = 0.04   
        self.size_e = 0.04    
        self.size_o = 0.2     # size of obstacles
        
        # Sensing range
        self.d_sen_l = 5   # sensing distance for left team
        self.d_sen_r = 5   # sensing distance for right team

        # Movement constraints
        self.linVel_l_max = 0.8  # maximum linear velocity for left team
        self.linVel_r_max = 0.8  # maximum linear velocity for right team
        self.linAcc_max = 1      # maximum linear acceleration

        # Attack/combat parameters
        self.attack_radius = 0.4  # attack range
        self.attack_angle = 0.4   # attack angle cone
        self.attack_hp = 1        # damage per attack
        self.recover_hp = 0.1     # health recovery rate
        self.attack_max = 2       # maximum simultaneous attacks

        # Health/energy system
        self.hp_l_max = 80.  # maximum health for left team
        self.hp_r_max = 80.  # maximum health for right team

        # Obstacle configuration
        self.obstacles_cannot_move = True   # static obstacles flag
        self.obstacles_is_constant = False  # fixed obstacle positions flag
        if self.obstacles_is_constant:   # then specify their locations:
            self.p_o = np.array([[-0.5,0.5], [0,0]])
        ## ======================================== end ========================================

        # Environment boundaries
        self.boundary_L_half = 1.6  # half-length of square boundary
        self.bound_center = np.zeros(2)  # center of boundary

        # Physics parameters
        self.L = self.boundary_L_half
        self.k_ball = 30       # sphere-sphere contact stiffness  N/m 
        # self.c_ball = 5      # sphere-sphere contact damping N/m/s
        self.k_wall = 100      # sphere-wall contact stiffness  N/m
        self.c_wall = 5        # sphere-wall contact damping N/m/s
        self.c_aero = 2        # sphere aerodynamic drag coefficient N/m/s

        # Simulation parameters
        self.simulation_time = 0
        self.dt = 0.1      # time step
        self.n_frames = 1  # frames per step
        self.sensitivity = 1 

        # Rendering configuration
        self.traj_len = 12           # trajectory history length
        self.plot_initialized = 0    # rendering initialization flag
        self.center_view_on_swarm = False  # camera centering option
        self.fontsize = 20
        width = 13
        height = 13
        self.figure_handle = plt.figure(figsize=(width, height))
        
    def __reinit__(self, args):
        """
        Reinitialize environment with specific configuration arguments.
        Called after __init__ to set up team sizes and strategies.
        """
        # Team sizes
        self.n_l = args.n_l      # number of left team agents
        self.n_r = args.n_r      # number of right team agents
        self.n_lr = self.n_l + self.n_r  # total agents
        
        # Store initial team sizes
        self.n_l_init = self.n_l
        self.n_r_init = self.n_r
        self.n_lr_init = self.n_lr
        self.n_lro = self.n_l + self.n_r + self.n_o  # total entities including obstacles
        
        # Agent labels for rendering
        self.s_text_l = np.char.mod('%d', np.arange(self.n_l))
        self.s_text_r = np.char.mod('%d', np.arange(self.n_r))
        
        # Rendering options
        self.render_traj = args.render_traj
        self.traj_len = args.traj_len
        self.video = args.video

        # Boundary conditions
        self.is_boundary = args.is_boundary
        self.is_periodic = False if self.is_boundary else True

        # Dynamics and strategy configuration
        self.dynamics_mode = args.dynamics_mode
        self.billiards_mode = args.billiards_mode
        self.is_con_self_state = args.is_con_self_state
        
        # Training mode flags for each team
        self.is_training = np.array([False, False], dtype=bool)
        self.l_strategy = args.l_strategy
        self.r_strategy = args.r_strategy
        
        # Set training flags based on strategy
        if self.l_strategy == 'input':
            self.is_training[0] = True
        if self.r_strategy == 'input':
            self.is_training[1] = True

        # Collision detection matrices
        self.is_collide_b2b = np.random.choice([True, False], size=(self.n_lr, self.n_lr))
        self.is_collide_b2w = np.zeros((4, self.n_lr), dtype=bool)
        self.d_b2w = np.ones((4, self.n_lro))

        # Initialize gym spaces
        self.observation_space = self._get_observation_space()  
        self.action_space = self._get_action_space()   
        
        # Physical properties
        self.m = self._get_mass()  
        self.size, self.sizes = self._get_size()  

        # Billiards mode physics adjustments
        if self.billiards_mode:
            self.c_wall = 0.2   # reduced wall damping
            self.c_aero = 0.01  # reduced air resistance

        # Cartesian dynamics mode setup
        if self.dynamics_mode == 'Cartesian':
            self.is_Cartesian = True
            self.linAcc_min = -1    
            assert (self.linAcc_min, self.linAcc_max) == (-1, 1)

        # Team colors for visualization
        self.color_l = np.tile(np.array([0, 0, 1]), (self.n_l, 1))      # blue for left team
        self.color_r = np.tile(np.array([1, 0.5, 0]), (self.n_r, 1))   # orange for right team
        self.color = np.concatenate((self.color_l, self.color_r), axis=0)

        # Video recording setup
        if self.video:
            self.video = VideoWriter(output_rate=self.dt, fps=20)
            self.video.video.setup(self.figure_handle, args.video_path) 

    def reset(self):
        """
        Reset environment to initial state.
        Returns initial observations for all agents.
        """
        # Reset simulation time
        self.simulation_time = 0
        
        # Initialize agent orientations
        self.heading = np.zeros((self.dim, self.n_lr))
        
        # Reset team sizes to initial values
        self.n_l = self.n_l_init
        self.n_r = self.n_r_init
        self.n_lr = self.n_lr_init
        
        # Agent indices for each team
        self.index_l = np.array([i for i in np.arange(self.n_l)], dtype=np.int32)
        self.index_r = np.array([i for i in np.arange(self.n_l, self.n_lr)], dtype=np.int32)
        
        # Store previous indices for comparison
        self.index_l_last = self.index_l
        self.index_r_last = self.index_r
        self.n_l_last = self.index_l_last.shape[0]
        self.n_r_last = self.index_r_last.shape[0]
        self.n_lr_last = self.n_l_last + self.n_r_last
        
        # Initialize combat-related arrays
        self.dead_index = -np.ones((self.n_lr,), dtype=np.int32)
        self.attack_neigh = -np.ones((self.n_lr, self.attack_max), dtype=np.int32)
        self.safe_max = np.max([self.n_l, self.n_r])
        self.safe_neigh = -np.ones((self.n_lr, self.safe_max), dtype=np.int32)

        # Set up environment boundaries
        self.bound_center = np.array([0, 0])
        # Boundary positions: x_min, y_max, x_max, y_min
        self.boundary_pos = np.array([self.bound_center[0] - self.boundary_L_half,
                                      self.bound_center[1] + self.boundary_L_half,
                                      self.bound_center[0] + self.boundary_L_half,
                                      self.bound_center[1] - self.boundary_L_half], dtype=np.float64) 

        # Initialize agent positions randomly within boundaries
        max_size = np.max(self.size)
        random_int = self.boundary_L_half
        self.p = np.random.uniform(-random_int + max_size, random_int - max_size, (2, self.n_lro))

        # Initialize trajectory history for rendering
        if self.render_traj == True:
            self.p_traj = np.zeros((self.traj_len, 2, self.n_lro))
            self.p_traj[0, :, :] = self.p
        
        # Initialize velocities
        self.dp = np.zeros((2, self.n_lro)) 
        if self.billiards_mode:
            self.dp = np.random.uniform(-1, 1, (2, self.n_lro))  # random initial velocities in billiards mode
        
        # Ensure obstacles remain stationary
        if self.obstacles_cannot_move:
            self.dp[:, self.n_lr:self.n_lro] = 0

        # Initialize accelerations
        self.ddp = np.zeros((2, self.n_lro))  

        # Initialize health points for each team
        self.hp = np.array([self.hp_l_max for _ in range(self.n_l)] + 
                          [self.hp_r_max for _ in range(self.n_r)]).reshape(1, self.n_lr)

        # Get initial observations
        obs = self._get_obs()

        return obs

    def _get_obs(self):
        """
        Generate observations for all agents using relative positions and velocities.
        Uses C library for efficient computation.
        """
        # Initialize observation array
        self.obs = np.zeros((self.observation_space.shape[0], self.n_lr))
        conditions = np.array([self.is_periodic, self.is_Cartesian, self.is_con_self_state])

        # Call C library function for efficient observation computation
        _LIB._get_observation(as_double_c_array(self.p), 
                              as_double_c_array(self.dp), 
                              as_double_c_array(self.heading),
                              as_double_c_array(self.obs),
                              as_double_c_array(self.boundary_pos),
                              as_double_c_array(self.hp),
                              as_int32_c_array(self.index_l), 
                              as_int32_c_array(self.index_r), 
                              ctypes.c_double(self.d_sen_l), 
                              ctypes.c_double(self.d_sen_r),
                              ctypes.c_double(self.hp_l_max), 
                              ctypes.c_double(self.hp_r_max), 
                              ctypes.c_int(self.topo_nei_l2l), 
                              ctypes.c_int(self.topo_nei_l2r), 
                              ctypes.c_int(self.topo_nei_r2l), 
                              ctypes.c_int(self.topo_nei_r2r), 
                              ctypes.c_int(self.n_l),
                              ctypes.c_int(self.n_r),
                              ctypes.c_int(self.n_lr_init), 
                              ctypes.c_int(self.obs_dim_max), 
                              ctypes.c_int(self.dim), 
                              as_bool_c_array(conditions))

        # # Process left team agents
        # for left_idx in range(self.n_l):
        #     left_id = self.index_l[left_idx]  # Current left agent's global ID
            
        #     # Compute relative positions/velocities to other left agents
        #     relPos_left2left = self.p[:, self.index_l] - self.p[:, [left_id]]
        #     if self.is_periodic:
        #         relPos_left2left = self._make_periodic(relPos_left2left, is_rel=True)
            
        #     if self.dynamics_mode == 'Cartesian':
        #         relVel_left2left = self.dp[:, self.index_l] - self.dp[:, [left_id]]
        #     else:  # Polar dynamics
        #         relVel_left2left = self.heading[:, self.index_l] - self.heading[:, [left_id]]
            
        #     # Filter to neighbors within sensing range
        #     relPos_left2left, relVel_left2left, _ = self._get_focused(
        #         relPos_left2left, relVel_left2left, 
        #         self.d_sen_l, self.topo_nei_l2l, True
        #     )
            
        #     # Compute relative states to right team agents
        #     relPos_left2right = self.p[:, self.index_r] - self.p[:, [left_id]]
        #     if self.is_periodic:
        #         relPos_left2right = self._make_periodic(relPos_left2right, is_rel=True)
            
        #     if self.dynamics_mode == 'Cartesian':
        #         relVel_left2right = self.dp[:, self.index_r] - self.dp[:, [left_id]]
        #     else:  # Polar dynamics
        #         relVel_left2right = self.heading[:, self.index_r] - self.heading[:, [left_id]]
            
        #     # Filter to relevant right agents
        #     relPos_left2right, relVel_left2right, _ = self._get_focused(
        #         relPos_left2right, relVel_left2right, 
        #         self.d_sen_l, self.topo_nei_l2r, False
        #     )
            
        #     # Construct left agent's observation
        #     if self.is_con_self_state:
        #         # Include self state
        #         obs_left_pos = np.concatenate((self.p[:, [left_id]], relPos_left2left, relPos_left2right), axis=1)
        #         obs_left_vel = np.concatenate((self.dp[:, [left_id]], relVel_left2left, relVel_left2right), axis=1)
        #         obs_left = np.concatenate((obs_left_pos, obs_left_vel), axis=0)
        #     else:
        #         # Observation without self state
        #         obs_left_pos = np.concatenate((relPos_left2left, relPos_left2right), axis=1)
        #         obs_left_vel = np.concatenate((relVel_left2left, relVel_left2right), axis=1)
        #         obs_left = np.concatenate((obs_left_pos, obs_left_vel), axis=0)
            
        #     # Store observation based on dynamics mode
        #     if self.dynamics_mode == 'Cartesian':
        #         self.obs[:, left_idx] = obs_left.T.reshape(-1)
        #     elif self.dynamics_mode == 'Polar':
        #         obs_length = self.obs_dim_l - self.dim
        #         self.obs[:obs_length, left_idx] = obs_left.T.reshape(-1)
        #         self.obs[obs_length:obs_length + self.dim, left_idx] = self.heading[:, left_id]

        # # Process right team agents
        # for right_idx in range(self.n_l, self.n_lr):
        #     right_id = self.index_r[right_idx - self.n_l]  # Current right agent's global ID
            
        #     # Compute relative states to left agents
        #     relPos_right2left = self.p[:, self.index_l] - self.p[:, [right_id]]
        #     if self.is_periodic:
        #         relPos_right2left = self._make_periodic(relPos_right2left, is_rel=True)
            
        #     if self.dynamics_mode == 'Cartesian':
        #         relVel_right2left = self.dp[:, self.index_l] - self.dp[:, [right_id]]
        #     else:  # Polar dynamics
        #         relVel_right2left = self.heading[:, self.index_l] - self.heading[:, [right_id]]
            
        #     # Filter to relevant left agents
        #     relPos_right2left, relVel_right2left, _ = self._get_focused(
        #         relPos_right2left, relVel_right2left, 
        #         self.d_sen_r, self.topo_nei_r2l, False
        #     )
            
        #     # Compute relative states to other right agents
        #     relPos_right2right = self.p[:, self.index_r] - self.p[:, [right_id]]
        #     if self.is_periodic:
        #         relPos_right2right = self._make_periodic(relPos_right2right, is_rel=True)
            
        #     if self.dynamics_mode == 'Cartesian':
        #         relVel_right2right = self.dp[:, self.index_r] - self.dp[:, [right_id]]
        #     else:  # Polar dynamics
        #         relVel_right2right = self.heading[:, self.index_r] - self.heading[:, [right_id]]
            
        #     # Filter to relevant right agents
        #     relPos_right2right, relVel_right2right, _ = self._get_focused(
        #         relPos_right2right, relVel_right2right, 
        #         self.d_sen_r, self.topo_nei_r2r, True
        #     )
            
        #     # Construct right agent's observation
        #     if self.is_con_self_state:
        #         # Include self state
        #         obs_right_pos = np.concatenate((self.p[:, [right_id]], relPos_right2right, relPos_right2left), axis=1)
        #         obs_right_vel = np.concatenate((self.dp[:, [right_id]], relVel_right2right, relVel_right2left), axis=1)
        #         obs_right = np.concatenate((obs_right_pos, obs_right_vel), axis=0)
        #     else:
        #         # Observation without self state
        #         obs_right_pos = np.concatenate((relPos_right2right, relPos_right2left), axis=1)
        #         obs_right_vel = np.concatenate((relVel_right2right, relVel_right2left), axis=1)
        #         obs_right = np.concatenate((obs_right_pos, obs_right_vel), axis=0)
            
        #     # Store observation based on dynamics mode
        #     if self.dynamics_mode == 'Cartesian':
        #         self.obs[:, right_idx] = obs_right.T.reshape(-1)
        #     elif self.dynamics_mode == 'Polar':
        #         obs_length = self.obs_dim_r - self.dim
        #         self.obs[:obs_length, right_idx] = obs_right.T.reshape(-1)
        #         self.obs[obs_length:obs_length + self.dim, right_idx] = self.heading[:, right_id]

        return self.obs
      
    def _get_reward(self, a):
        # Calculate rewards for agents based on actions and environment state.
        reward_l = np.zeros((1, self.n_l))
        reward_r = np.zeros((1, self.n_r))
        coefficients = np.array([5, 1, 0] + [0.2, 0.2, 0] + [0.05, 0.05] + [0.05, 0.05] + [1, 1], dtype=np.float64)
        conditions = np.concatenate((np.array([self.penalize_control_effort, self.penalize_collide_agents, self.penalize_collide_opponents, self.penalize_collide_walls, 
                                    self.is_Cartesian], dtype=bool), self.is_training))
        
        # Call C library function to compute rewards efficiently
        _LIB._get_reward(as_double_c_array(a),
                         as_double_c_array(self.boundary_pos), 
                         as_double_c_array(self.hp),
                         as_double_c_array(reward_l),
                         as_double_c_array(reward_r), 
                         as_double_c_array(coefficients),
                         as_bool_c_array(conditions),
                         as_bool_c_array(self.is_collide_b2b),
                         as_bool_c_array(self.is_collide_b2w),
                         as_int32_c_array(self.index_l),
                         as_int32_c_array(self.index_r),
                         as_int32_c_array(self.attack_neigh),
                         as_int32_c_array(self.safe_neigh),
                         ctypes.c_int(self.attack_max),
                         ctypes.c_int(self.safe_max), 
                         ctypes.c_int(self.n_l), 
                         ctypes.c_int(self.n_r),
                         ctypes.c_int(self.n_lr_init),
                         ctypes.c_int(self.dim))
 
        # The commented Python implementation below shows the equivalent logic
        # reward_l = np.zeros((1, self.n_l))  # left team rewards
        # reward_r = np.zeros((1, self.n_r))  # right team rewards

        # def calculate_combat_rewards(num_agents, agent_indices, team_reward, opponent_reward):
        #     """
        #     Calculate attack-based rewards for a team of agents
            
        #     Args:
        #         num_agents: Number of agents in the current team
        #         agent_indices: Global indices of team members
        #         team_reward: Reward vector for the current team
        #         opponent_reward: Reward vector for the opposing team
        #     """
        #     for i in range(num_agents):
        #         agent_id = agent_indices[i]
                
        #         # Get attacking and protected neighbors
        #         attackers = self.attack_neigh[agent_id][self.attack_neigh[agent_id] != -1]
        #         protected = self.safe_neigh[agent_id][self.safe_neigh[agent_id] != -1]
                
        #         # Reward opponents when attacking their targets
        #         opponent_reward[0, attackers] += coefficients[1]
                
        #         # Penalize opponents for protecting their targets
        #         opponent_reward[0, protected] -= coefficients[2]
                
        #         # Agent destruction penalties
        #         if self.hp[0, agent_id] <= 0:
        #             # Penalize own team for agent destruction
        #             team_reward[0, i] -= coefficients[0]
        #             # Reward opponents for successful destruction
        #             opponent_reward[0, attackers] += coefficients[0]
                    
        #     return team_reward, opponent_reward

        # # Apply combat rewards based on training mode
        # if self.is_training[1]:  # Right team training mode
        #     reward_l, reward_r = calculate_combat_rewards(self.n_l, self.index_l, reward_l, reward_r)
        # if self.is_training[0]:  # Left team training mode
        #     reward_r, reward_l = calculate_combat_rewards(self.n_r, self.index_r, reward_r, reward_l)

        # # Control effort penalties
        # if self.penalize_control_effort:
        #     if self.dynamics_mode == 'Cartesian':
        #         # Magnitude-based penalty
        #         leader_control_mag = np.sqrt(a[0, :self.n_l]**2 + a[1, :self.n_l]**2)
        #         follower_control_mag = np.sqrt(a[0, self.n_l:self.n_lr]**2 + a[1, self.n_l:self.n_lr]**2)
        #         reward_l -= coefficients[3] * leader_control_mag
        #         reward_r -= coefficients[4] * follower_control_mag
        #     elif self.dynamics_mode == 'Polar':
        #         # Separate linear/angular penalties
        #         leader_linear = np.abs(a[0, :self.n_l])
        #         leader_angular = np.abs(a[1, :self.n_l])
        #         reward_l -= coefficients[3] * leader_linear + coefficients[5] * leader_angular
                
        #         follower_linear = np.abs(a[0, self.n_l:self.n_lr])
        #         follower_angular = np.abs(a[1, self.n_l:self.n_lr])
        #         reward_r -= coefficients[4] * follower_linear + coefficients[5] * follower_angular

        # # Intra-team collision penalties
        # if self.penalize_collide_agents:
        #     all_agents = np.concatenate((self.index_l, self.index_r))
        #     collide_matrix = self.is_collide_b2b[np.ix_(all_agents, all_agents)]
            
        #     # Leader-leader collisions
        #     l2l_collisions = collide_matrix[:self.n_l, :self.n_l].sum(axis=0)
        #     reward_l -= coefficients[6] * l2l_collisions
            
        #     # Follower-follower collisions
        #     r2r_collisions = collide_matrix[self.n_l:self.n_lr, self.n_l:self.n_lr].sum(axis=0)
        #     reward_r -= coefficients[7] * r2r_collisions

        # # Inter-team collision penalties (combat collisions)
        # if self.penalize_collide_opponents:
        #     all_agents = np.concatenate((self.index_l, self.index_r))
        #     collide_matrix = self.is_collide_b2b[np.ix_(all_agents, all_agents)]
            
        #     # Leaders hitting followers
        #     l2r_collisions = collide_matrix[self.n_l:self.n_lr, :self.n_l].sum(axis=0)
        #     reward_l -= coefficients[8] * l2r_collisions
            
        #     # Followers hitting leaders
        #     r2l_collisions = collide_matrix[:self.n_l, self.n_l:self.n_lr].sum(axis=0)
        #     reward_r -= coefficients[9] * r2l_collisions

        # # Wall collision penalties (if enabled)
        # if self.penalize_collide_walls and not self.is_periodic:
        #     all_agents = np.concatenate((self.index_l, self.index_r))
        #     wall_collide_matrix = self.is_collide_b2w[:, all_agents]
            
        #     # Leader wall collisions
        #     l_walls = wall_collide_matrix[:, :self.n_l].sum(axis=0)
        #     reward_l -= coefficients[10] * l_walls
            
        #     # Follower wall collisions
        #     r_walls = wall_collide_matrix[:, self.n_l:self.n_lr].sum(axis=0)
        #     reward_r -= coefficients[11] * r_walls

        # Combine team rewards into single vector
        reward = np.concatenate((reward_l, reward_r), axis=1)

        return reward

    def _process_attack(self):
        """
        Process attacks and defenses between agents.
        """
        self.attack_neigh = -np.ones((self.n_lr_init, self.attack_max), dtype=np.int32)
        self.safe_neigh = -np.ones((self.n_lr_init, self.safe_max), dtype=np.int32)
        _LIB._process_attack(as_int32_c_array(self.index_l), 
                            as_int32_c_array(self.index_r),
                            as_int32_c_array(self.dead_index), 
                            as_int32_c_array(self.attack_neigh),
                            as_int32_c_array(self.safe_neigh),
                            as_double_c_array(self.p),
                            as_double_c_array(self.dp),
                            as_double_c_array(self.hp),
                            as_double_c_array(self.boundary_pos),
                            as_bool_c_array(self.is_training),
                            ctypes.c_bool(self.is_periodic),
                            ctypes.c_double(self.attack_radius), 
                            ctypes.c_double(self.attack_angle),
                            ctypes.c_double(self.attack_hp),
                            ctypes.c_double(self.recover_hp), 
                            ctypes.c_int(self.attack_max),
                            ctypes.c_int(self.safe_max),
                            ctypes.c_int(self.n_lr_init), 
                            ctypes.c_int(self.n_l),
                            ctypes.c_int(self.n_r),
                            ctypes.c_int(self.dim))

        # def process_attack(victims, opponents):
        #     for i in victims:
        #         p_op = self.p[:, opponents]
        #         v_op = self.dp[:, opponents]
        #         pos_rel = p_op - self.p[:, [i]]
        #         if self.is_periodic:
        #             pos_rel = self._make_periodic(pos_rel, is_rel=True)
        #         pos_rel_norm = np.linalg.norm(pos_rel, axis=0)
        #         sorted_seq = np.argsort(pos_rel_norm)
        #         pos_rel_norm_sorted = pos_rel_norm[sorted_seq] 
        #         threat_index_pos = sorted_seq[pos_rel_norm_sorted < self.attack_radius]

        #         # angle between agent's velocity and opponents' relative position
        #         dot_products = np.dot(self.dp[:,[i]].T, pos_rel).flatten()
        #         norm_v = np.linalg.norm(self.dp[:,[i]])
        #         cos_angles = dot_products / (norm_v * pos_rel_norm + 1e-8)
        #         angles_vp = np.arccos(np.clip(cos_angles, -1.0, 1.0))
        #         threat_index_dir_p = np.where(angles_vp > ((1 - self.attack_angle) * np.pi))[0]
        #         safe_index_dir_p = np.where(angles_vp < (self.attack_angle * np.pi))[0]

        #         # angle between agent's velocity and opponents' relative velocity
        #         dot_products = np.sum(-pos_rel * v_op, axis=0)
        #         norm_v_op = np.linalg.norm(v_op, axis=0)
        #         cos_angles = dot_products / (pos_rel_norm * norm_v_op + 1e-8)
        #         angles_vv = np.arccos(np.clip(cos_angles, -1.0, 1.0))
        #         threat_index_dir_v = np.where(angles_vv < (self.attack_angle * np.pi))[0]
        #         safe_index_dir_v = np.where(angles_vv > ((1 - self.attack_angle) * np.pi))[0]

        #         threat_index_dir = np.intersect1d(threat_index_dir_p, threat_index_dir_v)
        #         safe_index_dir = np.intersect1d(safe_index_dir_p, safe_index_dir_v)

        #         common_elements, pos_indices, _ = np.intersect1d(threat_index_pos, threat_index_dir, assume_unique=True, return_indices=True)
        #         threat_index = common_elements[np.argsort(pos_indices)]
        #         common_elements, pos_indices, _ = np.intersect1d(threat_index_pos, safe_index_dir, assume_unique=True, return_indices=True)
        #         safe_index = common_elements[np.argsort(pos_indices)]
        #         if len(threat_index) > 0:
        #             num_attack = np.min([len(threat_index), self.attack_max])
        #             self.attack_neigh[i,:num_attack] = threat_index[:num_attack]
        #             self.hp[0, i] -= self.attack_hp * num_attack
        #             # self.hp[0, opponents[threat_index[0]]] -= self.attack_hp
        #         else:
        #             self.hp[0, i] += self.recover_hp
        #         if len(safe_index) > 0:
        #             num_safe = len(safe_index)
        #             self.safe_neigh[i,:num_safe] = safe_index[:num_safe]
        #         if self.hp[0, i] <= 0:
        #             self.dead_index[i] = i

        # self.attack_neigh = -np.ones((self.n_lr_init, self.attack_max), dtype=np.int32)
        # self.safe_neigh = -np.ones((self.n_lr_init, self.safe_max), dtype=np.int32)
        # # if self.is_training[1]:
        # process_attack(self.index_l, self.index_r)
        # # if self.is_training[0]:
        # process_attack(self.index_r, self.index_l)

    def _process_action(self, a):
        """
        Process actions for agents, mapping them to the correct indices and dimensions.
        """
        a_com = np.zeros((self.dim, self.n_lr_init))
        a_true = np.zeros((self.dim, self.n_lr))
        _LIB._process_act(as_int32_c_array(self.index_l_last), 
                          as_int32_c_array(self.index_r_last),
                          as_int32_c_array(self.index_l), 
                          as_int32_c_array(self.index_r),
                          as_double_c_array(a_com),
                          as_double_c_array(a_true),
                          as_double_c_array(a.astype(np.float64)),
                          ctypes.c_int(self.dim), 
                          ctypes.c_int(self.n_lr), 
                          ctypes.c_int(self.n_lr_init),
                          ctypes.c_int(self.n_l_last), 
                          ctypes.c_int(self.n_r_last), 
                          ctypes.c_int(self.n_l),
                          ctypes.c_int(self.n_r))
        
        # for a_i in range(self.n_l):
        #     act_index = np.where(self.index_l_last == self.index_l[a_i])[0]
        #     a_com[:, [self.index_l[a_i]]] = a[:, act_index]
        #     a_true[:, [a_i]] = a[:, act_index]
        # for a_i in range(self.n_r):
        #     act_index = np.where(self.index_r_last == self.index_r[a_i])[0]
        #     a_com[:, [self.index_r[a_i]]] = a[:, act_index + self.index_l_last.shape[0]]
        #     a_true[:, [a_i + self.n_l]] = a[:, act_index + self.index_l_last.shape[0]]

        return a_com, a_true

    def _get_elastic_force(self):
        """
        Calculate elastic forces between agents based on their positions and distances.
        """
        sf_b2b = np.zeros((2, self.n_lro))
        _LIB._sf_b2b_all(as_double_c_array(self.p), 
                            as_double_c_array(sf_b2b), 
                            as_double_c_array(self.d_b2b_edge), 
                            as_bool_c_array(self.is_collide_b2b),
                            as_double_c_array(self.boundary_pos),
                            as_double_c_array(self.d_b2b_center),
                            as_int32_c_array(self.dead_index),
                            ctypes.c_int(self.n_lro), 
                            ctypes.c_int(self.dim), 
                            ctypes.c_double(self.k_ball),
                            ctypes.c_bool(self.is_periodic))

        # sf_b2b_all = np.zeros((2*self.n_lro, self.n_lro))   
        # for i in range(self.n_lro):
        #     for j in range(i):
        #         delta = self.p[:,j] - self.p[:,i]
        #         if self.is_periodic:
        #             delta = self._make_periodic(delta, is_rel=True)
        #         if i in self.dead_index:
        #             sf_b2b_all[2*i:2*(i+1),j] = 0
        #             sf_b2b_all[2*j:2*(j+1),i] = 0
        #         else:
        #             dir = delta / (self.d_b2b_center[i,j] + 1e-8)
        #             sf_b2b_all[2*i:2*(i+1),j] = self.is_collide_b2b[i,j] * self.d_b2b_edge[i,j] * self.k_ball * (-dir)
        #             sf_b2b_all[2*j:2*(j+1),i] = - sf_b2b_all[2*i:2*(i+1),j]  
                
        # sf_b2b = np.sum(sf_b2b_all, axis=1, keepdims=True).reshape(self.n_lro,2).T

        return sf_b2b

    def _get_dist_b2b(self):
        """
        Calculate distances between agents and check for collisions.
        """
        all_pos = np.tile(self.p, (self.n_lro, 1))   
        my_pos = self.p.T.reshape(2*self.n_lro, 1) 
        my_pos = np.tile(my_pos, (1, self.n_lro))   
        relative_p_2n_n =  all_pos - my_pos
        if self.is_periodic == True:
            relative_p_2n_n = self._make_periodic(relative_p_2n_n, is_rel=True)
        d_b2b_center = np.sqrt(relative_p_2n_n[::2,:]**2 + relative_p_2n_n[1::2,:]**2)  
        d_b2b_edge = d_b2b_center - self.sizes
        isCollision = (d_b2b_edge < 0)
        d_b2b_edge = np.abs(d_b2b_edge)

        self.d_b2b_center = d_b2b_center
        self.d_b2b_edge = d_b2b_edge
        self.is_collide_b2b = isCollision
        return self.d_b2b_center, self.d_b2b_edge, self.is_collide_b2b

    def _get_dist_b2w(self):
        """
        Calculate distances between agents and walls, and check for collisions.
        """
        _LIB._get_dist_b2w(as_double_c_array(self.p), 
                           as_double_c_array(self.size), 
                           as_double_c_array(self.d_b2w), 
                           as_bool_c_array(self.is_collide_b2w),
                           ctypes.c_int(self.dim), 
                           ctypes.c_int(self.n_lr_init), 
                           as_double_c_array(self.boundary_pos))
        # p = self.p
        # r = self.size
        # d_b2w = np.zeros((4, self.n_lr))
        # # isCollision = np.zeros((4,self.n_ao))
        # for i in range(self.n_lr_init):
        #     d_b2w[:,i] = np.array([ p[0,i] - r[i] - self.boundary_pos[0], 
        #                             self.boundary_pos[1] - (p[1,i] + r[i]),
        #                             self.boundary_pos[2] - (p[0,i] + r[i]),
        #                             p[1,i] - r[i] - self.boundary_pos[3]])  
        # self.is_collide_b2w = d_b2w < 0
        # self.d_b2w = np.abs(d_b2w) 

    def _get_done(self):
        """Check if episode is done for all agents."""
        all_done = np.zeros((1, self.n_lr)).astype(bool)
        return all_done

    def _get_info(self):
        """Placeholder for additional information. Returns empty array."""
        return np.array([None, None, None]).reshape(3, 1)

    def step(self, a):  
        """
        Main simulation step. Updates agent states based on actions.
        """
        # Update simulation time
        self.simulation_time += self.dt 

        # Perform physics updates for multiple frames per step
        for _ in range(self.n_frames): 
            # Calculate inter-agent distances and collision states
            self._get_dist_b2b()

            # Compute agent-agent interaction forces
            sf_b2b = self._get_elastic_force()

            # Wall interactions (if not periodic boundary)
            if self.is_periodic == False:
                self._get_dist_b2w()

                # Spring force from walls
                sf_b2w = np.array([[1, 0, -1, 0], [0, -1, 0, 1]]).dot(self.is_collide_b2w * self.d_b2w) * self.k_wall
                df_b2w = np.array([[-1, 0, -1, 0], [0, -1, 0, -1]]).dot(self.is_collide_b2w * np.concatenate((self.dp, self.dp), axis=0)) * self.c_wall

            def apply_strategy(strategy, is_left, a, n_last, n_lr_last):
                """Apply different action strategies to agents.
                """
                # Select appropriate agent subset
                if is_left:
                    a_rec = a[:, :n_last]
                else:
                    a_rec = a[:, n_last:n_lr_last]
                    
                # Apply different behavior strategies
                if strategy == 'input':
                    # Use provided actions unchanged
                    pass
                elif strategy == 'static':
                    # Zero action
                    if is_left:
                        a_rec = np.zeros((self.act_dim_l, n_last))
                    else:
                        a_rec = np.zeros((self.act_dim_r, n_lr_last - n_last))
                elif strategy == 'random':
                    # Random actions
                    if is_left:
                        a_rec = np.random.uniform(-1, 1, (self.act_dim_l, n_last))
                    else:
                        a_rec = np.random.uniform(-1, 1, (self.act_dim_r, n_lr_last - n_last))
                elif strategy == 'nearest':
                    # Move toward nearest agent
                    if self.n_l_last > 0 and self.n_r_last > 0:
                        # Find nearest neighbors
                        index_concat = np.concatenate((self.index_l_last, self.index_r_last))
                        d_b2b_center_active = self.d_b2b_center[np.ix_(index_concat, index_concat)]
                        if is_left:
                            # Left agents follow right
                            ind_nearest = np.argmin(d_b2b_center_active[:n_last, n_last:n_lr_last], axis=1)
                            off_set = -0.2 * self.dp[:, self.index_r_last[ind_nearest]]/(
                                np.linalg.norm(self.dp[:, self.index_r_last[ind_nearest]], axis=0) + 1e-6)
                            delta_p = self.p[:, self.index_r_last[ind_nearest]] + off_set - self.p[:, self.index_l_last]
                            delta_v = self.dp[:, self.index_r_last[ind_nearest]] - self.dp[:, self.index_l_last]
                        else:
                            # Right agents follow left
                            ind_nearest = np.argmin(d_b2b_center_active[n_last:n_lr_last, :n_last], axis=1)
                            off_set = -0.2 * self.dp[:, self.index_l_last[ind_nearest]]/(
                                np.linalg.norm(self.dp[:, self.index_l_last[ind_nearest]], axis=0) + 1e-6)
                            delta_p = self.p[:, self.index_l_last[ind_nearest]] + off_set - self.p[:, self.index_r_last]
                            delta_v = self.dp[:, self.index_l_last[ind_nearest]] - self.dp[:, self.index_r_last]

                        if self.is_periodic:
                            delta_p = self._make_periodic(delta_p, is_rel=True)

                        # Calculate approach vector
                        goto_dir = 1 * delta_p + 2 * delta_v
                        a_rec = goto_dir

                # Update action vector with modified actions
                if is_left:
                    a[:, :n_last] = a_rec
                else:
                    a[:, n_last:n_lr_last] = a_rec

            # Apply strategies to left and right agent groups
            apply_strategy(self.l_strategy, True, a, self.n_l_last, self.n_lr_last)
            apply_strategy(self.r_strategy, False, a, self.n_l_last, self.n_lr_last)

            # Process actions into actual commands
            a_com, a_true = self._process_action(a)

            # Dynamics mode selection
            if self.dynamics_mode == 'Cartesian':
                u = a_com
            else:
                print('Wrong in updating dynamics')

            # Sum all forces acting on agents
            if self.is_periodic == True:
                F = self.sensitivity * u + sf_b2b
            elif self.is_periodic == False:
                F = self.sensitivity * u + sf_b2b + sf_b2w + df_b2w 
            else:
                print('Wrong in consider walls !!!')

            # Update agent dynamics
            self.ddp = F/self.m  # Acceleration
            self.dp += self.ddp * self.dt  # Velocity
            
            # Zero velocity for dead agents
            dead_index = self.dead_index[self.dead_index != -1]
            self.dp[:, dead_index] = 0
            
            # Velocity clamping
            self.dp[:, :self.n_l_init] = np.clip(self.dp[:, :self.n_l_init], -self.linVel_l_max, self.linVel_l_max)
            self.dp[:, self.n_l_init:self.n_lr_init] = np.clip(
                self.dp[:, self.n_l_init:self.n_lr_init], -self.linVel_r_max, self.linVel_r_max)

            # Update positions
            self.p += self.dp * self.dt
            if self.is_periodic:
                self.p = self._make_periodic(self.p, is_rel=False)

            # Update trajectory history
            if self.render_traj == True:
                self.p_traj = np.concatenate((self.p_traj[1:, :, :], self.p.reshape(1, 2, self.n_lro)), axis=0)

            # Process agent attacks and health
            self._process_attack()
            self.hp[0, :self.n_l_init][self.hp[0, :self.n_l_init] > self.hp_l_max] = self.hp_l_max
            self.hp[0, self.n_l_init:][self.hp[0, self.n_l_init:] > self.hp_r_max] = self.hp_r_max

            # Prepare return values
            obs = self._get_obs()
            rew = self._get_reward(a_true)
            done = self._get_done()
            info = self._get_info()

            # Update agent indices and counts
            self.index_l_last = self.index_l
            self.index_r_last = self.index_r
            self.n_l_last = self.index_l_last.shape[0]
            self.n_r_last = self.index_r_last.shape[0]
            self.n_lr_last = self.n_l_last + self.n_r_last

            # Remove dead agents from active lists
            mask = np.isin(self.index_l, self.dead_index, invert=True)
            self.index_l = self.index_l[mask]
            mask = np.isin(self.index_r, self.dead_index, invert=True)
            self.index_r = self.index_r[mask]
            self.n_l = self.index_l.shape[0]
            self.n_r = self.index_r.shape[0]
            self.n_lr = self.n_l + self.n_r

        return obs, rew, done, info

    def render(self, mode="human"):
        """
        Displays agents, velocities, trails, and boundaries.
        """
        # Size settings for agents
        size_l = 1200
        size_r = 1200

        # Initialize plot if needed
        if self.plot_initialized == 0:
            plt.ion()
            left, bottom, width, height = 0.09, 0.06, 0.9, 0.9
            ax = self.figure_handle.add_axes([left, bottom, width, height])
            
            # Set view limits
            if self.center_view_on_swarm == False:
                axes_lim = self.axis_lim_view_static()
            else:
                axes_lim = self.axis_lim_view_dynamic()
            
            ax.set_xlim(axes_lim[0], axes_lim[1])
            ax.set_ylim(axes_lim[2], axes_lim[3])
            ax.set_xlabel('X position [m]')
            ax.set_ylabel('Y position [m]')
            ax.set_title(f'Simulation time: {self.simulation_time:.2f} seconds')
            ax.axis('equal')
            self.plot_initialized = 1
        else:
            # Clear and update plot
            self.figure_handle.axes[0].cla()
            ax = self.figure_handle.axes[0]
            plt.ion()

            # Plot agent positions
            ax.scatter(self.p[0, self.index_l], self.p[1, self.index_l], 
                    s=size_l, c=self.color[self.index_l], marker=".", alpha=1)
            ax.scatter(self.p[0, self.index_r], self.p[1, self.index_r], 
                    s=size_r, c=self.color[self.index_r], marker=".", alpha=1)

            # Plot velocity vectors
            v_norm_l = np.linalg.norm(self.dp[:, self.index_l], axis=0) + 1e-6
            v_norm_r = np.linalg.norm(self.dp[:, self.index_r], axis=0) + 1e-6
            ax.quiver(self.p[0, self.index_l], self.p[1, self.index_l],
                    self.dp[0, self.index_l]/v_norm_l, self.dp[1, self.index_l]/v_norm_l,
                    scale=30, color=self.color[self.index_l], width=0.002)
            ax.quiver(self.p[0, self.index_r], self.p[1, self.index_r],
                    self.dp[0, self.index_r]/v_norm_r, self.dp[1, self.index_r]/v_norm_r,
                    scale=30, color=self.color[self.index_r], width=0.002)
            
            # Plot trajectories if enough history exists
            if self.simulation_time / self.dt > self.traj_len:
                for l_index in self.index_l:
                    distance_index = self._calculate_distances(l_index)
                    if distance_index:
                        ax.plot(self.p_traj[distance_index:, 0, l_index], 
                                self.p_traj[distance_index:, 1, l_index],
                                linestyle='-', color=self.color[l_index], alpha=0.4)
                    else:
                        ax.plot(self.p_traj[:, 0, l_index], self.p_traj[:, 1, l_index],
                                linestyle='-', color=self.color[l_index], alpha=0.4)

                for r_index in self.index_r:
                    distance_index = self._calculate_distances(r_index)
                    if distance_index:
                        ax.plot(self.p_traj[distance_index:, 0, r_index],
                                self.p_traj[distance_index:, 1, r_index],
                                linestyle='-', color=self.color[r_index], alpha=0.4)
                    else:
                        ax.plot(self.p_traj[:, 0, r_index], self.p_traj[:, 1, r_index],
                                linestyle='-', color=self.color[r_index], alpha=0.4)

            # Draw boundary
            ax.plot([self.boundary_pos[0], self.boundary_pos[0], self.boundary_pos[2], self.boundary_pos[2], self.boundary_pos[0]],
                    [self.boundary_pos[3], self.boundary_pos[1], self.boundary_pos[1], self.boundary_pos[3], self.boundary_pos[3]])
            
            # Update view limits
            if self.center_view_on_swarm == False:
                axes_lim = self.axis_lim_view_static()
            else:
                axes_lim = self.axis_lim_view_dynamic()
                
            ax.set_xlim(axes_lim[0], axes_lim[1])
            ax.set_ylim(axes_lim[2], axes_lim[3])
            ax.set_xlabel('X position [m]', fontsize=self.fontsize)
            ax.set_ylabel('Y position [m]', fontsize=self.fontsize)
            ax.set_title(f'Simulation time: {self.simulation_time:.2f} seconds', fontsize=self.fontsize)
            ax.tick_params(axis='both', labelsize=self.fontsize)
            ax.grid(True)

            # Update video recording if enabled
            if self.video:
                self.video.update()

        plt.ioff()
        plt.pause(0.01)

    def axis_lim_view_static(self):
        """Get static view boundaries with padding."""
        indent = 0.05
        return [
            self.boundary_pos[0] - indent,
            self.boundary_pos[2] + indent,
            self.boundary_pos[3] - indent,
            self.boundary_pos[1] + indent
        ]
        
    def axis_lim_view_dynamic(self):
        """Get dynamic view boundaries centered on swarm with padding."""
        indent = 0.1
        return [
            np.min(self.p[0]) - indent,
            np.max(self.p[0]) + indent,
            np.min(self.p[1]) - indent,
            np.max(self.p[1]) + indent
        ]

    def _make_periodic(self, x, is_rel):
        """Apply periodic boundary conditions.
        """
        if is_rel:
            # Wrap relative vectors
            x[x > self.L] -= 2 * self.L 
            x[x < -self.L] += 2 * self.L
        else:
            # Wrap absolute positions
            x[0, x[0, :] < self.boundary_pos[0]] += 2 * self.L
            x[0, x[0, :] > self.boundary_pos[2]] -= 2 * self.L
            x[1, x[1, :] < self.boundary_pos[3]] += 2 * self.L
            x[1, x[1, :] > self.boundary_pos[1]] -= 2 * self.L
        return x

    def _normalize_angle(self, x):
        """Normalize angle to [-π, π] range."""
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    def _get_size(self):
        """Get agent sizes and pairwise size sums."""
        # Concatenate sizes for different agent types
        size = np.concatenate((
            np.array([self.size_p] * self.n_l),
            np.array([self.size_e] * self.n_r),
            np.array([self.size_o] * self.n_o)
        ))
        
        # Create pairwise size matrix (ignoring self-interactions)
        sizes = np.tile(size.reshape(self.n_lro, 1), (1, self.n_lro))
        sizes = sizes + sizes.T
        np.fill_diagonal(sizes, 0)  # Zero out diagonal
        return size, sizes
        
    def _get_mass(self):
        """Get agent masses concatenated by type."""
        return np.concatenate((
            np.array([self.m_p] * self.n_l),
            np.array([self.m_e] * self.n_r),
            np.array([self.m_o] * self.n_o)
        ))

    def _get_observation_space(self):
        """Configure observation space based on topology parameters."""
        self_flag = 1 if self.is_con_self_state else 0
        topo_n_l = self.topo_nei_l2l + self.topo_nei_l2r
        topo_n_r = self.topo_nei_r2l + self.topo_nei_r2r 
        self.obs_dim_l = (self_flag + topo_n_l) * (self.dim * 2)
        self.obs_dim_r = (self_flag + topo_n_r) * (self.dim * 2)
        self.obs_dim_max = max(self.obs_dim_l, self.obs_dim_r)
        return spaces.Box(
            low=-np.inf, high=+np.inf, 
            shape=(self.obs_dim_max, self.n_lr),
            dtype=np.float32
        )

    def _get_action_space(self):
        """Configure action space based on agent types."""
        act_dim_max = max(self.act_dim_l, self.act_dim_r)
        return spaces.Box(
            low=-np.inf, high=+np.inf, 
            shape=(act_dim_max, self.n_lr),
            dtype=np.float32
        )

    def _get_focused(self, Pos, Vel, norm_threshold, width, remove_self):
        """
        Find nearest neighbors within observation range.
        """
        # Calculate distances and sort
        norms = np.linalg.norm(Pos, axis=0)
        sorted_seq = np.argsort(norms)
        norms = norms[sorted_seq]
        
        # Filter by distance
        valid_mask = norms < norm_threshold
        Pos = Pos[:, sorted_seq][:, valid_mask]
        sorted_seq = sorted_seq[valid_mask]
        Vel = Vel[:, sorted_seq]
        
        # Remove self if requested
        if remove_self:
            Pos = Pos[:, 1:]
            sorted_seq = sorted_seq[1:]
        
        # Prepare outputs
        target_Pos = np.zeros((2, width))
        target_Vel = np.zeros((2, width))
        until_idx = min(Pos.shape[1], width)
        target_Pos[:, :until_idx] = Pos[:, :until_idx]
        target_Vel[:, :until_idx] = Vel[:, :until_idx]
        target_Nei = sorted_seq[:until_idx]
        return target_Pos, target_Vel, target_Nei

    def _calculate_distances(self, id_self):
        """
        Calculate when an agent's path crosses boundary.
        """
        x_coords = self.p_traj[:, 0, id_self]
        y_coords = self.p_traj[:, 1, id_self]
        distances = np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2)
        points_greater = np.where(distances > self.boundary_L_half)[0]

        if points_greater.size > 0:
            return points_greater[-1] + 1
        return False