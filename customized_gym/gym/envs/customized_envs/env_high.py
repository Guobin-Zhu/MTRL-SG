__credits__ = ["zhugb@buaa.edu.cn"]

import gym
from gym import error, spaces
import numpy as np
import matplotlib.pyplot as plt
from .VideoWriter import VideoWriter
from .envs_cplus.c_lib import as_double_c_array, as_bool_c_array, as_int32_c_array, _load_lib
import ctypes

_LIB_f = _load_lib(env_name='Flocking')
_LIB_a = _load_lib(env_name='Adversarial')

class EnvHighSwarmEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}
    def __init__(self):
        # dimension
        self.dim = 2 

        # Numbers of agents
        self.n_o = 0     # number of obstacles

        # Observation adversarial
        self.topo_nei_l2l = 3     # left agents to left agents 
        self.topo_nei_l2r = 3     # left agents to right agents 
        self.topo_nei_r2l = 3     # right agents to left agents 
        self.topo_nei_r2r = 3     # right agents to right agents  

        # Observation flocking
        self.topo_nei_max = 5   # agent to agent
        
        # Action
        self.act_dim_l = 2
        self.act_dim_r = 2
        
        # Mass
        self.m_p = 1
        self.m_e = 1     
        self.m_o = 10
        
        # Size
        self.size_p = 0.08    
        self.size_e = 0.08 
        self.size_o = 0.2
        
        # radius of FoV adversarial
        self.d_sen_l = 14   
        self.d_sen_r = 14

        # radius of Fov flocking
        self.d_sen = 2 
        self.d_ref = 0.4 

        # physical constraints
        self.linVel_l_max = 0.8  
        self.linVel_r_max = 0.8
       
        self.linAcc_max = 1

        # attack
        self.attack_radius = 0.4
        self.attack_angle = 0.4
        self.attack_hp = 2
        self.recover_hp = 0.1
        self.attack_max = 3

        # Energy
        self.hp_l_max = 80. 
        self.hp_r_max = 80.

        # virtual leader
        self.center_leader_state = np.zeros((2 * self.dim, 2))
        self.way_count_1 = 1
        self.way_count_2 = 1
        self.leader_v = 0.7

        ## ======================================== end ========================================

        # Half boundary length
        self.boundary_L_half = 6
        self.bound_center = np.zeros(2)

        ## Venue
        self.L = self.boundary_L_half
        self.k_ball = 30       # sphere-sphere contact stiffness  N/m 
        # self.c_ball = 5      # sphere-sphere contact damping N/m/s
        self.k_wall = 100      # sphere-wall contact stiffness  N/m
        self.c_wall = 5        # sphere-wall contact damping N/m/s
        self.c_aero = 2        # sphere aerodynamic drag coefficient N/m/s

        ## Simulation setup
        self.simulation_time = 0
        self.dt = 0.1
        self.n_frames = 1  
        self.sensitivity = 1 
        self.stage_index = 0
        self.stage_flocking_count = 0
        self.stop_flag = 0

        ## Rendering
        self.traj_len = 12
        self.plot_initialized = 0
        self.center_view_on_swarm = False
        self.fontsize = 24
        width = 16
        height = 16
        self.figure_handle = plt.figure(figsize = (width,height))

    def __reinit__(self, args):
        self.n_l = args.n_l
        self.n_r = args.n_r
        self.n_lr = self.n_l + self.n_r
        self.n_l_init = self.n_l
        self.n_r_init = self.n_r
        self.n_lr_init = self.n_lr
        self.n_lro = self.n_l + self.n_r + self.n_o
        self.n_leader = args.n_leader
        self.s_text_l = np.char.mod('%d',np.arange(self.n_l))
        self.s_text_r = np.char.mod('%d',np.arange(self.n_r))
        self.render_traj = args.render_traj
        self.traj_len = args.traj_len
        self.video = args.video

        self.is_boundary = args.is_boundary
        if self.is_boundary:
            self.is_periodic = False
        else:
            self.is_periodic = True

        self.is_leader = args.is_leader[0]
        if self.is_leader:
            self.is_remarkable_leader = args.is_leader[1]
        else:
            self.is_remarkable_leader = False

        self.leader_waypoint_origin_1 = args.leader_waypoint[:self.dim]
        self.leader_waypoint_origin_2 = args.leader_waypoint[:self.dim][:, ::-1]

        self.dynamics_mode = args.dynamics_mode
        self.billiards_mode = args.billiards_mode
        self.is_augmented = args.augmented
        self.is_con_self_state = args.is_con_self_state
        self.is_training = np.array([False, False], dtype=bool) # what are the train mode of left and right agents
        self.l_strategy = args.l_strategy
        self.r_strategy = args.r_strategy
        if self.l_strategy == 'input':
            self.is_training[0] = True
        if self.r_strategy == 'input':
            self.is_training[1] = True

        self.is_collide_b2b = np.random.choice([True, False], size=(self.n_lr, self.n_lr))
        self.is_collide_b2w = np.zeros((4, self.n_lr), dtype=bool)
        self.d_b2w = np.ones((4, self.n_lro))

        self.observation_space_flocking = self._get_observation_space_flocking()  
        self.observation_space_adversarial = self._get_observation_space_adversarial()
        self.observation_space_high = self._get_observation_space_high()
        self.action_space = self._get_action_space()  
        self.action_space_high = self._get_action_space_high() 
        self.m = self._get_mass()  
        self.size, self.sizes = self._get_size()  

        if self.billiards_mode:
            self.c_wall = 0.2
            self.c_aero = 0.01

        if self.dynamics_mode == 'Cartesian':
            self.is_Cartesian = True
            self.linAcc_min = -1    
            assert (self.linAcc_min, self.linAcc_max) == (-1, 1)
        elif self.dynamics_mode == 'Polar':
            self.is_Cartesian = False
            self.linAcc_min = 0    
            self.angle_max = 0.5   
            assert self.linAcc_min >= 0
        else:
            print('Wrong in linAcc_min') 

        self.color_l = np.tile(np.array([0, 0, 1.0]), (self.n_l, 1))
        self.color_r = np.tile(np.array([1, 0.5, 0]), (self.n_r, 1))
        if self.is_leader:
            self.color_l[:self.n_leader] = np.tile(np.array([0.603, 0.803, 0.196]), (self.n_leader, 1))
            self.color_r[:self.n_leader] = np.tile(np.array([0.729, 0.333, 0.827]), (self.n_leader, 1))
        self.color = np.concatenate((self.color_l, self.color_r), axis=0)

        if self.video:
            self.video = VideoWriter(output_rate=self.dt, fps=20)
            self.video.video.setup(self.figure_handle, args.video_path) 

    def reset(self):
        # initial variables
        self.simulation_time = 0
        self.n_l = self.n_l_init
        self.n_r = self.n_r_init
        self.n_lr = self.n_lr_init
        self.index_l = np.array([i for i in np.arange(self.n_l)], dtype=np.int32)
        self.index_r = np.array([i for i in np.arange(self.n_l, self.n_lr)], dtype=np.int32)
        self.index_l_last = self.index_l
        self.index_r_last = self.index_r
        self.n_l_last = self.index_l_last.shape[0]
        self.n_r_last = self.index_r_last.shape[0]
        self.n_lr_last = self.n_l_last + self.n_r_last
        self.dead_index = -np.ones((self.n_lr,), dtype=np.int32)
        self.attack_neigh = -np.ones((self.n_lr, self.attack_max), dtype=np.int32)
        self.safe_max = np.max([self.n_l, self.n_r])
        self.safe_neigh = -np.ones((self.n_lr, self.safe_max), dtype=np.int32)
        self.heading = np.zeros((self.dim, self.n_lr))
        self.stage_index = 0
        self.stage_flocking_count = 0
        self.stop_flag = 0

        self.bound_center = np.array([0, 0])
        # x_min, y_max, x_max, y_min
        self.boundary_pos = np.array([self.bound_center[0] - self.boundary_L_half,
                                      self.bound_center[1] + self.boundary_L_half,
                                      self.bound_center[0] + self.boundary_L_half,
                                      self.bound_center[1] - self.boundary_L_half], dtype=np.float64) 

        # position
        max_size = np.max(self.size)
        W_group = 1.7
        L_group = 2
        off_set = self.boundary_L_half - W_group
        p_l = np.concatenate((np.random.uniform(-L_group, L_group, (1, self.n_l)),
                              np.random.uniform(-W_group, W_group, (1, self.n_l))), axis=0) + np.array([[4, -off_set]]).T
        p_r = np.concatenate((np.random.uniform(-L_group, L_group, (1, self.n_r)),
                              np.random.uniform(-W_group, W_group, (1, self.n_r))), axis=0) + np.array([[-4, off_set]]).T
        self.p = np.concatenate((p_l, p_r), axis=1)

        if self.render_traj == True:
            self.p_traj = np.zeros((self.traj_len, 2, self.n_lro))
            self.p_traj[0,:,:] = self.p
        
        # velocity
        self.dp = np.random.uniform(-0.005, 0.005, (2, self.n_lro))  
        if self.billiards_mode:
            self.dp = np.random.uniform(-1, 1, (2,self.n_lro))                      

        # acceleration
        self.ddp = np.zeros((2, self.n_lro))  

        # leader state
        affine_coe_x = np.random.uniform(-1.4, -0.3) if np.random.rand() < 0.5 else np.random.uniform(0.3, 1.4)
        affine_coe_y = np.random.uniform(-1.4, -0.3) if np.random.rand() < 0.5 else np.random.uniform(0.3, 1.4)
        self.leader_waypoint_1 = self.leader_waypoint_origin_1.copy()
        self.leader_waypoint_2 = self.leader_waypoint_origin_2.copy()
        self.leader_waypoint_1[0] *= affine_coe_x
        self.leader_waypoint_1[1] *= affine_coe_y
        self.leader_waypoint_2[0] *= affine_coe_x
        self.leader_waypoint_2[1] *= affine_coe_y
        self.leader_state = np.zeros((2*self.dim, 2*self.n_leader))
        self.center_leader_state[:self.dim,0] = self.leader_waypoint_1[:,0]
        self.center_leader_state[:self.dim,1] = self.leader_waypoint_2[:,0]
        offset = 0.85
        leader_pos_offset = np.array([[0, offset], 
                                      [offset, 0], 
                                      [0, -offset], 
                                      [-offset, 0]]).T
        self.leader_state[:self.dim,:self.n_leader] = self.center_leader_state[:self.dim,[0]] + leader_pos_offset
        self.leader_state[:self.dim,self.n_leader:2*self.n_leader] = self.center_leader_state[:self.dim,[1]] + leader_pos_offset
        self.initial_leader_offset = np.concatenate((self.leader_state[:self.dim,:self.n_leader] - self.leader_waypoint_1[:,[0]],
                                                     self.leader_state[:self.dim,self.n_leader:2*self.n_leader] - self.leader_waypoint_2[:,[0]]), axis=1)
        self.random_leader_offset = np.random.uniform(-0.4, 0.4, (self.dim, 2*self.n_leader))

        # energy                                   
        self.hp = np.array([self.hp_l_max for _ in range(self.n_l)] + [self.hp_r_max for _ in range(self.n_r)]).reshape(1, self.n_lr)

        if self.dynamics_mode == 'Polar': 
            self.theta = np.pi * np.random.uniform(-1, 1, (1, self.n_lro))
            self.heading = np.concatenate((np.cos(self.theta), np.sin(self.theta)), axis=0)  

        obs_l = self._get_obs_flocking(group_index=0)
        obs_r = self._get_obs_flocking(group_index=1)
        obs = np.concatenate((obs_l, obs_r), axis=1)

        return obs

    def _get_obs_flocking(self, group_index):
        if group_index == 0:
            self.obs = np.zeros((self.observation_space_flocking.shape[0], self.n_l))
            self.neighbor_index = -1 * np.ones((self.n_l, self.topo_nei_max), dtype=np.int32)
            self.nearest_leader_index = -1 * np.ones((self.n_l), dtype=np.int32)
            p = self.p[:, self.index_l].copy()
            dp = self.dp[:, self.index_l].copy()
            heading = self.heading[:, self.index_l].copy()
            leader_state = self.leader_state[:,:self.n_leader].copy()
            num_agents = self.n_l
        else:
            self.obs = np.zeros((self.observation_space_flocking.shape[0], self.n_r)) 
            self.neighbor_index = -1 * np.ones((self.n_r, self.topo_nei_max), dtype=np.int32)
            self.nearest_leader_index = -1 * np.ones((self.n_r), dtype=np.int32)
            p = self.p[:, self.index_r].copy()
            dp = self.dp[:, self.index_r].copy()
            heading = self.heading[:, self.index_r].copy()
            leader_state = self.leader_state[:,self.n_leader:2*self.n_leader].copy()
            num_agents = self.n_r

        random_permutation = np.random.permutation(self.topo_nei_max).astype(np.int32)
        conditions = np.array([self.is_periodic, self.is_leader, self.is_remarkable_leader, self.is_Cartesian, self.is_augmented, self.is_con_self_state])

        _LIB_f._get_observation(as_double_c_array(p), 
                                as_double_c_array(dp), 
                                as_double_c_array(heading),
                                as_double_c_array(self.obs),
                                as_double_c_array(leader_state), 
                                as_int32_c_array(self.neighbor_index),
                                as_int32_c_array(self.nearest_leader_index),
                                as_int32_c_array(random_permutation),
                                as_double_c_array(self.boundary_pos),
                                ctypes.c_double(self.d_sen), 
                                ctypes.c_int(self.topo_nei_max), 
                                ctypes.c_int(num_agents),
                                ctypes.c_int(self.n_leader), 
                                ctypes.c_int(self.obs_dim_agent), 
                                ctypes.c_int(self.dim), 
                                as_bool_c_array(conditions))
                
        return self.obs

    def _get_obs_adversarial(self):

        self.obs = np.zeros((self.observation_space_adversarial.shape[0], self.n_lr)) 
        conditions = np.array([self.is_periodic, self.is_Cartesian, True])

        _LIB_a._get_observation(as_double_c_array(self.p), 
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

        return self.obs

    def get_observation_rl(self):
        obs = np.zeros((6, 2))
        obs[2*self.dim,0] = self.n_l
        obs[2*self.dim,1] = self.n_r
        obs[:2*self.dim,0] = np.random.uniform(-0.1, 0.1, (4,))
        obs[:2*self.dim,1] = np.random.uniform(-0.1, 0.1, (4,))
        obs[2*self.dim + 1,0] = np.random.uniform(-0.1, 0.1)
        obs[2*self.dim + 1,1] = np.random.uniform(-0.1, 0.1)

        obs_l_r = self.center_leader_state[:,1] - self.center_leader_state[:,0]
        rel_p = self.p[:,self.index_l] - self.center_leader_state[:self.dim,[1]]
        rel_p_norm = np.linalg.norm(rel_p, axis=0)
        if np.min(rel_p_norm) < self.d_sen:
            obs[:2*self.dim,0] = obs_l_r
            obs[:2*self.dim,1] = -obs_l_r
            obs[2*self.dim + 1,0] = self.n_r
            obs[2*self.dim + 1,1] = self.n_l

        return obs

    def get_reward_rl(self, a):
        rew = np.zeros((1, 2))
        a = a.flatten()
        if self.stage_index == 0 and a[0] != 0:
            rew[0][0] -= 1
        if self.stage_index == 1 and a[0] != 1:
            rew[0][0] -= 1 
        if self.stage_index == 0 and a[1] != 0:
            rew[0][1] -= 1
        if self.stage_index == 1 and a[1] != 1:
            rew[0][1] -= 1

        return rew

    def step_rl(self, a):
        obs_rl = self.get_observation_rl()
        rew_rl = self.get_reward_rl(a)

        return obs_rl, rew_rl

    def _judge_stage(self):
        if self.stage_index == 0:
            if self.n_l > 0 and self.n_r > 0:
                rel_p = self.p[:,self.index_l] - self.center_leader_state[:self.dim,[1]]
                rel_p_norm = np.linalg.norm(rel_p, axis=0)
                if np.min(rel_p_norm) < self.d_sen:
                    self.stage_index = 1
        
        if self.stage_index == 1:
            if self.n_l == 0 or self.n_r == 0:
                self.stage_index = 0
                self.stage_flocking_count = 1

    def _process_attack(self):
        self.attack_neigh = -np.ones((self.n_lr_init, self.attack_max), dtype=np.int32)
        self.safe_neigh = -np.ones((self.n_lr_init, self.safe_max), dtype=np.int32)
        _LIB_a._process_attack(as_int32_c_array(self.index_l), 
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

    def _process_action(self, a):
        a_com = np.zeros((self.dim, self.n_lr_init))
        a_true = np.zeros((self.dim, self.n_lr))
        for a_i in range(self.n_l):
            act_index = np.where(self.index_l_last == self.index_l[a_i])[0]
            a_com[:, [self.index_l[a_i]]] = a[:, act_index]
            a_true[:, [a_i]] = a[:, act_index]
        for a_i in range(self.n_r):
            act_index = np.where(self.index_r_last == self.index_r[a_i])[0]
            a_com[:, [self.index_r[a_i]]] = a[:, act_index + self.index_l_last.shape[0]]
            a_true[:, [a_i + self.n_l]] = a[:, act_index + self.index_l_last.shape[0]]

        return a_com, a_true

    def _get_elastic_force(self):
        sf_b2b = np.zeros((2, self.n_lro))
        _LIB_a._sf_b2b_all(as_double_c_array(self.p), 
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

        return sf_b2b

    def _get_dist_b2b(self):
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
        _LIB_a._get_dist_b2w(as_double_c_array(self.p), 
                             as_double_c_array(self.size), 
                             as_double_c_array(self.d_b2w), 
                             as_bool_c_array(self.is_collide_b2w),
                             ctypes.c_int(self.dim), 
                             ctypes.c_int(self.n_lr_init), 
                             as_double_c_array(self.boundary_pos))

    def _get_done(self):
        all_done = np.zeros((1, self.n_lr)).astype(bool)
        return all_done

    def _get_info(self):
        return np.array( [None, None, None] ).reshape(3,1)

    def step(self, a):  
        self.simulation_time += self.dt 

        for _ in range(self.n_frames): 
            if self.dynamics_mode == 'Polar':  
                a[0,:self.n_l_last] *= self.angle_max
                a[0,self.n_l_last:self.n_lr_last] *= self.angle_max
                a[1,:self.n_l_last] = (self.linAcc_max-self.linAcc_min)/2 * a[1,:self.n_l_last] + (self.linAcc_max+self.linAcc_min)/2 
                a[1,self.n_l_last:self.n_lr_last] = (self.linAcc_max-self.linAcc_min)/2 * a[1,self.n_l_last:self.n_lr_last] + (self.linAcc_max+self.linAcc_min)/2 

            # inter-agent distance and collision
            self._get_dist_b2b()

            # inter-agent elastic force
            sf_b2b = self._get_elastic_force()

            # agent-wall elastic force
            if self.is_periodic == False:
                self._get_dist_b2w()
                sf_b2w = np.array([[1, 0, -1, 0], [0, -1, 0, 1]]).dot(self.is_collide_b2w * self.d_b2w) * self.k_wall   
                df_b2w = np.array([[-1, 0, -1, 0], [0, -1, 0, -1]]).dot(self.is_collide_b2w*np.concatenate((self.dp, self.dp), axis=0))  *  self.c_wall   

            def apply_strategy(strategy, is_left, a, n_last, n_lr_last):
                if is_left:
                    a_rec = a[:, :n_last]
                else:
                    a_rec = a[:, n_last:n_lr_last]
                    
                if strategy == 'input':
                    pass
                elif strategy == 'static':
                    a_rec = np.zeros((self.act_dim_l, n_last)) if is_left else np.zeros((self.act_dim_r, n_lr_last - n_last))
                elif strategy == 'random':
                    a_rec = np.random.uniform(-1, 1, (self.act_dim_l, n_last)) if is_left else np.random.uniform(-1, 1, (self.act_dim_r, n_lr_last - n_last))
                    if self.dynamics_mode == 'Polar':
                        a_rec[0] *= self.angle_max
                        a_rec[1] = (self.linAcc_max - self.linAcc_min) / 2 * a_rec[1] + (self.linAcc_max + self.linAcc_min) / 2
                elif strategy == 'nearest':
                    if self.n_l_last > 0 and self.n_r_last > 0:
                        index_concat = np.concatenate((self.index_l_last, self.index_r_last))
                        d_b2b_center_active = self.d_b2b_center[np.ix_(index_concat, index_concat)]
                        if is_left:
                            ind_nearest = np.argmin(d_b2b_center_active[:n_last, n_last:n_lr_last], axis=1)
                            off_set = -0.2 * self.dp[:, self.index_r_last[ind_nearest]]/np.linalg.norm(self.dp[:, self.index_r_last[ind_nearest]], axis=0)
                            delta_p = self.p[:, self.index_r_last[ind_nearest]] + off_set - self.p[:, self.index_l_last]
                            delta_v = self.dp[:, self.index_r_last[ind_nearest]] - self.dp[:, self.index_l_last]
                        else:
                            ind_nearest = np.argmin(d_b2b_center_active[n_last:n_lr_last, :n_last], axis=1)
                            off_set = -0.2 * self.dp[:, self.index_l_last[ind_nearest]]/np.linalg.norm(self.dp[:, self.index_l_last[ind_nearest]], axis=0)
                            delta_p = self.p[:, self.index_l_last[ind_nearest]] + off_set - self.p[:, self.index_r_last]
                            delta_v = self.dp[:, self.index_l_last[ind_nearest]] - self.dp[:, self.index_r_last]

                        if self.is_periodic:
                            delta_p = self._make_periodic(delta_p, is_rel=True)

                        goto_dir = 1 * delta_p + 2 * delta_v # 1, 2
                        if self.dynamics_mode == 'Cartesian':
                            a_rec = goto_dir
                        elif self.dynamics_mode == 'Polar':
                            heading = self.heading[:, :n_last] if is_left else self.heading[:, n_last:n_lr_last]
                            dot_products = np.sum(heading * goto_dir, axis=0)
                            norms_heading = np.linalg.norm(heading, axis=0)
                            norms_goto = np.linalg.norm(goto_dir, axis=0)
                            cos_angles = dot_products / (norms_heading * norms_goto + 1e-8)
                            angles = np.arccos(np.clip(cos_angles, -1.0, 1.0))
                            cross_products_z = heading[0, :] * goto_dir[1, :] - heading[1, :] * goto_dir[0, :]
                            angles[cross_products_z < 0] = -angles[cross_products_z < 0]
                            a_rec[0] = angles
                            a_rec[1] = np.linalg.norm(goto_dir, axis=0)
                if is_left:
                    a[:, :n_last] = a_rec
                else:
                    a[:, n_last:n_lr_last] = a_rec

            self.r_strategy = 'nearest' if self.stage_index == 1 else 'input'
            apply_strategy(self.l_strategy, True, a, self.n_l_last, self.n_lr_last)
            apply_strategy(self.r_strategy, False, a, self.n_l_last, self.n_lr_last)

            # get the plus action to the desired position of local leader
            if self.stage_index == 0:
                k_p = 0.8 if self.stop_flag == 0 else 0.0
                k_d = 1.2 if self.stop_flag == 0 else 0.0
                if self.n_l_last > 0:
                    a_plus = k_p * (self.leader_state[:self.dim,:self.n_leader] - self.p[:,self.index_l_last[:self.n_leader]]) + \
                             k_d * (self.leader_state[self.dim:,:self.n_leader] - self.dp[:,self.index_l_last[:self.n_leader]])
                    a[:,:self.n_leader] = a_plus
                if self.n_r_last > 0:
                    a_plus = k_p * (self.leader_state[:self.dim,self.n_leader:2*self.n_leader] - self.p[:,self.index_r_last[:self.n_leader]]) + \
                             k_d * (self.leader_state[self.dim:,self.n_leader:2*self.n_leader] - self.dp[:,self.index_r_last[:self.n_leader]])
                    a[:,self.n_l_last:self.n_l_last + self.n_leader] = a_plus

            # get the actual action to perform
            a_com, _ = self._process_action(a)
            a_com = np.clip(a_com, self.linAcc_min, self.linAcc_max)

            if self.dynamics_mode == 'Cartesian':
                u = a_com   
            elif self.dynamics_mode == 'Polar':      
                self.theta += a_com[[0],:]
                self.theta = self._normalize_angle(self.theta)
                self.heading = np.concatenate((np.cos(self.theta), np.sin(self.theta)), axis=0) 
                u = a_com[[1], :] * self.heading 
            else:
                print('Wrong in updating dynamics')

            # sum-force
            if self.is_periodic == True:
                F = self.sensitivity * u + sf_b2b
            elif self.is_periodic == False:
                F = self.sensitivity * u + sf_b2b + sf_b2w + df_b2w 
            else:
                print('Wrong in consider walls !!!')

            # acceleration
            self.ddp = F/self.m

            # velocity
            self.dp += self.ddp * self.dt
            if self.stop_flag == 1:
                self.dp[:,self.index_l[:self.n_leader]] = 0.9 * self.dp[:,self.index_l[:self.n_leader]]
                self.dp[:,self.index_r[:self.n_leader]] = 0.9 * self.dp[:,self.index_r[:self.n_leader]]
            dead_index = self.dead_index[self.dead_index != -1]
            self.dp[:, dead_index] = 0
            self.dp[:,:self.n_l_init] = np.clip(self.dp[:,:self.n_l_init], -self.linVel_l_max, self.linVel_l_max)
            self.dp[:,self.n_l_init:self.n_lr_init] = np.clip(self.dp[:,self.n_l_init:self.n_lr_init], -self.linVel_r_max, self.linVel_r_max)

            # position
            self.p += self.dp * self.dt
            if self.is_periodic:
                self.p = self._make_periodic(self.p, is_rel=False)

            # The leader state needs to be updated separately in boundary case
            if self.stage_index == 0:
                self._get_way_count()
                self._update_leader_state()
                if self.simulation_time % 1 < 0.1:
                    self.random_leader_offset = np.random.uniform(-0.3, 0.3, (self.dim, 2*self.n_leader))
                random_pos_offset = self.initial_leader_offset + self.random_leader_offset
                self.leader_state[:self.dim,:self.n_leader] = self.center_leader_state[:self.dim,[0]] + random_pos_offset[:,:self.n_leader]
                self.leader_state[:self.dim,self.n_leader:2*self.n_leader] = self.center_leader_state[:self.dim,[1]] + random_pos_offset[:,self.n_leader:2*self.n_leader]
                self.leader_state[self.dim:,:self.n_leader] = np.tile(self.center_leader_state[self.dim:,[0]], (1, self.n_leader))
                self.leader_state[self.dim:,self.n_leader:2*self.n_leader] = np.tile(self.center_leader_state[self.dim:,[1]], (1, self.n_leader))

            if self.render_traj == True:
                self.p_traj = np.concatenate((self.p_traj[1:,:,:], self.p.reshape(1, 2, self.n_lro)), axis=0)

            # execute attack
            if self.stage_index == 1:
                self._process_attack()
                self.hp[0,:self.n_l_init][self.hp[0,:self.n_l_init]>self.hp_l_max] = self.hp_l_max
                self.hp[0,self.n_l_init:][self.hp[0,self.n_l_init:]>self.hp_r_max] = self.hp_r_max

            # judge stage
            if self.stage_flocking_count == 1:
                self.d_sen = 10
                self.d_ref = 0.6
            self._judge_stage()

            # output
            if self.stage_index == 0:
                obs_l = self._get_obs_flocking(group_index=0)
                obs_r = self._get_obs_flocking(group_index=1)
                obs = np.concatenate((obs_l, obs_r), axis=1)
            else:
                obs = self._get_obs_adversarial()

            # update agent index
            self.index_l_last = self.index_l
            self.index_r_last = self.index_r
            self.n_l_last = self.index_l_last.shape[0]
            self.n_r_last = self.index_r_last.shape[0]
            self.n_lr_last = self.n_l_last + self.n_r_last

            mask = np.isin(self.index_l, self.dead_index, invert=True)
            self.index_l = self.index_l[mask]
            mask = np.isin(self.index_r, self.dead_index, invert=True)
            self.index_r = self.index_r[mask]
            self.n_l = self.index_l.shape[0]
            self.n_r = self.index_r.shape[0]
            self.n_lr = self.n_l + self.n_r

        return obs

    def render(self, mode="human"): 

        size_l = 600
        size_r = 600

        if self.plot_initialized == 0:

            plt.ion()

            left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
            ax = self.figure_handle.add_axes([left, bottom, width, height],projection = None)

            # Plot agents position
            ax.scatter(self.p[0,self.index_l], self.p[1,self.index_l], s = size_l, c = self.color[self.index_l], marker = ".", alpha = 1)
            ax.scatter(self.p[0,self.index_r], self.p[1,self.index_r], s = size_r, c = self.color[self.index_r], marker = ".", alpha = 1)
                
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
            # ax.axis('equal')
            ax.grid

            plt.ioff()
            plt.pause(0.01)

            self.plot_initialized = 1
        else:
            self.figure_handle.axes[0].cla()
            ax = self.figure_handle.axes[0]

            plt.ion()

            ax.scatter(self.p[0,self.index_l], self.p[1,self.index_l], s = size_l, c = self.color[self.index_l], marker = ".", alpha = 1)
            ax.scatter(self.p[0,self.index_r], self.p[1,self.index_r], s = size_r, c = self.color[self.index_r], marker = ".", alpha = 1)

            # for l_index in self.index_l:
            #     ax.text(self.p[0,l_index], self.p[1,l_index], self.s_text_l[l_index], fontsize=self.fontsize)
            # for r_index in self.index_r:
            #     ax.text(self.p[0,r_index], self.p[1,r_index], self.s_text_r[r_index - self.n_l_init], fontsize=self.fontsize)

            # ax.scatter(self.leader_state[0], self.leader_state[1], s = size_l, c = self.color[:8], marker = ".", alpha = 0.5)

            if self.simulation_time / self.dt > self.traj_len:
                for l_index in self.index_l:
                    distance_index = self._calculate_distances(l_index)
                    if distance_index:
                        ax.plot(self.p_traj[distance_index:,0,l_index], self.p_traj[distance_index:,1,l_index], linestyle='-', color=self.color[l_index], alpha=0.4)
                    else:
                        ax.plot(self.p_traj[:,0,l_index], self.p_traj[:,1,l_index], linestyle='-', color=self.color[l_index], alpha=0.4)

                for r_index in self.index_r:
                    distance_index = self._calculate_distances(r_index)
                    if distance_index:
                        ax.plot(self.p_traj[distance_index:,0,r_index], self.p_traj[distance_index:,1,r_index], linestyle='-', color=self.color[r_index], alpha=0.4)
                    else:
                        ax.plot(self.p_traj[:,0,r_index], self.p_traj[:,1,r_index], linestyle='-', color=self.color[r_index], alpha=0.4)

            # if self.dynamics_mode == 'Polar':
            #     ax.quiver(self.p[0,self.index_l], self.p[1,self.index_l], self.heading[0,self.index_l], self.heading[1,self.index_l], scale=30, color=self.color[self.index_l], width = 0.002)
            #     ax.quiver(self.p[0,self.index_r], self.p[1,self.index_r], self.heading[0,self.index_r], self.heading[1,self.index_r], scale=30, color=self.color[self.index_r], width = 0.002)
            # elif self.dynamics_mode == 'Cartesian':
            v_norm_l = np.linalg.norm(self.dp[:,self.index_l], axis=0) + 1e-6
            v_norm_r = np.linalg.norm(self.dp[:,self.index_r], axis=0) + 1e-6
            ax.quiver(self.p[0,self.index_l], self.p[1,self.index_l], self.dp[0,self.index_l]/v_norm_l, self.dp[1,self.index_l]/v_norm_l, scale=30, color=self.color[self.index_l], width = 0.002)
            ax.quiver(self.p[0,self.index_r], self.p[1,self.index_r], self.dp[0,self.index_r]/v_norm_r, self.dp[1,self.index_r]/v_norm_r, scale=30, color=self.color[self.index_r], width = 0.002)
            
            ax.plot(np.array([self.boundary_pos[0], self.boundary_pos[0], self.boundary_pos[2], self.boundary_pos[2], self.boundary_pos[0]]), 
                    np.array([self.boundary_pos[3], self.boundary_pos[1], self.boundary_pos[1], self.boundary_pos[3], self.boundary_pos[3]]))

            ax.scatter(self.leader_waypoint_1[0], self.leader_waypoint_1[1], color='blue', label='waypoint1')

            for index_i in range(self.n_l):
                for index_j in range(index_i):
                    dist_ij = np.linalg.norm(self.p[:, self.index_l[index_i]] - self.p[:, self.index_l[index_j]])
                    if (dist_ij > self.d_ref - 0.05) and (dist_ij < self.d_ref + 0.05):
                        ax.plot(np.array([self.p[0][self.index_l[index_i]], self.p[0][self.index_l[index_j]]]), np.array([self.p[1][self.index_l[index_i]], self.p[1][self.index_l[index_j]]]), linestyle='-.', color=(0.1, 0.2, 0.3))
            
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
            # ax.axis('equal')
            ax.grid(True)

            plt.ioff()
            plt.pause(0.01)
        
            if self.video:
                self.video.update()

    def axis_lim_view_static(self):
        indent = 0.2
        x_min = self.boundary_pos[0] - indent
        x_max = self.boundary_pos[2] + indent
        y_min = self.boundary_pos[3] - indent
        y_max = self.boundary_pos[1] + indent
        return [x_min, x_max, y_min, y_max]
    
    def axis_lim_view_dynamic(self):
        indent = 0.2
        x_min = np.min(self.p[0]) - indent
        x_max = np.max(self.p[0]) + indent
        y_min = np.min(self.p[1]) - indent
        y_max = np.max(self.p[1]) + indent

        return [x_min, x_max, y_min, y_max]

    def _make_periodic(self, x, is_rel):
        if is_rel:
            x[x >  self.L] -= 2*self.L 
            x[x < -self.L] += 2*self.L
        else:
            x[0, x[0,:] < self.boundary_pos[0]] += 2*self.L
            x[0, x[0,:] > self.boundary_pos[2]] -= 2*self.L
            x[1, x[1,:] < self.boundary_pos[3]] += 2*self.L
            x[1, x[1,:] > self.boundary_pos[1]] -= 2*self.L
        return x

    def _normalize_angle(self, x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    def _get_size(self):
        size = np.concatenate((np.array([self.size_p for _ in range(self.n_l)]), 
                               np.array([self.size_e for _ in range(self.n_r)]), 
                               np.array([self.size_o for _ in range(self.n_o)])))  
        sizes = np.tile(size.reshape(self.n_lro,1), (1,self.n_lro))
        sizes = sizes + sizes.T
        sizes[np.arange(self.n_lro), np.arange(self.n_lro)] = 0
        return size, sizes
    
    def _get_mass(self):
        m = np.concatenate((np.array([self.m_p for _ in range(self.n_l)]), 
                            np.array([self.m_e for _ in range(self.n_r)]), 
                            np.array([self.m_o for _ in range(self.n_o)]))) 
        return m

    def _get_focused(self, Pos, Vel, norm_threshold, width, remove_self):
        # assert A.shape[0] == 2
        norms = np.linalg.norm(Pos, axis=0)
        sorted_seq = np.argsort(norms)    
        Pos = Pos[:, sorted_seq]   
        norms = norms[sorted_seq] 
        Pos = Pos[:, norms < norm_threshold] 
        sorted_seq = sorted_seq[norms < norm_threshold]   
        if remove_self == True:
            Pos = Pos[:,1:]  
            sorted_seq = sorted_seq[1:]                    
        Vel = Vel[:, sorted_seq]
        target_Pos = np.zeros((2, width))
        target_Vel = np.zeros((2, width))
        until_idx = np.min([Pos.shape[1], width])
        target_Pos[:, :until_idx] = Pos[:, :until_idx] 
        target_Vel[:, :until_idx] = Vel[:, :until_idx]
        target_Nei = sorted_seq[:until_idx]
        return target_Pos, target_Vel, target_Nei

    def _get_observation_space_adversarial(self):
        # if self.is_con_self_state:
        self_flag = 1
        # else:
        #     self_flag = 0

        topo_n_l = self.topo_nei_l2l + self.topo_nei_l2r
        topo_n_r = self.topo_nei_r2l + self.topo_nei_r2r 
        self.obs_dim_l = 2 * self.dim * (self_flag + topo_n_l)
        self.obs_dim_r = 2 * self.dim * (self_flag + topo_n_r)
        if self.dynamics_mode == 'Polar':
            self.obs_dim_l += 2
            self.obs_dim_r += 2  
        self.obs_dim_max = np.max([self.obs_dim_l, self.obs_dim_r])   
        observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(self.obs_dim_max, self.n_lr), dtype=np.float32)
        return observation_space

    def _get_observation_space_flocking(self):
        if self.is_con_self_state:
            self_flag = 1
        else:
            self_flag = 0

        if self.is_leader and self.is_remarkable_leader:
            self.obs_dim_agent = 2 * self.dim * (self.topo_nei_max + 1 + self_flag)
        else:
            self.obs_dim_agent = 2 * self.dim * (self.topo_nei_max + 0 + self_flag)   

        if self.dynamics_mode == 'Polar':
            self.obs_dim_agent += self.dim

        observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(self.obs_dim_agent, self.n_lr), dtype=np.float32)
        return observation_space

    def _get_observation_space_high(self):
        observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(6, 2), dtype=np.float32)
        return observation_space

    def _get_action_space(self):
        act_dim_max = np.max([self.act_dim_l, self.act_dim_r])
        action_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(act_dim_max, self.n_lr), dtype=np.float32)
        return action_space

    def _get_action_space_high(self):
        action_space = spaces.Discrete(2)
        return action_space

    def _calculate_distances(self, id_self):
        x_coords = self.p_traj[:, 0, id_self]
        y_coords = self.p_traj[:, 1, id_self]
        distances = np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2)
        points_greater = np.where(distances > self.boundary_L_half)[0]

        if len(points_greater) > 0:
            return points_greater[-1] + 1
        else:
            return False

    def _get_way_count(self):
        dist_1 = np.linalg.norm(self.center_leader_state[:self.dim,0] - self.leader_waypoint_1[:,self.way_count_1])
        dist_2 = np.linalg.norm(self.center_leader_state[:self.dim,1] - self.leader_waypoint_2[:,self.way_count_2])
        # Switching waypoint by distance
        if dist_1 < 0.1:
            if self.way_count_1 < self.leader_waypoint_1.shape[1] - 1:
                self.way_count_1 += 1
            else:
                self.stop_flag = 1
        if dist_2 < 0.1:
            if self.way_count_2 < self.leader_waypoint_2.shape[1] - 1:
                self.way_count_2 += 1
            else:
                self.stop_flag = 1

    def _update_leader_state(self):
        d_shill_target_1 = np.linalg.norm(self.leader_waypoint_1[:,self.way_count_1] - self.center_leader_state[:self.dim,0])
        if d_shill_target_1 > 0.1:
            self.center_leader_state[self.dim:2*self.dim,0] = self.leader_v*(self.leader_waypoint_1[:,self.way_count_1] - self.center_leader_state[0:self.dim,0])/d_shill_target_1
        else:
            self.center_leader_state[self.dim:2*self.dim,0] = 0.7*self.center_leader_state[self.dim:2*self.dim,0]

        d_shill_target_2 = np.linalg.norm(self.leader_waypoint_2[:,self.way_count_2] - self.center_leader_state[:self.dim,1])
        if d_shill_target_2 > 0.1:
            self.center_leader_state[self.dim:2*self.dim,1] = self.leader_v*(self.leader_waypoint_2[:,self.way_count_2] - self.center_leader_state[0:self.dim,1])/d_shill_target_2
        else:
            self.center_leader_state[self.dim:2*self.dim,1] = 0.7*self.center_leader_state[self.dim:2*self.dim,1]

        self.center_leader_state[:self.dim] = self.center_leader_state[:self.dim] + self.center_leader_state[self.dim:2*self.dim]*self.dt


    