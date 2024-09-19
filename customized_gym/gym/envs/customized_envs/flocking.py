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
    ''' A kind of MPE env.
    ball1 ball2 ball3 ...     (in order of pursuers, escapers and obstacles)
    sf = spring force,  df = damping force
    the forces include: u, ball(spring forces), aerodynamic forces
    x, dx: (2,n), position and vel of agent i  
    '''

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 45}
    def __init__(self):
        
        self.reward_sharing_mode = 'individual'   # select one from ['sharing_mean', 'sharing_max', 'individual'] 
        
        self.penalize_control_effort = False         
        self.penalize_inter_agent_distance = True
        self.penalize_collide_agents = False       
        self.penalize_collide_obstacles = False   
        self.penalize_collide_walls = False

        # dimension
        self.dim = 2

        # Numbers of agents
        self.n_a = 10   # number of agents
        self.n_o = 0     # number of obstacles

        # Observation 
        self.topo_nei_max = 6   # agent to agent 
        
        # Action
        self.act_dim_agent = self.dim
        
        # Mass
        self.m_a = 1     
        self.m_o = 10
        
        # Size  
        self.size_a = 0.035 
        self.size_o = 0.2
        
        # radius 
        self.d_sen = 3
        self.d_ref = 0.6

        # physical constraint
        self.Vel_max = 0.8
        self.Vel_min = 0.4
        self.Acc_max = 1

        # virtual leader
        self.center_leader_state = np.zeros((2*self.dim, 1))
        self.way_count = 1
        self.leader_v = 0.6

        # Properties of obstacles
        self.obstacles_cannot_move = True 
        self.obstacles_is_constant = False
        if self.obstacles_is_constant:   # then specify their locations:
            self.p_o = np.array([[-0.5, 0.5],[0, 0]])
        ## ======================================== end ========================================

        # Half boundary length
        self.boundary_L_half = 3
        self.bound_center = np.zeros(2)

        ## Venue
        self.L = self.boundary_L_half
        self.k_ball = 30       # sphere-sphere contact stiffness  N/m 
        # self.c_ball = 5      # sphere-sphere contact damping N/m/s
        self.k_wall = 100      # sphere-wall contact stiffness  N/m
        self.c_wall = 5        # sphere-wall contact damping N/m/s
        self.c_aero = 1.2      # sphere aerodynamic drag coefficient N/m/s

        ## Simulation Steps
        self.simulation_time = 0
        self.dt = 0.1
        self.n_frames = 1  
        self.sensitivity = 1

        ## Rendering
        self.traj_len = 15
        self.plot_initialized = 0
        self.center_view_on_swarm = False
        self.fontsize = 24
        width = 16
        height = 16
        self.figure_handle = plt.figure(figsize = (width,height))

    def __reinit__(self, args):
        self.n_a = args.n_a
        self.n_l = args.n_l
        self.s_text = np.char.mod('%d',np.arange(self.n_a))
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
        self.dynamics_mode = args.dynamics_mode
        self.agent_strategy = args.agent_strategy
        self.is_augmented = args.augmented
        self.is_con_self_state = args.is_con_self_state

        self.leader_waypoint_origin = args.leader_waypoint[:self.dim]
        self.leader_waypoint = args.leader_waypoint[:self.dim]

        self.n_ao = self.n_a + self.n_o
        self.B = np.zeros((self.n_ao, self.n_ao))
        self.I = np.zeros((self.n_ao, self.n_ao))
        self.is_collide_b2b = np.random.choice([True, False], size=(self.n_a, self.n_a))
        self.is_collide_b2w = np.zeros((4, self.n_a), dtype=bool)
        self.d_b2w = np.ones((4, self.n_ao))

        self.observation_space = self._get_observation_space()  
        self.action_space = self._get_action_space()   
        self.m = self._get_mass()  
        self.size, self.sizes = self._get_size()  
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
            print('Wrong in linAcc_p_min')
        
        self.color = np.tile(np.array([1, 0.5, 0]), (self.n_a, 1))
        if self.is_leader:
            self.color[:self.n_l] = np.tile(np.array([0, 0, 1]), (self.n_l, 1))
        if self.video:
            self.video = VideoWriter(output_rate=self.dt, fps=40)
            self.video.video.setup(self.figure_handle, args.video_path)

    def reset(self):
        self.simulation_time = 0
        self.d_sen = 3
        self.heading = np.zeros((self.dim, self.n_a))
        self.leader_state = np.zeros((2 * self.dim, self.n_l))

        # bound position
        self.bound_center = np.array([0, 0])
        # x_min, y_max, x_max, y_min
        self.boundary_pos = np.array([self.bound_center[0] - self.boundary_L_half,
                                      self.bound_center[1] + self.boundary_L_half,
                                      self.bound_center[0] + self.boundary_L_half,
                                      self.bound_center[1] - self.boundary_L_half], dtype=np.float64) 

        # initialize position
        random_int = 3
        if self.is_leader:
            self.p = np.concatenate((np.random.uniform(-random_int, random_int, (2, self.n_l)), 
                                     np.random.uniform(-random_int, random_int, (2, self.n_ao - self.n_l))), axis=1)   # Initialize self.p
        else:
            self.p = np.random.uniform(-random_int, random_int, (2, self.n_ao))   # Initialize self.p

        if self.render_traj == True:
            self.p_traj = np.zeros((self.traj_len, 2, self.n_ao))
            self.p_traj[-1,:,:] = self.p
         
        # initilize velocity
        if self.dynamics_mode == 'Cartesian':
            self.dp = np.random.uniform(-0.5, 0.5, (self.dim, self.n_a))
        else:
            self.dp = np.zeros((self.dim, self.n_ao))

        if self.obstacles_cannot_move:
            self.dp[:, self.n_a:self.n_ao] = 0

        # initilize acceleration
        self.ddp = np.zeros((2, self.n_ao))         

        if self.is_boundary:
            self.leader_state[:self.dim] = self.leader_waypoint[:,[0]] + self.p[:,:self.n_l]

            # static random position
            # self.leader_state[:self.dim] = np.random.uniform(-self.boundary_L_half, self.boundary_L_half, (2, self.n_l))

            # dyanmic random position
            # theta_rotate = np.random.uniform(-np.pi, np.pi)
            # matrix_rotate = np.array([[np.cos(theta_rotate), np.sin(theta_rotate)], [-np.sin(theta_rotate), np.cos(theta_rotate)]])
            # self.leader_waypoint = np.dot(matrix_rotate, self.leader_waypoint_origin)
            # # self.leader_waypoint = np.random.uniform(-1, 1, (2, 1)) + self.leader_waypoint
            # self.leader_state[:self.dim] = self.leader_waypoint[:,[0]] + self.p[:,:self.n_l]

            self.center_leader_state[:self.dim] = self.leader_waypoint[:,[0]]
            self.initial_p = self.p.copy()
        else:
            self.leader_state[:self.dim] = np.zeros((2, self.n_l))
            random_integer = np.random.randint(0, 16, size=1) * 2 *np.pi / 15 * np.ones(self.n_l)
            self.leader_state[self.dim:] = self.leader_v * np.array([np.cos(random_integer), np.sin(random_integer)]) 

        if self.dynamics_mode == 'Polar': 
            self.theta = np.pi * np.random.uniform(-1, 1, (1, self.n_ao))
            self.heading = np.concatenate((np.cos(self.theta), np.sin(self.theta)), axis=0)                                 

        obs = self._get_obs()
        # cent_obs = self._get_cent_obs()

        return obs
        # return obs, cent_obs

    def _get_obs(self):

        self.obs = np.zeros(self.observation_space.shape) 
        self.neighbor_index = -1 * np.ones((self.n_a, self.topo_nei_max), dtype=np.int32)
        self.nearest_leader_index = -1 * np.ones((self.n_a), dtype=np.int32)
        random_permutation = np.random.permutation(self.topo_nei_max).astype(np.int32)
        conditions = np.array([self.is_periodic, self.is_leader, self.is_remarkable_leader, self.is_Cartesian, self.is_augmented, self.is_con_self_state])

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
                
        return self.obs

    def _get_cent_obs(self):
        local_obs = self.obs
        state = np.concatenate((self.p, self.dp), axis=0)
        flatten_state = state.T.reshape(-1)
        global_obs = np.tile(flatten_state, (self.n_a, 1)).T
        cent_obs = np.concatenate((local_obs, global_obs), axis=0)
        
        return cent_obs
      
    def _get_reward(self, a):

        reward_a = np.zeros((1, self.n_ao))
        coefficients = np.array([15, 1.5, 7] + [1.4, 6] + [4] + [1, 1, 1] + [5] + [12], dtype=np.float64) # 15, 2.5, 2
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
    
        return reward_a

    def _get_dist_b2b(self):
        all_pos = np.tile(self.p, (self.n_ao, 1))   
        my_pos = self.p.T.reshape(2 * self.n_ao, 1) 
        my_pos = np.tile(my_pos, (1, self.n_ao))   
        relative_p_2n_n =  all_pos - my_pos
        if self.is_periodic == True:
            relative_p_2n_n = self._make_periodic(relative_p_2n_n, is_rel=True)
        d_b2b_center = np.sqrt(relative_p_2n_n[::2,:]**2 + relative_p_2n_n[1::2,:]**2)  
        d_b2b_edge = d_b2b_center - self.sizes
        isCollision = (d_b2b_edge < 0)
        d_b2b_edge = np.abs(d_b2b_edge)
        return d_b2b_center, d_b2b_edge, isCollision
    
    def _get_dist_b2w(self):
        _LIB._get_dist_b2w(as_double_c_array(self.p), 
                           as_double_c_array(self.size), 
                           as_double_c_array(self.d_b2w), 
                           as_bool_c_array(self.is_collide_b2w),
                           ctypes.c_int(self.dim), 
                           ctypes.c_int(self.n_a), 
                           as_double_c_array(self.boundary_pos))

    def _get_done(self):
        all_done = np.zeros( (1, self.n_a) ).astype(bool)
        return all_done

    def _get_info(self):
        return np.array( [None, None, None] ).reshape(3,1)

    def step(self, a): 
        self.simulation_time += self.dt 
        for _ in range(self.n_frames): 
            if self.dynamics_mode == 'Polar':  
                a[0, :self.n_a] *= self.angle_max
                a[1, :self.n_a] = (self.Acc_max - self.Acc_min)/2 * a[1,:self.n_a] + (self.Acc_max + self.Acc_min)/2 

            ########################################################################################################
            self.d_b2b_center, self.d_b2b_edge, self.is_collide_b2b = self._get_dist_b2b() 
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
            
            if self.is_boundary:
                self._get_dist_b2w()
                sf_b2w = np.array([[1, 0, -1, 0], [0, -1, 0, 1]]).dot(self.is_collide_b2w * self.d_b2w) * self.k_wall 
                df_b2w = np.array([[-1, 0, -1, 0], [0, -1, 0, -1]]).dot(self.is_collide_b2w*np.concatenate((self.dp, self.dp), axis=0))  *  self.c_wall

            if self.agent_strategy == 'input':
                pass               
            elif self.agent_strategy == 'random':
                a = np.random.uniform(-1, 1, (self.act_dim_agent, self.n_a)) 
                if self.dynamics_mode == 'Polar': 
                    a[0, :self.n_a] *= self.angle_max
                    a[1, :self.n_a] = (self.Acc_max - self.Acc_min)/2 * a[1,:self.n_a] + (self.Acc_max + self.Acc_min)/2
            elif self.agent_strategy == 'rule':
                a = np.zeros((2, self.n_ao))
                dist = pdist(self.p.T)
                dist_mat = squareform(dist)
                sorted_indices = np.argsort(dist_mat, axis=0)
                self.B = np.take_along_axis(dist_mat, sorted_indices, axis=0)
                self.I = sorted_indices
                for agent in range(self.n_a):
                    list_nei_indices = self.B[:, agent] <= self.d_sen
                    list_nei = self.I[list_nei_indices, agent]
                    list_nei = list_nei[1:]
                    if len(list_nei) > self.topo_nei_max:
                        list_nei = list_nei[:self.topo_nei_max]

                    if len(list_nei) > 0:
                        pos_rel = self.p - self.p[:, [agent]]
                        for agent2 in list_nei:
                            a[:, agent] += 0.4 * pos_rel[:,agent2] + self.dp[:,agent2] / (np.linalg.norm(self.dp[:,agent2]) + 0.0001)
            else:
                print('Wrong in Step function')

            if self.dynamics_mode == 'Cartesian':
                u = a   
            elif self.dynamics_mode == 'Polar':      
                self.theta += a[[0],:]
                self.theta = self._normalize_angle(self.theta)
                self.heading = np.concatenate((np.cos(self.theta), np.sin(self.theta)), axis=0) 
                u = a[[1], :] * self.heading 
            else:
                print('Wrong in updating dynamics')
            
            #########################################################################################
            if self.is_boundary:
                F = self.sensitivity * u + sf_b2b + sf_b2w + df_b2w
            else: 
                F = self.sensitivity * u + sf_b2b

            # acceleration
            self.ddp = F/self.m

            # velocity
            self.dp += self.ddp * self.dt
            if self.is_leader: # local leader
                self.dp[:,:self.n_l] = self.leader_state[self.dim:,:self.n_l]
                self.heading[:,:self.n_l] = self.dp[:,:self.n_l] / np.linalg.norm(self.dp[:,:self.n_l], axis=0)
            if self.is_boundary and (self.is_leader == False):
                self._regularize_min_velocity()
            self.dp = np.clip(self.dp, -self.Vel_max, self.Vel_max)

            if self.obstacles_cannot_move:
                self.dp[:, self.n_a:self.n_ao] = 0
        
            # position
            self.p += self.dp * self.dt
            if self.is_leader: # local leader
                if self.is_periodic:
                    self.leader_state[:self.dim,:self.n_l] = self.p[:,:self.n_l]
                else:
                    self.p[:,:self.n_l] = self.leader_state[:self.dim,:self.n_l]
            if self.obstacles_is_constant:
                self.p[:, self.n_a:self.n_ao] = self.p_o
            if self.is_periodic:
                self.p = self._make_periodic(self.p, is_rel=False)

            # The leader state needs to be updated separately in boundary case
            if self.is_leader and self.is_boundary:
                self._get_way_count()
                self._update_leader_state()
                self.leader_state[:self.dim] = self.center_leader_state[:self.dim,[0]] + self.initial_p[:,:self.n_l]
                self.leader_state[self.dim:] = np.tile(self.center_leader_state[self.dim:,[0]], (1, self.n_l))

            if self.render_traj == True:
                self.p_traj = np.concatenate( (self.p_traj[1:,:,:], self.p.reshape(1, 2, self.n_ao)), axis=0 )

            # output
            obs = self._get_obs()
            rew = self._get_reward(a)
            done = self._get_done()
            info = self._get_info()
            # cent_obs = self._get_cent_obs()

        return obs, rew, done, info
        # return obs, cent_obs, rew, done, info

    def render(self, mode="human"): 

        size_agents = 1200

        if self.plot_initialized == 0:

            plt.ion()

            left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
            ax = self.figure_handle.add_axes([left, bottom, width, height],projection = None)

            # Plot agents position
            ax.scatter(self.p[0,:], self.p[1,:], s = size_agents, c = self.color, marker = ".", alpha = 1)
                
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
            #         # if dist_ij < 1:
            #         #     print(dist_ij)
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
        indent = 0.5
        x_min = self.boundary_pos[0] - indent
        x_max = self.boundary_pos[2] + indent
        y_min = self.boundary_pos[3] - indent
        y_max = self.boundary_pos[1] + indent
        return [x_min, x_max, y_min, y_max]
    
    def axis_lim_view_dynamic(self):
        indent = 0.5
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
        size = np.concatenate((np.array([self.size_a for _ in range(self.n_a)]),
                               np.array([self.size_o for _ in range(self.n_o)])))  
        sizes = np.tile(size.reshape(self.n_ao,1), (1,self.n_ao))
        sizes = sizes + sizes.T
        sizes[np.arange(self.n_ao), np.arange(self.n_ao)] = 0
        return size, sizes
    
    def _get_mass(self):
        m = np.concatenate((np.array([self.m_a for _ in range(self.n_a)]), 
                            np.array([self.m_o for _ in range(self.n_o)]))) 
        return m

    def _get_observation_space(self):
        if self.is_con_self_state:
            self_flag = 1
        else:
            self_flag = 0

        if self.is_leader and self.is_remarkable_leader:
            self.obs_dim_agent = 2*self.dim*(self.topo_nei_max + 1 + self_flag)
        else:
            self.obs_dim_agent = 2*self.dim*(self.topo_nei_max + 0 + self_flag)   

        if self.dynamics_mode == 'Polar':
            self.obs_dim_agent += self.dim

        observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(self.obs_dim_agent, self.n_a), dtype=np.float32)
        return observation_space

    def _get_action_space(self):
        action_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(self.act_dim_agent, self.n_a), dtype=np.float32)
        return action_space

    def _get_focused(self, Pos, Vel, norm_threshold, width, remove_self):
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
        target_Pos = np.zeros( (2, width) )
        target_Vel = np.zeros( (2, width) )
        until_idx = np.min( [Pos.shape[1], width] )
        target_Pos[:, :until_idx] = Pos[:, :until_idx] 
        target_Vel[:, :until_idx] = Vel[:, :until_idx]
        target_Nei = sorted_seq[:until_idx]
        return target_Pos, target_Vel, target_Nei

    def _regularize_min_velocity(self):
        norms = np.linalg.norm(self.dp, axis=0)
        mask = norms < self.Vel_min
        self.dp[:, mask] *= self.Vel_min / (norms[mask] + 0.00001)

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
        dist = np.linalg.norm(self.center_leader_state[:self.dim] - self.leader_waypoint[:,[self.way_count]])
        # Switching waypoint by distance
        if dist < 0.1:
            if self.way_count < self.leader_waypoint.shape[1] - 1:
                self.way_count += 1

    def _update_leader_state(self):
        d_shill_target = np.linalg.norm(self.leader_waypoint[:,self.way_count].reshape(self.dim,1) - self.center_leader_state[0:self.dim])
        if d_shill_target > 0.1:
            self.center_leader_state[self.dim:2*self.dim] = self.leader_v*(self.leader_waypoint[:,self.way_count].reshape(self.dim,1) - self.center_leader_state[0:self.dim])/d_shill_target
        else:
            self.center_leader_state[self.dim:2*self.dim] = 0.001*self.center_leader_state[self.dim:2*self.dim]

        self.center_leader_state[:self.dim] = self.center_leader_state[:self.dim] + self.center_leader_state[self.dim:2*self.dim]*self.dt
            


