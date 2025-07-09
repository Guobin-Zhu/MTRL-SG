"""
Configuration file for Comprehension Swarm Environment
Defines environment and training parameters for multi-agent comprehension simulation
"""

import numpy as np
import argparse
import numpy.linalg as lg
import sympy as sp

def _compute_waypoint():
    """
    Compute waypoints for target agent navigation with smooth cornering
    """
    # Define waypoint intervals forming an irregular closed path
    # Current configuration: octagonal path with tight corners
    waypoint_interval1 = np.array([[2, -1.5, 0], 
                                [-1, -1.5, 0],
                                [-1.5, -0.8, 0], 
                                [-1, 0, 0],
                                [1, 0, 0],
                                [1.5, 0.8, 0],
                                [1, 1.5, 0],
                                [-2, 1.5, 0]])

    waypoint1 = np.empty((3,1))
    way_interval = 0.2          # Distance between waypoints
    delta_theta = 0.24          # Angular resolution for corners
    r_corner = 0.8              # Corner radius for smooth turns

    # Generate waypoints with smooth corners for each path segment
    for l in np.arange(1, waypoint_interval1.shape[0] - 1):
        # Get three consecutive points for corner calculation
        p1 = waypoint_interval1[l - 1,...].reshape(3,1)
        p2 = waypoint_interval1[l,...].reshape(3,1)
        p3 = waypoint_interval1[l + 1,...].reshape(3,1)
        
        # Calculate vectors between points
        p1p2 = p2 - p1
        p2p3 = p3 - p2
        
        # Calculate angle between vectors
        theta_p1p2p3 = np.pi - np.arccos(np.dot(p1p2.T,p2p3)/(lg.norm(p1p2)*lg.norm(p2p3)))
        
        # Calculate distance from corner to circle tangent points
        d = r_corner/np.tan(theta_p1p2p3/2)
        w1p2 = d*p1p2/lg.norm(p1p2)
        p2w2 = d*p2p3/lg.norm(p2p3)
        w1 = p2 - w1p2  # First tangent point
        w2 = p2 + p2w2  # Second tangent point

        # Compute the center of circle for smooth corner
        normal_vec = np.cross(p1p2.T,p2p3.T).T
        if lg.norm(normal_vec) == 0:
            normal_vec = np.array([[0,0,1]]).T
        else:
            normal_vec = normal_vec/lg.norm(normal_vec)
        
        # Solve for circle center using symbolic computation
        x_o, y_o, z_o = sp.symbols('x_o, y_o,z_o',real = True)
        eqn1 = p1p2[0][0]*(w1[0][0] - x_o) + p1p2[1][0]*(w1[1][0] - y_o) + p1p2[2][0]*(w1[2][0] - z_o)
        eqn2 = normal_vec[0][0]*(w1[0][0] - x_o) + normal_vec[1][0]*(w1[1][0] - y_o) + normal_vec[2][0]*(w1[2][0] - z_o)
        eqn3 = sp.sqrt((w1[0][0] - x_o)**2 + (w1[1][0] - y_o)**2 + (w1[2][0] - z_o)**2) - r_corner
        eqn4 = p2p3[0][0]*(w2[0][0] - x_o) + p2p3[1][0]*(w2[1][0] - y_o) + p2p3[2][0]*(w2[2][0] - z_o)
        eqn5 = normal_vec[0][0]*(w2[0][0] - x_o) + normal_vec[1][0]*(w2[1][0] - y_o) + normal_vec[2][0]*(w2[2][0] - z_o)
        eqn6 = sp.sqrt((w2[0][0] - x_o)**2 + (w2[1][0] - y_o)**2 + (w2[2][0] - z_o)**2) - r_corner
        
        # Solve the circle center and convert to float
        o1 = sp.solve([eqn1,eqn2,eqn3],[x_o,y_o,z_o])
        o1_c = np.array([[o1[0][0],o1[0][1],o1[0][2]],[o1[1][0],o1[1][1],o1[1][2]]],dtype = np.float64).T
        o2 = sp.solve([eqn4,eqn5,eqn6],[x_o,y_o,z_o])
        o2_c = np.array([[o2[0][0],o2[0][1],o2[0][2]],[o2[1][0],o2[1][1],o2[1][2]]],dtype = np.float64).T

        # Find matching circle centers
        o1_c_o2_c = np.sqrt(np.sum((o1_c - o2_c)**2, axis = 0))
        if o1_c_o2_c[o1_c_o2_c < 1e-6].size == 0:
            o2_c = np.append(o2_c[...,1].reshape(3,1),o2_c[...,0].reshape(3,1), axis = 1)
            o1_c_o2_c = np.sqrt(np.sum((o1_c - o2_c)**2, axis = 0))
        
        circle_center = o2_c[...,np.argwhere(o1_c_o2_c < 1e-6)[0]]

        # Generate waypoints along the circular arc
        center_w1 = w1 - circle_center
        circle_waypoint = np.empty((3,1))
        for alpha_m in np.arange(0,np.pi - theta_p1p2p3 + delta_theta, delta_theta):
            center_m = center_w1*np.cos(alpha_m) + lg.norm(center_w1)*np.sin(alpha_m)*w1p2/d
            m = circle_center + center_m
            circle_waypoint = np.append(circle_waypoint,m,axis = 1)
        circle_waypoint = np.delete(circle_waypoint,0,axis = 1)
        
        # Generate waypoints on straight segments before corner
        wmew1_waypoint = np.empty((3,1))
        wmew1_norm = lg.norm(p1p2)/2 - d
        number_interval_wmew1 = np.floor(wmew1_norm/way_interval)
        for waypoint_m in np.arange(1,number_interval_wmew1 + 1):
            wmew1_waypoint_m = (p1 + p2)/2 + waypoint_m*way_interval*w1p2/d
            wmew1_waypoint = np.append(wmew1_waypoint,wmew1_waypoint_m,axis = 1)
        wmew1_waypoint = np.delete(wmew1_waypoint,0,axis = 1)
        
        # Generate waypoints on straight segments after corner
        w2wme_waypoint = np.empty((3,1))
        w2wme_norm = lg.norm(p2p3)/2 - d
        number_interval_w2wme = np.floor(w2wme_norm/way_interval)
        for waypoint_m in np.arange(1,number_interval_w2wme + 1):
            w2wme_waypoint_m = w2 + waypoint_m*way_interval*p2w2/d
            w2wme_waypoint = np.append(w2wme_waypoint,w2wme_waypoint_m,axis = 1)
        w2wme_waypoint = np.delete(w2wme_waypoint,0,axis = 1)
        
        # Combine all waypoints for this segment
        waypoint_l = np.concatenate((wmew1_waypoint,circle_waypoint,w2wme_waypoint), axis = 1)
        waypoint1 = np.append(waypoint1,waypoint_l,axis = 1)

        # Handle first segment specially
        if l == 1:
            p1wme_waypoint = np.empty((3,1))
            p1wme_norm = lg.norm(p1p2)/2
            number_interval_p1wme = np.floor(p1wme_norm/way_interval)
            for waypoint_m in np.arange(1,number_interval_p1wme + 1):
                p1wme_waypoint_m = p1 + waypoint_m*way_interval*(p2 - p1)/lg.norm(p1p2)
                p1wme_waypoint = np.append(p1wme_waypoint,p1wme_waypoint_m, axis = 1)
            p1wme_waypoint = np.delete(p1wme_waypoint,0,axis = 1)
    
    # Add initial segment waypoints
    waypoint1 = np.append(p1wme_waypoint,waypoint1,axis = 1)
    waypoint1 = np.delete(waypoint1,p1wme_waypoint.shape[1],axis = 1)

    # Create symmetric waypoints (mirror image) for figure-eight pattern
    waypoint2 = np.array([-waypoint1[0], -waypoint1[1], waypoint1[2]])
    waypoint2 = waypoint2[:,::-1]  # Reverse order for continuous path
    waypoint = np.concatenate((waypoint1, waypoint2), axis=1)

    return waypoint

# Parse all arguments
def get_comprehension_cfg():
    """
    Parse command line arguments for Comprehension Swarm Environment configuration
    """

    # Initialize argument parser for command line configuration
    parser = argparse.ArgumentParser("Gym-ComprehensionSwarm Arguments")

    ## ==================== Environment Parameters ====================

    # Agent population settings
    parser.add_argument("--n-l", type=int, default=50, help='Number of left agents') 
    parser.add_argument("--n-r", type=int, default=50, help='Number of right agents') 
    parser.add_argument("--n_leader", type=int, default=1, help='Number of leader') 

    # Environment physics and boundaries
    parser.add_argument("--is_boundary", type=bool, default=True, help='Set whether has wall or periodic boundaries')
    parser.add_argument("--is_leader", type=list, default=[True, True], help='Set whether has virtual leader and remarkable/non-remarkable')
    parser.add_argument("--is_con_self_state", type=bool, default=True, help="Whether contain myself state in the observation")

    # Movement dynamics
    parser.add_argument("--dynamics-mode", type=str, default='Cartesian', help="Select one from ['Cartesian', 'Polar']") 

    # Agent behavior strategies
    parser.add_argument("--l-strategy", type=str, default='input', help="Select one from ['input', 'static', 'random', 'nearest']") 
    parser.add_argument("--r-strategy", type=str, default='input', help="Select one from ['input', 'static', 'random', 'nearest']") 

    # Simulation modes and algorithms
    parser.add_argument("--mode", type=int, default=0, help="Select one from [0, 1, 2]")
    parser.add_argument("--algorithm", type=str, default='sg', help="Select one from [sg, mappo]")

    # Visualization settings
    parser.add_argument("--render-traj", type=bool, default=True, help=" Whether render trajectories of agents") 
    parser.add_argument("--traj_len", type=int, default=10, help="Length of the trajectory") 

    # Special modes and features
    parser.add_argument("--billiards-mode", type=bool, default=False, help="Billiards mode") 
    parser.add_argument("--augmented", type=bool, default=False, help="Whether has data augmentation")
    parser.add_argument("--leader_waypoint", type=_compute_waypoint, default=_compute_waypoint(), help="The agent's trajectory")
    parser.add_argument("--video", type=bool, default=False, help="Record video")

    ## ==================== Environment Parameters ====================

    ## ==================== Training Parameters ====================

    # Basic training configuration
    parser.add_argument("--env_name", default="env_high", type=str)
    parser.add_argument("--seed", default=226, type=int, help="Random seed")

    # Threading and parallelization
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=5, type=int)

    # Training schedule and memory
    parser.add_argument("--data_buffer_length", default=1e4, type=int)
    parser.add_argument("--n_episodes", default=200, type=int)
    parser.add_argument("--episode_length", default=300, type=int)
    parser.add_argument("--batch_size", default=1028, type=int)
    parser.add_argument("--sample_index_start", default=5e3, type=int)   

    # Neural network architecture
    parser.add_argument("--hidden_dim", default=64, type=int)

    # Learning rates and optimization
    parser.add_argument("--lr_actor", default=1e-3, type=float)
    parser.add_argument("--lr_critic", default=1e-3, type=float)

    # Algorithm configuration
    parser.add_argument("--action_space_class", default='Discrete', type=str)
    parser.add_argument("--agent_alg", default="mappo", type=str, choices=["mappo"])
    parser.add_argument("--adversary_alg", default="mappo", type=str, choices=["mappo"])

    # Hardware and logging
    parser.add_argument("--device", default="cpu", type=str, choices=['cpu', 'gpu'])
    parser.add_argument("--save_interval", default=10, type=int, help="Save data for every 'save_interval' episodes")

    # Prepare parameters for MAPPO
    parser.add_argument("--cuda", action='store_false', default=True, help="By default True, will use GPU to train; or else will use CPU;")
    parser.add_argument("--cuda_deterministic", action='store_false', default=True, help="By default, make sure random seed effective. if set, bypass such function.")
    parser.add_argument("--use_centralized_V", action='store_false', default=True, help="Whether to use centralized V function")

    # Neural network architecture parameters
    parser.add_argument("--layer_N", type=int, default=3, help="Number of layers for actor/critic networks")
    parser.add_argument("--activate_func_index", type=int, default=2, choices=['Tanh', 'ReLU', 'Leaky_ReLU'])
    parser.add_argument("--use_valuenorm", action='store_false', default=True, help="by default True, use running mean and std to normalize rewards.")
    parser.add_argument("--use_feature_normalization", action='store_false', default=False, help="Whether to apply layernorm to the inputs")
    parser.add_argument("--use_orthogonal", action='store_false', default=True, help="Whether to use Orthogonal initialization for weights")
    parser.add_argument("--gain", type=float, default=5/3, help="The gain # of last action layer")

    # Recurrent policy parameters
    parser.add_argument("--use_recurrent_policy", action='store_false',default=False, help='Use a recurrent policy')
    parser.add_argument("--recurrent_N", type=int, default=1, help="The number of recurrent layers.")
    parser.add_argument("--data_chunk_length", type=int, default=10, help="Time length of chunks used to train a recurrent_policy")

    # Optimizer parameters
    parser.add_argument("--opti_eps", type=float, default=1e-5,help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--weight_decay", type=float, default=0)

    # PPO-specific parameters
    parser.add_argument("--ppo_epoch", type=int, default=20, help='Number of ppo epochs (default: 15)')
    parser.add_argument("--use_clipped_value_loss", action='store_false', default=False, help="By default, clip loss value. If set, do not clip loss value.")
    parser.add_argument("--clip_param", type=float, default=0.2, help='ppo clip parameter (default: 0.2)')
    parser.add_argument("--num_mini_batch", type=int, default=1, help='Number of batches for ppo (default: 1)')
    parser.add_argument("--entropy_coef", type=float, default=0.01, help='Entropy term coefficient (default: 0.01)')
    parser.add_argument("--value_loss_coef", type=float, default=1, help='Value loss coefficient (default: 0.5)')
    parser.add_argument("--use_max_grad_norm", action='store_false', default=True, help="By default, use max norm of gradients. If set, do not use.")
    parser.add_argument("--max_grad_norm", type=float, default=10.0, help='Max norm of gradients (default: 0.5)')

    # Advantage computation parameters
    parser.add_argument("--advantage_method", type=str, default="GAE", choices=['GAE', 'TD', 'n_step_TD'])
    parser.add_argument("--gamma", type=float, default=0.99, help='Discount factor for rewards (default: 0.99)')
    parser.add_argument("--gae_lambda", type=float, default=0.95, help='Gae lambda parameter (default: 0.95)')
    parser.add_argument("--use_huber_loss", action='store_false', default=True, help="By default, use huber loss. If set, do not use huber loss.")
    parser.add_argument("--huber_delta", type=float, default=10.0, help=" Coefficience of huber loss.")

    # Learning rate schedule
    parser.add_argument("--use_linear_lr_decay", action='store_true', default=False, help='Use a linear schedule on the learning rate')

    ## ==================== Training Parameters ====================

    return parser.parse_args()