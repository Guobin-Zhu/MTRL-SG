"""
Configuration file for Flocking Swarm Environment
Defines environment and training parameters for multi-agent flocking simulation
"""

import numpy as np
import numpy.linalg as lg
import sympy as sp
import argparse

def _compute_waypoint():
    """
    Compute waypoints for agent navigation with smooth cornering
    Creates a path with curved corners using circle interpolation
    
    Returns:
        waypoint1: 3xN array of waypoints in 3D space
    """
    # Define initial waypoint intervals (triangle path)
    waypoint_interval1 = 1.4 * np.array([[3, 0, 0],
                                        [0, 3, 0], 
                                        [-3, 0, 0]])

    waypoint1 = np.empty((3,1))
    way_interval = 0.2          # Distance between waypoints
    delta_theta = 0.24          # Angular resolution for corners
    r_corner = 0.8              # Corner radius for smooth turns

    # Generate waypoints with smooth corners
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

    # Alternative waypoint patterns (commented out)
    # waypoint1 = 1.2*np.array([[2, -2, 0], 
    #                         [-2, 2, 0],
    #                         [2, -2, 0], 
    #                         [-2, -2, 0],
    #                         [2, 2, 0],
    #                         [-2, -2, 0],
    #                         [-2, 2, 0],
    #                         [2, -2, 0],
    #                         [-2, 2, 0],
    #                         [2, 2, 0],
    #                         [-2, -2, 0],
    #                         [2, 2, 0],
    #                         [2, -2, 0]]).T

    # waypoint1 = np.array([[0, 0, 0], 
    #                       [1, 0, 0],
    #                       [2, 0, 0],
    #                       [3, 0, 0]]).T

    return waypoint1

# Parse all arguments
def get_flocking_args():
    """
    Parse command line arguments for flocking environment configuration.
    
    Returns:
        argparse.Namespace: Parsed arguments as a namespace object.
    """

    # Initialize argument parser for command line configuration
    parser = argparse.ArgumentParser("Gym-FlockingSwarm Arguments")

    ## ==================== User settings ====================

    # Agent population and leadership
    parser.add_argument("--n_a", type=int, default=50, help='Number of agents')
    parser.add_argument("--n_l", type=int, default=1, help='Number of leader') 

    # Environment physics and boundaries
    parser.add_argument("--is_boundary", type=bool, default=False, help='Set whether has wall or periodic boundaries')
    parser.add_argument("--is_leader", type=list, default=[False, False], help='Set whether has virtual leader and remarkable/non-remarkable') 
    parser.add_argument("--is_con_self_state", type=bool, default=True, help="Whether contain myself state in the observation") 

    # Movement dynamics
    parser.add_argument("--dynamics_mode", type=str, default='Cartesian', help="Select one from ['Cartesian', 'Polar']")

    # Visualization settings
    parser.add_argument("--render-traj", type=bool, default=True, help="Whether render trajectories of agents") 
    parser.add_argument("--traj_len", type=int, default=12, help="Length of the trajectory")  

    # Agent behavior and strategy
    parser.add_argument("--agent_strategy", type=str, default='input', help="The agent's strategy, please select one from ['input','random','rule']")
    parser.add_argument("--augmented", type=bool, default=False, help="Whether has data augmentation")
    parser.add_argument("--leader_waypoint", type=_compute_waypoint, default=_compute_waypoint(), help="The agent's strategy")

    # Recording options
    parser.add_argument("--video", type=bool, default=False, help="Record video")

    ## ==================== End of User settings ====================

    ## ==================== Training Parameters ====================

    # Basic training configuration
    parser.add_argument("--env_name", default="flocking", type=str)
    parser.add_argument("--seed", default=226, type=int, help="Random seed")

    # Threading and parallelization
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=5, type=int)

    # Training schedule and memory
    parser.add_argument("--buffer_length", default=int(5e3), type=int)  # Experience replay buffer size
    parser.add_argument("--n_episodes", default=2000, type=int)
    parser.add_argument("--episode_length", default=200, type=int)
    parser.add_argument("--batch_size", default=512, type=int, help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=25000, type=int)  # Exploration episodes

    # Neural network architecture
    parser.add_argument("--hidden_dim", default=64, type=int)

    # Learning rates and optimization
    parser.add_argument("--lr_actor", default=1e-4, type=float)
    parser.add_argument("--lr_critic", default=1e-3, type=float)

    # Regularization and smoothness
    parser.add_argument("--lambda_s", default=30, type=float, help="The coefficient of smoothness-inducing regularization")
    parser.add_argument("--epsilon_p", default=0.03, type=float, help="The amplitude of state perturbation")

    # Training hyperparameters
    parser.add_argument("--epsilon", default=0.1, type=float)      # Exploration parameter
    parser.add_argument("--noise_scale", default=0.8, type=float) # Action noise scale
    parser.add_argument("--tau", default=0.01, type=float)        # Target network update rate

    # Algorithm selection
    parser.add_argument("--agent_alg", default="MADDPG", type=str, choices=['MADDPG', 'DDPG'])

    # Hardware and logging
    parser.add_argument("--device", default="cpu", type=str, choices=['cpu', 'gpu'])
    parser.add_argument("--save_interval", default=50, type=int, help="Save data for every 'save_interval' episodes")

    ## ==================== Training Parameters ====================

    return parser.parse_args()