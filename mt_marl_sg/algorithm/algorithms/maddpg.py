import torch
from algorithm.utils import DDPGAgent, soft_update, average_gradients

MSELoss = torch.nn.MSELoss()

class MADDPG(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """
    def __init__(self, agent_init_params, alg_types, epsilon, noise, gamma=0.95, tau=0.01, lr_actor=1e-4, lr_critic=1e-3, lambda_s=500, epsilon_p=0.06,  
                 hidden_dim=64, device='cpu', discrete_action=False):
        """
        Initialize MADDPG with multiple DDPG agents
        
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                dim_input_policy (int): Input dimensions to policy
                dim_output_policy (int): Output dimensions to policy
                dim_input_critic (int): Input dimensions to critic
            alg_types (list of str): Learning algorithm for each agent (DDPG
                                       or MADDPG)
            epsilon (float): Exploration probability for random actions
            noise (float): Exploration noise scale
            gamma (float): Discount factor
            tau (float): Target update rate
            lr_actor (float): Learning rate for actor networks
            lr_critic (float): Learning rate for critic networks
            lambda_s (float): Weight for spatial regularization loss
            epsilon_p (float): Perturbation magnitude for spatial loss
            hidden_dim (int): Number of hidden dimensions for networks
            device (str): Device to run on ('cpu' or 'gpu')
            discrete_action (bool): Whether or not to use discrete action space
        """
        self.nagents = len(alg_types)
        self.alg_types = alg_types
        self.epsilon = epsilon
        self.noise = noise
        
        # Create homogeneous agents with different initialization parameters
        self.agents = [DDPGAgent(lr_actor=lr_actor, 
                                 lr_critic=lr_critic, 
                                 discrete_action=discrete_action, 
                                 hidden_dim=hidden_dim, 
                                 epsilon=self.epsilon, 
                                 noise=self.noise,
                                 **params) for params in agent_init_params]   
        
        # Store hyperparameters
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.lambda_s = lambda_s
        self.epsilon_p = epsilon_p
        self.discrete_action = discrete_action
        
        # Device tracking for different network components
        self.pol_dev = 'cpu'  
        self.critic_dev = 'cpu' 
        self.trgt_pol_dev = 'cpu' 
        self.trgt_critic_dev = 'cpu' 
        
        # Regularization loss flags
        self.spatial_loss = False
        self.temporal_loss = False 
        
        # Training iteration counter
        self.niter = 0

    @property           
    def policies(self):
        """Get all agent policies"""
        return [a.policy for a in self.agents]

    def target_policies(self, agent_i, obs):
        """Get target policy output for specific agent"""
        return self.agents[agent_i].target_policy(obs)

    def scale_noise(self, scale, new_epsilon):
        """
        Scale noise for each agent
        
        Inputs:
            scale (float): scale of noise
            new_epsilon (float): new epsilon value for exploration
        """
        for a in self.agents:
            a.scale_noise(scale)       
            a.epsilon = new_epsilon

    def reset_noise(self):
        """Reset exploration noise for all agents"""
        for a in self.agents:
            a.reset_noise()

    def step(self, observations, start_stop_num, explore=False):
        """
        Take a step forward in environment with all agents
        
        Inputs:
            observations: Observations for all agents
            start_stop_num: Index ranges for each agent's observations
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """                                                           
        return [self.agents[i].step(observations[:, start_stop_num[i]].t(), explore=explore) for i in range(len(start_stop_num))]

    def update(self, obs, acs, rews, next_obs, dones, agent_i, parallel=False, logger=None):
        """
        Update parameters of agent model based on sample from replay buffer
        
        Inputs:
            obs: Current observations
            acs: Actions taken
            rews: Rewards received
            next_obs: Next observations
            dones: Episode termination flags
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter): Logger for tracking training metrics
        """
        curr_agent = self.agents[agent_i]    

        ######################### update critic #########################       
        curr_agent.critic_optimizer.zero_grad()     
        
        # Compute target Q-value using target networks
        all_trgt_acs = self.target_policies(agent_i, next_obs)  
        trgt_vf_in = torch.cat((next_obs, all_trgt_acs), dim=1)  
        target_value = (rews + self.gamma * curr_agent.target_critic(trgt_vf_in) *  (1 - dones))                                               
        
        # Compute current Q-value
        vf_in = torch.cat((obs, acs), dim=1)
        actual_value = curr_agent.critic(vf_in)
        
        # Critic loss (TD error)
        vf_loss = MSELoss(actual_value, target_value.detach()) 

        vf_loss.backward()
        if parallel:
            average_gradients(curr_agent.critic)

        # Gradient clipping 
        # torch.nn.utils.clip_grad_norm(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()

        ######################### update actor #########################
        curr_agent.policy_optimizer.zero_grad()  

        # Get current policy actions
        if not self.discrete_action:
            curr_pol_out = curr_agent.policy(obs)
            curr_pol_vf_in = curr_pol_out

        all_pol_acs = curr_pol_vf_in  
        vf_in = torch.cat((obs, all_pol_acs), dim=1)
        
        # Policy loss (maximize Q-value)
        pol_loss = -curr_agent.critic(vf_in).mean()
        
        # Spatial regularization loss (optional)
        spat_act_loss = 0
        if self.spatial_loss:
            obs_pert = (2 *torch.rand_like(obs) - 1) * self.epsilon_p + obs
            curr_pol_out_pert = curr_agent.policy(obs_pert)
            spat_act_loss = self.lambda_s * MSELoss(all_pol_acs, curr_pol_out_pert)
        
        # Temporal regularization loss (optional)
        temp_act_loss = 0
        if self.temporal_loss:
            next_pol_out = curr_agent.policy(next_obs)
            temp_act_loss = 5 * MSELoss(all_pol_acs, next_pol_out)

        # Total policy loss
        all_pol_loss = pol_loss + spat_act_loss + temp_act_loss
        all_pol_loss.backward()     
                                
        if parallel:
            average_gradients(curr_agent.policy)

        # Gradient clipping
        # torch.nn.utils.clip_grad_norm(curr_agent.policy.parameters(), 0.5)   
        curr_agent.policy_optimizer.step() 
        
        # Log training metrics
        if logger is not None:
            logger.add_scalars('agent%i/losses' % agent_i, {'vf_loss': vf_loss, 'pol_loss': pol_loss}, self.niter)

    def update_all_targets(self):    
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        for a in self.agents:
            soft_update(a.target_critic, a.critic, self.tau)   
            soft_update(a.target_policy, a.policy, self.tau)
        self.niter += 1

    def prep_training(self, device='gpu'):
        """
        Prepare networks for training mode and move to specified device
        
        Inputs:
            device (str): Device to move networks to ('gpu' or 'cpu')
        """
        # Set all networks to training mode
        for a in self.agents:
            a.policy.train()  
            a.target_policy.train()
            a.target_critic.train()

        # Device transformation function
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
            
        # Move networks to device if needed
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            for a in self.agents:
                a.critic = fn(a.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            for a in self.agents:
                a.target_critic = fn(a.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        """
        Prepare networks for rollout/evaluation mode
        
        Inputs:
            device (str): Device to move networks to ('gpu' or 'cpu')
        """
        # Set policies to evaluation mode
        for a in self.agents:
            a.policy.eval()   
            
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
            
        # Only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        
        Inputs:
            filename (str): Path to save file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)

    @classmethod      
    def init_from_env(cls, env, agent_alg="MADDPG", adversary_alg="MADDPG", gamma=0.95, tau=0.01, lr_actor=1e-4, lr_critic=1e-3, lambda_s=500, epsilon_p=0.06,  
                        hidden_dim=64, device='cpu', epsilon=0.1, noise=0.1):
        """
        Instantiate instance of this class from multi-agent environment
        
        Inputs:
            env: Multi-agent environment
            agent_alg (str): Algorithm type for regular agents
            adversary_alg (str): Algorithm type for adversarial agents
            Other parameters: Same as __init__ method
        """
        agent_init_params = []
        
        # Extract dimensions from environment
        dim_input_policy=env.observation_space.shape[0]
        dim_output_policy=env.action_space.shape[0]
        dim_input_critic=env.observation_space.shape[0] + env.action_space.shape[0]

        # Set algorithm types based on agent roles
        alg_types = [adversary_alg if atype == 'adversary' else agent_alg for atype in env.agent_types]   
                     
        # Create initialization parameters for each agent
        for algtype in alg_types:  
            agent_init_params.append({'dim_input_policy': dim_input_policy,
                                      'dim_output_policy': dim_output_policy,
                                      'dim_input_critic': dim_input_critic})

        # Create initialization dictionary
        init_dict = {'gamma': gamma, 'tau': tau, 'lr_actor': lr_actor, 'lr_critic': lr_critic, 'lambda_s': lambda_s, 'epsilon_p': epsilon_p, 'epsilon': epsilon, 
                     'noise': noise, 'hidden_dim': hidden_dim, 'device': device, 'alg_types': alg_types, 'agent_init_params': agent_init_params}
        instance = cls(**init_dict)    
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename):    
        """
        Instantiate instance of this class from file created by 'save' method
        
        Inputs:
            filename (str): Path to saved model file
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        
        # Load parameters for each agent
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)
        return instance

    @classmethod
    def init_from_save_with_id(cls, filename, list_id):    
        """
        Instantiate instance of this class from file with selective agent loading
        
        Inputs:
            filename (str): Path to saved model file
            list_id (list): List of agent IDs to load (2 means skip loading)
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        
        # Load parameters for specific agents only
        for i in range(len(instance.agents)):
            a = instance.agents[i]
            policy_id = list_id[i]
            if policy_id == 2:  # Skip loading if ID is 2
                continue
            params = save_dict['agent_params'][policy_id]
            a.load_params(params)
        return instance