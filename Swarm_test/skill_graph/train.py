import numpy as np
import torch
from datetime import datetime
import os
import random
from config import get_config, get_model_config, get_train_config
from skill_graph.algorithm import SkillGraph

def seed(seed_value: int) -> None:
    """Set seed for all RNGs to ensure reproducibility.
    
    Args:
        seed_value: Seed value for RNG initialization
    """
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)

if __name__ == "__main__":
    # Parse command line arguments
    args = get_config()
    
    # Create timestamped directory for experiment logs
    time_dir = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    # Set up paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(current_dir, 'runs', time_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize environment with seed for reproducibility
    seed(args.seed)
    
    # Configure and initialize skill graph model
    model_config = get_model_config(args)
    model_config["log_dir"] = log_dir  # Add log directory
    sg = SkillGraph(model_config)
    
    # Get training configuration
    train_config = get_train_config(args)
    
    # Train skill graph model
    print(f'Training starting with config: {train_config}')
    sg.train_skill_graph(train_config)
    print('Training completed successfully')