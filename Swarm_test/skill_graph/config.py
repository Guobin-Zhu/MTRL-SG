"""
Simple configuration file for skill graph training and testing.
"""

import argparse

def get_config():
    """Get configuration parameters using argparse."""
    parser = argparse.ArgumentParser(description='Skill Graph Training and Testing Configuration')
    
    # ================================ Model Configuration ================================
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden layer dimensionality')
    parser.add_argument('--skill_dim', type=int, default=96, help='Skill representation size')
    parser.add_argument('--read_skill', action='store_true', default=True)
    parser.add_argument('--eval', action='store_true', default=False)
    
    # ================================ Training Configuration ================================
    parser.add_argument('--batch_size', type=int, default=1028, help='Samples per training iteration')
    parser.add_argument('--train_iters', type=int, default=500, help='Number of training iterations')
    
    # ================================ Reproducibility Configuration ================================
    parser.add_argument('--seed', type=int, default=226, help='Random seed for reproducibility')
    
    return parser.parse_args()

def get_model_config(args):
    """Get model configuration dictionary."""
    return {
        "hidden_dim": args.hidden_dim,
        "skill_dim": args.skill_dim,
        "read_skill": args.read_skill,
        "eval": args.eval
    }

def get_train_config(args):
    """Get training configuration dictionary."""
    return {
        "batch_size": args.batch_size,
        "train_iters": args.train_iters
    }