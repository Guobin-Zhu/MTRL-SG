import matplotlib.pyplot as plt
import numpy as np
import torch
import json
import os
import random
import pickle
import time
from config import get_config, get_model_config
from skill_graph.algorithm import SkillGraph

def seed(seed_value: int) -> None:
    """Set seeds for all RNGs to ensure reproducibility"""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)

if __name__ == "__main__":
    # Parse command line arguments (for evaluation mode)
    args = get_config()
    args.eval = True  # Force evaluation mode for testing
    
    # ================================ Initialization ================================
    seed(args.seed)
    
    # Set up directories and paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(current_dir, 'runs', 'your_experiment_name')
    results_dir = os.path.join(log_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Load pre-trained model
    model_path = os.path.join(log_dir, 'logs', 'save', '495')
    model_config = get_model_config(args)
    sg = SkillGraph(model_config)
    sg.load(model_path)
    
    # ============================== Skill Inference ==============================
    # Define task properties
    task_flocking = np.array([1.0, 0, 0, 3, 0.4])
    
    # Initialize timing statistics
    inference_times = []
    
    for i, s in enumerate([task_flocking]):
        # Record start time for inference
        start_time = time.time()
        
        # Perform skill inference
        (skills, scores, (e_scores, t_scores)) = sg.kgc(
            env_property=np.array([1, 5]), 
            task_property=s
        )
        
        # Record end time and calculate duration
        end_time = time.time()
        inference_time = end_time - start_time
        inference_times.append(inference_time)
        
        # Print timing information
        print(f"Inference {i+1} completed in {inference_time:.4f} seconds")
    
    # Calculate and display timing statistics
    if inference_times:
        avg_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)
        
        print(f"\n========== Inference Timing Statistics ==========")
        print(f"Number of inferences: {len(inference_times)}")
        print(f"Average inference time: {avg_time:.4f} ± {std_time:.4f} seconds")
        print(f"Min inference time: {min_time:.4f} seconds")
        print(f"Max inference time: {max_time:.4f} seconds")
        print(f"Total inference time: {sum(inference_times):.4f} seconds")
        print(f"================================================\n")
    
    # Print top skills
    top_skills = [(scores[i].item(), s.desc) for i, s in enumerate(skills[:12])]
    print(f'Top six skills: {top_skills}')
    
    # Save results including timing data
    with open(os.path.join(results_dir, 'skill_data.pkl'), 'wb') as f:
        pickle.dump(skills, f)
        pickle.dump(scores, f)
        pickle.dump(inference_times, f)
    
    # Save timing statistics and skill scores to JSON
    timing_stats = {
        'inference_times': inference_times,
        'average_time': avg_time if inference_times else 0,
        'std_time': std_time if inference_times else 0,
        'min_time': min_time if inference_times else 0,
        'max_time': max_time if inference_times else 0,
        'total_time': sum(inference_times) if inference_times else 0,
        'num_inferences': len(inference_times),
        'skill_scores': {
            'top_6_skills': [(float(scores[i].item()), skills[i].desc) for i in range(min(6, len(skills)))],
            'all_scores': [float(score.item()) for score in scores],
            'skill_descriptions': [skill.desc for skill in skills]
        }
    }
    
    with open(os.path.join(results_dir, 'timing_and_scores.json'), 'w') as f:
        json.dump(timing_stats, f, indent=2)
    
    print(f"Timing statistics and skill scores saved to: {os.path.join(results_dir, 'timing_and_scores.json')}")

    # ============================== Skill Visualization ==============================
    # Prepare plot settings
    plt.style.use('seaborn-v0_8-whitegrid')
    fontsize = 20
    plt.rcParams.update({'font.size': fontsize})
    
    # Prepare data
    scores = scores.detach().cpu().numpy()[:24]
    skill_descs = [s.desc for s in skills[:24]]
    
    # Create skill score plot
    fig, ax = plt.subplots(figsize=(18, 10))
    bars = ax.bar(range(len(scores)), scores, color='royalblue', alpha=0.8)
    
    # Add reference lines and labels
    ax.axhline(y=0.85, color='tomato', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.axhline(y=0.95, color='green', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.text(len(scores)*0.05, 0.86, 'Minimum Threshold', color='tomato')
    ax.text(len(scores)*0.05, 0.96, 'Target Performance', color='green')
    
    # Configure plot appearance
    ax.set_xticks(range(len(scores)))
    ax.set_xticklabels(skill_descs, rotation=45, ha='right', fontsize=fontsize-2)
    ax.set_ylabel('Relevance Score', fontsize=fontsize)
    ax.set_title('Skill Relevance Ranking', fontsize=fontsize+4)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save and show
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'skill_scores.png'), dpi=300, bbox_inches='tight')
    
    # ============================== Training Analysis ==============================
    # Load training summary data
    summary_path = os.path.join(log_dir, 'logs', 'summary', 'summary.json')
    with open(summary_path, 'r') as f:
        data = json.load(f)
        
    # Extract loss metrics
    loss_key = next(k for k in data if 'total_loss' in k)
    loss_data = data[loss_key]
    timestamps = np.array([entry[1] for entry in loss_data])
    loss_values = np.array([entry[2] for entry in loss_data])
        
    # Find minimum loss point
    min_index = np.argmin(loss_values)
    min_timestamp = timestamps[min_index]
    min_value = loss_values[min_index]
        
    # Create loss curve plot
    plt.figure(figsize=(18, 10))
    plt.plot(timestamps, loss_values, 'b-', linewidth=2.5, label='Training Loss')
        
    # Highlight minimum point
    plt.scatter(min_timestamp, min_value, color='red', s=120,
                zorder=5, label=f'Minimum Loss ({min_value:.4f} at iter {min_timestamp})')

    # Annotate minimum loss point
    plt.annotate(f'Min Loss: {min_value:.4f}\nIteration: {min_timestamp}',
                xy=(min_timestamp, min_value),
                xytext=(min_timestamp - 20, min_value + 0.2),
                arrowprops=dict(facecolor='red', shrink=0.05, width=2),
                fontsize=fontsize,
                color='red',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
    # Configure plot appearance
    plt.xlabel('Training Iterations', fontsize=fontsize)
    plt.ylabel('Loss Value', fontsize=fontsize)
    plt.xticks(fontsize=fontsize-2)
    plt.yticks(fontsize=fontsize-2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title('Training Loss Progression', fontsize=fontsize+4)
    plt.legend(loc='upper right', fontsize=fontsize-2)
        
    # Save and show
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_loss.png'), dpi=300)
    plt.show()