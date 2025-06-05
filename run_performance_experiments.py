#!/usr/bin/env python
"""
This script runs an experiment to compare different model configurations
and loss functions to find the best performance for time series forecasting.
"""

import os
import sys
import subprocess
import datetime
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

# Create experiment directory
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
experiment_name = f"performance_optimization_{timestamp}"
experiment_dir = os.path.join("experiments", experiment_name)
os.makedirs(experiment_dir, exist_ok=True)

# Define experiment configurations
configurations = [
    {
        "name": "baseline",
        "args": ["--train_epochs", "30", "--batch_size", "64", "--learning_rate", "0.0005", 
                "--loss_function", "mse"]
    },
    {
        "name": "directional_loss",
        "args": ["--train_epochs", "30", "--batch_size", "64", "--learning_rate", "0.0005",
                "--loss_function", "directional", "--directional_weight", "0.5"]
    },
    {
        "name": "financial_loss",
        "args": ["--train_epochs", "30", "--batch_size", "64", "--learning_rate", "0.0005",
                "--loss_function", "financial", "--directional_weight", "0.7"]
    },
    {
        "name": "deeper_model",
        "args": ["--train_epochs", "30", "--batch_size", "64", "--learning_rate", "0.0005",
                "--e_layers", "3", "--d_layers", "2", "--loss_function", "mse"]
    },
    {
        "name": "wider_model",
        "args": ["--train_epochs", "30", "--batch_size", "64", "--learning_rate", "0.0005",
                "--d_model", "256", "--d_ff", "1024", "--loss_function", "mse"]
    },
    {
        "name": "optimized",
        "args": ["--train_epochs", "30", "--batch_size", "128", "--learning_rate", "0.001",
                "--grad_clip", "1.0", "--e_layers", "2", "--d_layers", "1", 
                "--d_model", "192", "--loss_function", "financial", "--optimize_for_mps"]
    }
]

# Function to run a configuration
def run_configuration(config, output_dir):
    config_dir = os.path.join(experiment_dir, config["name"])
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(os.path.join(config_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(config_dir, "output"), exist_ok=True)
    
    # Base command
    cmd = [
        "python", "direct_run.py",
        "--checkpoints", os.path.join(config_dir, "checkpoints"),
        "--output_dir", os.path.join(config_dir, "output")
    ]
    
    # Add configuration-specific arguments
    cmd.extend(config["args"])
    
    # Save configuration
    with open(os.path.join(config_dir, "config.json"), 'w') as f:
        json.dump({"name": config["name"], "args": config["args"]}, f, indent=4)
    
    # Run the command
    print(f"Running configuration: {config['name']}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                  universal_newlines=True)
        
        # Stream output in real-time
        with open(os.path.join(config_dir, "log.txt"), 'w') as log_file:
            for line in iter(process.stdout.readline, ""):
                print(line, end="")
                log_file.write(line)
                
        process.wait()
        print(f"Configuration {config['name']} completed with return code {process.returncode}")
        return True
    except Exception as e:
        print(f"Error running configuration {config['name']}: {e}")
        return False

# Run all configurations
results = []
for config in configurations:
    success = run_configuration(config, experiment_dir)
    if success:
        # Load metrics from output
        metrics_path = os.path.join(experiment_dir, config["name"], "output", "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                metrics["configuration"] = config["name"]
                results.append(metrics)

# Combine results into a comparison table
if results:
    # Convert to DataFrame for easy manipulation
    results_df = pd.DataFrame(results)
    
    # Save comparison to CSV
    comparison_path = os.path.join(experiment_dir, "metrics_comparison.csv")
    results_df.to_csv(comparison_path, index=False)
    print(f"Comparison saved to {comparison_path}")
    
    # Create visualizations for each metric
    metrics_to_compare = ["RMSE", "MAE", "MAPE", "DirectionalAccuracy"]
    for metric in metrics_to_compare:
        if metric in results_df.columns:
            plt.figure(figsize=(12, 6))
            plt.bar(results_df["configuration"], results_df[metric], color='skyblue')
            plt.title(f"Comparison of {metric} Across Configurations")
            plt.xlabel("Configuration")
            plt.ylabel(metric)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(experiment_dir, f"comparison_{metric}.png"))
            plt.close()
    
    # Create a normalized comparison plot
    plt.figure(figsize=(12, 8))
    
    # Normalize each metric to [0,1] for fair comparison
    normalized_df = results_df.copy()
    for metric in metrics_to_compare:
        if metric in normalized_df.columns:
            if metric == "DirectionalAccuracy":
                # Higher is better for directional accuracy
                normalized_df[metric] = (normalized_df[metric] - normalized_df[metric].min()) / \
                                     (normalized_df[metric].max() - normalized_df[metric].min())
            else:
                # Lower is better for error metrics
                normalized_df[metric] = 1 - (normalized_df[metric] - normalized_df[metric].min()) / \
                                     (normalized_df[metric].max() - normalized_df[metric].min())
    
    # Plot normalized metrics
    for i, metric in enumerate(metrics_to_compare):
        if metric in normalized_df.columns:
            plt.subplot(2, 2, i+1)
            plt.bar(normalized_df["configuration"], normalized_df[metric], color='skyblue')
            plt.title(f"Normalized {metric}")
            plt.xlabel("Configuration")
            plt.ylabel("Score (higher is better)")
            plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, "comparison_normalized.png"))
    plt.close()
    
    # Identify the best configuration
    best_configs = {}
    for metric in metrics_to_compare:
        if metric in results_df.columns:
            if metric == "DirectionalAccuracy":
                # Higher is better
                best_idx = results_df[metric].idxmax()
            else:
                # Lower is better
                best_idx = results_df[metric].idxmin()
            
            best_configs[metric] = results_df.loc[best_idx, "configuration"]
    
    # Print and save the best configurations
    print("\nBest configurations by metric:")
    for metric, config in best_configs.items():
        print(f"{metric}: {config}")
    
    with open(os.path.join(experiment_dir, "best_configs.json"), 'w') as f:
        json.dump(best_configs, f, indent=4)
else:
    print("No results to compare.")

print(f"Experiment completed. Results saved to {experiment_dir}")
