"""
Visualization utilities for time series forecasting results.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
from datetime import datetime, timedelta

def plot_predictions(y_true, y_pred, sample_indices=None, max_samples=5, 
                    output_dir=None, time_labels=None, title_prefix="Sample",
                    time_freq=None):
    """
    Plot predictions vs. true values for a subset of samples.
    
    Parameters:
    -----------
    y_true : array-like
        The true values with shape (samples, sequence_length, features)
    y_pred : array-like
        The predicted values with shape (samples, sequence_length, features)
    sample_indices : list, optional
        Indices of specific samples to plot. If None, random samples are chosen.
    max_samples : int, optional
        Maximum number of samples to plot
    output_dir : str, optional
        Directory to save plots. If None, plots are displayed but not saved.
    time_labels : array-like, optional
        Time labels for x-axis. If None, indices are used.
    title_prefix : str, optional
        Prefix for plot titles
    time_freq : str, optional
        Frequency of the time series for datetime formatting (e.g., 'D', 'H', 'min')
    
    Returns:
    --------
    None
    """
    # Make sure output_dir exists if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Ensure data is numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Determine number of samples to plot
    n_samples = min(max_samples, y_true.shape[0])
    
    # Select sample indices if not provided
    if sample_indices is None:
        if n_samples >= y_true.shape[0]:
            # Use all samples if we want more than available
            sample_indices = list(range(y_true.shape[0]))
        else:
            # Randomly select samples
            sample_indices = np.random.choice(
                y_true.shape[0], size=n_samples, replace=False
            )
    
    # Create time labels if not provided
    if time_labels is None:
        time_labels = np.arange(y_true.shape[1])
    
    # Create datetime objects if time_freq is provided
    if time_freq:
        base_date = datetime.now()
        if time_freq == 'D':
            time_labels = [base_date + timedelta(days=i) for i in range(y_true.shape[1])]
        elif time_freq == 'H':
            time_labels = [base_date + timedelta(hours=i) for i in range(y_true.shape[1])]
        elif time_freq == 'min':
            time_labels = [base_date + timedelta(minutes=i) for i in range(y_true.shape[1])]
    
    # Plot each sample
    for i, idx in enumerate(sample_indices[:n_samples]):
        plt.figure(figsize=(12, 6))
        
        # Extract the data for this sample
        if idx < y_true.shape[0]:
            # Get the target feature (first feature by default)
            true_vals = y_true[idx, :, 0] if y_true.ndim > 2 else y_true[idx]
            pred_vals = y_pred[idx, :, 0] if y_pred.ndim > 2 else y_pred[idx]
            
            # Plot
            plt.plot(time_labels, true_vals, 'b-', label='True', linewidth=2)
            plt.plot(time_labels, pred_vals, 'r--', label='Predicted', linewidth=2)
            
            # Calculate error metrics for this sample
            rmse = np.sqrt(np.mean((true_vals - pred_vals) ** 2))
            mae = np.mean(np.abs(true_vals - pred_vals))
            
            # Format dates on x-axis if time_labels are datetime objects
            if time_freq:
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M' if time_freq == 'min' 
                                                                    else '%m-%d' if time_freq == 'D' 
                                                                    else '%H:%M'))
                plt.gcf().autofmt_xdate()
            
            # Add metrics to the title
            plt.title(f"{title_prefix} {i+1} (RMSE: {rmse:.4f}, MAE: {mae:.4f})")
            
            # Add grid and legend
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(loc='best')
            
            # Add labels
            plt.xlabel('Time Steps')
            plt.ylabel('Value')
            
            # Improve x-axis display
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
            
            # Save or show
            if output_dir:
                plt.savefig(os.path.join(output_dir, f'sample_{i+1}.png'), dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.tight_layout()
                plt.show()

def plot_loss_curves(train_losses, val_losses=None, output_dir=None, filename='loss_curve.png'):
    """
    Plot training and validation loss curves.
    
    Parameters:
    -----------
    train_losses : array-like
        Training loss values per epoch
    val_losses : array-like, optional
        Validation loss values per epoch
    output_dir : str, optional
        Directory to save the plot. If None, plot is displayed but not saved.
    filename : str, optional
        Filename for the saved plot
    
    Returns:
    --------
    None
    """
    plt.figure(figsize=(12, 6))
    epochs = range(1, len(train_losses) + 1)
    
    # Plot training loss with marker
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=6)
    
    # Plot validation loss if provided
    if val_losses is not None:
        plt.plot(epochs, val_losses, 'r--s', label='Validation Loss', linewidth=2, markersize=6)
    
    # Add title and labels
    plt.title('Training and Validation Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    
    # Improve x-axis display
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Save or show
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def plot_metrics_comparison(metrics_dict, output_dir=None, exclude_metrics=None):
    """
    Plot bar charts for each metric to compare models.
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary where keys are model names and values are dictionaries of metrics
    output_dir : str, optional
        Directory to save plots. If None, plots are displayed but not saved.
    exclude_metrics : list, optional
        List of metrics to exclude from plotting
    
    Returns:
    --------
    None
    """
    if exclude_metrics is None:
        exclude_metrics = []
    
    # Get all unique metrics
    all_metrics = set()
    for model_metrics in metrics_dict.values():
        all_metrics.update(model_metrics.keys())
    
    # Remove excluded metrics
    all_metrics = [m for m in all_metrics if m not in exclude_metrics]
    
    # Make sure output_dir exists if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Plot each metric
    for metric in all_metrics:
        plt.figure(figsize=(12, 6))
        
        # Collect values for this metric across all models
        models = []
        values = []
        
        for model_name, model_metrics in metrics_dict.items():
            if metric in model_metrics:
                models.append(model_name)
                values.append(model_metrics[metric])
        
        # Create bar chart
        bars = plt.bar(models, values, color='skyblue', edgecolor='black')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(values),
                    f'{height:.4f}', ha='center', va='bottom')
        
        # Add title and labels
        plt.title(f'Comparison of {metric} Across Models')
        plt.xlabel('Models')
        plt.ylabel(metric)
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.3, axis='y')
        
        # Rotate x-axis labels if there are many models
        if len(models) > 3:
            plt.xticks(rotation=45, ha='right')
        
        # Save or show
        if output_dir:
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'comparison_{metric}.png'), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
