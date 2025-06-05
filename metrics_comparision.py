import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import glob

def calculate_metrics(actual, predicted):
    """Calculate various time series metrics"""
    # Ensure arrays are flattened for calculation
    actual = actual.reshape(-1)
    predicted = predicted.reshape(-1)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(actual - predicted))
    
    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((actual - predicted)**2))
    
    # Mean Absolute Percentage Error
    # Adding small epsilon to avoid division by zero
    mape = np.mean(np.abs((actual - predicted) / (np.abs(actual) + 1e-10))) * 100
    
    # Directional Accuracy
    direction_actual = np.diff(actual) >= 0
    direction_pred = np.diff(predicted) >= 0
    dir_acc = np.mean(direction_actual == direction_pred) * 100
    
    # Mean Squared Error
    mse = np.mean((actual - predicted)**2)
    
    # R-squared (Coefficient of Determination)
    ss_tot = np.sum((actual - np.mean(actual))**2)
    ss_res = np.sum((actual - predicted)**2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))
    
    # Symmetric Mean Absolute Percentage Error (SMAPE)
    # Better for financial data as it's bounded between 0-200%
    smape = 100 * np.mean(2.0 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted) + 1e-10))
    
    # Theil's U-statistic (Forecast Accuracy)
    # 0 = perfect forecast, 1 = naive forecast, >1 = worse than naive
    # Calculate changes
    actual_changes = np.diff(actual)
    pred_changes = np.diff(predicted)
    # Add small epsilon to avoid division by zero
    theil_u = np.sqrt(np.mean((pred_changes - actual_changes)**2)) / (np.sqrt(np.mean(actual_changes**2)) + 1e-10)
    
    # Maximum Error
    max_error = np.max(np.abs(predicted - actual))
    
    # Trading Strategy Return (simplified)
    # Assuming: buy when predicted up, sell when predicted down
    # This is a very simplified version
    signals = np.sign(np.diff(predicted))  # 1 for up, -1 for down
    # Shift actual returns to align with signals (can't trade on future knowledge)
    actual_returns = np.diff(actual)
    # Calculate strategy returns (signal * next_period_return)
    strategy_returns = signals[:-1] * actual_returns[1:]
    # Cumulative return
    cumulative_return = np.sum(strategy_returns)
    
    # Profit Factor (sum of positive returns / sum of negative returns)
    positive_returns = strategy_returns[strategy_returns > 0]
    negative_returns = strategy_returns[strategy_returns < 0]
    if len(negative_returns) > 0 and np.sum(np.abs(negative_returns)) > 0:
        profit_factor = np.sum(positive_returns) / np.sum(np.abs(negative_returns))
    else:
        profit_factor = np.inf if len(positive_returns) > 0 else 0
    
    # Hit Ratio (percentage of profitable trades)
    hit_ratio = len(positive_returns) / len(strategy_returns) * 100 if len(strategy_returns) > 0 else 0
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'DirectionalAccuracy': dir_acc,
        'MSE': mse,
        'R2': r2,
        'SMAPE': smape,
        'TheilU': theil_u,
        'MaxError': max_error,
        'CumulativeReturn': cumulative_return,
        'ProfitFactor': profit_factor if profit_factor != np.inf else 999.99,  # Cap infinite values
        'HitRatio': hit_ratio
    }

def load_experiment_results(exp_dir):
    """Load results from a single experiment"""
    # Extract experiment name
    exp_name = os.path.basename(exp_dir)
    print(f"Processing {exp_name}...")
    
    # Load experiment config
    config_path = os.path.join(exp_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        print(f"No config.json found in {exp_dir}, using directory name as experiment name")
        config = {'name': exp_name}
    
    # Load predictions and actual values
    # First try with 'true_values.npy' (as saved in direct_run.py)
    predictions_path = os.path.join(exp_dir, 'output', 'predictions.npy')
    actuals_path = os.path.join(exp_dir, 'output', 'true_values.npy')
    
    # If not found, try with 'actuals.npy'
    if not os.path.exists(actuals_path):
        actuals_path = os.path.join(exp_dir, 'output', 'actuals.npy')
    
    if os.path.exists(predictions_path) and os.path.exists(actuals_path):
        try:
            predictions = np.load(predictions_path, allow_pickle=True)
            actuals = np.load(actuals_path, allow_pickle=True)
            
            # Check array dimensions and types
            print(f"Predictions shape: {predictions.shape}, dtype: {predictions.dtype}")
            print(f"Actuals shape: {actuals.shape}, dtype: {actuals.dtype}")
            
            # Convert to float if necessary
            if predictions.dtype != np.float32 and predictions.dtype != np.float64:
                predictions = predictions.astype(np.float32)
            if actuals.dtype != np.float32 and actuals.dtype != np.float64:
                actuals = actuals.astype(np.float32)
            
            # Calculate metrics
            metrics = calculate_metrics(actuals, predictions)
            
            # Add configuration details to metrics
            metrics['experiment'] = exp_name
            for key in ['d_model', 'n_heads', 'e_layers', 'learning_rate', 'train_epochs']:
                if key in config:
                    metrics[key] = config[key]
                    
            return metrics
        except Exception as e:
            print(f"Error processing {exp_name}: {str(e)}")
            return None
    else:
        if not os.path.exists(predictions_path):
            print(f"Missing predictions file: {predictions_path}")
        if not os.path.exists(actuals_path):
            print(f"Missing actuals file: {actuals_path}")
        return None

def compare_all_experiments(base_dir='./experiments'):
    """Compare metrics across all experiments"""
    print(f"Looking for experiment directories in {base_dir}")
    
    # Find all experiment directories
    exp_dirs = [d for d in glob.glob(os.path.join(base_dir, '*')) if os.path.isdir(d)]
    print(f"Found {len(exp_dirs)} experiment directories")
    
    # Collect metrics from each experiment
    all_metrics = []
    for exp_dir in exp_dirs:
        metrics = load_experiment_results(exp_dir)
        if metrics:
            all_metrics.append(metrics)
    
    # Convert to DataFrame for easy comparison
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        
        # Sort by MAE (or preferred metric)
        df = df.sort_values('MAE')
        
        # Save to CSV
        csv_path = os.path.join(base_dir, 'metrics_comparison.csv')
        df.to_csv(csv_path, index=False)
        
        # Create visualizations
        # Individual metric plots
        for metric in ['MAE', 'RMSE', 'MAPE', 'DirectionalAccuracy']:
            plt.figure(figsize=(12, 6))
            bars = plt.bar(df['experiment'], df[metric])
            plt.title(f'Comparison of {metric} Across Experiments')
            plt.xlabel('Experiment')
            plt.ylabel(metric)
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels on top of each bar
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(df[metric]),
                        f'{height:.4f}', ha='center', va='bottom', rotation=0)
            
            plt.tight_layout()
            plt.savefig(os.path.join(base_dir, f'comparison_{metric}.png'))
            plt.close()
        
        # Combined plot (normalized for fair comparison)
        plt.figure(figsize=(14, 8))
        metrics_to_plot = ['MAE', 'RMSE', 'MAPE', 'DirectionalAccuracy']
        
        # Normalize each metric (lower is better for all except DirectionalAccuracy)
        df_norm = df.copy()
        for metric in metrics_to_plot:
            if metric == 'DirectionalAccuracy':  # Higher is better
                df_norm[metric] = df[metric] / df[metric].max()
            else:  # Lower is better
                min_val = df[metric].min()
                df_norm[metric] = min_val / df[metric]  # Invert so higher is better
        
        # Plot normalized metrics
        width = 0.2
        x = np.arange(len(df))
        
        for i, metric in enumerate(metrics_to_plot):
            plt.bar(x + (i-1.5)*width, df_norm[metric], width, label=metric)
        
        plt.xlabel('Experiment')
        plt.ylabel('Normalized Score (higher is better)')
        plt.title('Comparison of All Metrics (Normalized)')
        plt.xticks(x, df['experiment'], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, 'comparison_normalized.png'))
        
        print(f"Comparison saved to {csv_path}")
        print(f"Visualizations saved to {base_dir}")
        
        # Print top 3 experiments for each metric
        print("\nTop Experiments by Metric:")
        for metric in metrics_to_plot:
            sort_ascending = metric != 'DirectionalAccuracy'  # True for metrics where lower is better
            top_df = df.sort_values(metric, ascending=sort_ascending).head(3)
            print(f"\n{metric}:")
            for idx, row in top_df.iterrows():
                print(f"  {row['experiment']}: {row[metric]:.4f}")
        
        return df
    
    print("No experiment results found.")
    return None

if __name__ == "__main__":
    compare_all_experiments()