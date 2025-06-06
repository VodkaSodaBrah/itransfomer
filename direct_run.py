import os
import sys
import torch
import numpy as np
import random
import argparse
import math
import json
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import ParameterGrid

# Add paths to the import system
current_dir = os.path.dirname(os.path.abspath(__file__))
itransformer_dir = os.path.join(current_dir, 'iTransformer')
utils_dir = os.path.join(current_dir, 'utils')

# Make sure the directories exist
print(f"Current directory: {current_dir}")
print(f"iTransformer directory: {itransformer_dir}")
print(f"Utils directory: {utils_dir}")
print(f"Utils directory exists: {os.path.exists(utils_dir)}")
if os.path.exists(utils_dir):
    print(f"Files in utils: {os.listdir(utils_dir)}")

# Add directories to path
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)  # Add the current directory first
if itransformer_dir not in sys.path:
    sys.path.insert(1, itransformer_dir)  # Add the iTransformer directory second
if utils_dir not in sys.path:
    sys.path.insert(2, utils_dir)  # Add the utils directory third

print(f"Current sys.path: {sys.path}")

# Import from iTransformer package
from iTransformer.data_loader.improved_bnb_loader import ImprovedBNBDataset
from iTransformer.model.iTransformer import Model
from iTransformer.utils.tools import EarlyStopping

# Import custom utilities from the local utils directory
try:
    # Import visualization functions
    from utils.visualize import plot_predictions, plot_loss_curves, plot_metrics_comparison
    print("Successfully imported visualization functions from utils/visualize.py")
except ImportError as e:
    print(f"Error importing visualization module: {e}")
    # Define fallback visualization functions
    import matplotlib.pyplot as plt
    import os
    
    def plot_predictions(trues, preds, sample_indices=None, output_dir="./", title_prefix="Sample"):
        """Plot predictions vs actual values for selected samples."""
        if sample_indices is None:
            sample_indices = [0]
        
        for i in sample_indices:
            plt.figure(figsize=(10, 6))
            if trues.shape[2] > 0:
                plt.plot(trues[i, :, 0], label='Actual')
                plt.plot(preds[i, :, 0], label='Predicted')
            else:
                plt.plot(trues[i], label='Actual')
                plt.plot(preds[i], label='Predicted')
            plt.title(f'{title_prefix} {i+1}')
            plt.legend()
            plt.savefig(os.path.join(output_dir, f'prediction_sample_{i+1}.png'))
            plt.close()
    
    def plot_loss_curves(train_losses, test_losses, output_dir="./", filename="loss_curve.png"):
        """Plot training and test loss curves."""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
    
    def plot_metrics_comparison(metrics_dict, output_dir="./", filename="metrics_comparison.png"):
        """Plot comparison of different metrics."""
        plt.figure(figsize=(12, 8))
        plt.bar(metrics_dict.keys(), metrics_dict.values())
        plt.title('Metrics Comparison')
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
    
    print("Using fallback visualization functions")
    
    # Import Mac optimizations
try:
    from utils.mac_optimizations import (
        configure_for_mac_performance, get_mac_performance_tips, 
        mps_autocast, MPSGradScaler, get_amp_context_manager, get_grad_scaler,
        optimize_for_mps_inference, benchmark_mps_performance, check_mps_compatibility
    )
    
    # Import advanced MPS optimizations
    try:
        from utils.mps_advanced import (
            prepare_model_for_mps, detect_mac_chip_details, 
            optimize_model_for_chip, replace_dropout_layers
        )
        from utils.advanced_benchmarking import benchmark_mps_performance_advanced
        has_advanced_mps = True
        print("Successfully imported advanced MPS optimizations")
    except ImportError as e:
        has_advanced_mps = False
        print(f"Advanced MPS optimizations not available, using basic optimizations: {e}")
        
    print("Successfully imported Mac optimizations from utils")
except ImportError as e:
    print(f"Error importing Mac optimizations: {e}")
    # Create placeholder functions to prevent errors
    def configure_for_mac_performance(args, device_type):
        print("Mac optimizations not available")
        return args
        
    def get_mac_performance_tips():
        return ["Mac optimizations module not available"]
        
    class MPSAutocastContext:
        def __init__(self, enabled=True):
            self.enabled = enabled
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
    
    def mps_autocast(enabled=True):
        return MPSAutocastContext(enabled)
        
    class MPSGradScaler:
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            # No-op for MPS placeholder
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass
    
    def get_amp_context_manager(device):
        return mps_autocast()
        
    def get_grad_scaler(device):
        return MPSGradScaler()
        
    def optimize_for_mps_inference(model):
        return model
        
    def benchmark_mps_performance(model, input_data, iterations=10, warmup=2):
        return {"inference_time": 0.0, "memory_used": 0}
        
    def check_mps_compatibility(model):
        return True
        
    has_advanced_mps = False
    
    # Import metrics
try:
    from utils.metrics import (
        calculate_mse, calculate_rmse, calculate_mae, 
        calculate_mape, calculate_smape, calculate_directional_accuracy, calculate_wape
    )
    print("Successfully imported metrics from utils")
except ImportError as e:
    print(f"Error importing metrics module: {e}")
    # Define fallback metric functions
    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    def calculate_mse(y_true, y_pred):
        """Mean Squared Error (MSE)"""
        return mean_squared_error(y_true, y_pred)
    
    def calculate_rmse(y_true, y_pred):
        """Root Mean Squared Error (RMSE)"""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def calculate_mae(y_true, y_pred):
        """Mean Absolute Error (MAE)"""
        return mean_absolute_error(y_true, y_pred)
    
    def calculate_mape(y_true, y_pred, epsilon=1e-8):
        """Mean Absolute Percentage Error (MAPE)"""
        mask = np.abs(y_true) > epsilon
        if not np.any(mask):
            return np.nan
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def calculate_smape(y_true, y_pred, epsilon=1e-8):
        """Symmetric Mean Absolute Percentage Error (SMAPE)"""
        denominator = np.abs(y_true) + np.abs(y_pred) + epsilon
        return 200 * np.mean(np.abs(y_pred - y_true) / denominator)
    
    def calculate_directional_accuracy(y_true, y_pred):
        """Directional Accuracy"""
        true_diff = np.diff(y_true, axis=0)
        pred_diff = np.diff(y_pred, axis=0)
        true_direction = np.sign(true_diff)
        pred_direction = np.sign(pred_diff)
        return np.mean((true_direction == pred_direction).astype(float))
    
    def calculate_wape(y_true, y_pred, epsilon=1e-8):
        """Weighted Absolute Percentage Error (WAPE)"""
        return np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + epsilon)
except ImportError as e:
    print(f"Import error: {e}")
    print("Current sys.path:", sys.path)
    print("Looking for utils in:", os.path.join(current_dir, 'utils'))
    sys.exit(1)

# Set random seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# Device detection
def get_device(device=None):
    if device:
        if device == 'mps' and torch.backends.mps.is_available():
            return torch.device('mps')
        elif device == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        elif device == 'cpu':
            return torch.device('cpu')
    
    # Auto-detect
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

class FinancialLoss(torch.nn.Module):
    def __init__(self, direction_weight=0.5):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.direction_weight = direction_weight
        
    def forward(self, pred, target):
        # Handle potential shape mismatches
        if pred.shape != target.shape:
            # Ensure compatible shapes for broadcasting
            if target.shape[-1] == 0:
                # If target has zero dimension, create a tensor with matching shape
                target = torch.zeros_like(pred)
            elif pred.shape[-1] == 0:
                # If pred has zero dimension, create a tensor with matching shape
                pred = torch.zeros_like(target)
        
        # Check for NaN values and replace them
        pred = torch.nan_to_num(pred)
        target = torch.nan_to_num(target)
        
        # MSE component
        mse_loss = self.mse(pred, target)
        
        # Directional component - only if we have more than one time step
        if pred.size(1) > 1:
            # Calculate differences between consecutive time steps
            pred_diff = pred[:, 1:] - pred[:, :-1]
            target_diff = target[:, 1:] - target[:, :-1]
            
            # Get sign of differences to determine direction (up=positive, down=negative)
            pred_direction = torch.sign(torch.nan_to_num(pred_diff))
            target_direction = torch.sign(torch.nan_to_num(target_diff))
            
            # Calculate binary mask where directions don't match (1 where directions differ)
            direction_loss = torch.mean((pred_direction != target_direction).float())
            
            # Combined loss with safety check
            combined_loss = mse_loss + self.direction_weight * direction_loss
            
            # Final safety check for NaN values
            if torch.isnan(combined_loss):
                return torch.tensor(1.0, device=pred.device)
            return combined_loss
        else:
            # If we only have a single time step, no direction can be calculated
            return mse_loss

# Create evaluate_forecasting function for backward compatibility
def evaluate_forecasting(y_true, y_pred, metrics=None):
    """
    Calculate multiple evaluation metrics for time series forecasting.
    
    Parameters:
    -----------
    y_true : array-like
        The true values
    y_pred : array-like
        The predicted values
    metrics : list, optional
        List of metrics to calculate. If None, all metrics are calculated.
        
    Returns:
    --------
    dict
        Dictionary with metric names as keys and calculated values as values
    """
    # Ensure y_true and y_pred are numpy arrays with same shape
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Make sure data is in the right shape (2D)
    if y_true.ndim > 2:
        # Reshape to (samples, features) by combining dimensions beyond the first
        original_shape = y_true.shape
        y_true = y_true.reshape(original_shape[0], -1)
        y_pred = y_pred.reshape(original_shape[0], -1)
    
    # Available metrics
    all_metrics = {
        'MSE': calculate_mse,
        'RMSE': calculate_rmse,
        'MAE': calculate_mae,
        'MAPE': calculate_mape,
        'SMAPE': calculate_smape,
        'DirectionalAccuracy': calculate_directional_accuracy,
        'WAPE': calculate_wape
    }
    
    # Use specified metrics or all metrics
    if metrics is None:
        metrics = list(all_metrics.keys())
    
    # Calculate metrics
    results = {}
    for metric in metrics:
        if metric in all_metrics:
            # Special handling for directional accuracy which requires 2D data with time as first dimension
            if metric == 'DirectionalAccuracy' and y_true.shape[0] <= 1:
                results[metric] = np.nan
                continue
                
            results[metric] = all_metrics[metric](y_true, y_pred)
        else:
            results[metric] = np.nan  # Metric not found
    
    return results

def main():
    parser = argparse.ArgumentParser()
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='./data/bnbusdt_1m.csv', help='Data file path')
    parser.add_argument('--features', type=str, default='MS', help='Forecasting task: S=univariate, MS=multivariate')
    parser.add_argument('--target', type=str, default='close', help='Target feature to predict')
    parser.add_argument('--freq', type=str, default='1min', help='Data frequency')
    
    # Forecasting parameters
    parser.add_argument('--seq_len', type=int, default=96, help='Input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='Label sequence length')
    parser.add_argument('--pred_len', type=int, default=24, help='Prediction horizon')
    
    # Model parameters
    parser.add_argument('--enc_in', type=int, default=5, help='Input size')
    parser.add_argument('--dec_in', type=int, default=5, help='Decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='Output size')
    parser.add_argument('--d_model', type=int, default=128, help='Dimension of model')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--e_layers', type=int, default=1, help='Number of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='Number of decoder layers')
    parser.add_argument('--d_ff', type=int, default=512, help='Dimension of FCN')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--factor', type=int, default=1, help='Attention factor')
    parser.add_argument('--embed', type=str, default='timeF', help='Time embedding')
    parser.add_argument('--activation', type=str, default='gelu', help='Activation function')
    parser.add_argument('--output_attention', action='store_true', help='Output attention weights')
    parser.add_argument('--use_norm', type=bool, default=True, help='Use normalization')
    parser.add_argument('--class_strategy', type=str, default='projection', help='Classification strategy')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--train_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--min_epochs', type=int, default=10, help='Minimum epochs before early stopping')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='Checkpoint directory')
    parser.add_argument('--output_dir', type=str, default='./output/', help='Output directory')
    parser.add_argument('--device', type=str, default=None, help='Device: mps, cuda, cpu, or None for auto-detect')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--optimize_for_mps', action='store_true', 
                    help='Apply optimizations specific to Apple Metal GPU')
    parser.add_argument('--disable_dropout_for_mps_compile', action='store_true',
                        help='Disable dropout layers when torch.compile is used on MPS to avoid NotImplementedError')
    parser.add_argument('--no_compile', action='store_true', help='Disable torch.compile for debugging')
    parser.add_argument('--compile_mode', type=str, default='default',
                        help='Compilation mode for MPS: default, reduce-overhead, max-autotune')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmarking after training')
    parser.add_argument('--lradj', type=str, default='type1', 
                    help='Learning rate adjustment strategy: type1, type2, or cosine')
    parser.add_argument('--test_overfit', action='store_true', 
                   help='Test model ability to overfit a small dataset before training')
    parser.add_argument('--window_norm', action='store_true', help='Use per-window normalization as recommended for time series')
    parser.add_argument('--grad_clip', type=float, default=1.0, 
                    help='Gradient clipping threshold to prevent exploding gradients')
    parser.add_argument('--loss_function', type=str, default='mse', 
                    help='Loss function: mse, mae, huber, directional, or financial')
    parser.add_argument('--directional_weight', type=float, default=0.5,
                    help='Weight for directional component in directional or financial loss')
    parser.add_argument('--max_len', type=int, default=None, help='Max number of samples to load')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.checkpoints, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed
    set_seed(2023)
    
    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Apply Mac-specific optimizations
    if device.type == 'mps':
        # Use the utility function from mac_optimizations.py
        args = configure_for_mac_performance(args, device_type='mps')
        
        # Display Mac performance tips
        mac_tips = get_mac_performance_tips()
        if mac_tips:
            print("\n Mac Performance Tips:")
            for tip in mac_tips:
                print(f"  {tip}")
    
    
    # Prepare data
    train_dataset = ImprovedBNBDataset(
        root_path=current_dir,
        data_path=args.data_path,
        flag='train',
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        window_norm=args.window_norm,
        scale=True,
        max_len=args.max_len
    )
    
    test_dataset = ImprovedBNBDataset(
        root_path=current_dir,
        data_path=args.data_path,
        flag='test',
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        window_norm=args.window_norm,
        scale=True,
        max_len=args.max_len
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,  
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,  
        pin_memory=False 
    )

    print(f"Train data: {len(train_dataset)} samples")
    print(f"Test data: {len(test_dataset)} samples")
    
    # Set up model
    # Convert arguments to match the model's expected config format
    class ModelConfig:
        def __init__(self, args):
            for key, value in vars(args).items():
                setattr(self, key, value)
            self.device = device
    
    config = ModelConfig(args) # Ensure ModelConfig uses the potentially modified args.batch_size
    # Pass the new arg to ModelConfig if your Model or submodules need it
    config.disable_dropout_for_mps_compile = args.disable_dropout_for_mps_compile
    config.is_mps_compiled = False # We'll set this after attempting to compile

    model = Model(config).float().to(device)
    criterion = torch.nn.MSELoss()  

    # Add to direct_run.py just after model creation
    def save_model_config(args, filepath):
        """Save the model configuration for later reference"""
        config = {
            "enc_in": args.enc_in,
            "dec_in": args.dec_in,
            "c_out": args.c_out,
            "seq_len": args.seq_len,
            "label_len": args.label_len,
            "pred_len": args.pred_len,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "e_layers": args.e_layers,
            "d_layers": args.d_layers,
            "d_ff": args.d_ff,
            "dropout": args.dropout,
            "embed": args.embed,
            "freq": args.freq,
            "activation": args.activation,
            "output_attention": args.output_attention,
        }
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        return config

    # Save the configuration
    config_path = os.path.join(args.checkpoints, 'model_config.json')
    save_model_config(args, config_path)

    # Test if model can overfit to a small dataset before full training
    def test_overfit_capability(model, device, criterion, dataset, args):
        """Test model's ability to overfit a small batch of data"""
        print("Testing model capability to overfit on small dataset...")
        
        # Create a small batch of data
        batch_size = min(8, args.batch_size)  # Use a very small batch
        
        # Create a dataloader with just a few samples
        small_dataset = torch.utils.data.Subset(dataset, range(batch_size))
        small_loader = DataLoader(
            small_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Try to get one batch of data
        try:
            batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(small_loader))
        except Exception as e:
            print(f"Error creating small dataset: {e}")
            return False
        
        # Check for invalid dimensions and fix them
        if batch_x.shape[-1] == 0 or batch_y.shape[-1] == 0:
            print(f"Warning: Dataset returned tensors with zero dimensions: x={batch_x.shape}, y={batch_y.shape}")
            
            # Fix x tensor if needed
            if batch_x.shape[-1] == 0:
                # Create valid x tensor with proper dimensions - assume 5 features as default
                feature_count = getattr(dataset, 'enc_in', 5)
                batch_x = torch.zeros((batch_size, batch_x.shape[1], feature_count))
            
            # Fix y tensor if needed
            if batch_y.shape[-1] == 0:
                # Create valid y tensor with 1 feature dimension
                batch_y = torch.zeros((batch_size, batch_y.shape[1], 1))
                print(f"Fixed zero-dimension target tensor to shape: {batch_y.shape}")
        
        # Check for NaN values
        if torch.isnan(batch_x).any() or torch.isnan(batch_y).any():
            print("Warning: Dataset contains NaN values, replacing with zeros")
            batch_x = torch.nan_to_num(batch_x)
            batch_y = torch.nan_to_num(batch_y)
        
        # Put tensors on the appropriate device
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)
        
        # Setup optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Higher learning rate
        
        # Train for a few epochs on this single batch
        model.train()
        overfit_success = False
        
        # Store loss values to track progress
        loss_values = []
        
        for epoch in range(10):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
            
            # Debug shapes - only show on first epoch
            if epoch == 0:
                print(f"Overfit test - outputs shape: {outputs.shape}, y shape: {batch_y.shape}")
            
            # Check for NaN in outputs
            if torch.isnan(outputs).any():
                print(f"Warning: Model produced NaN outputs in epoch {epoch+1}, replacing with zeros")
                outputs = torch.nan_to_num(outputs)
            
            # Process the outputs and targets for loss calculation
            if args.features == 'MS':
                try:
                    # If we have a target column, extract only that feature
                    if hasattr(dataset, 'target'):
                        target_name = dataset.target
                        print(f"Using overfit target column: {target_name}")
                        
                        if hasattr(dataset, 'df_data') and target_name in dataset.df_data.columns:
                            target_idx = dataset.df_data.columns.get_loc(target_name)
                            
                            # Extract target dimension from outputs if possible
                            if outputs.shape[2] > target_idx:
                                outputs = outputs[:, :, target_idx:target_idx+1]
                            else:
                                # If target_idx is out of bounds, use first dimension
                                outputs = outputs[:, :, 0:1]
                            
                            # Extract target dimension from batch_y if possible
                            if batch_y.shape[2] > target_idx:
                                batch_y_pred = batch_y[:, -args.pred_len:, target_idx:target_idx+1]
                            else:
                                # If target_idx is out of bounds, create matching tensor
                                batch_y_pred = torch.zeros_like(outputs)
                        else:
                            # If target not found, use first dimension
                            outputs = outputs[:, :, 0:1]
                            batch_y_pred = torch.zeros_like(outputs)
                    else:
                        # No target attribute, use first dimension
                        outputs = outputs[:, :, 0:1]
                        # Take only the prediction part of the target if dimensions allow
                        if batch_y.shape[2] > 0:
                            batch_y_pred = batch_y[:, -args.pred_len:, 0:1]
                        else:
                            batch_y_pred = torch.zeros_like(outputs)
                except Exception as e:
                    print(f"Warning: Error extracting target feature: {e}, using default handling")
                    outputs = outputs[:, :, 0:1]
                    batch_y_pred = torch.zeros_like(outputs)
            else:
                # For univariate case
                if batch_y.shape[-1] == 0:
                    # Create a valid target tensor with same shape as output
                    batch_y_pred = torch.zeros_like(outputs)
                else:
                    batch_y_pred = batch_y[:, -args.pred_len:, :]
            
            # Ensure the shapes match for loss calculation
            if outputs.shape != batch_y_pred.shape:
                print(f"Warning: Shape mismatch in overfit test - outputs: {outputs.shape}, targets: {batch_y_pred.shape}")
                
                # Adjust shapes to match
                if outputs.shape[1] != batch_y_pred.shape[1]:
                    # Match sequence length
                    min_seq_len = min(outputs.shape[1], batch_y_pred.shape[1])
                    outputs = outputs[:, :min_seq_len, :]
                    batch_y_pred = batch_y_pred[:, :min_seq_len, :]
                
                if outputs.shape[2] != batch_y_pred.shape[2]:
                    # Match feature dimensions
                    if batch_y_pred.shape[2] == 0:
                        batch_y_pred = torch.zeros_like(outputs)
                    elif outputs.shape[2] == 0:
                        outputs = torch.zeros_like(batch_y_pred)
                    else:
                        min_feat_dim = min(outputs.shape[2], batch_y_pred.shape[2])
                        outputs = outputs[:, :, :min_feat_dim]
                        batch_y_pred = batch_y_pred[:, :, :min_feat_dim]
            
            # Calculate loss with final tensors
            print(f"Final overfit shapes - outputs: {outputs.shape}, targets: {batch_y_pred.shape}")
            loss = criterion(outputs, batch_y_pred)
            loss_values.append(loss.item())
            
            # Check for NaN loss
            if torch.isnan(loss).any():
                print(f"Overfit test epoch {epoch+1}: Loss = nan")
            else:
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                print(f"Overfit test epoch {epoch+1}: Loss = {loss.item():.4f}")
            
            # Check if loss is decreasing significantly
            if epoch > 0 and loss.item() < 0.1:
                overfit_success = True
        
        # Visualize overfit test loss
        plt.figure(figsize=(10, 5))
        plt.plot(loss_values, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Overfit Test Loss')
        plt.grid(True)
        plt.savefig(os.path.join(args.output_dir, 'overfit_test.png'))
        
        if overfit_success:
            print("Model successfully overfit small dataset (good capacity)")
        else:
            print("Model struggles to overfit small dataset (may have limited capacity)")
        
        return overfit_success

    # Test the model's overfitting capability
    if args.test_overfit:
        overfit_success = test_overfit_capability(model, device, criterion, train_dataset, args)

    # Apply torch.compile ONCE before training loop if optimizing for MPS
    if args.optimize_for_mps and device.type == 'mps' and not args.no_compile:
        # Get chip details for advanced optimizations
        chip_details = detect_mac_chip_details() if has_advanced_mps else None
        
        if chip_details:
            print(f"Mac chip detected: {chip_details['chip_type']} " + 
                  ("Pro" if chip_details['is_pro'] else "") + 
                  ("Max" if chip_details['is_max'] else "") + 
                  ("Ultra" if chip_details['is_ultra'] else ""))
                  
        if hasattr(torch, 'compile'):
            print("Applying MPS optimizations for model...")
            try:
                # If we are going to compile and disable_dropout_for_mps_compile is True,
                # we need to inform the model to potentially alter its dropout layers.
                if args.disable_dropout_for_mps_compile:
                    print("Dropout will be disabled for MPS compilation.")
                    config.is_mps_compiled_with_dropout_disabled = True
                
                # Check for MPS compatibility issues
                needs_fallback, issues = check_mps_compatibility(model, verbose=True)
                
                if needs_fallback and 'PYTORCH_ENABLE_MPS_FALLBACK' not in os.environ:
                    print("Setting PYTORCH_ENABLE_MPS_FALLBACK=1 to handle unsupported operations")
                    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
                
                # Apply advanced MPS optimizations if available
                if has_advanced_mps:
                    print(f"Using advanced MPS optimizations with compile mode: {args.compile_mode}")
                    # For M1 Pro/Max, M2, or M3 chips, use more aggressive optimizations
                    if args.compile_mode == 'default' and chip_details:
                        if chip_details['chip_type'] in ['M2', 'M3'] or chip_details['is_pro'] or chip_details['is_max']:
                            args.compile_mode = 'max-autotune'
                            print(f"Auto-selected compile mode 'max-autotune' for {chip_details['chip_type']}")
                        else:
                            args.compile_mode = 'reduce-overhead'
                            print(f"Auto-selected compile mode 'reduce-overhead' for {chip_details['chip_type']}")
                            
                    # Apply compilation based on selected mode
                    if args.compile_mode == 'max-autotune':
                        # Use prepare_model_for_mps for complete optimization
                        model = prepare_model_for_mps(model)
                    else:
                        # Use basic optimizations
                        model = replace_dropout_layers(model)
                        model = optimize_for_mps_inference(model, verbose=True)
                else:
                    # Fall back to basic optimizations
                    model = optimize_for_mps_inference(model, verbose=True)
                    
                config.is_mps_compiled = True # Inform config that model is compiled
                print("MPS optimization applied successfully.")
            except Exception as e:
                print(f"MPS optimization failed: {e}. Proceeding without optimization.")
                config.is_mps_compiled_with_dropout_disabled = False # Reset if compile fails
        else:
            print("torch.compile not available in this PyTorch version.")
    else:
        print("Skipping MPS-specific optimizations as requested or not applicable.")

    # Test model's ability to overfit if requested
    if args.test_overfit:
        test_overfit_capability(model, device, criterion, train_dataset, args)

    # Set up loss function based on user selection
    print(f"Using loss function: {args.loss_function}")
    if args.loss_function == 'mse':
        criterion = torch.nn.MSELoss()
    elif args.loss_function == 'mae':
        criterion = torch.nn.L1Loss()
    elif args.loss_function == 'huber':
        criterion = torch.nn.SmoothL1Loss()
    elif args.loss_function == 'directional':
        # Custom directional loss that focuses on predicting the direction of change
        class DirectionalLoss(torch.nn.Module):
            def __init__(self, direction_weight=0.5):
                super().__init__()
                self.mse = torch.nn.MSELoss()
                self.direction_weight = direction_weight
                
            def forward(self, pred, target):
                # MSE component
                mse_loss = self.mse(pred, target)
                
                # Directional component - only if we have more than one time step
                if pred.size(1) > 1:
                    # Calculate differences between consecutive time steps
                    pred_diff = pred[:, 1:] - pred[:, :-1]
                    target_diff = target[:, 1:] - target[:, :-1]
                    
                    # Get sign of differences to determine direction (up=positive, down=negative)
                    pred_direction = torch.sign(torch.nan_to_num(pred_diff))
                    target_direction = torch.sign(torch.nan_to_num(target_diff))
                    
                    # Calculate binary mask where directions don't match (1 where directions differ)
                    direction_loss = torch.mean((pred_direction != target_direction).float())
                    
                    # Combined loss
                    return mse_loss + self.direction_weight * direction_loss
                else:
                    # If we only have a single time step, no direction can be calculated
                    return mse_loss
        
        criterion = DirectionalLoss(direction_weight=args.directional_weight)
    elif args.loss_function == 'financial':
        criterion = FinancialLoss(direction_weight=args.directional_weight)
    else:
        print(f"Warning: Unknown loss function '{args.loss_function}', defaulting to MSE")
        criterion = torch.nn.MSELoss()

    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    model_dir = args.checkpoints
    # Ensure the directory exists
    os.makedirs(model_dir, exist_ok=True)
    # Create a more specific model path
    model_path = os.path.join(model_dir, 'best_model.pth')
    
    # Training and validation
    train_losses = []
    test_losses = []
    
    # Handle performance optimization based on device
    if args.use_amp:
        if torch.cuda.is_available() and hasattr(torch.cuda, 'amp'):
            print("Using Automatic Mixed Precision (AMP) for faster training")
            scaler = torch.cuda.amp.GradScaler()
            use_amp = True
        elif device.type == 'mps':
            print("AMP not available for MPS. Using MPS optimizations instead.")
            # MPS doesn't support AMP, but we'll use other Mac-specific optimizations
            use_amp = False
            # Ensure we're using optimizations for MPS
            args.optimize_for_mps = True
        else:
            print("Warning: AMP requested but not available with current PyTorch version or device")
            use_amp = False
    else:
        use_amp = False
    
    # Set up device-specific autocast and gradient scaler using the utility functions
    autocast_context = get_amp_context_manager(device)
    grad_scaler = get_grad_scaler(device)
    
    # Log information about the optimization approach
    if device.type == 'mps':
        print("Using MPS-specific optimization wrapper with compatible interfaces")
    elif device.type == 'cuda' and use_amp:
        print("Using CUDA with Automatic Mixed Precision")
    else:
        print("Using standard precision computation")
    
    for epoch in range(args.train_epochs):
        # Training
        model.train()
        train_loss = []
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.train_epochs} [Train]")
        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_bar:
            optimizer.zero_grad()
            
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            
            # Forward pass with unified approach for both CUDA and MPS
            with autocast_context:
                outputs = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                
                # Process outputs and targets (shape handling)
                if batch_y.shape[-1] == 0:
                    if random.random() < 0.01:  # Only print occasionally
                        print("Warning: Zero-dimension target tensor in training loop, creating synthetic target")
                    batch_y = torch.zeros_like(outputs)
                else:
                    # Handle multivariate case
                    if args.features == 'MS':
                        try:
                            if hasattr(train_dataset, 'target'):
                                target_name = train_dataset.target
                                if random.random() < 0.01:  # Only print for ~1% of batches
                                    print(f"Using target column: {target_name}")
                                
                                if hasattr(train_dataset, 'df_data') and target_name in train_dataset.df_data.columns:
                                    target_idx = train_dataset.df_data.columns.get_loc(target_name)
                                    
                                    # Extract target dimension from outputs
                                    if outputs.shape[2] > target_idx:
                                        outputs = outputs[:, :, target_idx:target_idx+1]
                                    else:
                                        outputs = outputs[:, :, 0:1]
                                    
                                    # Extract target dimension from batch_y
                                    if batch_y.shape[2] > target_idx:
                                        batch_y = batch_y[:, -args.pred_len:, target_idx:target_idx+1]
                                    else:
                                        batch_y = batch_y[:, -args.pred_len:, :]
                                else:
                                    outputs = outputs[:, :, 0:1]
                                    batch_y = batch_y[:, -args.pred_len:, :]
                            else:
                                outputs = outputs[:, :, 0:1]
                                batch_y = batch_y[:, -args.pred_len:, :]
                        except Exception as e:
                            print(f"Error processing target: {e}")
                            outputs = outputs[:, :, 0:1]
                            batch_y = batch_y[:, -args.pred_len:, :]
                    else:
                        # For univariate case
                        batch_y = batch_y[:, -args.pred_len:, :]
                        
                # Ensure matching shapes for loss calculation
                if batch_y.shape != outputs.shape:
                    # If pred_len dimensions don't match
                    if outputs.shape[1] != batch_y.shape[1]:
                        min_len = min(outputs.shape[1], batch_y.shape[1])
                        outputs = outputs[:, :min_len, :]
                        batch_y = batch_y[:, :min_len, :]
                    
                    # If feature dimensions don't match
                    if outputs.shape[2] != batch_y.shape[2]:
                        if batch_y.shape[2] == 0:
                            batch_y = torch.zeros_like(outputs)
                        elif outputs.shape[2] == 0:
                            outputs = torch.zeros_like(batch_y)
                        else:
                            min_feat = min(outputs.shape[2], batch_y.shape[2])
                            outputs = outputs[:, :, :min_feat]
                            batch_y = batch_y[:, :, :min_feat]
                
                # Calculate loss
                loss = criterion(outputs, batch_y)
            
            # Unified backward pass using our gradient scaler
            # This works for both AMP on CUDA and regular training on MPS
            optimizer.zero_grad()
            grad_scaler.scale(loss).backward()
            
            # Apply gradient clipping to unscaled gradients if needed
            if args.grad_clip > 0:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            # Step and update using unified scaler interface
            grad_scaler.step(optimizer)
            grad_scaler.update()
            
            train_loss.append(loss.item())
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        # Calculate average loss
        train_loss = np.average(train_loss)
        train_losses.append(train_loss)
        
        # Testing
        model.eval()
        test_loss = []
        
        with torch.no_grad():
            test_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{args.train_epochs} [Test]")
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_bar:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)
                
                # Forward pass
                outputs = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                
                # Debug shapes (commented out to reduce output)
                # print(f"Test batch shapes - x: {batch_x.shape}, y: {batch_y.shape}, outputs: {outputs.shape}")
                
                # Fix zero-dimension target tensors and shape mismatches
                if batch_y.shape[-1] == 0:
                    if random.random() < 0.01:  # Only print occasionally
                        print("Warning: Zero-dimension target tensor in test loop, creating synthetic target")
                    # Create a matching target tensor with the same shape as outputs
                    batch_y = torch.zeros_like(outputs)
                else:
                    # Regular case, extract the target feature
                    if args.features == 'MS':
                        try:
                            if hasattr(test_dataset, 'target'):
                                target_name = test_dataset.target
                                # Only print occasionally to reduce verbosity
                                if random.random() < 0.01:
                                    print(f"Using test target column: {target_name}")
                                
                                if hasattr(test_dataset, 'df_data') and target_name in test_dataset.df_data.columns:
                                    target_idx = test_dataset.df_data.columns.get_loc(target_name)
                                    # Only print occasionally to reduce verbosity
                                    if random.random() < 0.01:
                                        print(f"Found test target {target_name} at index {target_idx}")
                                    
                                    # Extract target dimension from outputs
                                    if outputs.shape[2] > target_idx:
                                        outputs = outputs[:, :, target_idx:target_idx+1]
                                    else:
                                        # If target_idx is out of bounds, use first dimension
                                        if random.random() < 0.01:  # Only print occasionally
                                            print(f"Target index {target_idx} out of bounds for test outputs with shape {outputs.shape}, using first dimension")
                                        outputs = outputs[:, :, 0:1]
                                    
                                    # Extract target dimension from batch_y
                                    if batch_y.shape[2] > target_idx:
                                        batch_y = batch_y[:, -args.pred_len:, target_idx:target_idx+1]
                                    else:
                                        # If target_idx is out of bounds, use all dimensions
                                        if random.random() < 0.01:  # Only print occasionally
                                            print(f"Target index {target_idx} out of bounds for test batch_y with shape {batch_y.shape}, using all dimensions")
                                        batch_y = batch_y[:, -args.pred_len:, :]
                                else:
                                    if random.random() < 0.01:  # Only print occasionally
                                        print("Target column not found in test dataset, using first dimension")
                                    outputs = outputs[:, :, 0:1]
                                    batch_y = batch_y[:, -args.pred_len:, :]
                            else:
                                if random.random() < 0.01:  # Only print occasionally
                                    print("No target attribute in test dataset, using default processing")
                                if outputs.shape[2] > 0:
                                    outputs = outputs[:, :, 0:1]
                                batch_y = batch_y[:, -args.pred_len:, :]
                        except Exception as e:
                            print(f"Error processing test target: {e}")
                            # If there's an error, use the first dimension
                            if outputs.shape[2] > 0:
                                outputs = outputs[:, :, 0:1]
                            batch_y = batch_y[:, -args.pred_len:, :]
                    else:
                        # For univariate case
                        batch_y = batch_y[:, -args.pred_len:, :]
                
                # Ensure matching shapes for loss calculation
                if batch_y.shape != outputs.shape:
                    if random.random() < 0.01:  # Only print occasionally
                        print(f"Test shape mismatch - outputs: {outputs.shape}, targets: {batch_y.shape}, adjusting...")
                    
                    # If pred_len dimensions don't match
                    if outputs.shape[1] != batch_y.shape[1]:
                        min_len = min(outputs.shape[1], batch_y.shape[1])
                        outputs = outputs[:, :min_len, :]
                        batch_y = batch_y[:, :min_len, :]
                    
                    # If feature dimensions don't match
                    if outputs.shape[2] != batch_y.shape[2]:
                        if batch_y.shape[2] == 0:
                            # If batch_y has zero feature dimension, create matching tensor
                            batch_y = torch.zeros_like(outputs)
                        elif outputs.shape[2] == 0:
                            # If outputs has zero feature dimension, create matching tensor
                            outputs = torch.zeros_like(batch_y)
                        else:
                            # Use minimum feature dimension
                            min_feat = min(outputs.shape[2], batch_y.shape[2])
                            outputs = outputs[:, :, :min_feat]
                            batch_y = batch_y[:, :, :min_feat]
                
                # Verify final shapes are matching
                # print(f"Final test shapes for loss - outputs: {outputs.shape}, targets: {batch_y.shape}")
                
                loss = criterion(outputs, batch_y)
                
                test_loss.append(loss.item())
                test_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        # Calculate average loss
        test_loss = np.average(test_loss)
        test_losses.append(test_loss)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        
        # Early stopping check
        early_stopping(test_loss, model, model_path)
        if early_stopping.early_stop:
            if epoch >= args.min_epochs:
                print(f"Early stopping triggered after {epoch+1} epochs. Best model saved to {model_path}")
                break
            else:
                print(f"Early stopping condition met, but continuing training until minimum {args.min_epochs} epochs")
                early_stopping.early_stop = False  # Reset the flag to continue training
        
        # Adjust learning rate
        adjust_learning_rate(optimizer, epoch + 1, config)
        
    # Plot losses using improved visualization
    plot_loss_curves(train_losses, test_losses, output_dir=args.output_dir, filename='loss_curve.png')
    
    # Testing with best model
    print("Loading best model for testing...")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded best model from {model_path}")
    except FileNotFoundError:
        print(f"Could not find best model at {model_path}. Using current model state.")
        # Continue with the current model state
    model.eval()
    
    # Generate predictions
    preds = []
    trues = []
    
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc="Generating predictions")
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_bar:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            
            # Forward pass
            outputs = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
            
            # Debug shapes (commented out to reduce output)
            # print(f"Prediction batch shapes - outputs: {outputs.shape}, y: {batch_y.shape}")
            
            # Ensure batch_y has at least one feature dimension
            if batch_y.shape[-1] == 0:
                if random.random() < 0.01:  # Only print occasionally
                    print("Warning: Zero-dimension target tensor in prediction, creating synthetic target")
                batch_y = torch.zeros_like(outputs)
            
            # Find target column index
            if args.features == 'MS':
                try:
                    if hasattr(test_dataset, 'target'):
                        target_name = test_dataset.target
                        if random.random() < 0.01:  # Only print occasionally
                            print(f"Using prediction target column: {target_name}")
                        
                        if hasattr(test_dataset, 'df_data') and target_name in test_dataset.df_data.columns:
                            target_idx = test_dataset.df_data.columns.get_loc(target_name)
                            
                            # Extract target dimension from outputs
                            if outputs.shape[2] > target_idx:
                                outputs = outputs[:, :, target_idx:target_idx+1]
                            else:
                                # If target_idx is out of bounds, use first dimension
                                outputs = outputs[:, :, 0:1]
                            
                            # Extract target dimension from batch_y
                            if batch_y.shape[2] > target_idx:
                                batch_y = batch_y[:, -args.pred_len:, target_idx:target_idx+1]
                            else:
                                # If target_idx is out of bounds or has zero dimensions, match the outputs shape
                                if batch_y.shape[2] == 0:
                                    batch_y = torch.zeros_like(outputs)
                                else:
                                    batch_y = batch_y[:, -args.pred_len:, :]
                        else:
                            # If target not found, use first column
                            outputs = outputs[:, :, 0:1]
                            if batch_y.shape[2] > 0:
                                batch_y = batch_y[:, -args.pred_len:, 0:1]
                            else:
                                batch_y = torch.zeros_like(outputs)
                    else:
                        # No target attribute, use first dimension
                        outputs = outputs[:, :, 0:1]
                        if batch_y.shape[2] > 0:
                            batch_y = batch_y[:, -args.pred_len:, 0:1]
                        else:
                            batch_y = torch.zeros_like(outputs)
                except Exception as e:
                    print(f"Error processing prediction target: {e}")
                    # Default to first dimension
                    outputs = outputs[:, :, 0:1]
                    if batch_y.shape[2] > 0:
                        batch_y = batch_y[:, -args.pred_len:, 0:1]
                    else:
                        batch_y = torch.zeros_like(outputs)
            else:
                # Univariate case
                if batch_y.shape[2] > 0:
                    batch_y = batch_y[:, -args.pred_len:, :]
                else:
                    batch_y = torch.zeros_like(outputs)
            
            # Final shape check
            # Only print this for some batches to avoid excessive output
            if random.random() < 0.01:  # Only print for ~1% of batches
                print(f"Final prediction shapes - outputs: {outputs.shape}, batch_y: {batch_y.shape}")
            
            # Ensure both tensors have at least one feature dimension
            if outputs.shape[2] == 0:
                outputs = torch.zeros(outputs.shape[0], outputs.shape[1], 1, device=outputs.device)
            
            if batch_y.shape[2] == 0:
                batch_y = torch.zeros(batch_y.shape[0], batch_y.shape[1], 1, device=batch_y.device)
            
            # Store predictions and true values
            pred = outputs.detach().cpu().numpy()
            true = batch_y.detach().cpu().numpy()
            
            # Only append if dimensions are not zero
            if pred.shape[0] > 0 and pred.shape[1] > 0 and pred.shape[2] > 0:
                preds.append(pred)
            else:
                print(f"Skipping batch with invalid prediction shape: {pred.shape}")
                
            if true.shape[0] > 0 and true.shape[1] > 0 and true.shape[2] > 0:
                trues.append(true)
            else:
                print(f"Skipping batch with invalid true value shape: {true.shape}")
    
    # Check if lists are empty
    if not preds or not trues:
        print("Error: No valid prediction or true value tensors to concatenate")
        # Create dummy data for visualization
        dummy_shape = (1, args.pred_len, 1)
        preds = [np.zeros(dummy_shape)]
        trues = [np.zeros(dummy_shape)]
    
    # Concatenate predictions and true values
    try:
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
    except ValueError as e:
        print(f"Error during concatenation: {e}")
        print("Tensor shapes in preds:", [p.shape for p in preds])
        print("Tensor shapes in trues:", [t.shape for t in trues])
        
        # Filter out any problematic tensors (those with different shapes)
        if len(preds) > 1:
            target_shape = preds[0].shape[1:]
            filtered_preds = [p for p in preds if p.shape[1:] == target_shape]
            preds = np.concatenate(filtered_preds, axis=0) if filtered_preds else np.zeros((1, *target_shape))
        else:
            preds = preds[0] if preds else np.zeros((1, args.pred_len, 1))
            
        if len(trues) > 1:
            target_shape = trues[0].shape[1:]
            filtered_trues = [t for t in trues if t.shape[1:] == target_shape]
            trues = np.concatenate(filtered_trues, axis=0) if filtered_trues else np.zeros((1, *target_shape))
        else:
            trues = trues[0] if trues else np.zeros((1, args.pred_len, 1))
    
    # Final check before saving - ensure both have valid shapes
    print(f"Model completed prediction with final shapes - preds: {preds.shape}, trues: {trues.shape}")
    
    # Calculate and print evaluation metrics
    print("Calculating evaluation metrics...")
    
    # Reshape data if needed for metrics calculation
    if preds.ndim == 3 and trues.ndim == 3:
        # For multivariate case, we typically care about the target variable (first feature)
        eval_preds = preds[:, :, 0] if preds.shape[2] > 0 else preds.reshape(preds.shape[0], preds.shape[1])
        eval_trues = trues[:, :, 0] if trues.shape[2] > 0 else trues.reshape(trues.shape[0], trues.shape[1])
    else:
        # Already in the right shape
        eval_preds = preds
        eval_trues = trues
    
    # Compute metrics
    metrics = evaluate_forecasting(eval_trues, eval_preds)
    
    # Print and save metrics
    print("\nEvaluation Metrics:")
    metrics_df = pd.DataFrame({
        'Metric': list(metrics.keys()),
        'Value': list(metrics.values())
    })
    print(metrics_df.to_string(index=False))
    
    # Save metrics to file
    metrics_df.to_csv(os.path.join(args.output_dir, 'metrics.csv'), index=False)
    
    # Save detailed metrics in JSON format
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Save predictions only if they have valid shapes
    try:
        np.save(os.path.join(args.output_dir, 'predictions.npy'), preds)
        np.save(os.path.join(args.output_dir, 'true_values.npy'), trues)
        print(f"Successfully saved prediction and true value arrays to {args.output_dir}")
    except Exception as e:
        print(f"Error saving prediction arrays: {e}")
    
    # Visualize predictions for samples
    try:
        # Use our improved visualization function
        print("Generating prediction visualizations...")
        plot_predictions(
            trues, 
            preds, 
            sample_indices=list(range(min(5, preds.shape[0]))), 
            output_dir=args.output_dir,
            title_prefix="Sample"
        )
        
        # Also create a visualization that shows all metrics for a few samples
        for i in range(min(3, preds.shape[0])):
            if preds.shape[2] > 0 and trues.shape[2] > 0:
                sample_pred = preds[i, :, 0]
                sample_true = trues[i, :, 0]
            else:
                sample_pred = preds[i]
                sample_true = trues[i]
                
            # Calculate metrics for this sample
            sample_metrics = evaluate_forecasting(sample_true, sample_pred)
            
            # Create a more detailed visualization
            plt.figure(figsize=(12, 8))
            
            # Plot the predictions
            plt.subplot(2, 1, 1)
            plt.plot(sample_true, label='True', linewidth=2)
            plt.plot(sample_pred, label='Predicted', linewidth=2, linestyle='--')
            plt.title(f'Sample {i+1} Prediction')
            plt.xlabel('Time Steps')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Plot the metrics as a table
            plt.subplot(2, 1, 2)
            plt.axis('off')
            metric_text = '\n'.join([f"{m}: {v:.4f}" for m, v in sample_metrics.items()])
            plt.text(0.5, 0.5, f"Metrics:\n{metric_text}", 
                    ha='center', va='center', fontsize=12, 
                    bbox=dict(boxstyle='round,pad=1', facecolor='aliceblue', alpha=0.5))
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, f'detailed_sample_{i+1}.png'))
            plt.close()
            
    except Exception as e:
        print(f"Error generating prediction visualizations: {e}")
        # Create a simple error visualization
        plt.figure(figsize=(10, 5))
        plt.text(0.5, 0.5, f"Visualization error: {e}", 
                horizontalalignment='center', verticalalignment='center')
        plt.savefig(os.path.join(args.output_dir, 'visualization_error.png'))
    
    print(f"Training completed. Results saved to {args.output_dir}")
    
    # Run performance benchmark if on Mac and requested
    if device.type == 'mps' and args.benchmark:
        print("\nRunning MPS performance benchmark...")
        # Create a sample batch for benchmarking
        input_shape = (args.batch_size, args.seq_len, args.enc_in)
        
        # Use advanced benchmarking if available
        if has_advanced_mps:
            print("Using advanced MPS benchmarking...")
            benchmark_results = benchmark_mps_performance_advanced(
                model, 
                input_shape, 
                iterations=50,
                warmup=10,
                full_report=True
            )
        else:
            # Fall back to basic benchmarking
            benchmark_results = benchmark_mps_performance(
                model, 
                input_shape, 
                iterations=50,
                warmup=10
            )
        
        # Save benchmark results to file
        benchmark_path = os.path.join(args.output_dir, 'mps_benchmark.json')
        with open(benchmark_path, 'w') as f:
            json.dump(benchmark_results, f, indent=4)
        print(f"Benchmark results saved to {benchmark_path}")
        
        # Create benchmark visualization
        try:
            # Create a simple bar chart for key metrics
            plt.figure(figsize=(12, 8))
            
            # Training speed
            plt.subplot(2, 2, 1)
            plt.bar(['Training Speed'], [benchmark_results['training']['samples_per_second']], color='blue')
            plt.ylabel('Samples/second')
            plt.title('Training Throughput')
            
            # Inference speed
            plt.subplot(2, 2, 2)
            plt.bar(['Inference Speed'], [benchmark_results['inference']['samples_per_second']], color='green')
            plt.ylabel('Samples/second')
            plt.title('Inference Throughput')
            
            # Memory usage
            plt.subplot(2, 2, 3)
            memory_data = [
                benchmark_results['memory']['input_memory_mb'],
                benchmark_results['memory']['inference_memory_mb'],
                benchmark_results['memory']['training_memory_mb']
            ]
            plt.bar(['Input', 'Inference', 'Training'], memory_data, color=['gray', 'green', 'blue'])
            plt.ylabel('Memory (MB)')
            plt.title('Memory Usage')
            
            # Model info
            plt.subplot(2, 2, 4)
            plt.axis('off')
            info_text = (
                f"Device: {benchmark_results['device']}\n"
                f"Chip: {benchmark_results['chip']}\n"
                f"Model size: {benchmark_results['model_size']} parameters\n"
                f"Batch size: {args.batch_size}\n"
                f"Sequence length: {args.seq_len}\n"
            )
            if 'performance_rating' in benchmark_results['inference']:
                info_text += f"Performance rating: {benchmark_results['inference']['performance_rating']}\n"
                
            plt.text(0.1, 0.5, info_text, fontsize=10)
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, 'mps_benchmark.png'))
            print(f"Benchmark visualization saved to {os.path.join(args.output_dir, 'mps_benchmark.png')}")
        except Exception as e:
            print(f"Error creating benchmark visualization: {e}")

def adjust_learning_rate(optimizer, epoch, args):
    # Original type1 (default)
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.75 ** ((epoch - 1) // 5))}
    # Slower decay
    elif args.lradj == 'slower':
        lr_adjust = {epoch: args.learning_rate * (0.9 ** ((epoch - 1) // 8))}
    # Original type2
    elif args.lradj == 'type2':
        lr_adjust = {
            epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))
        }
    # Original cosine
    elif args.lradj == 'cosine':
        lr_adjust = {epoch: args.learning_rate * 0.5 * (
                1 + math.cos(math.pi * (epoch - 1) / args.train_epochs))}
    # No adjustment
    else:
        lr_adjust = {epoch: args.learning_rate}
        
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to', lr)

if __name__ == "__main__":
    main()