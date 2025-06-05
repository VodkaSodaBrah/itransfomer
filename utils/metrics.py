"""
Time series forecasting evaluation metrics.
This module provides a comprehensive set of metrics for evaluating time series forecasting models.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import warnings

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
    # Handle zeros in y_true to avoid division by zero
    mask = np.abs(y_true) > epsilon
    if not np.any(mask):
        return np.nan  # All values are too close to zero
    
    # Use sklearn's implementation with a fallback for potential issues
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore sklearn's warning about division by zero
            return mean_absolute_percentage_error(
                y_true[mask], y_pred[mask]
            ) * 100  # Convert to percentage
    except:
        # Manual calculation as a fallback
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calculate_smape(y_true, y_pred, epsilon=1e-8):
    """Symmetric Mean Absolute Percentage Error (SMAPE)"""
    denominator = np.abs(y_true) + np.abs(y_pred) + epsilon
    return 200 * np.mean(np.abs(y_pred - y_true) / denominator)

def calculate_directional_accuracy(y_true, y_pred):
    """Directional Accuracy (DA) - percentage of times the direction of change is correctly predicted"""
    # Calculate the direction of change
    true_diff = np.diff(y_true, axis=0)
    pred_diff = np.diff(y_pred, axis=0)
    
    # Get the sign of the direction (up=positive, down=negative)
    true_direction = np.sign(true_diff)
    pred_direction = np.sign(pred_diff)
    
    # Calculate how many times the direction matches
    correct_direction = (true_direction == pred_direction).astype(int)
    
    # Return the percentage of correct directions
    return np.mean(correct_direction) * 100

def calculate_wape(y_true, y_pred):
    """Weighted Absolute Percentage Error (WAPE)"""
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100

def evaluate_forecasting(y_true, y_pred, metrics=None):
    """
    Evaluate time series forecasting predictions using multiple metrics.
    
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
