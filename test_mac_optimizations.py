#!/usr/bin/env python
"""
This is a simplified version of direct_run.py focusing on demonstrating the Mac optimizations.
"""
import os
import sys
import torch
import numpy as np
import random
import argparse
import time

print("Script starting...")

# Add the project root to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print(f"Current directory: {current_dir}")
print(f"Files in utils: {os.listdir(os.path.join(current_dir, 'utils'))}")
print(f"sys.path: {sys.path}")

try:
    # Import Mac optimizations
    from utils.mac_optimizations import (
        configure_for_mac_performance, get_mac_performance_tips, 
        mps_autocast, MPSGradScaler, get_amp_context_manager, get_grad_scaler,
        optimize_for_mps_inference, benchmark_mps_performance
    )
    print("Successfully imported Mac optimizations")
except ImportError as e:
    print(f"Error importing Mac optimizations: {e}")
    sys.exit(1)

def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_device(device=None):
    """Get appropriate compute device"""
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

def main():
    # Simple argument parser
    parser = argparse.ArgumentParser(description='Test Mac optimizations')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default=None, help='Device: mps, cuda, cpu, or None for auto-detect')
    parser.add_argument('--optimize_for_mps', action='store_true', help='Apply optimizations for MPS')
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping threshold')
    parser.add_argument('--train_epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--lradj', type=str, default='type1', help='Learning rate adjustment strategy')
    args = parser.parse_args()
    
    # Set random seed
    set_seed(2023)
    
    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Apply Mac-specific optimizations
    if device.type == 'mps':
        original_args = vars(args).copy()
        args = configure_for_mac_performance(args, device_type='mps')
        
        # Print changes made by configure_for_mac_performance
        print("\nChanges made by Mac optimization:")
        for key, new_value in vars(args).items():
            if key in original_args and original_args[key] != new_value:
                print(f"  {key}: {original_args[key]} -> {new_value}")
        
        # Display Mac performance tips
        mac_tips = get_mac_performance_tips()
        if mac_tips:
            print("\nMac Performance Tips:")
            for tip in mac_tips:
                print(f"  {tip}")
    
    # Create a simple model for testing
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(10, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 1)
            )
            
        def forward(self, x):
            return self.layers(x)
    
    # Create and move model to device
    model = SimpleModel().to(device)
    print(f"Created simple model: {model}")
    
    # Apply torch.compile if optimizing for MPS
    if args.optimize_for_mps and device.type == 'mps':
        try:
            model = optimize_for_mps_inference(model, verbose=True)
            print("MPS optimization applied successfully.")
        except Exception as e:
            print(f"MPS optimization failed: {e}. Proceeding without optimization.")
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.MSELoss()
    
    # Set up context managers directly
    if device.type == 'cuda' and torch.cuda.is_available() and hasattr(torch.cuda, 'amp'):
        print("Using CUDA with Automatic Mixed Precision")
        autocast_context = torch.cuda.amp.autocast()
        grad_scaler = torch.cuda.amp.GradScaler()
    else:
        print("Using MPS-specific optimization with compatible interfaces")
        autocast_context = mps_autocast()
        grad_scaler = MPSGradScaler()
    
    # Simulate training
    print("\n--- Starting training simulation ---")
    for epoch in range(args.train_epochs):
        model.train()
        train_losses = []
        
        # Simulate batches
        for i in range(10):
            # Create fake data
            batch_x = torch.randn(args.batch_size, 10, device=device)
            batch_y = torch.randn(args.batch_size, 1, device=device)
            
            # Forward pass with unified approach for both CUDA and MPS
            try:
                print(f"Using autocast_context: {type(autocast_context)}")
                with autocast_context:
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                print(f"Forward pass successful, loss={loss.item()}")
            except Exception as e:
                print(f"Error in forward pass: {e}")
                import traceback
                traceback.print_exc()
                break
            
            # Unified backward pass using our gradient scaler
            optimizer.zero_grad()
            grad_scaler.scale(loss).backward()
            
            # Apply gradient clipping if needed
            if args.grad_clip > 0:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            # Step and update using unified scaler interface
            grad_scaler.step(optimizer)
            grad_scaler.update()
            
            train_losses.append(loss.item())
            print(f"Epoch {epoch+1}, Batch {i+1}: Loss = {loss.item():.4f}")
            
        print(f"Epoch {epoch+1} completed. Average loss: {np.mean(train_losses):.4f}")
    
    # Run performance benchmark if on Mac
    if device.type == 'mps':
        print("\nRunning MPS performance benchmark...")
        # Create a sample batch for benchmarking
        input_shape = (args.batch_size, 10)
        
        # Define benchmark function for this simple model
        def benchmark_simple_model(model, input_shape, iterations=50, warmup=10):
            """Simple benchmark for the model"""
            device = next(model.parameters()).device
            model.eval()
            
            # Create dummy input
            dummy_input = torch.randn(*input_shape, device=device)
            
            # Warmup
            for _ in range(warmup):
                with torch.no_grad():
                    _ = model(dummy_input)
            
            # Benchmark
            torch.cuda.synchronize() if device.type == 'cuda' else torch.mps.synchronize() if device.type == 'mps' else None
            start_time = time.time()
            
            for _ in range(iterations):
                with torch.no_grad():
                    _ = model(dummy_input)
            
            torch.cuda.synchronize() if device.type == 'cuda' else torch.mps.synchronize() if device.type == 'mps' else None
            end_time = time.time()
            
            # Calculate metrics
            total_time = end_time - start_time
            avg_time = total_time / iterations
            throughput = iterations / total_time
            
            return {
                'total_time_ms': total_time * 1000,
                'avg_inference_time_ms': avg_time * 1000,
                'throughput_inferences_per_sec': throughput,
                'device': str(device),
                'batch_size': input_shape[0],
                'iterations': iterations
            }
        
        # Run benchmark
        benchmark_results = benchmark_simple_model(model, input_shape)
        
        # Print benchmark results
        print("\nMPS Performance Benchmark Results:")
        for key, value in benchmark_results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
