"""
Mac-specific optimization guide for the time series forecasting model.
For running on Apple M-series chips (MPS backend).
"""

import os
import sys
import torch
import platform
import time
import subprocess
import contextlib

# Check if running on macOS
is_mac = platform.system() == 'Darwin'

# Get Mac chip information if available
def get_mac_chip_info():
    """Get information about the Mac's M-series chip if available"""
    if not is_mac:
        return "Not macOS"
    
    try:
        # Use sysctl to get chip information
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], 
            capture_output=True, text=True, check=True
        )
        chip_info = result.stdout.strip()
        
        # Determine if it's M1, M2, or M3 series
        if "Apple M1" in chip_info:
            return "Apple M1 series"
        elif "Apple M2" in chip_info:
            return "Apple M2 series"
        elif "Apple M3" in chip_info:
            return "Apple M3 series"
        else:
            return chip_info
    except Exception:
        return "Unknown Mac chip"

def configure_for_mac_performance(args, device_type):
    """
    Configure performance settings specifically for Mac/MPS.
    
    Parameters:
    -----------
    args : argparse.Namespace
        The command-line arguments
    device_type : str
        The device type ('mps', 'cuda', or 'cpu')
        
    Returns:
    --------
    args : argparse.Namespace
        Modified args with Mac-specific optimizations
    """
    if not is_mac or device_type != 'mps':
        return args
    
    chip_info = get_mac_chip_info()
    print(f"Configuring Mac-specific performance optimizations for {chip_info}")
    
    # Batch size optimizations based on Mac chip
    optimal_batch_size = 512  # Default for newer chips
    
    if "M1" in chip_info and "Pro" not in chip_info and "Max" not in chip_info:
        # Base M1 has less RAM, use smaller batch size
        optimal_batch_size = 256
    
    if args.batch_size < optimal_batch_size:
        original_batch_size = args.batch_size
        args.batch_size = optimal_batch_size
        print(f"  - Increased batch size from {original_batch_size} to {args.batch_size} for better MPS performance")
    
    # Learning rate stability for MPS
    if args.learning_rate > 0.001:
        original_lr = args.learning_rate
        args.learning_rate = 0.001
        print(f"  - Adjusted learning rate from {original_lr} to {args.learning_rate} for MPS stability")
    
    # Thread settings for MPS
    torch.set_num_threads(1)
    print(f"  - Set PyTorch threads to 1 for better MPS coordination")
    
    # Enable float16 computations if available (newer PyTorch versions)
    if hasattr(torch.backends.mps, 'enable_float16_computations'):
        torch.backends.mps.enable_float16_computations(True)
        print("  - Enabled float16 computations for MPS acceleration")
    
    # Disable AMP as it's not supported on MPS
    if args.use_amp:
        args.use_amp = False
        print("  - Disabled AMP (not supported on MPS) - using MPS-specific optimizations instead")
    
    # Enable MPS-specific optimizations
    args.optimize_for_mps = True
    
    # Set gradient clipping for stability
    if args.grad_clip == 0.0:
        args.grad_clip = 1.0
        print("  - Enabled gradient clipping (1.0) for training stability")
    
    # Recommend specific learning rate scheduler for MPS
    if args.lradj != 'cosine':
        print("  - Recommendation: Consider using --lradj cosine for better convergence on MPS")
    
    # Memory optimization recommendations
    print("  - Recommendation: Close other intensive applications when training")
    
    return args

class MPSAutocastContext:
    """
    A context manager class that mimics torch.amp.autocast for MPS devices.
    
    While MPS doesn't support true Automatic Mixed Precision (AMP), this provides
    a compatible interface so code can remain consistent between devices.
    
    Usage:
    ```
    with MPSAutocastContext():
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    ```
    """
    def __init__(self, enabled=True, dtype=torch.float16, cache_enabled=True):
        """
        Initialize the MPS autocast context with parameters matching torch.cuda.amp.autocast
        
        Args:
            enabled (bool): Whether autocast is enabled
            dtype (torch.dtype): The desired dtype (ignored on MPS, included for API compatibility)
            cache_enabled (bool): Whether to cache autocast operations (ignored on MPS)
        """
        self.enabled = enabled
        self.dtype = dtype
        self.cache_enabled = cache_enabled
        self._previous_dtype = None
        
    def __enter__(self):
        # While MPS doesn't support true autocast, we can at least try to use float16
        # if explicitly available in newer PyTorch versions
        if self.enabled and hasattr(torch.backends.mps, 'enable_float16_computations'):
            # Only attempt to enable float16 on MPS if the autocast was requested with float16
            if self.dtype == torch.float16:
                try:
                    # Store previous state to restore it later
                    self._previous_dtype = torch.backends.mps.is_float16_computations_enabled()
                    # Enable float16 computations
                    torch.backends.mps.enable_float16_computations(True)
                except Exception:
                    # Silently continue if this fails
                    pass
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous state if we changed it
        if self.enabled and self._previous_dtype is not None and hasattr(torch.backends.mps, 'enable_float16_computations'):
            try:
                torch.backends.mps.enable_float16_computations(self._previous_dtype)
            except Exception:
                # Silently continue if this fails
                pass

def mps_autocast():
    """
    Creates an autocast context for MPS devices.
    
    Returns an instance of MPSAutocastContext.
    """
    return MPSAutocastContext()

class MPSGradScaler:
    """
    A stub class that mimics torch.cuda.amp.GradScaler for MPS devices.
    
    This provides a compatible interface so code can remain consistent 
    between different devices (CUDA vs MPS).
    
    Usage:
    ```
    scaler = MPSGradScaler()
    
    # In training loop:
    with mps_autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    ```
    """
    
    def __init__(self):
        pass
        
    def scale(self, loss):
        """Return the loss unchanged for MPS devices"""
        return loss
        
    def unscale_(self, optimizer):
        """No-op for MPS devices"""
        pass
        
    def step(self, optimizer):
        """Just call optimizer.step() for MPS devices"""
        optimizer.step()
        
    def update(self):
        """No-op for MPS devices"""
        pass

def get_amp_context_manager(device):
    """
    Get the appropriate autocast context manager based on device type.
    
    Parameters:
    -----------
    device : torch.device
        The device being used
        
    Returns:
    --------
    context_manager : context manager
        Either torch.cuda.amp.autocast() or MPSAutocastContext()
    """
    if device.type == 'cuda' and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
        return torch.cuda.amp.autocast()
    else:
        return MPSAutocastContext()

def get_grad_scaler(device):
    """
    Get the appropriate gradient scaler based on device type.
    
    Parameters:
    -----------
    device : torch.device
        The device being used
        
    Returns:
    --------
    scaler : GradScaler or MPSGradScaler
        Either torch.cuda.amp.GradScaler or MPSGradScaler
    """
    if device.type == 'cuda' and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'GradScaler'):
        return torch.cuda.amp.GradScaler()
    else:
        return MPSGradScaler()

def optimize_for_mps_inference(model, verbose=True):
    """
    Apply MPS-specific optimizations for inference
    
    Parameters:
    -----------
    model : torch.nn.Module
        The PyTorch model to optimize
    verbose : bool
        Whether to print information about optimizations
        
    Returns:
    --------
    model : torch.nn.Module
        Optimized model
    """
    if not is_mac or not torch.backends.mps.is_available():
        return model
    
    # First check if the PYTORCH_ENABLE_MPS_FALLBACK environment variable is set
    if 'PYTORCH_ENABLE_MPS_FALLBACK' not in os.environ:
        if verbose:
            print("Setting PYTORCH_ENABLE_MPS_FALLBACK=1 to handle unsupported MPS operations")
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    # Get PyTorch version to apply version-specific optimizations
    pytorch_version = torch.__version__
    
    if verbose:
        print(f"Optimizing for MPS inference on PyTorch {pytorch_version}")
    
    # Apply PyTorch version-specific optimizations
    if hasattr(torch.backends.mps, 'enable_float16_computations'):
        torch.backends.mps.enable_float16_computations(True)
        if verbose:
            print("Enabled float16 computations for MPS acceleration")
    
    # Detect Mac chip type for specific optimizations
    chip_info = get_mac_chip_info()
    
    # Different compilation strategies based on Mac chip and PyTorch version
    if hasattr(torch, 'compile'):
        try:
            if verbose:
                print(f"Applying torch.compile for MPS inference acceleration on {chip_info}...")
            
            # Configure compilation options for MPS
            # Different backends work better for different model types and PyTorch versions
            compilation_backend = "aot_eager"  # Default backend
            
            # For newer PyTorch versions and newer chips, inductor can be faster
            if pytorch_version >= '2.1.0' and any(chip in chip_info for chip in ["M2", "M3"]):
                compilation_backend = "inductor"
            
            # For time series models, we've found these options work well on MPS
            compilation_options = {
                "backend": compilation_backend,
                "mode": "reduce-overhead", # Optimizes for inference
                "fullgraph": False,  # Partial graph compilation works better with MPS
                "options": {
                    "allow_rnn": True,  # Allow RNN operations to be compiled
                    "allow_dropout": True,  # More stable with dropout allowed
                }
            }
            
            if verbose:
                print(f"Using compilation backend: {compilation_backend}")
                
            # Apply compilation
            compiled_model = torch.compile(model, **compilation_options)
            
            if verbose:
                print("Model compilation successful")
            
            return compiled_model
        except Exception as e:
            if verbose:
                print(f"torch.compile failed on MPS: {e}")
                print("This is normal on some PyTorch versions or with certain model architectures.")
                print("Proceeding without compilation.")
    
    # If compilation failed or isn't available, apply other optimizations
    if verbose:
        print("Applying non-compilation MPS optimizations")
    
    # Set MPS-specific environment variables
    if 'PYTORCH_MPS_HIGH_WATERMARK_RATIO' not in os.environ:
        # Allow MPS to use more memory when needed
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.8'
        if verbose:
            print("Set PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8 for better memory utilization")
    
    return model

# Add a function to check for MPS-compatibility issues
def check_mps_compatibility(model, sample_input=None, verbose=True):
    """
    Check if a model has operations that might not be supported by MPS.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The PyTorch model to check
    sample_input : tuple or None
        A sample input to trace the model (optional)
    verbose : bool
        Whether to print information about compatibility issues
        
    Returns:
    --------
    needs_fallback : bool
        Whether the model likely needs CPU fallback for some operations
    issues : list
        List of potential issues detected
    """
    if not is_mac or not torch.backends.mps.is_available():
        return False, []
    
    issues = []
    
    # Check 1: Look for known problematic operations in the model
    dropout_count = 0
    layernorm_count = 0
    embedding_count = 0
    rnn_count = 0
    
    # Count potentially problematic layers for MPS
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            dropout_count += 1
        elif isinstance(module, torch.nn.LayerNorm):
            layernorm_count += 1
        elif isinstance(module, torch.nn.Embedding):
            embedding_count += 1
        elif any(isinstance(module, rnn_type) for rnn_type in 
                [torch.nn.RNN, torch.nn.LSTM, torch.nn.GRU]):
            rnn_count += 1
            
    if dropout_count > 0:
        issues.append(f"Model contains {dropout_count} dropout layers which may need CPU fallback on MPS")
    
    if layernorm_count > 0 and torch.__version__ < '2.0.0':
        issues.append(f"Model contains {layernorm_count} LayerNorm layers which may have reduced performance on older PyTorch versions")
    
    if rnn_count > 0:
        issues.append(f"Model contains {rnn_count} RNN/LSTM/GRU layers which may have compatibility issues on MPS")
    
    # Check 2: Check for custom operations or attention mechanisms
    attention_count = 0
    for name, module in model.named_modules():
        if 'attention' in name.lower() or hasattr(module, 'attention'):
            attention_count += 1
    
    if attention_count > 0:
        issues.append(f"Model contains {attention_count} attention mechanisms which may need performance tuning on MPS")
    
    # Check 3: Model size check
    param_count = sum(p.numel() for p in model.parameters())
    param_size_mb = param_count * 4 / (1024 * 1024)  # Assuming float32 parameters (4 bytes each)
    
    # Get Mac chip information to provide tailored advice
    chip_info = get_mac_chip_info()
    
    # Determine recommended maximum model size based on chip
    max_recommended_size_mb = 1000  # Default for base M1
    if "M1 Pro" in chip_info or "M1 Max" in chip_info:
        max_recommended_size_mb = 4000
    elif "M2" in chip_info and not any(x in chip_info for x in ["Pro", "Max", "Ultra"]):
        max_recommended_size_mb = 2000
    elif "M2 Pro" in chip_info or "M2 Max" in chip_info:
        max_recommended_size_mb = 6000
    elif "M3" in chip_info:
        max_recommended_size_mb = 8000
    
    if param_size_mb > max_recommended_size_mb:
        issues.append(f"Model size ({param_size_mb:.1f} MB) exceeds recommended maximum ({max_recommended_size_mb:.1f} MB) for {chip_info}")
    
    # Determine if fallback is likely needed
    needs_fallback = len(issues) > 0
    
    if verbose and issues:
        print("MPS compatibility issues detected:")
        for issue in issues:
            print(f"  - {issue}")
        print("  Recommendation: Set PYTORCH_ENABLE_MPS_FALLBACK=1 for CPU fallback on unsupported operations")
        
        # Provide chip-specific advice
        if "M1" in chip_info and not any(x in chip_info for x in ["Pro", "Max", "Ultra"]):
            print("\n💡 Base M1 Chip Recommendations:")
            print("  - Consider reducing model size (--d_model 192 or lower)")
            print("  - Use batch_size 256 or lower")
            print("  - Limit sequence length to under 128")
        elif "M3" in chip_info:
            print("\n💡 M3 Chip Advantages:")
            print("  - Can use larger batch sizes (up to 1024)")
            print("  - Supports larger models (--d_model up to 512)")
            print("  - Benefits from newer PyTorch optimizations")
    
    return needs_fallback, issues

def get_mac_performance_tips():
    """
    Returns a list of performance tips for Mac users
    """
    if not is_mac:
        return []
    
    chip_info = get_mac_chip_info()
    
    tips = [
        f"To maximize performance on your Mac ({chip_info}):",
        "1. Increase batch size (256-512 works well on M1/M2 chips)",
        "2. Use cosine learning rate scheduler (--lradj cosine)",
        "3. Keep sequence lengths moderate (--seq_len 96 is a good balance)",
        "4. Use financial loss for better predictions (--loss_function financial)",
        "5. Close other GPU-intensive applications during training",
        "6. Consider increasing model width rather than depth (--d_model 192 --d_ff 768)",
        "7. For M1 Pro/Max/Ultra or M2/M3 chips, you can try larger models",
        "8. Use gradient clipping (--grad_clip 1.0) for training stability"
    ]
    
    return tips

def benchmark_mps_performance(model, input_shape, iterations=100, warmup=10):
    """
    Benchmark model performance on MPS device
    
    Parameters:
    -----------
    model : torch.nn.Module
        The PyTorch model to benchmark
    input_shape : tuple
        The shape of the input tensor (batch_size, seq_len, features)
    iterations : int
        Number of iterations to run
    warmup : int
        Number of warmup iterations
        
    Returns:
    --------
    results : dict
        Dictionary with benchmark results
    """
    if not is_mac or not torch.backends.mps.is_available():
        return {"error": "MPS not available"}
    
    device = torch.device('mps')
    model = model.to(device)
    model.eval()
    
    # Measure memory before
    memory_before = get_mps_memory_usage()
    
    # Create dummy inputs
    batch_x = torch.randn(*input_shape).to(device)
    batch_x_mark = torch.randn(input_shape[0], input_shape[1], 4).to(device)
    batch_y = torch.randn(input_shape[0], input_shape[1], input_shape[2]).to(device)
    batch_y_mark = torch.randn(input_shape[0], input_shape[1], 4).to(device)
    
    # Measure memory after input creation
    memory_after_inputs = get_mps_memory_usage()
    
    # Warm-up
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
    
    # Synchronize before timing
    torch.mps.synchronize()
    
    # Benchmark inference
    start_time = time.time()
    
    for _ in range(iterations):
        with torch.no_grad():
            _ = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
    
    # Synchronize after inference
    torch.mps.synchronize()
    end_time = time.time()
    
    # Measure memory after inference
    memory_after_inference = get_mps_memory_usage()
    
    # Benchmark training (one iteration)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    # Warm-up for training
    for _ in range(5):
        optimizer.zero_grad()
        outputs = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    # Synchronize before timing training
    torch.mps.synchronize()
    train_start_time = time.time()
    
    # Single training step benchmark
    optimizer.zero_grad()
    outputs = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
    loss = criterion(outputs, batch_y)
    loss.backward()
    optimizer.step()
    
    # Synchronize after training
    torch.mps.synchronize()
    train_end_time = time.time()
    
    # Measure memory after training
    memory_after_training = get_mps_memory_usage()
    
    # Calculate metrics
    total_inference_time = end_time - start_time
    avg_inference_time = total_inference_time / iterations
    training_step_time = train_end_time - train_start_time
    
    # Memory deltas
    input_memory_usage = memory_after_inputs["used"] - memory_before["used"]
    inference_memory_usage = memory_after_inference["used"] - memory_after_inputs["used"]
    training_memory_usage = memory_after_training["used"] - memory_after_inference["used"]
    
    return {
        "device": str(device),
        "chip": get_mac_chip_info(),
        "model_size": sum(p.numel() for p in model.parameters()),
        "inference": {
            "iterations": iterations,
            "total_time_sec": total_inference_time,
            "avg_time_per_batch_sec": avg_inference_time,
            "batches_per_second": 1.0 / avg_inference_time,
            "samples_per_second": input_shape[0] / avg_inference_time
        },
        "training": {
            "step_time_sec": training_step_time,
            "batches_per_second": 1.0 / training_step_time,
            "samples_per_second": input_shape[0] / training_step_time
        },
        "memory": {
            "input_memory_mb": input_memory_usage / (1024 * 1024),
            "inference_memory_mb": inference_memory_usage / (1024 * 1024),
            "training_memory_mb": training_memory_usage / (1024 * 1024),
            "total_used_mb": memory_after_training["used"] / (1024 * 1024),
            "total_available_mb": memory_after_training["total"] / (1024 * 1024)
        },
        "input_shape": input_shape
    }

def get_mps_memory_usage():
    """Get MPS memory usage information if available"""
    try:
        # This is a simple approximation since PyTorch doesn't expose detailed MPS memory stats
        # We use system memory as a proxy since Metal can use a large portion of system RAM
        import psutil
        vm = psutil.virtual_memory()
        return {
            "total": vm.total,
            "used": vm.used,
            "free": vm.available,
            "percent": vm.percent
        }
    except:
        # Fallback when psutil is not available
        return {
            "total": 0,
            "used": 0,
            "free": 0,
            "percent": 0
        }

# Example usage in README or documentation:
"""
# Mac-specific performance settings:
python direct_run.py --optimize_for_mps --batch_size 512 --d_model 192 --loss_function financial --lradj cosine --grad_clip 1.0
"""
