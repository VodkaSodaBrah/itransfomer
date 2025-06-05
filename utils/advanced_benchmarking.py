import time
import os
import torch
import platform
import subprocess

# Check if running on macOS
is_mac = platform.system() == 'Darwin'

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

def benchmark_mps_performance_advanced(model, input_shape, iterations=100, warmup=10, full_report=False):
    """
    Enhanced benchmark for model performance on MPS device with detailed metrics
    
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
    full_report : bool
        Whether to include additional detailed metrics
        
    Returns:
    --------
    results : dict
        Dictionary with benchmark results
    """
    import time
    try:
        import psutil
    except ImportError:
        print("psutil not found. Install with: pip install psutil")
        psutil = None
    
    if not is_mac or not torch.backends.mps.is_available():
        return {"error": "MPS not available"}
    
    device = torch.device('mps')
    model = model.to(device)
    model.eval()
    
    # Track max memory usage
    def get_max_memory():
        if psutil:
            return psutil.virtual_memory().percent
        return 0
    
    # Measure memory before
    memory_before = get_mps_memory_usage()
    
    # Record starting time for overall benchmark
    benchmark_start_time = time.time()
    
    # Create dummy inputs based on model structure
    batch_x = torch.randn(*input_shape).to(device)
    batch_x_mark = torch.randn(input_shape[0], input_shape[1], 4).to(device)
    batch_y = torch.randn(input_shape[0], input_shape[1], input_shape[2]).to(device)
    batch_y_mark = torch.randn(input_shape[0], input_shape[1], 4).to(device)
    
    # Measure memory after input creation
    memory_after_inputs = get_mps_memory_usage()
    
    # Track max memory usage for each phase
    max_memory_percent = get_max_memory()
    
    # Warm-up - important for MPS to initialize graph
    print("Performing warm-up iterations...")
    warmup_times = []
    for i in range(warmup):
        with torch.no_grad():
            warmup_start = time.time()
            _ = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
            torch.mps.synchronize()
            warmup_end = time.time()
            warmup_times.append(warmup_end - warmup_start)
            max_memory_percent = max(max_memory_percent, get_max_memory())
            if i == 0:
                print(f"  First warm-up iteration: {warmup_times[0]*1000:.2f} ms")
    
    # Average warmup time - useful to detect compilation overhead
    avg_warmup_time = sum(warmup_times) / len(warmup_times)
    first_warmup_time = warmup_times[0] if warmup_times else 0
    
    # Synchronize before timing
    torch.mps.synchronize()
    
    # Benchmark inference with per-iteration tracking
    print(f"Running {iterations} inference iterations...")
    start_time = time.time()
    inference_times = []
    
    for i in range(iterations):
        iter_start = time.time()
        with torch.no_grad():
            _ = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
        torch.mps.synchronize()
        iter_end = time.time()
        inference_times.append(iter_end - iter_start)
        max_memory_percent = max(max_memory_percent, get_max_memory())
        
        # Show progress for long benchmarks
        if iterations > 20 and i % 20 == 0:
            print(f"  Completed {i}/{iterations} iterations")
    
    # Synchronize after inference
    torch.mps.synchronize()
    end_time = time.time()
    
    # Calculate inference statistics
    inference_times_ms = [t * 1000 for t in inference_times]
    avg_inference_time = sum(inference_times) / len(inference_times)
    min_inference_time = min(inference_times)
    max_inference_time = max(inference_times)
    p50_inference_time = sorted(inference_times)[len(inference_times)//2]
    p95_inference_time = sorted(inference_times)[int(len(inference_times)*0.95)]
    
    # Measure memory after inference
    memory_after_inference = get_mps_memory_usage()
    
    # Benchmark training (one iteration)
    print("Running training benchmark...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    # Warm-up for training (shorter)
    train_warmup_times = []
    for i in range(5):
        optimizer.zero_grad()
        train_warmup_start = time.time()
        outputs = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        torch.mps.synchronize()
        train_warmup_end = time.time()
        train_warmup_times.append(train_warmup_end - train_warmup_start)
        max_memory_percent = max(max_memory_percent, get_max_memory())
    
    # Synchronize before timing training
    torch.mps.synchronize()
    
    # Train for multiple iterations to get better statistics
    train_iterations = 20
    train_times = []
    train_start_time = time.time()
    
    print(f"Running {train_iterations} training iterations...")
    for i in range(train_iterations):
        train_iter_start = time.time()
        optimizer.zero_grad()
        outputs = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        torch.mps.synchronize()
        train_iter_end = time.time()
        train_times.append(train_iter_end - train_iter_start)
        max_memory_percent = max(max_memory_percent, get_max_memory())
    
    # Synchronize after training
    torch.mps.synchronize()
    train_end_time = time.time()
    
    # Calculate training statistics
    train_times_ms = [t * 1000 for t in train_times]
    avg_train_time = sum(train_times) / len(train_times)
    min_train_time = min(train_times)
    max_train_time = max(train_times)
    p50_train_time = sorted(train_times)[len(train_times)//2]
    p95_train_time = sorted(train_times)[int(len(train_times)*0.95)]
    
    # Measure memory after training
    memory_after_training = get_mps_memory_usage()
    
    # Overall benchmark time
    benchmark_end_time = time.time()
    total_benchmark_time = benchmark_end_time - benchmark_start_time
    
    # Memory deltas
    input_memory_usage = memory_after_inputs["used"] - memory_before["used"]
    inference_memory_usage = memory_after_inference["used"] - memory_after_inputs["used"]
    training_memory_usage = memory_after_training["used"] - memory_after_inference["used"]
    
    # Chip-specific performance analysis
    chip_info = get_mac_chip_info()
    inference_rating = "Unknown"
    
    # Define some baselines for different chips (these would be based on empirical data)
    if "M1" in chip_info and not any(x in chip_info for x in ["Pro", "Max", "Ultra"]):
        if avg_inference_time < 0.005:  # Less than 5ms
            inference_rating = "Excellent"
        elif avg_inference_time < 0.010:  # Less than 10ms
            inference_rating = "Good"
        else:
            inference_rating = "Could be optimized"
    elif "M2" in chip_info:
        if avg_inference_time < 0.004:  # Less than 4ms
            inference_rating = "Excellent"
        elif avg_inference_time < 0.008:  # Less than 8ms
            inference_rating = "Good"
        else:
            inference_rating = "Could be optimized"
    elif "M3" in chip_info:
        if avg_inference_time < 0.003:  # Less than 3ms
            inference_rating = "Excellent"
        elif avg_inference_time < 0.006:  # Less than 6ms
            inference_rating = "Good"
        else:
            inference_rating = "Could be optimized"
    
    # Basic results dictionary
    results = {
        "device": str(device),
        "chip": chip_info,
        "pytorch_version": torch.__version__,
        "model_size": sum(p.numel() for p in model.parameters()),
        "model_size_mb": sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024),
        "inference": {
            "iterations": iterations,
            "total_time_sec": end_time - start_time,
            "avg_time_per_batch_sec": avg_inference_time,
            "batches_per_second": 1.0 / avg_inference_time,
            "samples_per_second": input_shape[0] / avg_inference_time,
            "performance_rating": inference_rating
        },
        "training": {
            "iterations": train_iterations,
            "total_time_sec": train_end_time - train_start_time,
            "avg_step_time_sec": avg_train_time,
            "batches_per_second": 1.0 / avg_train_time,
            "samples_per_second": input_shape[0] / avg_train_time
        },
        "memory": {
            "input_memory_mb": input_memory_usage / (1024 * 1024),
            "inference_memory_mb": inference_memory_usage / (1024 * 1024),
            "training_memory_mb": training_memory_usage / (1024 * 1024),
            "total_used_mb": memory_after_training["used"] / (1024 * 1024),
            "total_available_mb": memory_after_training["total"] / (1024 * 1024),
            "max_memory_percent": max_memory_percent
        },
        "input_shape": input_shape,
        "warmup": {
            "iterations": warmup,
            "first_iteration_time_sec": first_warmup_time,
            "avg_time_sec": avg_warmup_time,
            "compilation_overhead_sec": first_warmup_time - avg_inference_time
        }
    }
    
    # Add detailed statistics if requested
    if full_report:
        results["inference"]["min_time_ms"] = min_inference_time * 1000
        results["inference"]["max_time_ms"] = max_inference_time * 1000
        results["inference"]["p50_time_ms"] = p50_inference_time * 1000
        results["inference"]["p95_time_ms"] = p95_inference_time * 1000
        results["inference"]["times_ms"] = inference_times_ms
        
        results["training"]["min_time_ms"] = min_train_time * 1000
        results["training"]["max_time_ms"] = max_train_time * 1000
        results["training"]["p50_time_ms"] = p50_train_time * 1000
        results["training"]["p95_time_ms"] = p95_train_time * 1000
        results["training"]["times_ms"] = train_times_ms
        
        results["benchmark_total_time_sec"] = total_benchmark_time
        
        # Add optimization recommendations based on measurements
        recommendations = []
        
        if max_memory_percent > 90:
            recommendations.append("Memory usage is high. Consider reducing batch size or model size.")
        
        if first_warmup_time > avg_inference_time * 5:
            recommendations.append("High compilation overhead. Consider using 'aot_eager' backend for compilation.")
        
        if "M1" in chip_info and input_shape[0] > 256:
            recommendations.append("Batch size may be too large for this chip. Consider reducing to 256.")
        
        if p95_inference_time > avg_inference_time * 2:
            recommendations.append("High variance in inference times. Check for background processes or thermal throttling.")
        
        results["optimization_recommendations"] = recommendations
    
    # Generate an overall summary
    print("\n--- MPS Performance Benchmark Summary ---")
    print(f"Device: {device} ({chip_info})")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Model size: {results['model_size_mb']:.2f} MB")
    print(f"Inference: {results['inference']['samples_per_second']:.1f} samples/sec (avg: {avg_inference_time*1000:.2f} ms)")
    print(f"Training: {results['training']['samples_per_second']:.1f} samples/sec (avg: {avg_train_time*1000:.2f} ms)")
    print(f"Memory usage: {results['memory']['total_used_mb']:.1f} MB / {results['memory']['total_available_mb']:.1f} MB (max: {max_memory_percent}%)")
    print(f"Performance rating: {inference_rating}")
    
    if 'optimization_recommendations' in results:
        print("\nOptimization recommendations:")
        for rec in results['optimization_recommendations']:
            print(f"- {rec}")
    
    return results
