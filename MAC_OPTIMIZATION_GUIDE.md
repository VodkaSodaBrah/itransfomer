# Mac Optimization Guide for Time Series Forecasting with iTransformer

This guide explains the Mac-specific optimizations implemented for running the iTransformer model efficiently on Apple M-series chips using the Metal Performance Shaders (MPS) backend.

## Background

Apple's M-series chips provide excellent performance for machine learning tasks, but they require specific optimizations since they don't support CUDA and have some limitations with PyTorch's MPS backend. This implementation addresses those limitations and provides a seamless experience for training and inferencing on Mac.

## Key Optimizations

### 1. Automatic Mixed Precision (AMP) Compatibility

Since PyTorch's MPS backend doesn't support Automatic Mixed Precision, we've implemented compatibility layers:

- `MPSAutocastContext`: A context manager that mimics CUDA's `autocast` context
- `MPSGradScaler`: A class that provides the same interface as CUDA's `GradScaler`

These classes allow using a unified training loop for both CUDA and MPS devices without conditional code throughout the codebase.

### 2. Memory and Performance Optimizations

- **Chip-Specific Batch Sizing**: Automatically selects optimal batch sizes based on Mac chip capabilities (M1, M2, M3 series)
- **Dynamic Model Sizing**: Adjusts model dimensions based on available computational resources
- **Thread Configuration**: Sets optimal thread count for better coordination with MPS
- **Gradient Clipping**: Applies appropriate clipping to maintain training stability
- **Learning Rate Scheduling**: Uses cosine scheduling for better convergence on MPS

### 3. Model Compilation with Chip-Specific Strategies

- Uses `torch.compile()` with chip-specific optimization modes:
  - Base M1: `aot_eager` backend with `reduce-overhead` mode
  - M1 Pro/Max and M2: `aot_eager` with optimized parameters
  - M2 Pro/Max and M3: `inductor` backend with `max-autotune` mode when available
- Custom dropout implementation that avoids known MPS limitations

### 4. Environment Variable Optimization

- Sets `PYTORCH_ENABLE_MPS_FALLBACK=1` to handle unsupported operations
- Configures `PYTORCH_MPS_HIGH_WATERMARK_RATIO` for better memory utilization
- Enables chip-specific optimizations like `PYTORCH_MPS_ENABLE_NATIVE_SOFTMAX`
- Sets optimal OpenMP and MKL thread counts based on chip capabilities

### 5. Advanced Benchmarking

The implementation includes comprehensive benchmarking that measures:

- Inference speed with detailed statistics (p50, p95 latencies)
- Training throughput across multiple iterations
- Memory usage patterns and peak utilization
- Warmup overhead and compilation time
- Chip-specific performance recommendations

## Usage

### Basic Usage

Use the `run_on_mac.sh` script which applies all the necessary environment variables and optimizations:

```bash
./run_on_mac.sh --batch_size 256 --train_epochs 5 --seq_len 96 \
  --d_model 192 --loss_function financial --lradj cosine --grad_clip 1.0
```

### Advanced Options

```bash
# Run with benchmarking enabled
./run_on_mac.sh --benchmark

# Run benchmark only without training
./run_on_mac.sh --benchmark-only

# Disable model compilation (useful for debugging)
./run_on_mac.sh --no-compile

# Specify compilation mode 
./run_on_mac.sh --compile-mode reduce-overhead
```

## Chip-Specific Recommendations

### Base M1

- **Batch size**: 256
- **Model dimensions**: d_model=192, d_ff=768
- **Sequence length**: 96-128
- **Compilation**: Use `aot_eager` backend with `reduce-overhead` mode
- **Memory**: Keep under 6GB total model size for best performance

### M1 Pro/Max and M2

- **Batch size**: 512-768
- **Model dimensions**: d_model=256, d_ff=1024
- **Sequence length**: up to 256
- **Compilation**: Use `inductor` backend if available, otherwise `aot_eager`
- **Memory**: Can handle up to 12GB models efficiently

### M2 Pro/Max and M3 Series

- **Batch size**: 768-1024
- **Model dimensions**: d_model=320, d_ff=1280
- **Sequence length**: up to 384
- **Compilation**: Use `inductor` backend with `max-autotune` mode
- **Memory**: Can handle 20GB+ models efficiently

## Implementation Details

The optimizations are implemented across several files:

- `utils/mac_optimizations.py`: Core compatibility layers and utility functions
- `utils/mps_advanced.py`: Advanced chip-specific optimizations and custom implementations
- `utils/advanced_benchmarking.py`: Comprehensive benchmarking tools
- `direct_run.py`: Integration with the training loop
- `run_on_mac.sh`: Environment configuration and parameter selection

## Technical Implementation Notes

### Dropout Handling

PyTorch's MPS backend can have issues with the `native_dropout` operation. We address this with:

1. A custom dropout implementation that uses manual masking on MPS devices
2. Environment variable fallbacks to CPU when native implementation fails
3. Special handling during model compilation to maintain training stability

### Memory Management

MPS memory management differs from CUDA. Our optimizations include:

1. Setting higher watermark ratios for better memory utilization
2. Monitoring and reporting memory usage during training
3. Adjusting batch sizes based on chip capabilities to prevent OOM errors

### Model Compilation Strategies

Different compilation strategies work better for different chip architectures:

1. For base M1 chips, we use simpler compilation modes that prioritize stability
2. For newer chips like M2 Pro/Max and M3, we leverage advanced features like `inductor`
3. Compilation modes are automatically selected based on chip detection

## Known Limitations

1. Some operations may still use CPU fallback when not supported by MPS
2. The first training epoch is slower due to compilation and graph caching
3. Very large model sizes may still experience memory pressure on base M1 chips
4. Mixed precision training is emulated rather than natively supported

## Troubleshooting

If you encounter issues:

1. Try running with `--no-compile` to disable model compilation
2. Enable verbose mode with `--verbose` to see detailed output
3. Run benchmark mode with `--benchmark-only` to isolate performance issues
4. For memory errors, reduce batch size or model dimensions
5. Ensure PyTorch version is at least 2.0.0 for best compatibility
