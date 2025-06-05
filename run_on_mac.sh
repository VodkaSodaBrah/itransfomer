#!/bin/zsh
# Run script for Mac with necessary environment variables and optimizations

# Detect Mac chip type
MAC_CHIP=$(sysctl -n machdep.cpu.brand_string)

# Determine optimal threading based on chip
if [[ $MAC_CHIP == *"M1"* || $MAC_CHIP == *"M2"* || $MAC_CHIP == *"M3"* ]]; then
  # For Apple Silicon, limit OpenMP threads for better coordination with MPS
  OMP_THREADS=1
  PYTORCH_THREADS=1
else
  # For Intel Macs, use more threads
  OMP_THREADS=4
  PYTORCH_THREADS=4
fi

# Default values
BATCH_SIZE=256
EPOCHS=5
SEQ_LEN=96
D_MODEL=192
LOSS="financial"
LRADJ="cosine"
GRAD_CLIP=1.0
VERBOSE=false
BENCHMARK=false
BENCHMARK_ONLY=false
USE_COMPILE=true
COMPILE_MODE="default" # Options: default, reduce-overhead, max-autotune

# Set environment variables for Mac performance
export PYTORCH_ENABLE_MPS_FALLBACK=1  # Enable fallback for unsupported MPS operations
export OMP_NUM_THREADS=$OMP_THREADS    # Set OpenMP threads for better MPS coordination
export MKL_NUM_THREADS=$OMP_THREADS    # Set MKL threads to match OpenMP setting

# Set proper watermark ratios for MPS memory management
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7  # Must be between 0 and 1
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5   # Must be less than high watermark

# Additional Mac-specific optimizations
if [[ $MAC_CHIP == *"M2"* || $MAC_CHIP == *"M3"* ]]; then
  # Newer chips benefit from these settings
  export PYTORCH_MPS_ENABLE_NATIVE_SOFTMAX=1
fi

# If running on M1 Pro/Max or M2/M3, use system priority boost
if [[ $MAC_CHIP != *"M1"* || $MAC_CHIP == *"M1 Pro"* || $MAC_CHIP == *"M1 Max"* ]]; then
  # Prioritize app (requires sudo)
  if sudo -n true 2>/dev/null; then
    sudo renice -10 $$
    echo "Process priority boosted for better performance"
  else
    echo "Could not boost process priority (requires sudo)"
  fi
fi

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --train_epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --seq_len)
      SEQ_LEN="$2"
      shift 2
      ;;
    --d_model)
      D_MODEL="$2"
      shift 2
      ;;
    --loss_function)
      LOSS="$2"
      shift 2
      ;;
    --lradj)
      LRADJ="$2"
      shift 2
      ;;
    --grad_clip)
      GRAD_CLIP="$2"
      shift 2
      ;;
    --verbose)
      VERBOSE=true
      shift
      ;;
    --benchmark)
      BENCHMARK=true
      shift
      ;;
    --benchmark-only)
      BENCHMARK_ONLY=true
      BENCHMARK=true
      shift
      ;;
    --no-compile)
      USE_COMPILE=false
      shift
      ;;
    --compile-mode)
      COMPILE_MODE="$2"
      shift 2
      ;;
    *)
      # Pass through any other arguments
      ARGS="$ARGS $1"
      shift
      ;;
  esac
done

# Detect Mac chip type
MAC_CHIP=$(sysctl -n machdep.cpu.brand_string)
echo "Detected Mac chip: $MAC_CHIP"

# Adjust batch size based on Mac chip
if [[ $MAC_CHIP == *"M1"* && $MAC_CHIP != *"Pro"* && $MAC_CHIP != *"Max"* ]]; then
  # Base M1 chip
  if [[ $BATCH_SIZE -gt 256 ]]; then
    echo "Reducing batch size to 256 for better performance on base M1 chip"
    BATCH_SIZE=256
  fi
  # Adjust model dimensions for base M1
  if [[ $D_MODEL -gt 192 ]]; then
    echo "Reducing model dimension to 192 for better performance on base M1 chip"
    D_MODEL=192
  fi
elif [[ $MAC_CHIP == *"M1 Pro"* || $MAC_CHIP == *"M1 Max"* || $MAC_CHIP == *"M2"* ]]; then
  # M1 Pro/Max or M2 chips can handle larger batch sizes
  if [[ $BATCH_SIZE -lt 512 ]]; then
    echo "Increasing batch size to 512 for better performance on M1 Pro/Max or M2 chip"
    BATCH_SIZE=512
  fi
  # Adjust model dimensions for mid-tier chips
  if [[ $D_MODEL -lt 256 ]]; then
    echo "Increasing model dimension to 256 for better utilization of M1 Pro/Max or M2 chip"
    D_MODEL=256
  fi
elif [[ $MAC_CHIP == *"M2 Pro"* || $MAC_CHIP == *"M2 Max"* || $MAC_CHIP == *"M3"* ]]; then
  # M2 Pro/Max or M3 chips can handle even larger batch sizes
  if [[ $BATCH_SIZE -lt 768 ]]; then
    echo "Increasing batch size to 768 for better performance on M2 Pro/Max or M3 chip"
    BATCH_SIZE=768
  fi
  # Adjust model dimensions for high-end chips
  if [[ $D_MODEL -lt 320 ]]; then
    echo "Increasing model dimension to 320 for better utilization of M2 Pro/Max or M3 chip"
    D_MODEL=320
  fi
fi

# Determine optimal threading based on chip
if [[ $MAC_CHIP == *"M1"* || $MAC_CHIP == *"M2"* || $MAC_CHIP == *"M3"* ]]; then
  # For Apple Silicon, limit OpenMP threads for better coordination with MPS
  OMP_THREADS=1
  PYTORCH_THREADS=1
else
  # For Intel Macs, use more threads
  OMP_THREADS=4
  PYTORCH_THREADS=4
fi

# Determine optimal compilation mode if not specified
if [[ $COMPILE_MODE == "default" ]]; then
  if [[ $MAC_CHIP == *"M3"* ]]; then
    COMPILE_MODE="max-autotune"
  elif [[ $MAC_CHIP == *"M2"* ]]; then
    COMPILE_MODE="reduce-overhead"
  else
    COMPILE_MODE="reduce-overhead"
  fi
fi

# Attempt to prioritize app (may require sudo)
if sudo -n true 2>/dev/null; then
  sudo renice -10 $$
  echo "Process priority boosted for better performance"
else
  echo "Could not boost process priority (requires sudo)"
fi

# Echo the configuration
echo ""
echo "Running iTransformer with Mac optimizations"
echo "================================================"
echo "Environment variables:"
echo "  PYTORCH_ENABLE_MPS_FALLBACK=1"
echo "  OMP_NUM_THREADS=$OMP_THREADS"
echo "  MKL_NUM_THREADS=$OMP_THREADS"
echo "  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7"
echo "  PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5"
if [[ -n "${PYTORCH_MPS_ENABLE_NATIVE_SOFTMAX}" ]]; then
  echo "  PYTORCH_MPS_ENABLE_NATIVE_SOFTMAX=1"
fi
echo ""
echo "Model configuration:"
echo "  --optimize_for_mps --batch_size $BATCH_SIZE --train_epochs $EPOCHS"
echo "  --seq_len $SEQ_LEN --d_model $D_MODEL --loss_function $LOSS"
echo "  --lradj $LRADJ --grad_clip $GRAD_CLIP"
if [[ "$USE_COMPILE" == "false" ]]; then
  echo "  --no_compile"
else
  echo "  --compile_mode $COMPILE_MODE"
fi
if [[ -n "$ARGS" ]]; then
  echo "  $ARGS"
fi
echo "================================================"

# Close background apps suggestion
echo "💡 Tip: For best performance, close other GPU-intensive applications before training"
echo ""

# Run benchmark only if requested
if [[ "$BENCHMARK_ONLY" == "true" ]]; then
  echo "Running benchmark only mode..."
  python -c "
import sys
sys.path.append('/Users/mchildress/Active Code/itransformer')
from utils.advanced_benchmarking import benchmark_mps_performance_advanced
from iTransformer.model.iTransformer import Model
import torch

# Create a small model for benchmarking
configs = type('configs', (), {
    'seq_len': $SEQ_LEN,
    'label_len': $SEQ_LEN // 2,
    'pred_len': $SEQ_LEN // 2,
    'enc_in': 12,
    'dec_in': 12,
    'c_out': 12,
    'd_model': $D_MODEL,
    'embed': 'timeF',
    'freq': 'h',
    'dropout': 0.1,
    'factor': 1,
    'n_heads': 8,
    'batch_size': $BATCH_SIZE,
    'e_layers': 2,
    'd_layers': 1,
    'd_ff': $D_MODEL * 4,
})

model = Model(configs).to('mps')
batch_shape = ($BATCH_SIZE, $SEQ_LEN, 12)
print(f'Running benchmark with model size: {sum(p.numel() for p in model.parameters())} parameters')
results = benchmark_mps_performance_advanced(model, batch_shape, iterations=50, warmup=10, full_report=True)
"
  exit 0
fi

# Construct run command with the correct parameters
RUN_CMD="python direct_run.py --optimize_for_mps --batch_size $BATCH_SIZE --train_epochs $EPOCHS \
  --seq_len $SEQ_LEN --d_model $D_MODEL --loss_function $LOSS \
  --lradj $LRADJ --grad_clip $GRAD_CLIP"

# Add compile settings
if [[ "$USE_COMPILE" == "false" ]]; then
  RUN_CMD="$RUN_CMD --no_compile"
else
  RUN_CMD="$RUN_CMD --compile_mode $COMPILE_MODE"
fi

# Add benchmark flag if needed
if [[ "$BENCHMARK" == "true" ]]; then
  RUN_CMD="$RUN_CMD --benchmark"
fi

# Add any additional arguments
if [[ -n "$ARGS" ]]; then
  RUN_CMD="$RUN_CMD $ARGS"
fi

# Display the command
echo "Running: $RUN_CMD"
echo ""

# Execute the command
eval $RUN_CMD
