import os
import torch
import subprocess
import sys

def detect_and_print_device():
    """Detect available devices and print information"""
    print("Device detection:")
    
    # Check MPS (Mac)
    if torch.backends.mps.is_available():
        print("✅ MPS (Apple Silicon GPU) is available")
        recommended = "mps"
    else:
        print("❌ MPS (Apple Silicon GPU) is not available")
        
    # Check CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA is available with {torch.cuda.device_count()} devices:")
        for i in range(torch.cuda.device_count()):
            print(f"   - Device {i}: {torch.cuda.get_device_name(i)}")
        recommended = "cuda"
    else:
        print("❌ CUDA is not available")
        
    # Fallback to CPU
    if not torch.backends.mps.is_available() and not torch.cuda.is_available():
        print("⚠️ No GPU acceleration available, using CPU")
        recommended = "cpu"
    
    return recommended

def main():
    # Detect the best available device
    recommended_device = detect_and_print_device()
    
    # Set default parameters for BNB data
    default_params = [
        "--is_training", "1",
        "--model_id", "bnb_model",
        "--model", "iTransformer",
        "--data", "bnb",
        "--root_path", "./data/",
        "--data_path", "bnbusdt_1m.csv",
        "--features", "MS",
        "--target", "close",
        "--freq", "1min",
        "--seq_len", "96",
        "--label_len", "48",
        "--pred_len", "24",
        "--enc_in", "5",
        "--dec_in", "5",
        "--c_out", "1",
        "--d_model", "128",
        "--n_heads", "4",
        "--e_layers", "1",
        "--d_layers", "1",
        "--d_ff", "512",
        "--batch_size", "16",
        "--train_epochs", "5",
        "--learning_rate", "0.0005",
        "--num_workers", "0",
        "--device", recommended_device
    ]
    
    # Allow command line arguments to override defaults
    if len(sys.argv) > 1:
        # Get command line arguments (skip the script name)
        user_args = sys.argv[1:]
        
        # Add user args to the command (they will override defaults when duplicated)
        all_args = default_params + user_args
    else:
        all_args = default_params
    
    # Create the full command
    command = [sys.executable, "iTransformer/run.py"] + all_args
    
    print("\nRunning command:")
    print(" ".join(command))
    print("\n" + "=" * 50)
    
    # Execute the command
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()