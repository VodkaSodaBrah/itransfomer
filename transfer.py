import torch
import os
import json
from datetime import datetime
from iTransformer.model.iTransformer import Model

# Create new directories
exp_name = f"transfer_learning_{datetime.now().strftime('%Y%m%d_%H%M')}"
exp_dir = f"./experiments/{exp_name}"
checkpoints_dir = f"{exp_dir}/checkpoints"
output_dir = f"{exp_dir}/output"
os.makedirs(checkpoints_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Path to best model and its config
best_model_path = './checkpoints/best_model.pth'

# Either load saved config or specify manually
try:
    with open('./checkpoints/model_config.json', 'r') as f:
        config = json.load(f)
    print("Loaded model configuration from config file")
except FileNotFoundError:
    # If no config file, set parameters manually based on your original run
    config = {
        "enc_in": 5,  # Number of input features
        "dec_in": 5,  # Number of input features for decoder
        "c_out": 5,   # Number of output features
        "seq_len": 96,  # Input sequence length
        "label_len": 48,  # Label sequence length
        "pred_len": 24,  # Prediction sequence length
        "d_model": 512,  # Model dimension
        "n_heads": 8,    # Number of attention heads
        "e_layers": 2,   # Number of encoder layers
        "d_layers": 1,   # Number of decoder layers
        "d_ff": 2048,    # Dimension of FCN
        "dropout": 0.0,  # Dropout
        "embed": "timeF",  # Time embedding
        "freq": "1min",    # Time frequency
        "activation": "gelu",  # Activation
        "output_attention": False  # Whether to output attention
    }
    print("Using manually configured parameters")

# Initialize model with loaded or specified configuration
model = Model(**config)
model.load_state_dict(torch.load(best_model_path, map_location='mps'))

# Save initial model to new directory
torch.save(model.state_dict(), f"{checkpoints_dir}/initial_model.pth")

# Save config to new directory too
with open(f"{checkpoints_dir}/model_config.json", 'w') as f:
    json.dump(config, f, indent=2)

# Run training with new parameters and starting from best model
cmd = [
    "python", "direct_run.py",
    "--device", "mps",
    "--batch_size", "512",
    "--train_epochs", "20",
    "--d_model", "512",
    "--n_heads", "8",
    "--e_layers", "2",
    "--learning_rate", "0.0001",  # Lower learning rate for fine-tuning
    "--patience", "5",
    "--checkpoints", checkpoints_dir,
    "--output_dir", output_dir,
    "--optimize_for_mps",
    "--no_compile",
    "--load_model", f"{checkpoints_dir}/initial_model.pth"
]

# You'll need to add a --load_model flag to direct_run.py to support this

import subprocess
subprocess.run(cmd)