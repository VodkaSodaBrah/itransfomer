import os
import subprocess
import json
from datetime import datetime

# Base configuration
base_config = {
    "device": "mps",
    "batch_size": 512,
    "train_epochs": 20,
    "d_model": 512,
    "n_heads": 8,
    "e_layers": 2,
    "learning_rate": 0.001,
    "optimize_for_mps": True,
    "no_compile": True,
    "disable_dropout_for_mps_compile": True
}

# Experiments to run
experiments = [
    {
        "name": "baseline_repeat",
        "config": {}  # Uses base config
    },
    # Already run manually - commenting out
    # {
    #     "name": "deeper_model",
    #     "config": {"e_layers": 4}
    # },
    {
        "name": "wider_model",
        "config": {"d_model": 768, "n_heads": 12, "batch_size": 256}
    },
    {
        "name": "lr_cosine",
        "config": {"learning_rate": 0.0001, "lradj": "cosine"}
    },
    {
        "name": "longer_train",
        "config": {"train_epochs": 30, "patience": 5}
    }
]

# Run each experiment
for experiment in experiments:
    # Create experiment directory
    exp_dir = f"./experiments/{experiment['name']}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    checkpoints_dir = f"{exp_dir}/checkpoints"
    output_dir = f"{exp_dir}/output"
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create experiment config
    config = base_config.copy()
    config.update(experiment["config"])
    config["checkpoints"] = checkpoints_dir
    config["output_dir"] = output_dir
    
    # Save experiment config
    with open(f"{exp_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Build command
    cmd = ["python", "direct_run.py"]
    for k, v in config.items():
        if isinstance(v, bool) and v:
            cmd.append(f"--{k}")
        elif not isinstance(v, bool):
            cmd.append(f"--{k}")
            cmd.append(str(v))
    
    # Run experiment
    print(f"Running experiment: {experiment['name']}")
    print(" ".join(cmd))
    subprocess.run(cmd)
    
    print(f"Completed experiment: {experiment['name']}")
    print("-" * 50)