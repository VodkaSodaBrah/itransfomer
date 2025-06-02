import os
import sys
import torch
import numpy as np
import random
import argparse
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Add the iTransformer directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'iTransformer'))

# Now import the modules using the correct paths
from iTransformer.model.iTransformer import Model
from iTransformer.data_loader.improved_bnb_loader import ImprovedBNBDataset
from iTransformer.utils.tools import EarlyStopping, adjust_learning_rate

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
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
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
    parser.add_argument('--lradj', type=str, default='type1', 
                    help='Learning rate adjustment strategy: type1, type2, or cosine')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.checkpoints, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed
    set_seed(2023)
    
    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # MPS-specific optimizations - before creating DataLoader AND before model creation
    if args.optimize_for_mps and device.type == 'mps':
        print("Applying MPS-specific optimizations...")
        # Use larger batch sizes for better parallelism
        if args.batch_size < 512: # This is okay here, as it modifies args before DataLoader
            print(f"Increasing batch size from {args.batch_size} to 512 for better MPS performance")
            args.batch_size = 512
    
    # Prepare data
    train_dataset = ImprovedBNBDataset(
        root_path='./data/',
        data_path=os.path.basename(args.data_path),
        flag='train',
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=1,
        freq=args.freq
    )
    
    test_dataset = ImprovedBNBDataset(
        root_path='./data/',
        data_path=os.path.basename(args.data_path),
        flag='test',
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=1,
        freq=args.freq
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,  # Crucial for MPS
        pin_memory=False # Crucial for MPS
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,  # Crucial for MPS
        pin_memory=False # Crucial for MPS
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
            # Add any other parameters needed for model initialization
        }
        import json
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        return config

    # Save the configuration
    config_path = os.path.join(args.checkpoints, 'model_config.json')
    save_model_config(args, config_path)

    # Apply torch.compile ONCE before training loop if optimizing for MPS
    if args.optimize_for_mps and device.type == 'mps' and not args.no_compile:
        if hasattr(torch, 'compile'):
            print("Attempting to use torch.compile for model optimization on MPS...")
            try:
                # If we are going to compile and disable_dropout_for_mps_compile is True,
                # we need to inform the model to potentially alter its dropout layers.
                # This is a bit tricky as the model is already instantiated.
                # A cleaner way is to pass this config down during model instantiation.
                if args.disable_dropout_for_mps_compile:
                    print("Dropout will be disabled for MPS compilation.")
                    # This flag should ideally be used by the model's __init__
                    # to replace nn.Dropout with nn.Identity if this flag is True AND device is MPS.
                    # For now, we'll rely on the model checking this config.
                    config.is_mps_compiled_with_dropout_disabled = True


                model_compiled = torch.compile(model, backend="aot_eager")
                # If compilation is successful, use the compiled model
                model = model_compiled
                config.is_mps_compiled = True # Inform config that model is compiled
                print("torch.compile applied successfully.")
            except Exception as e:
                print(f"torch.compile failed on MPS: {e}. Proceeding without compilation.")
                config.is_mps_compiled_with_dropout_disabled = False # Reset if compile fails
        else:
            print("torch.compile not available in this PyTorch version.")
    else:
        print("Skipping torch.compile as requested.")

    # Set up optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.MSELoss()
    
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
            
            # Forward pass
            outputs = model(batch_x, batch_x_mark, batch_y, batch_y_mark)

            # Find target column index
            if args.features == 'MS':
                target_idx = train_dataset.df_data.columns.get_loc(args.target)
                outputs = outputs[:, :, target_idx:target_idx+1]
                batch_y = batch_y[:, -args.pred_len:, target_idx:target_idx+1]
            else:
                batch_y = batch_y[:, -args.pred_len:, :]

            loss = criterion(outputs, batch_y)

            # Regular backward pass
            loss.backward()
            optimizer.step()
            
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
                
                # Find target column index
                if args.features == 'MS':
                    target_idx = test_dataset.df_data.columns.get_loc(args.target)
                    outputs = outputs[:, :, target_idx:target_idx+1]
                    batch_y = batch_y[:, -args.pred_len:, target_idx:target_idx+1]
                else:
                    batch_y = batch_y[:, -args.pred_len:, :]
                
                loss = criterion(outputs, batch_y)
                
                test_loss.append(loss.item())
                test_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        # Calculate average loss
        test_loss = np.average(test_loss)
        test_losses.append(test_loss)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        
        # Early stopping check
        early_stopping(test_loss, model, model_path)  # Pass the full path, not just directory
        if early_stopping.early_stop:
            print(f"Early stopping triggered. Best model saved to {model_path}")
            break
        
        # Adjust learning rate
        adjust_learning_rate(optimizer, epoch + 1, config)
        
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'loss_curve.png'))
    
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
            
            # Find target column index
            if args.features == 'MS':
                target_idx = test_dataset.df_data.columns.get_loc(args.target)
                outputs = outputs[:, :, target_idx:target_idx+1]
                batch_y = batch_y[:, -args.pred_len:, target_idx:target_idx+1]
            else:
                batch_y = batch_y[:, -args.pred_len:, :]
            
            # Store predictions and true values
            pred = outputs.detach().cpu().numpy()
            true = batch_y.detach().cpu().numpy()
            
            preds.append(pred)
            trues.append(true)
    
    # Concatenate predictions and true values
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    
    # Save predictions
    np.save(os.path.join(args.output_dir, 'predictions.npy'), preds)
    np.save(os.path.join(args.output_dir, 'true_values.npy'), trues)
    
    # Visualize predictions for a sample
    for i in range(min(5, preds.shape[0])):
        plt.figure(figsize=(10, 5))
        plt.plot(trues[i, :, 0], label='True')
        plt.plot(preds[i, :, 0], label='Predicted')
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.title(f'Sample {i+1}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(args.output_dir, f'sample_{i+1}.png'))
    
    print(f"Training completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()