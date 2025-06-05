# iTransformer: Time Series Forecasting

A comprehensive implementation of the iTransformer architecture for time series forecasting, with specific optimizations for financial time series data. This project is based on the paper [iTransformer: Inverted Transformers Are Effective for Time Series Forecasting](https://arxiv.org/abs/2310.06625).

## Project Overview

iTransformer introduces a novel approach to time series forecasting by inverting the traditional Transformer architecture. Instead of treating time steps as tokens, it treats variables as tokens, allowing the model to better capture inter-variable relationships while maintaining temporal patterns.

This implementation includes:

- The core iTransformer architecture
- Financial time series specific adaptations
- Optimizations for different hardware (CUDA, MPS, CPU)
- Comprehensive evaluation metrics
- Tools for hyperparameter tuning and model analysis

## Project Structure

- `iTransformer/`: Core implementation of the iTransformer model
  - `data_loader/`: Data loading and processing utilities
  - `model/`: Model architecture implementation
  - `utils/`: Utility functions for the core model
  - `run.py`: Main script for running the core model
- `utils/`: Additional utility functions for this implementation
  - `metrics.py`: Forecasting evaluation metrics
  - `visualize.py`: Visualization utilities
  - `advanced_benchmarking.py`: Performance benchmarking tools
  - `mac_optimizations.py`: Optimizations for Apple Silicon
- `data/`: Default directory for datasets
- `output/`: Default directory for model outputs and visualizations
- `experiments/`: Directory for experiment results
- `checkpoints/`: Directory for model checkpoints
- `tests/`: Unit and integration tests

## Data Format

The project primarily works with time series data in CSV format. Financial data like OHLCV (Open, High, Low, Close, Volume) is supported with the BNB dataset loader. The data processing pipeline includes:

- Time feature encoding
- Z-score normalization (default) or per-window normalization
- Train/validation/test splitting

## Key Components

### Data Loaders

The `ImprovedBNBDataset` in `iTransformer/data_loader/improved_bnb_loader.py` handles financial time series data with the following features:

- Automatic detection of date/time columns
- Technical indicator calculation
- Robust handling of missing or invalid data
- Two normalization methods: global Z-score and per-window

### Model Architecture

The iTransformer model inverts the traditional Transformer by:

- Treating variables as tokens instead of time steps
- Using attention mechanisms to capture inter-variable relationships
- Employing layer normalization and feedforward networks to learn time series representations

### Loss Functions

Multiple loss functions are available:

- MSE (Mean Squared Error) - default
- Directional Loss - focuses on correctly predicting price movement directions
- Financial Loss - combines MSE with directional accuracy for financial applications

### Evaluation Metrics

The system calculates multiple metrics for comprehensive evaluation:

- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- SMAPE (Symmetric Mean Absolute Percentage Error)
- DirectionalAccuracy - percentage of correctly predicted direction changes
- WAPE (Weighted Absolute Percentage Error)

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- pandas, numpy, matplotlib, scikit-learn

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/itransformer.git
cd itransformer
```

1. Install dependencies:

```bash
pip install -r iTransformer/requirements.txt
```

### Basic Usage

#### Running with default BNB dataset

The simplest way to get started is using the `run_bnb.py` script which automatically configures common parameters:

```bash
python run_bnb.py
```

This will:

- Detect the best available device (CUDA, MPS, or CPU)
- Load the BNB cryptocurrency dataset
- Train an iTransformer model with default parameters
- Save the model and predictions

#### Customizing parameters

You can customize parameters by passing them to `run_bnb.py`:

```bash
python run_bnb.py --train_epochs 30 --batch_size 128 --pred_len 48
```

#### Direct model running

For more advanced usage, you can use `direct_run.py` which provides additional options:

```bash
python direct_run.py --model iTransformer --data bnb --seq_len 96 --pred_len 24 --loss_function financial
```

### Fine-tuning

The `fine_tune.py` script provides a framework for running multiple experiments with different hyperparameters:

```bash
python fine_tune.py
```

This will run a series of experiments with different configurations and save the results in the `experiments/` directory.

## Workflow Steps

### 1. Data Preparation

Place your CSV data file in the `data/` directory. For financial data, ensure it contains at least OHLC (Open, High, Low, Close) columns.

### 2. Model Training

Run the training using either `run_bnb.py` for the BNB dataset or `direct_run.py` for custom configurations:

```bash
python run_bnb.py --train_epochs 20
```

The model will be trained and saved to the `checkpoints/` directory.

### 3. Model Evaluation

During and after training, the model will be evaluated on the test set. Results are saved to:

- `output/metrics.json` - Numerical performance metrics
- `output/predictions.npy` - Raw model predictions
- `output/true_values.npy` - Ground truth values

### 4. Visualization

Prediction visualizations are automatically generated in the `output/` directory:

- Loss curves
- Sample predictions
- Detailed analysis plots

### 5. Performance Experiments

For performance optimization, run:

```bash
python run_performance_experiments.py
```

This will test different model configurations and loss functions, comparing their performance.

## Optimizations

### Hardware-specific Optimizations

- **CUDA**: Optimized for NVIDIA GPUs
- **MPS**: Special optimizations for Apple Silicon (M1/M2/M3)
- **CPU**: Fallback for systems without GPU support

To use specific hardware:

```bash
python run_bnb.py --device cuda  # For NVIDIA GPUs
python run_bnb.py --device mps   # For Apple Silicon
python run_bnb.py --device cpu   # For CPU only
```

### Apple Silicon Optimizations

For Mac users with M1/M2/M3 chips, special optimizations are available:

```bash
python run_bnb.py --optimize_for_mps
```

## Advanced Features

### Benchmarking

To benchmark performance across different devices:

```bash
python direct_run.py --benchmark_devices
```

### Transfer Learning

To use a pre-trained model for a new dataset:

```bash
python transfer.py --source_model checkpoints/best_model.pth --target_data new_data.csv
```

## Performance Metrics

Model evaluation uses multiple metrics to provide a comprehensive assessment:

- **MSE/RMSE**: Overall prediction accuracy
- **MAE**: Absolute error measurement
- **MAPE/SMAPE**: Percentage-based error metrics
- **DirectionalAccuracy**: Correctly predicted trend directions
- **WAPE**: Weighted error measurement

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the original iTransformer paper by [original authors]
- Includes optimizations for Apple Silicon devices
