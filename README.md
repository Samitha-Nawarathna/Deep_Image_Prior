# Deep Image Prior

A PyTorch implementation of Deep Image Prior for image restoration tasks. This project provides a modular and extensible framework for implementing and experimenting with deep image prior techniques, particularly focused on image denoising.

## Overview

Deep Image Prior is a technique that leverages the structure of convolutional neural networks as an implicit prior for image restoration tasks. Instead of learning from a large dataset, the network structure itself acts as a handcrafted prior that captures low-level image statistics. This implementation provides a clean, modular architecture for experimenting with various image restoration tasks.

## Features

- **Modular Architecture**: Clean separation of concerns with abstractions for trainers, loggers, networks, and operators
- **Image Denoising**: Complete implementation for image denoising experiments
- **Hourglass Network**: Encoder-decoder architecture with skip connections
- **Configurable Training**: YAML-based configuration management
- **Comprehensive Logging**: Built-in logging for metrics, images, and training progress
- **Debugging Tools**: Utilities for analyzing network behavior (memorization, spectral analysis, weight saturation)
- **Metrics & Visualization**: PSNR calculation and image plotting utilities
- **GPU Support**: Optional CUDA acceleration

## Project Structure

```
Deep_Image_Prior/
├── dip_core/                       # Core framework modules
│   ├── abstractions/               # Abstract base classes
│   │   ├── logger.py              # Logging interface
│   │   ├── trainer.py             # Training interface
│   │   ├── operator.py            # Operator interface
│   │   ├── network_factory.py    # Network factory interface
│   │   ├── metric.py              # Metrics interface
│   │   └── ...
│   ├── base_implementations/       # Default implementations
│   │   ├── default_logger.py      # Default logger
│   │   ├── default_trainer.py     # Default trainer
│   │   ├── default_operator.py    # Default operator
│   │   ├── default_optimizer.py   # Default optimizer factory
│   │   ├── default_stopper.py     # Default stopping criterion
│   │   ├── config_manager.py      # Configuration management
│   │   └── run_folder_manager.py  # Run folder management
│   ├── networks/                   # Network architectures
│   │   ├── hourglass.py           # Hourglass/U-Net architecture
│   │   └── factory.py             # Network factory
│   ├── training/                   # Training components
│   │   ├── default_trainer.py     # Training loop implementation
│   │   └── default_training_step.py # Single training step
│   └── utils/                      # Utility functions
│       ├── image_io.py            # Image loading/saving
│       ├── losses.py              # Loss functions
│       ├── metrics.py             # Evaluation metrics (PSNR)
│       ├── plotting.py            # Visualization tools
│       ├── debugging.py           # Debugging utilities
│       └── seeds.py               # Random seed management
├── experiments/                    # Experiment implementations
│   └── denoising/                 # Denoising experiment
│       ├── run.py                 # Main script
│       ├── run.ipynb              # Jupyter notebook
│       ├── config.yml             # User configuration
│       ├── config_template.yml    # Configuration template
│       └── denoising_logger.py    # Custom logger for denoising
├── data/                           # Data directory
│   └── denoising/                 # Denoising data
│       └── ground_truth.png       # Sample image
├── logs/                           # Training logs
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Installation

### Prerequisites

- Python 3.7+
- PyTorch 1.7+
- CUDA (optional, for GPU acceleration)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Samitha-Nawarathna/Deep_Image_Prior.git
   cd Deep_Image_Prior
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Requirements

The project requires the following Python packages:

- `torch` - PyTorch deep learning framework
- `torchvision` - PyTorch vision utilities
- `numpy` - Numerical computing
- `matplotlib` - Plotting and visualization
- `scipy` - Scientific computing
- `Pillow` - Image processing
- `tqdm` - Progress bars
- `scikit-image` - Image processing utilities
- `opencv-python` - Computer vision utilities
- `pyyaml` - YAML configuration parsing
- `ipykernel` - Jupyter notebook support
- `torchsummary` - Model summary utilities
- `graphviz` - Graph visualization
- `torchviz` - PyTorch visualization

See `requirements.txt` for complete list.

## Quick Start

Run the image denoising experiment with default settings:

```bash
cd experiments/denoising
python run.py
```

This will:
1. Load the ground truth image from `data/denoising/ground_truth.png`
2. Add Gaussian noise (sigma=25)
3. Train the network for 10,000 iterations
4. Save results and logs to the logs directory

## Usage

### Running Experiments

#### Image Denoising

1. **Navigate to the denoising experiment directory:**
   ```bash
   cd experiments/denoising
   ```

2. **Prepare your data:**
   - Place your ground truth image in `data/denoising/ground_truth.png`
   - Or modify `config.yml` to point to your image

3. **Configure the experiment:**
   Edit `config.yml` to customize parameters:
   ```yaml
   seed: 42                    # Random seed
   iterations: 10000           # Number of training iterations
   lr: 0.1                     # Learning rate
   network: 'hourglass'        # Network architecture
   sigma: 25                   # Noise standard deviation
   use_gpu: true               # Enable GPU acceleration
   image_per: 500              # Log images every N iterations
   ```

4. **Run the experiment:**
   ```bash
   python run.py
   ```

5. **Monitor progress:**
   - Training progress is displayed via tqdm progress bar
   - Logs are saved to the logs directory
   - Images are saved periodically based on `image_per` setting

### Configuration Guide

The configuration file (`config.yml`) controls all aspects of the experiment:

#### Basic Settings
```yaml
seed: 42                        # Random seed for reproducibility
iterations: 10000               # Total training iterations
lr: 0.1                         # Learning rate
use_gpu: true                   # Use CUDA if available
image_per: 500                  # Save images every N iterations
```

#### Network Architecture
```yaml
network: 'hourglass'            # Network type
description:
  n_u: [16, 32, 64, 128, 128, 128]      # Encoder channels
  n_d: [16, 32, 64, 128, 128, 128]      # Decoder channels
  k_d: [3, 3, 3, 3, 3, 3]               # Decoder kernel sizes
  k_u: [5, 5, 5, 5, 5, 5]               # Encoder kernel sizes
  n_s: [0, 0, 0, 0, 0, 0]               # Skip connection channels
  k_s: [0, 0, 0, 0, 0, 0]               # Skip kernel sizes
  upsampling: 'nearest'                  # Upsampling mode
  use_bn: true                           # Use batch normalization
  use_sigmoid: true                      # Use sigmoid output
  in_channel: 3                          # Input channels (RGB)
  out_channel: 3                         # Output channels (RGB)
  activation: 'leakyReLU'                # Activation function
```

#### Data Settings
```yaml
noise_shape: [1, 3, 256, 256]           # Input noise shape [batch, channels, height, width]
sigma: 25                                # Gaussian noise standard deviation
std_inp_noise: 0.1                       # Standard deviation of input noise
ground_truth_image: "data/denoising/ground_truth.png"  # Path to ground truth image
```

### Using Jupyter Notebook

Alternatively, you can run experiments interactively:

```bash
cd experiments/denoising
jupyter notebook run.ipynb
```

### Debugging Tools

The framework includes debugging utilities for analyzing network behavior:

```python
from dip_core.utils.debugging import Debugger

# Create debugger
debugger = Debugger(model)

# Run diagnostics
debugger.run("memorization", gt_image)      # Test memorization capacity
debugger.run("spectral", gt_image)          # Spectral analysis
debugger.run("weight_saturation", gt_image) # Weight saturation analysis
debugger.run("grad_norms", gt_image)        # Gradient norm analysis

# Get results
print(debugger.get_log())
```

### Visualization

Plot network architecture:
```python
from dip_core.utils.plotting import plot_network

plot_network(model)
```

Plot tensors/images:
```python
from dip_core.utils.plotting import plot_tensor

plot_tensor(image_tensor)
```

## Architecture Overview

### Core Components

1. **Abstractions**: Define interfaces for extensibility
   - `Trainer`: Training loop interface
   - `Logger`: Logging interface
   - `Operator`: Forward operator interface
   - `NetworkFactory`: Network creation interface
   - `Metric`: Evaluation metric interface

2. **Base Implementations**: Default implementations
   - `DefaultTrainer`: Standard training loop with tqdm progress
   - `DefaultLogger`: Logging scalars, images, and text
   - `DefaultOperator`: Identity operator (can be customized)
   - `DefaultStopper`: Iteration-based stopping criterion

3. **Networks**:
   - `HourglassNetwork`: Encoder-decoder with skip connections
   - Modular design allows easy addition of new architectures

4. **Training**:
   - `DefaultTrainingStep`: Single forward-backward pass
   - Configurable loss functions and optimizers

5. **Utils**:
   - Image I/O, metrics (PSNR), loss functions
   - Debugging and visualization tools

### Training Pipeline

1. **Configuration**: Load and validate YAML config
2. **Data Preparation**: Load images, add noise, create input noise tensor
3. **Network Creation**: Build network from factory
4. **Training Components**: Initialize optimizer, loss, metrics, logger
5. **Training Loop**: Iterate with progress tracking
6. **Logging**: Save metrics, images, and checkpoints

## Extending the Framework

### Adding New Experiments

1. Create a new directory under `experiments/`
2. Implement custom logger if needed (inherit from `DefaultLogger`)
3. Create `run.py` and `config.yml`
4. Follow the pattern from `experiments/denoising/run.py`

### Adding New Networks

1. Define network class in `dip_core/networks/`
2. Register in `dip_core/networks/factory.py`
3. Use in config: `network: 'your_network_name'`

### Adding New Metrics

1. Create metric class inheriting from `Metric` abstract class
2. Implement `__call__` and `name` methods
3. Add to metrics list in experiment script

## Output

After running an experiment, you'll find:

- **Logs directory**: Contains subdirectories for each run
  - `config.yml`: Saved configuration
  - `scalars/`: Training metrics (loss, PSNR)
  - `images/`: Periodic output images
  - `log.txt`: Text logs

## Tips

- Start with fewer iterations (e.g., 1000) for testing
- Adjust `image_per` for more/less frequent image logging
- Use GPU for faster training: `use_gpu: true`
- Monitor PSNR to track denoising quality
- Experiment with network architecture parameters
- Try different noise levels (sigma) for various difficulty levels

## Troubleshooting

**CUDA out of memory:**
- Reduce image size in `noise_shape`
- Set `use_gpu: false` to use CPU

**Poor denoising results:**
- Increase `iterations`
- Adjust learning rate `lr`
- Try different network architectures
- Modify noise standard deviation

**Import errors:**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.7+)

## Contributing

Contributions are welcome! Areas for improvement:
- Additional image restoration tasks (super-resolution, inpainting)
- New network architectures
- Advanced stopping criteria
- More evaluation metrics
- Documentation improvements

## License

This project is open source. Please check the repository for license details.

## References

- Ulyanov, D., Vedaldi, A., & Lempitsky, V. (2018). Deep Image Prior. In CVPR.

## Contact

For questions or issues, please open an issue on the GitHub repository.
