# Generative Design for Architecture

A neural network-based system for generating architectural floor plans using Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs).

## Overview

This project implements state-of-the-art generative models for architectural layout generation, providing tools for creating diverse and realistic floor plans. The system supports both VAE and GAN architectures, with comprehensive evaluation metrics and an interactive web interface.

## Features

- **Multiple Model Architectures**: VAE and GAN implementations optimized for architectural layouts
- **Synthetic Data Generation**: Procedural floor plan generation for training and evaluation
- **Comprehensive Metrics**: Room statistics, connectivity, compactness, and diversity analysis
- **Interactive Demo**: Streamlit-based web interface for real-time generation
- **Reproducible Research**: Deterministic seeding and comprehensive logging
- **Modern Stack**: PyTorch Lightning, mixed precision training, and device optimization

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended) or Apple Silicon (MPS) support

### Setup

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Generative-Design-for-Architecture.git
cd Generative-Design-for-Architecture
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install in development mode:
```bash
pip install -e .
```

3. Install pre-commit hooks (optional):
```bash
pre-commit install
```

## Quick Start

### 1. Generate Synthetic Data

The system automatically generates synthetic floor plan data for training:

```python
from generative_design.data import FloorPlanGenerator

generator = FloorPlanGenerator(
    width=32, height=32,
    min_rooms=3, max_rooms=8,
    room_types=5
)

# Generate training data
train_data = generator.generate_dataset(10000)
val_data = generator.generate_dataset(2000)
```

### 2. Train a Model

#### VAE Training
```bash
python scripts/train.py --model vae --epochs 100 --batch_size 32
```

#### GAN Training
```bash
python scripts/train.py --model gan --epochs 200 --batch_size 32
```

#### Using Configuration Files
```bash
python scripts/train.py --config configs/vae_config.yaml
python scripts/train.py --config configs/gan_config.yaml
```

### 3. Generate Samples

```bash
python scripts/sample.py --model_path models/vae-epoch=99-val_loss=0.1234.ckpt --num_samples 16
```

### 4. Launch Interactive Demo

```bash
streamlit run demo/app.py
```

## Project Structure

```
generative-design-architecture/
├── src/generative_design/          # Main package
│   ├── __init__.py
│   ├── config.py                  # Configuration management
│   ├── data.py                    # Data generation and loading
│   ├── models.py                  # VAE and GAN implementations
│   ├── metrics.py                 # Evaluation metrics
│   └── utils.py                   # Utility functions
├── scripts/                       # Training and sampling scripts
│   ├── train.py
│   └── sample.py
├── configs/                       # Configuration files
│   ├── vae_config.yaml
│   └── gan_config.yaml
├── demo/                          # Interactive demo
│   └── app.py
├── tests/                         # Unit tests
├── data/                          # Data storage
├── models/                        # Model checkpoints
├── logs/                          # Training logs
├── assets/                        # Generated samples
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Model Architectures

### Variational Autoencoder (VAE)

The VAE implementation includes:
- Convolutional encoder-decoder architecture
- Reparameterization trick for differentiable sampling
- Beta-VAE support with KL annealing
- Latent space interpolation capabilities

**Key Features:**
- Stable training with reconstruction and KL divergence losses
- Smooth latent space for interpolation
- Deterministic reconstruction from latent codes

### Generative Adversarial Network (GAN)

The GAN implementation includes:
- DCGAN-style generator with transposed convolutions
- Spectral normalization for training stability
- Hinge loss for improved convergence
- Separate optimizers for generator and discriminator

**Key Features:**
- High-quality sample generation
- Training stability improvements
- Flexible architecture for different floor plan sizes

## Evaluation Metrics

The system provides comprehensive evaluation metrics:

### Room Statistics
- Total number of rooms
- Coverage ratio (occupied vs. total area)
- Room size distribution and variance
- Room type diversity

### Connectivity Analysis
- Horizontal and vertical room connections
- Connection density
- Spatial relationship analysis

### Compactness Metrics
- Bounding box efficiency
- Aspect ratio analysis
- Space utilization metrics

### Diversity Assessment
- Pairwise sample differences
- Uniqueness scoring
- Distribution analysis

## Configuration

Models can be configured using YAML files or command-line arguments:

### Key Parameters

- **Data**: Floor plan dimensions, room constraints, dataset size
- **Model**: Architecture type, latent dimensions, hidden layers
- **Training**: Learning rates, batch size, optimization settings
- **Evaluation**: Metrics configuration, logging options

### Example Configuration

```yaml
data:
  width: 32
  height: 32
  min_rooms: 3
  max_rooms: 8
  room_types: 5

model:
  model_type: "vae"
  latent_dim: 128
  hidden_dims: [512, 256]

training:
  epochs: 100
  learning_rate: 0.001
  batch_size: 32
```

## Interactive Demo

The Streamlit demo provides:

- **Model Upload**: Load trained model checkpoints
- **Real-time Generation**: Generate floor plans with custom parameters
- **Parameter Control**: Adjust room constraints and generation settings
- **Analysis Tools**: Comprehensive metrics and visualization
- **Export Options**: Save generated samples and analysis results

### Demo Features

1. **Model Selection**: Choose between VAE and GAN models
2. **Parameter Tuning**: Adjust floor plan dimensions and room constraints
3. **Batch Generation**: Generate multiple samples simultaneously
4. **Quality Analysis**: Real-time evaluation of generated layouts
5. **Visualization**: Interactive plots and sample grids

## Training Tips

### VAE Training
- Start with beta=1.0 and enable KL annealing
- Use cosine annealing for learning rate scheduling
- Monitor reconstruction and KL losses separately
- Validate on held-out data regularly

### GAN Training
- Use spectral normalization for stability
- Balance generator and discriminator learning rates
- Monitor gradient norms and loss curves
- Consider using different loss functions (hinge, Wasserstein)

### General Recommendations
- Use mixed precision training (FP16) for faster training
- Implement early stopping based on validation metrics
- Log training progress with TensorBoard or Weights & Biases
- Save model checkpoints regularly

## Advanced Usage

### Custom Data Generation

```python
from generative_design.data import FloorPlanGenerator

# Custom floor plan generator
generator = FloorPlanGenerator(
    width=64, height=64,
    min_rooms=5, max_rooms=12,
    min_room_size=4, max_room_size=10,
    room_types=8
)

# Generate custom dataset
custom_data = generator.generate_dataset(5000)
```

### Model Evaluation

```python
from generative_design.metrics import ArchitecturalMetrics

metrics = ArchitecturalMetrics(room_types=5)

# Evaluate single floor plan
stats = metrics.evaluate_floor_plan(floor_plan)

# Evaluate dataset
dataset_metrics = metrics.evaluate_dataset(floor_plans)
```

### Latent Space Manipulation (VAE)

```python
# Interpolate between two samples
interpolations = model.interpolate(sample1, sample2, num_steps=10)

# Sample from specific latent region
z = torch.randn(1, latent_dim) * 0.5  # Smaller variance
sample = model.decode(z)
```

## Performance Considerations

### Hardware Requirements
- **Minimum**: CPU with 8GB RAM
- **Recommended**: GPU with 8GB+ VRAM (RTX 3070/4070 or better)
- **Optimal**: High-end GPU with 16GB+ VRAM (RTX 4080/4090, A100)

### Optimization Tips
- Use batch sizes that fit in GPU memory (32-128 for most GPUs)
- Enable mixed precision training for 2x speedup
- Use data loading with multiple workers
- Consider gradient accumulation for larger effective batch sizes

### Memory Usage
- VAE: ~2-4GB VRAM for 32x32 floor plans
- GAN: ~4-8GB VRAM for 32x32 floor plans
- Larger floor plans require proportionally more memory

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use gradient accumulation
   - Enable mixed precision training

2. **Training Instability**
   - Check learning rates
   - Use gradient clipping
   - Enable spectral normalization for GANs

3. **Poor Sample Quality**
   - Increase model capacity
   - Adjust loss function weights
   - Check data preprocessing

4. **Slow Training**
   - Enable mixed precision
   - Increase number of data loader workers
   - Use faster storage (SSD)

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite: `pytest`
6. Format code: `black . && ruff check .`
7. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{generative_design_architecture,
  title={Generative Design for Architecture},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Generative-Design-for-Architecture}
}
```

## Acknowledgments

- PyTorch Lightning team for the excellent training framework
- The generative modeling community for foundational research
- Streamlit team for the interactive demo framework

## Model Card

### Intended Use
This model is designed for research and educational purposes in architectural design and generative modeling. It can be used to:
- Generate diverse architectural floor plan layouts
- Explore design spaces and constraints
- Study generative model architectures
- Develop architectural design tools

### Limitations
- Generated layouts are synthetic and may not meet real-world building codes
- No consideration for structural engineering constraints
- Limited to 2D floor plan generation
- Performance may vary with different architectural styles

### Bias and Fairness
- Training data is procedurally generated and may not represent diverse architectural traditions
- No explicit bias mitigation measures implemented
- Users should be aware of potential cultural biases in generated designs

### Safety Considerations
- Generated designs should not be used for actual construction without professional review
- No safety or accessibility compliance validation
- Users assume responsibility for any design decisions based on generated layouts
# Generative-Design-for-Architecture
