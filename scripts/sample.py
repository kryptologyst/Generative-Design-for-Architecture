"""Sampling script for generating architectural layouts."""

import argparse
import os
from pathlib import Path
from typing import List, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from generative_design.config import Config
from generative_design.models import VAEArchitecturalGenerator, GANArchitecturalGenerator
from generative_design.utils import set_seed, get_device, visualize_floor_plan


def load_model(model_path: str, config: Config, device: torch.device):
    """Load a trained model.
    
    Args:
        model_path: Path to the model checkpoint.
        config: Configuration object.
        device: Device to load the model on.
        
    Returns:
        Loaded model.
    """
    if config.model.model_type == 'vae':
        model = VAEArchitecturalGenerator(
            input_size=config.model.input_size,
            latent_dim=config.model.latent_dim,
            hidden_dims=config.model.hidden_dims,
            beta=config.model.beta,
            kl_annealing=config.model.kl_annealing
        )
    elif config.model.model_type == 'gan':
        model = GANArchitecturalGenerator(
            input_size=config.model.input_size,
            latent_dim=config.model.latent_dim,
            generator_hidden_dims=config.model.hidden_dims,
            discriminator_hidden_dims=config.model.hidden_dims,
            use_spectral_norm=config.model.use_spectral_norm
        )
    else:
        raise ValueError(f"Unknown model type: {config.model.model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model


def generate_samples(
    model,
    num_samples: int,
    device: torch.device,
    model_type: str,
    save_dir: Optional[str] = None
) -> List[np.ndarray]:
    """Generate samples from the model.
    
    Args:
        model: Trained model.
        num_samples: Number of samples to generate.
        device: Device to generate samples on.
        model_type: Type of model ('vae' or 'gan').
        save_dir: Optional directory to save samples.
        
    Returns:
        List of generated floor plans.
    """
    samples = []
    
    with torch.no_grad():
        if model_type == 'vae':
            # Generate samples from VAE
            generated = model.sample(num_samples, device)
        elif model_type == 'gan':
            # Generate samples from GAN
            generated = model.sample(num_samples, device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Convert to numpy and denormalize
        generated_np = generated.cpu().numpy()
        
        # Denormalize from [0, 1] to [0, room_types]
        generated_np = generated_np * 5  # Assuming 5 room types
        generated_np = np.round(generated_np).astype(np.int32)
        
        for i in range(num_samples):
            sample = generated_np[i, 0]  # Remove channel dimension
            samples.append(sample)
            
            # Save individual samples
            if save_dir:
                save_path = Path(save_dir) / f"sample_{i:04d}.png"
                plt.figure(figsize=(6, 6))
                plt.imshow(sample, cmap='viridis', origin='upper')
                plt.title(f"Generated Floor Plan {i+1}")
                plt.colorbar(label='Room Type')
                plt.axis('off')
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
    
    return samples


def create_sample_grid(
    samples: List[np.ndarray],
    grid_size: tuple[int, int] = (4, 4),
    save_path: Optional[str] = None
) -> None:
    """Create a grid visualization of samples.
    
    Args:
        samples: List of floor plan samples.
        grid_size: Size of the grid (rows, cols).
        save_path: Optional path to save the grid.
    """
    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    
    for i in range(rows * cols):
        row, col = i // cols, i % cols
        
        if i < len(samples):
            axes[row, col].imshow(samples[i], cmap='viridis', origin='upper')
            axes[row, col].set_title(f'Sample {i+1}')
        else:
            axes[row, col].axis('off')
        
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def interpolate_samples(
    model,
    sample1: torch.Tensor,
    sample2: torch.Tensor,
    num_steps: int = 10,
    device: torch.device,
    model_type: str,
    save_path: Optional[str] = None
) -> List[np.ndarray]:
    """Interpolate between two samples.
    
    Args:
        model: Trained model.
        sample1: First sample.
        sample2: Second sample.
        num_steps: Number of interpolation steps.
        device: Device for computation.
        model_type: Type of model.
        save_path: Optional path to save interpolation.
        
    Returns:
        List of interpolated samples.
    """
    if model_type != 'vae':
        raise ValueError("Interpolation is only supported for VAE models")
    
    with torch.no_grad():
        interpolations = model.interpolate(sample1, sample2, num_steps)
        
        # Convert to numpy
        interp_np = interpolations.cpu().numpy()
        interp_np = interp_np * 5  # Denormalize
        interp_np = np.round(interp_np).astype(np.int32)
        
        samples = []
        for i in range(num_steps):
            sample = interp_np[i, 0]  # Remove channel dimension
            samples.append(sample)
        
        # Create interpolation visualization
        fig, axes = plt.subplots(1, num_steps, figsize=(20, 2))
        for i, sample in enumerate(samples):
            axes[i].imshow(sample, cmap='viridis', origin='upper')
            axes[i].set_title(f'Step {i+1}')
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return samples


def main():
    """Main sampling function."""
    parser = argparse.ArgumentParser(description='Generate architectural layouts')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--model_type', type=str, choices=['vae', 'gan'],
                       help='Model type (overrides config)')
    parser.add_argument('--num_samples', type=int, default=16,
                       help='Number of samples to generate')
    parser.add_argument('--grid_size', type=str, default='4x4',
                       help='Grid size for visualization (e.g., 4x4)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--save_dir', type=str, help='Directory to save samples')
    parser.add_argument('--interpolate', action='store_true',
                       help='Generate interpolation between two samples')
    parser.add_argument('--interpolation_steps', type=int, default=10,
                       help='Number of interpolation steps')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
            config = Config(**config_dict)
    
    # Override model type if specified
    if args.model_type:
        config.model.model_type = args.model_type
    
    # Set seed
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading {config.model.model_type.upper()} model from {args.model_path}")
    model = load_model(args.model_path, config, device)
    
    # Create save directory
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = None
    
    if args.interpolate and config.model.model_type == 'vae':
        # Generate interpolation
        print("Generating interpolation...")
        
        # Generate two random samples for interpolation
        sample1 = torch.randn(1, config.model.latent_dim, device=device)
        sample2 = torch.randn(1, config.model.latent_dim, device=device)
        
        # Generate samples from latent vectors
        with torch.no_grad():
            sample1_img = model.decode(sample1)
            sample2_img = model.decode(sample2)
        
        interpolations = interpolate_samples(
            model=model,
            sample1=sample1_img,
            sample2=sample2_img,
            num_steps=args.interpolation_steps,
            device=device,
            model_type=config.model.model_type,
            save_path=save_dir / "interpolation.png" if save_dir else None
        )
        
    else:
        # Generate regular samples
        print(f"Generating {args.num_samples} samples...")
        samples = generate_samples(
            model=model,
            num_samples=args.num_samples,
            device=device,
            model_type=config.model.model_type,
            save_dir=save_dir
        )
        
        # Create grid visualization
        grid_size = tuple(map(int, args.grid_size.split('x')))
        create_sample_grid(
            samples=samples,
            grid_size=grid_size,
            save_path=save_dir / "sample_grid.png" if save_dir else None
        )
        
        print(f"Generated {len(samples)} samples")
        if save_dir:
            print(f"Samples saved to {save_dir}")


if __name__ == '__main__':
    main()
