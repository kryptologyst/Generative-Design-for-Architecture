"""Training configuration for generative design models."""

from dataclasses import dataclass
from typing import Tuple, Optional
from pathlib import Path


@dataclass
class DataConfig:
    """Data configuration."""
    width: int = 32
    height: int = 32
    min_rooms: int = 3
    max_rooms: int = 8
    min_room_size: int = 3
    max_room_size: int = 8
    room_types: int = 5
    num_train_samples: int = 10000
    num_val_samples: int = 2000
    batch_size: int = 32
    num_workers: int = 4


@dataclass
class ModelConfig:
    """Model configuration."""
    model_type: str = "vae"  # "vae" or "gan"
    input_size: Tuple[int, int] = (32, 32)
    latent_dim: int = 128
    hidden_dims: Tuple[int, ...] = (512, 256)
    beta: float = 1.0
    kl_annealing: bool = True
    use_spectral_norm: bool = True


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    beta1: float = 0.5
    beta2: float = 0.999
    d_lr: float = 1e-4  # Discriminator learning rate for GAN
    g_lr: float = 1e-4  # Generator learning rate for GAN
    d_steps: int = 1  # Discriminator steps per generator step
    g_steps: int = 1  # Generator steps per discriminator step
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    val_check_interval: float = 1.0
    log_every_n_steps: int = 50
    save_top_k: int = 3
    monitor: str = "val_loss"


@dataclass
class Config:
    """Main configuration class."""
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    
    # Paths
    data_dir: str = "data"
    model_dir: str = "models"
    log_dir: str = "logs"
    assets_dir: str = "assets"
    
    # Reproducibility
    seed: int = 42
    
    # Device
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    
    # Logging
    use_wandb: bool = False
    wandb_project: str = "generative-design-architecture"
    wandb_entity: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Create directories
        for dir_path in [self.data_dir, self.model_dir, self.log_dir, self.assets_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
