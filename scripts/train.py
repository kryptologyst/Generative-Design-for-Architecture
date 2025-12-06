"""Training script for generative design models."""

import argparse
import os
from pathlib import Path
from typing import Dict, Any

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from generative_design.config import Config
from generative_design.data import FloorPlanGenerator, create_data_loaders
from generative_design.models import VAEArchitecturalGenerator, GANArchitecturalGenerator
from generative_design.utils import set_seed, get_device


class VAETrainingModule(pl.LightningModule):
    """PyTorch Lightning module for VAE training."""
    
    def __init__(self, config: Config):
        """Initialize the training module.
        
        Args:
            config: Configuration object.
        """
        super().__init__()
        self.config = config
        self.model = VAEArchitecturalGenerator(
            input_size=config.model.input_size,
            latent_dim=config.model.latent_dim,
            hidden_dims=config.model.hidden_dims,
            beta=config.model.beta,
            kl_annealing=config.model.kl_annealing
        )
        
        # KL annealing
        self.kl_annealing_factor = 0.0
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        return self.model(x)
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Training step."""
        # Update KL annealing factor
        if self.config.model.kl_annealing:
            self.kl_annealing_factor = min(1.0, self.current_epoch / 50.0)
        
        # Forward pass
        outputs = self(batch)
        
        # Calculate loss
        from generative_design.models import vae_loss
        losses = vae_loss(
            outputs['reconstruction'],
            batch,
            outputs['mu'],
            outputs['logvar'],
            beta=self.config.model.beta,
            kl_annealing_factor=self.kl_annealing_factor
        )
        
        # Log metrics
        self.log('train_loss', losses['total_loss'], on_step=True, on_epoch=True)
        self.log('train_recon_loss', losses['recon_loss'], on_step=True, on_epoch=True)
        self.log('train_kl_loss', losses['kl_loss'], on_step=True, on_epoch=True)
        self.log('kl_annealing_factor', self.kl_annealing_factor, on_step=False, on_epoch=True)
        
        return losses['total_loss']
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        outputs = self(batch)
        
        from generative_design.models import vae_loss
        losses = vae_loss(
            outputs['reconstruction'],
            batch,
            outputs['mu'],
            outputs['logvar'],
            beta=self.config.model.beta,
            kl_annealing_factor=1.0  # Full KL weight for validation
        )
        
        # Log metrics
        self.log('val_loss', losses['total_loss'], on_step=False, on_epoch=True)
        self.log('val_recon_loss', losses['recon_loss'], on_step=False, on_epoch=True)
        self.log('val_kl_loss', losses['kl_loss'], on_step=False, on_epoch=True)
        
        return losses['total_loss']
    
    def configure_optimizers(self):
        """Configure optimizers."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            betas=(self.config.training.beta1, self.config.training.beta2)
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.training.epochs
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }


class GANTrainingModule(pl.LightningModule):
    """PyTorch Lightning module for GAN training."""
    
    def __init__(self, config: Config):
        """Initialize the training module.
        
        Args:
            config: Configuration object.
        """
        super().__init__()
        self.config = config
        self.model = GANArchitecturalGenerator(
            input_size=config.model.input_size,
            latent_dim=config.model.latent_dim,
            generator_hidden_dims=config.model.hidden_dims,
            discriminator_hidden_dims=config.model.hidden_dims,
            use_spectral_norm=config.model.use_spectral_norm
        )
        
        # Separate optimizers for generator and discriminator
        self.g_optimizer = None
        self.d_optimizer = None
        
    def forward_generator(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through generator."""
        return self.model.forward_generator(z)
    
    def forward_discriminator(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through discriminator."""
        return self.model.forward_discriminator(x)
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Training step."""
        batch_size = batch.size(0)
        
        # Generate fake samples
        z = torch.randn(batch_size, self.config.model.latent_dim, device=self.device)
        fake_samples = self.forward_generator(z)
        
        # Discriminator loss
        real_pred = self.forward_discriminator(batch)
        fake_pred = self.forward_discriminator(fake_samples.detach())
        
        from generative_design.models import gan_loss
        losses = gan_loss(real_pred, fake_pred, loss_type='hinge')
        
        # Log discriminator metrics
        self.log('train_d_loss', losses['d_loss'], on_step=True, on_epoch=True)
        
        # Generator loss
        fake_pred_gen = self.forward_discriminator(fake_samples)
        g_loss = -fake_pred_gen.mean()
        
        # Log generator metrics
        self.log('train_g_loss', g_loss, on_step=True, on_epoch=True)
        
        return {
            'd_loss': losses['d_loss'],
            'g_loss': g_loss,
            'loss': losses['d_loss'] + g_loss  # For monitoring
        }
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        batch_size = batch.size(0)
        
        # Generate fake samples
        z = torch.randn(batch_size, self.config.model.latent_dim, device=self.device)
        fake_samples = self.forward_generator(z)
        
        # Calculate losses
        real_pred = self.forward_discriminator(batch)
        fake_pred = self.forward_discriminator(fake_samples)
        
        from generative_design.models import gan_loss
        losses = gan_loss(real_pred, fake_pred, loss_type='hinge')
        
        # Log metrics
        self.log('val_d_loss', losses['d_loss'], on_step=False, on_epoch=True)
        self.log('val_g_loss', losses['g_loss'], on_step=False, on_epoch=True)
        
        return losses['d_loss'] + losses['g_loss']
    
    def configure_optimizers(self):
        """Configure optimizers."""
        # Generator optimizer
        self.g_optimizer = torch.optim.Adam(
            self.model.generator.parameters(),
            lr=self.config.training.g_lr,
            betas=(self.config.training.beta1, self.config.training.beta2)
        )
        
        # Discriminator optimizer
        self.d_optimizer = torch.optim.Adam(
            self.model.discriminator.parameters(),
            lr=self.config.training.d_lr,
            betas=(self.config.training.beta1, self.config.training.beta2)
        )
        
        return [self.g_optimizer, self.d_optimizer]


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train generative design models')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--model', type=str, choices=['vae', 'gan'], default='vae',
                       help='Model type to train')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--wandb', action='store_true', help='Use wandb logging')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    if args.config:
        # Load from file if provided
        import yaml
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
            config = Config(**config_dict)
    
    # Override with command line arguments
    config.model.model_type = args.model
    config.training.epochs = args.epochs
    config.data.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.device = args.device
    config.seed = args.seed
    config.use_wandb = args.wandb
    
    # Set seed
    set_seed(config.seed)
    
    # Generate data
    print("Generating synthetic floor plan data...")
    generator = FloorPlanGenerator(
        width=config.data.width,
        height=config.data.height,
        min_rooms=config.data.min_rooms,
        max_rooms=config.data.max_rooms,
        min_room_size=config.data.min_room_size,
        max_room_size=config.data.max_room_size,
        room_types=config.data.room_types
    )
    
    train_data = generator.generate_dataset(config.data.num_train_samples)
    val_data = generator.generate_dataset(config.data.num_val_samples)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_data=train_data,
        val_data=val_data,
        batch_size=config.data.batch_size,
        image_size=(config.data.height, config.data.width),
        num_workers=config.data.num_workers
    )
    
    # Create model
    if config.model.model_type == 'vae':
        model = VAETrainingModule(config)
    elif config.model.model_type == 'gan':
        model = GANTrainingModule(config)
    else:
        raise ValueError(f"Unknown model type: {config.model.model_type}")
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=config.model_dir,
            filename=f'{config.model.model_type}-{{epoch:02d}}-{{val_loss:.4f}}',
            monitor=config.training.monitor,
            mode='min',
            save_top_k=config.training.save_top_k,
            save_last=True
        ),
        EarlyStopping(
            monitor=config.training.monitor,
            mode='min',
            patience=20,
            verbose=True
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]
    
    # Setup logger
    if config.use_wandb:
        logger = WandbLogger(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=f'{config.model.model_type}-training'
        )
    else:
        logger = TensorBoardLogger(
            save_dir=config.log_dir,
            name=f'{config.model.model_type}_training'
        )
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=config.training.epochs,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=config.training.gradient_clip_val,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        val_check_interval=config.training.val_check_interval,
        log_every_n_steps=config.training.log_every_n_steps,
        devices=1,
        accelerator='auto' if config.device == 'auto' else config.device,
        precision=16,  # Mixed precision
        deterministic=True
    )
    
    # Train model
    print(f"Starting training of {config.model.model_type.upper()} model...")
    trainer.fit(model, train_loader, val_loader)
    
    print("Training completed!")


if __name__ == '__main__':
    main()
