"""Variational Autoencoder for architectural layout generation."""

from typing import Tuple, Dict, Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class VAEArchitecturalGenerator(nn.Module):
    """Variational Autoencoder for generating architectural floor plans."""
    
    def __init__(
        self,
        input_size: Tuple[int, int] = (32, 32),
        latent_dim: int = 128,
        hidden_dims: Tuple[int, ...] = (512, 256),
        beta: float = 1.0,
        kl_annealing: bool = True
    ):
        """Initialize the VAE.
        
        Args:
            input_size: Input image size (height, width).
            latent_dim: Dimension of the latent space.
            hidden_dims: Hidden layer dimensions for encoder/decoder.
            beta: Beta parameter for beta-VAE.
            kl_annealing: Whether to use KL annealing.
        """
        super().__init__()
        
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.beta = beta
        self.kl_annealing = kl_annealing
        
        # Calculate the size after convolutions
        self.conv_output_size = self._calculate_conv_output_size()
        
        # Encoder
        self.encoder = self._build_encoder()
        
        # Latent space projections
        self.fc_mu = nn.Linear(self.conv_output_size, latent_dim)
        self.fc_logvar = nn.Linear(self.conv_output_size, latent_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, self.conv_output_size)
        self.decoder = self._build_decoder()
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _calculate_conv_output_size(self) -> int:
        """Calculate the output size after encoder convolutions."""
        # Simulate forward pass to get output size
        x = torch.zeros(1, 1, *self.input_size)
        
        # Encoder convolutions
        x = F.relu(self._conv_block(1, 32, 4, 2, 1)(x))
        x = F.relu(self._conv_block(32, 64, 4, 2, 1)(x))
        x = F.relu(self._conv_block(64, 128, 4, 2, 1)(x))
        x = F.relu(self._conv_block(128, 256, 4, 2, 1)(x))
        
        return x.numel()
    
    def _conv_block(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        stride: int, 
        padding: int
    ) -> nn.Conv2d:
        """Create a convolutional block."""
        return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    
    def _deconv_block(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        stride: int, 
        padding: int
    ) -> nn.ConvTranspose2d:
        """Create a transposed convolutional block."""
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
    
    def _build_encoder(self) -> nn.Module:
        """Build the encoder network."""
        layers = []
        in_channels = 1
        
        for hidden_dim in self.hidden_dims:
            layers.extend([
                self._conv_block(in_channels, hidden_dim, 4, 2, 1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ])
            in_channels = hidden_dim
        
        return nn.Sequential(*layers)
    
    def _build_decoder(self) -> nn.Module:
        """Build the decoder network."""
        layers = []
        hidden_dims = list(self.hidden_dims)[::-1]  # Reverse order
        
        for i, hidden_dim in enumerate(hidden_dims):
            if i == 0:
                layers.extend([
                    self._deconv_block(hidden_dim, hidden_dim, 4, 2, 1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True)
                ])
            else:
                layers.extend([
                    self._deconv_block(hidden_dim, hidden_dim, 4, 2, 1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True)
                ])
        
        # Final layer to output
        layers.append(self._deconv_block(hidden_dims[-1], 1, 4, 2, 1))
        layers.append(nn.Sigmoid())
        
        return nn.Sequential(*layers)
    
    def _init_weights(self, m: nn.Module) -> None:
        """Initialize network weights."""
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent space.
        
        Args:
            x: Input tensor of shape (batch_size, 1, height, width).
            
        Returns:
            Tuple of (mu, logvar) for the latent distribution.
        """
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE.
        
        Args:
            mu: Mean of the latent distribution.
            logvar: Log variance of the latent distribution.
            
        Returns:
            Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction.
        
        Args:
            z: Latent vector of shape (batch_size, latent_dim).
            
        Returns:
            Reconstructed tensor of shape (batch_size, 1, height, width).
        """
        h = self.decoder_fc(z)
        h = h.view(h.size(0), 256, 2, 2)  # Reshape to match encoder output
        return self.decoder(h)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass of the VAE.
        
        Args:
            x: Input tensor of shape (batch_size, 1, height, width).
            
        Returns:
            Dictionary containing reconstruction, mu, logvar, and z.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        
        return {
            'reconstruction': recon,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Sample from the VAE.
        
        Args:
            num_samples: Number of samples to generate.
            device: Device to generate samples on.
            
        Returns:
            Generated samples of shape (num_samples, 1, height, width).
        """
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=device)
            samples = self.decode(z)
        return samples
    
    def interpolate(
        self, 
        x1: torch.Tensor, 
        x2: torch.Tensor, 
        num_steps: int = 10
    ) -> torch.Tensor:
        """Interpolate between two inputs in latent space.
        
        Args:
            x1: First input tensor.
            x2: Second input tensor.
            num_steps: Number of interpolation steps.
            
        Returns:
            Interpolated samples.
        """
        mu1, logvar1 = self.encode(x1)
        mu2, logvar2 = self.encode(x2)
        
        z1 = self.reparameterize(mu1, logvar1)
        z2 = self.reparameterize(mu2, logvar2)
        
        interpolations = []
        for i in range(num_steps):
            alpha = i / (num_steps - 1)
            z_interp = (1 - alpha) * z1 + alpha * z2
            interp = self.decode(z_interp)
            interpolations.append(interp)
        
        return torch.cat(interpolations, dim=0)


def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    kl_annealing_factor: float = 1.0
) -> Dict[str, torch.Tensor]:
    """Calculate VAE loss.
    
    Args:
        recon: Reconstructed output.
        x: Original input.
        mu: Mean of latent distribution.
        logvar: Log variance of latent distribution.
        beta: Beta parameter for beta-VAE.
        kl_annealing_factor: KL annealing factor.
        
    Returns:
        Dictionary containing total loss and individual components.
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon, x, reduction='sum') / x.size(0)
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    
    # Apply beta and annealing
    kl_loss = beta * kl_annealing_factor * kl_loss
    
    total_loss = recon_loss + kl_loss
    
    return {
        'total_loss': total_loss,
        'recon_loss': recon_loss,
        'kl_loss': kl_loss
    }


class GANArchitecturalGenerator(nn.Module):
    """Generative Adversarial Network for architectural layout generation."""
    
    def __init__(
        self,
        input_size: Tuple[int, int] = (32, 32),
        latent_dim: int = 128,
        generator_hidden_dims: Tuple[int, ...] = (512, 256),
        discriminator_hidden_dims: Tuple[int, ...] = (256, 512),
        use_spectral_norm: bool = True
    ):
        """Initialize the GAN.
        
        Args:
            input_size: Input image size (height, width).
            latent_dim: Dimension of the latent space.
            generator_hidden_dims: Hidden layer dimensions for generator.
            discriminator_hidden_dims: Hidden layer dimensions for discriminator.
            use_spectral_norm: Whether to use spectral normalization.
        """
        super().__init__()
        
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.use_spectral_norm = use_spectral_norm
        
        # Generator
        self.generator = self._build_generator(generator_hidden_dims)
        
        # Discriminator
        self.discriminator = self._build_discriminator(discriminator_hidden_dims)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _build_generator(self, hidden_dims: Tuple[int, ...]) -> nn.Module:
        """Build the generator network."""
        layers = []
        
        # Initial projection from latent to feature map
        initial_size = 4  # Start with 4x4 feature map
        initial_channels = hidden_dims[0]
        
        layers.append(nn.Linear(self.latent_dim, initial_channels * initial_size * initial_size))
        layers.append(nn.BatchNorm1d(initial_channels * initial_size * initial_size))
        layers.append(nn.ReLU(inplace=True))
        
        # Reshape to feature map
        layers.append(Reshape(initial_channels, initial_size, initial_size))
        
        # Transposed convolutions
        in_channels = initial_channels
        for hidden_dim in hidden_dims[1:]:
            layers.extend([
                nn.ConvTranspose2d(in_channels, hidden_dim, 4, 2, 1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ])
            in_channels = hidden_dim
        
        # Final layer
        layers.extend([
            nn.ConvTranspose2d(in_channels, 1, 4, 2, 1),
            nn.Tanh()
        ])
        
        return nn.Sequential(*layers)
    
    def _build_discriminator(self, hidden_dims: Tuple[int, ...]) -> nn.Module:
        """Build the discriminator network."""
        layers = []
        
        # First layer
        layers.append(nn.Conv2d(1, hidden_dims[0], 4, 2, 1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Hidden layers
        in_channels = hidden_dims[0]
        for hidden_dim in hidden_dims[1:]:
            conv = nn.Conv2d(in_channels, hidden_dim, 4, 2, 1)
            if self.use_spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            layers.extend([
                conv,
                nn.BatchNorm2d(hidden_dim),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            in_channels = hidden_dim
        
        # Final layer
        final_conv = nn.Conv2d(in_channels, 1, 4, 1, 0)
        if self.use_spectral_norm:
            final_conv = nn.utils.spectral_norm(final_conv)
        layers.append(final_conv)
        
        return nn.Sequential(*layers)
    
    def _init_weights(self, m: nn.Module) -> None:
        """Initialize network weights."""
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            nn.init.constant_(m.bias, 0)
    
    def forward_generator(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through generator.
        
        Args:
            z: Latent vector of shape (batch_size, latent_dim).
            
        Returns:
            Generated tensor of shape (batch_size, 1, height, width).
        """
        return self.generator(z)
    
    def forward_discriminator(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through discriminator.
        
        Args:
            x: Input tensor of shape (batch_size, 1, height, width).
            
        Returns:
            Discriminator output of shape (batch_size, 1).
        """
        return self.discriminator(x).view(-1)
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Sample from the generator.
        
        Args:
            num_samples: Number of samples to generate.
            device: Device to generate samples on.
            
        Returns:
            Generated samples of shape (num_samples, 1, height, width).
        """
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=device)
            samples = self.forward_generator(z)
        return samples


class Reshape(nn.Module):
    """Reshape layer for neural networks."""
    
    def __init__(self, *shape: int):
        """Initialize reshape layer.
        
        Args:
            *shape: Target shape dimensions.
        """
        super().__init__()
        self.shape = shape
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            Reshaped tensor.
        """
        return x.view(x.size(0), *self.shape)


def gan_loss(
    real_pred: torch.Tensor,
    fake_pred: torch.Tensor,
    loss_type: str = 'hinge'
) -> Dict[str, torch.Tensor]:
    """Calculate GAN loss.
    
    Args:
        real_pred: Discriminator predictions on real data.
        fake_pred: Discriminator predictions on fake data.
        loss_type: Type of loss ('hinge', 'ns', 'wgan').
        
    Returns:
        Dictionary containing generator and discriminator losses.
    """
    if loss_type == 'hinge':
        # Hinge loss
        d_loss_real = F.relu(1.0 - real_pred).mean()
        d_loss_fake = F.relu(1.0 + fake_pred).mean()
        d_loss = d_loss_real + d_loss_fake
        
        g_loss = -fake_pred.mean()
        
    elif loss_type == 'ns':
        # Non-saturating loss
        d_loss_real = F.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred))
        d_loss_fake = F.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred))
        d_loss = d_loss_real + d_loss_fake
        
        g_loss = F.binary_cross_entropy_with_logits(fake_pred, torch.ones_like(fake_pred))
        
    elif loss_type == 'wgan':
        # Wasserstein loss
        d_loss = -real_pred.mean() + fake_pred.mean()
        g_loss = -fake_pred.mean()
        
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    return {
        'd_loss': d_loss,
        'g_loss': g_loss
    }
