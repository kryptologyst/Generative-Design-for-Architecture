"""Tests for generative design architecture package."""

import pytest
import torch
import numpy as np

from generative_design.data import FloorPlanGenerator, ArchitecturalDataset
from generative_design.models import VAEArchitecturalGenerator, GANArchitecturalGenerator
from generative_design.metrics import ArchitecturalMetrics
from generative_design.utils import set_seed, get_device, calculate_room_statistics


class TestFloorPlanGenerator:
    """Test floor plan generation."""
    
    def test_generate_floor_plan(self):
        """Test basic floor plan generation."""
        generator = FloorPlanGenerator(width=16, height=16)
        floor_plan = generator.generate_floor_plan()
        
        assert floor_plan.shape == (16, 16)
        assert floor_plan.dtype == np.int32
        assert np.all(floor_plan >= 0)
        assert np.any(floor_plan > 0)  # Should have some rooms
    
    def test_generate_dataset(self):
        """Test dataset generation."""
        generator = FloorPlanGenerator(width=16, height=16)
        dataset = generator.generate_dataset(10)
        
        assert len(dataset) == 10
        assert all(plan.shape == (16, 16) for plan in dataset)
    
    def test_room_constraints(self):
        """Test room constraints are respected."""
        generator = FloorPlanGenerator(
            width=20, height=20,
            min_rooms=3, max_rooms=5,
            min_room_size=2, max_room_size=4
        )
        
        floor_plan = generator.generate_floor_plan()
        unique_rooms = len(np.unique(floor_plan)) - 1  # Exclude background
        
        assert 3 <= unique_rooms <= 5


class TestArchitecturalDataset:
    """Test PyTorch dataset."""
    
    def test_dataset_creation(self):
        """Test dataset creation."""
        generator = FloorPlanGenerator(width=16, height=16)
        data = generator.generate_dataset(5)
        
        dataset = ArchitecturalDataset(data)
        
        assert len(dataset) == 5
        sample = dataset[0]
        assert isinstance(sample, torch.Tensor)
        assert sample.shape == (1, 16, 16)  # Channel dimension added
    
    def test_normalization(self):
        """Test data normalization."""
        generator = FloorPlanGenerator(width=16, height=16)
        data = generator.generate_dataset(3)
        
        dataset = ArchitecturalDataset(data, normalize=True)
        sample = dataset[0]
        
        assert torch.all(sample >= 0)
        assert torch.all(sample <= 1)


class TestVAEModel:
    """Test VAE model."""
    
    def test_vae_creation(self):
        """Test VAE model creation."""
        model = VAEArchitecturalGenerator(
            input_size=(32, 32),
            latent_dim=64,
            hidden_dims=(256, 128)
        )
        
        assert model.latent_dim == 64
        assert model.input_size == (32, 32)
    
    def test_vae_forward(self):
        """Test VAE forward pass."""
        model = VAEArchitecturalGenerator(input_size=(16, 16), latent_dim=32)
        
        x = torch.randn(2, 1, 16, 16)
        outputs = model(x)
        
        assert 'reconstruction' in outputs
        assert 'mu' in outputs
        assert 'logvar' in outputs
        assert 'z' in outputs
        
        assert outputs['reconstruction'].shape == x.shape
        assert outputs['mu'].shape == (2, 32)
        assert outputs['logvar'].shape == (2, 32)
    
    def test_vae_sampling(self):
        """Test VAE sampling."""
        model = VAEArchitecturalGenerator(input_size=(16, 16), latent_dim=32)
        device = torch.device('cpu')
        
        samples = model.sample(3, device)
        
        assert samples.shape == (3, 1, 16, 16)
        assert torch.all(samples >= 0)
        assert torch.all(samples <= 1)


class TestGANModel:
    """Test GAN model."""
    
    def test_gan_creation(self):
        """Test GAN model creation."""
        model = GANArchitecturalGenerator(
            input_size=(32, 32),
            latent_dim=64,
            generator_hidden_dims=(256, 128),
            discriminator_hidden_dims=(128, 256)
        )
        
        assert model.latent_dim == 64
        assert model.input_size == (32, 32)
    
    def test_gan_generator(self):
        """Test GAN generator."""
        model = GANArchitecturalGenerator(input_size=(16, 16), latent_dim=32)
        
        z = torch.randn(2, 32)
        generated = model.forward_generator(z)
        
        assert generated.shape == (2, 1, 16, 16)
        assert torch.all(generated >= -1)  # Tanh output
        assert torch.all(generated <= 1)
    
    def test_gan_discriminator(self):
        """Test GAN discriminator."""
        model = GANArchitecturalGenerator(input_size=(16, 16), latent_dim=32)
        
        x = torch.randn(2, 1, 16, 16)
        output = model.forward_discriminator(x)
        
        assert output.shape == (2,)
    
    def test_gan_sampling(self):
        """Test GAN sampling."""
        model = GANArchitecturalGenerator(input_size=(16, 16), latent_dim=32)
        device = torch.device('cpu')
        
        samples = model.sample(3, device)
        
        assert samples.shape == (3, 1, 16, 16)
        assert torch.all(samples >= -1)
        assert torch.all(samples <= 1)


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_room_statistics(self):
        """Test room statistics calculation."""
        metrics = ArchitecturalMetrics(room_types=5)
        
        # Create a simple floor plan
        floor_plan = np.array([
            [1, 1, 0, 0],
            [1, 1, 2, 2],
            [0, 0, 2, 2],
            [0, 0, 0, 0]
        ])
        
        stats = metrics.calculate_room_statistics(floor_plan)
        
        assert stats['total_rooms'] == 2
        assert stats['total_area'] == 6
        assert stats['coverage_ratio'] == 6 / 16
    
    def test_connectivity(self):
        """Test connectivity calculation."""
        metrics = ArchitecturalMetrics()
        
        floor_plan = np.array([
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        
        conn = metrics.calculate_connectivity(floor_plan)
        
        assert conn['horizontal_connections'] == 1  # Between rooms 1 and 2
        assert conn['vertical_connections'] == 0
        assert conn['total_connections'] == 1
    
    def test_compactness(self):
        """Test compactness calculation."""
        metrics = ArchitecturalMetrics()
        
        floor_plan = np.array([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        
        comp = metrics.calculate_compactness(floor_plan)
        
        assert comp['compactness'] == 1.0  # Perfectly compact
        assert comp['efficiency'] == 4 / 16  # 4 occupied out of 16 total
    
    def test_diversity(self):
        """Test diversity calculation."""
        metrics = ArchitecturalMetrics()
        
        floor_plans = [
            np.array([[1, 1], [1, 1]]),
            np.array([[2, 2], [2, 2]]),
            np.array([[1, 1], [1, 1]])  # Duplicate
        ]
        
        diversity = metrics.calculate_diversity(floor_plans)
        
        assert diversity['uniqueness'] == 2 / 3  # 2 unique out of 3
        assert diversity['num_unique'] == 2


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        val1 = np.random.random()
        
        set_seed(42)
        val2 = np.random.random()
        
        assert val1 == val2
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_calculate_room_statistics(self):
        """Test room statistics utility."""
        floor_plan = np.array([
            [1, 1, 0],
            [1, 1, 2],
            [0, 0, 2]
        ])
        
        stats = calculate_room_statistics(floor_plan)
        
        assert stats['total_rooms'] == 2
        assert stats['total_area'] == 5
        assert stats['coverage_ratio'] == 5 / 9


if __name__ == '__main__':
    pytest.main([__file__])
