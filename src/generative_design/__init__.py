"""Generative Design for Architecture Package.

This package provides neural network-based generative design tools for architectural
layout generation, including VAE and GAN implementations.
"""

__version__ = "0.1.0"
__author__ = "AI Projects"

from .models import VAEArchitecturalGenerator, GANArchitecturalGenerator
from .data import ArchitecturalDataset, FloorPlanGenerator
from .utils import set_seed, get_device, visualize_floor_plan

__all__ = [
    "VAEArchitecturalGenerator",
    "GANArchitecturalGenerator", 
    "ArchitecturalDataset",
    "FloorPlanGenerator",
    "set_seed",
    "get_device",
    "visualize_floor_plan",
]
