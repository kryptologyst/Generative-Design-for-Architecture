"""Utility functions for generative design architecture project."""

import random
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device for computation.
    
    Returns:
        torch.device: CUDA if available, else MPS (Apple Silicon), else CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def visualize_floor_plan(
    floor_plan: np.ndarray,
    title: str = "Generated Floor Plan",
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (8, 8)
) -> None:
    """Visualize a floor plan as a 2D grid.
    
    Args:
        floor_plan: 2D numpy array representing the floor plan.
        title: Title for the plot.
        save_path: Optional path to save the figure.
        figsize: Figure size tuple (width, height).
    """
    plt.figure(figsize=figsize)
    plt.imshow(floor_plan, cmap='viridis', origin='upper')
    plt.title(title)
    plt.colorbar(label='Room Type')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def calculate_room_statistics(floor_plan: np.ndarray) -> dict[str, float]:
    """Calculate basic statistics for a floor plan.
    
    Args:
        floor_plan: 2D numpy array representing the floor plan.
        
    Returns:
        Dictionary containing room statistics.
    """
    unique_rooms, counts = np.unique(floor_plan, return_counts=True)
    
    stats = {
        'total_rooms': len(unique_rooms) - 1,  # Exclude background (0)
        'total_area': np.sum(floor_plan > 0),
        'coverage_ratio': np.sum(floor_plan > 0) / floor_plan.size,
        'room_diversity': len(unique_rooms) - 1,
    }
    
    if len(unique_rooms) > 1:
        room_areas = counts[1:]  # Exclude background
        stats['avg_room_size'] = np.mean(room_areas)
        stats['room_size_std'] = np.std(room_areas)
        stats['largest_room'] = np.max(room_areas)
        stats['smallest_room'] = np.min(room_areas)
    
    return stats
