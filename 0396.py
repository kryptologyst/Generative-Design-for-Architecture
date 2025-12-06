#!/usr/bin/env python3
"""
Project 396: Generative Design for Architecture

This is a modernized version of the original generative design project.
The original simple genetic algorithm has been replaced with state-of-the-art
neural network architectures (VAE and GAN) for architectural layout generation.

For the full implementation, see the src/generative_design/ package.
This file serves as a backward-compatible demo of the original approach.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import the modern implementation
try:
    from generative_design.data import FloorPlanGenerator
    from generative_design.utils import visualize_floor_plan, calculate_room_statistics
    MODERN_IMPL_AVAILABLE = True
except ImportError:
    MODERN_IMPL_AVAILABLE = False
    print("Modern implementation not available. Using legacy version.")


def generate_floor_plan_legacy(width=10, height=10):
    """Legacy floor plan generation function (original implementation)."""
    # Randomly place rooms in a grid
    floor_plan = np.zeros((height, width))
    
    num_rooms = np.random.randint(3, 6)  # Random number of rooms
    for _ in range(num_rooms):
        room_width = np.random.randint(2, width // 2)
        room_height = np.random.randint(2, height // 2)
        
        x_pos = np.random.randint(0, width - room_width)
        y_pos = np.random.randint(0, height - room_height)
        
        # Place room in the grid
        floor_plan[y_pos:y_pos + room_height, x_pos:x_pos + room_width] = 1
    
    return floor_plan


def plot_floor_plan_legacy(floor_plan, title="Generated Floor Plan"):
    """Legacy visualization function."""
    plt.figure(figsize=(8, 8))
    plt.imshow(floor_plan, cmap='gray', origin='upper')
    plt.title(title)
    plt.axis('off')
    plt.show()


def demo_legacy_approach():
    """Demonstrate the original genetic algorithm approach."""
    print("=== Legacy Genetic Algorithm Approach ===")
    
    # Generate and display a random floor plan
    floor_plan = generate_floor_plan_legacy(width=10, height=10)
    plot_floor_plan_legacy(floor_plan)
    
    # Calculate basic statistics
    unique_rooms, counts = np.unique(floor_plan, return_counts=True)
    num_rooms = len(unique_rooms) - 1  # Exclude background
    
    print(f"Generated floor plan with {num_rooms} rooms")
    print(f"Total coverage: {np.sum(floor_plan > 0) / floor_plan.size:.1%}")
    
    return floor_plan


def demo_modern_approach():
    """Demonstrate the modern neural network approach."""
    if not MODERN_IMPL_AVAILABLE:
        print("Modern implementation not available. Install the package to use neural networks.")
        return
    
    print("\n=== Modern Neural Network Approach ===")
    
    # Create modern floor plan generator
    generator = FloorPlanGenerator(
        width=32, height=32,
        min_rooms=3, max_rooms=8,
        min_room_size=3, max_room_size=8,
        room_types=5
    )
    
    # Generate multiple floor plans
    floor_plans = generator.generate_dataset(4)
    
    # Visualize with modern tools
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, floor_plan in enumerate(floor_plans):
        axes[i].imshow(floor_plan, cmap='viridis', origin='upper')
        axes[i].set_title(f'Modern Floor Plan {i+1}')
        axes[i].axis('off')
        
        # Calculate statistics
        stats = calculate_room_statistics(floor_plan)
        print(f"Floor Plan {i+1}: {stats['total_rooms']} rooms, "
              f"{stats['coverage_ratio']:.1%} coverage, "
              f"avg room size: {stats['avg_room_size']:.1f}")
    
    plt.tight_layout()
    plt.show()
    
    return floor_plans


def main():
    """Main demonstration function."""
    print("Project 396: Generative Design for Architecture")
    print("=" * 50)
    
    # Demo legacy approach
    legacy_plan = demo_legacy_approach()
    
    # Demo modern approach
    modern_plans = demo_modern_approach()
    
    print("\n=== Comparison ===")
    print("Legacy approach:")
    print("- Simple random room placement")
    print("- Basic visualization")
    print("- Limited control over room properties")
    
    if MODERN_IMPL_AVAILABLE:
        print("\nModern approach:")
        print("- Neural network-based generation (VAE/GAN)")
        print("- Advanced evaluation metrics")
        print("- Configurable room constraints")
        print("- Interactive web interface")
        print("- Comprehensive training pipeline")
        print("\nTo use the full modern implementation:")
        print("1. Install: pip install -e .")
        print("2. Train: python scripts/train.py --model vae")
        print("3. Demo: streamlit run demo/app.py")
    else:
        print("\nTo use the modern implementation:")
        print("1. Install the package dependencies")
        print("2. Run: pip install -e .")
        print("3. Follow the README.md instructions")


if __name__ == "__main__":
    main()