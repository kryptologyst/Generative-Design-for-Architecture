"""Data generation and loading utilities for architectural layouts."""

import os
from typing import Optional, Tuple, List
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class FloorPlanGenerator:
    """Generator for synthetic floor plan data."""
    
    def __init__(
        self,
        width: int = 32,
        height: int = 32,
        min_rooms: int = 3,
        max_rooms: int = 8,
        min_room_size: int = 3,
        max_room_size: int = 8,
        room_types: int = 5
    ):
        """Initialize the floor plan generator.
        
        Args:
            width: Width of the floor plan grid.
            height: Height of the floor plan grid.
            min_rooms: Minimum number of rooms to generate.
            max_rooms: Maximum number of rooms to generate.
            min_room_size: Minimum room size (width or height).
            max_room_size: Maximum room size (width or height).
            room_types: Number of different room types (excluding background).
        """
        self.width = width
        self.height = height
        self.min_rooms = min_rooms
        self.max_rooms = max_rooms
        self.min_room_size = min_room_size
        self.max_room_size = max_room_size
        self.room_types = room_types
        
    def generate_floor_plan(self) -> np.ndarray:
        """Generate a random floor plan.
        
        Returns:
            2D numpy array representing the floor plan.
        """
        floor_plan = np.zeros((self.height, self.width), dtype=np.int32)
        
        num_rooms = np.random.randint(self.min_rooms, self.max_rooms + 1)
        
        for room_id in range(1, num_rooms + 1):
            # Try to place room with collision detection
            max_attempts = 50
            placed = False
            
            for _ in range(max_attempts):
                room_width = np.random.randint(
                    self.min_room_size, 
                    min(self.max_room_size + 1, self.width // 2)
                )
                room_height = np.random.randint(
                    self.min_room_size, 
                    min(self.max_room_size + 1, self.height // 2)
                )
                
                x_pos = np.random.randint(0, self.width - room_width + 1)
                y_pos = np.random.randint(0, self.height - room_height + 1)
                
                # Check for overlap
                if np.all(floor_plan[y_pos:y_pos + room_height, x_pos:x_pos + room_width] == 0):
                    # Assign room type (1 to room_types)
                    room_type = np.random.randint(1, self.room_types + 1)
                    floor_plan[y_pos:y_pos + room_height, x_pos:x_pos + room_width] = room_type
                    placed = True
                    break
            
            if not placed:
                # If we can't place a room, try a smaller one
                room_width = min(room_width, 2)
                room_height = min(room_height, 2)
                for x_pos in range(0, self.width - room_width + 1, 2):
                    for y_pos in range(0, self.height - room_height + 1, 2):
                        if np.all(floor_plan[y_pos:y_pos + room_height, x_pos:x_pos + room_width] == 0):
                            room_type = np.random.randint(1, self.room_types + 1)
                            floor_plan[y_pos:y_pos + room_height, x_pos:x_pos + room_width] = room_type
                            placed = True
                            break
                    if placed:
                        break
        
        return floor_plan
    
    def generate_dataset(self, num_samples: int) -> List[np.ndarray]:
        """Generate a dataset of floor plans.
        
        Args:
            num_samples: Number of floor plans to generate.
            
        Returns:
            List of floor plan arrays.
        """
        return [self.generate_floor_plan() for _ in range(num_samples)]


class ArchitecturalDataset(Dataset):
    """PyTorch dataset for architectural floor plans."""
    
    def __init__(
        self,
        data: List[np.ndarray],
        transform: Optional[A.Compose] = None,
        normalize: bool = True
    ):
        """Initialize the dataset.
        
        Args:
            data: List of floor plan arrays.
            transform: Albumentations transform pipeline.
            normalize: Whether to normalize data to [0, 1].
        """
        self.data = data
        self.transform = transform
        self.normalize = normalize
        
    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a floor plan by index.
        
        Args:
            idx: Index of the floor plan.
            
        Returns:
            Tensor representation of the floor plan.
        """
        floor_plan = self.data[idx].astype(np.float32)
        
        if self.normalize:
            # Normalize to [0, 1] range
            floor_plan = floor_plan / self.data[idx].max()
        
        if self.transform:
            # Apply transforms
            transformed = self.transform(image=floor_plan)
            floor_plan = transformed['image']
        else:
            # Convert to tensor
            floor_plan = torch.from_numpy(floor_plan).unsqueeze(0)  # Add channel dimension
        
        return floor_plan


def get_transforms(
    image_size: Tuple[int, int] = (32, 32),
    training: bool = True
) -> A.Compose:
    """Get data augmentation transforms.
    
    Args:
        image_size: Target image size (height, width).
        training: Whether to apply training augmentations.
        
    Returns:
        Albumentations transform pipeline.
    """
    if training:
        transforms = A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate90(p=0.5),
            A.GaussNoise(var_limit=(0.01, 0.05), p=0.3),
            ToTensorV2(),
        ])
    else:
        transforms = A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            ToTensorV2(),
        ])
    
    return transforms


def create_data_loaders(
    train_data: List[np.ndarray],
    val_data: List[np.ndarray],
    batch_size: int = 32,
    image_size: Tuple[int, int] = (32, 32),
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders.
    
    Args:
        train_data: Training floor plans.
        val_data: Validation floor plans.
        batch_size: Batch size for data loaders.
        image_size: Target image size.
        num_workers: Number of worker processes.
        
    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_transform = get_transforms(image_size, training=True)
    val_transform = get_transforms(image_size, training=False)
    
    train_dataset = ArchitecturalDataset(train_data, train_transform)
    val_dataset = ArchitecturalDataset(val_data, val_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
