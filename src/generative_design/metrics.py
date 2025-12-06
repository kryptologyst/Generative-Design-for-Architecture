"""Evaluation metrics for architectural layout generation."""

from typing import List, Dict, Any, Optional
import numpy as np
import torch
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat


class ArchitecturalMetrics:
    """Collection of metrics for evaluating architectural layouts."""
    
    def __init__(self, room_types: int = 5):
        """Initialize metrics.
        
        Args:
            room_types: Number of room types (excluding background).
        """
        self.room_types = room_types
    
    def calculate_room_statistics(self, floor_plan: np.ndarray) -> Dict[str, float]:
        """Calculate basic room statistics.
        
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
            stats['room_size_cv'] = np.std(room_areas) / np.mean(room_areas)  # Coefficient of variation
        else:
            stats.update({
                'avg_room_size': 0.0,
                'room_size_std': 0.0,
                'largest_room': 0,
                'smallest_room': 0,
                'room_size_cv': 0.0
            })
        
        return stats
    
    def calculate_connectivity(self, floor_plan: np.ndarray) -> Dict[str, float]:
        """Calculate connectivity metrics.
        
        Args:
            floor_plan: 2D numpy array representing the floor plan.
            
        Returns:
            Dictionary containing connectivity metrics.
        """
        height, width = floor_plan.shape
        
        # Count adjacent rooms
        horizontal_connections = 0
        vertical_connections = 0
        
        for i in range(height):
            for j in range(width - 1):
                if floor_plan[i, j] != floor_plan[i, j + 1] and floor_plan[i, j] > 0 and floor_plan[i, j + 1] > 0:
                    horizontal_connections += 1
        
        for i in range(height - 1):
            for j in range(width):
                if floor_plan[i, j] != floor_plan[i + 1, j] and floor_plan[i, j] > 0 and floor_plan[i + 1, j] > 0:
                    vertical_connections += 1
        
        total_connections = horizontal_connections + vertical_connections
        
        return {
            'horizontal_connections': horizontal_connections,
            'vertical_connections': vertical_connections,
            'total_connections': total_connections,
            'connection_density': total_connections / (height * width)
        }
    
    def calculate_compactness(self, floor_plan: np.ndarray) -> Dict[str, float]:
        """Calculate compactness metrics.
        
        Args:
            floor_plan: 2D numpy array representing the floor plan.
            
        Returns:
            Dictionary containing compactness metrics.
        """
        # Find bounding box of occupied areas
        occupied = floor_plan > 0
        if not np.any(occupied):
            return {
                'compactness': 0.0,
                'aspect_ratio': 1.0,
                'efficiency': 0.0
            }
        
        rows, cols = np.where(occupied)
        min_row, max_row = np.min(rows), np.max(rows)
        min_col, max_col = np.min(cols), np.max(cols)
        
        occupied_area = np.sum(occupied)
        bounding_area = (max_row - min_row + 1) * (max_col - min_col + 1)
        
        compactness = occupied_area / bounding_area if bounding_area > 0 else 0.0
        
        # Aspect ratio
        height = max_row - min_row + 1
        width = max_col - min_col + 1
        aspect_ratio = max(height, width) / min(height, width) if min(height, width) > 0 else 1.0
        
        # Efficiency (occupied area / total area)
        efficiency = occupied_area / floor_plan.size
        
        return {
            'compactness': compactness,
            'aspect_ratio': aspect_ratio,
            'efficiency': efficiency,
            'bounding_box_area': bounding_area,
            'occupied_area': occupied_area
        }
    
    def calculate_diversity(self, floor_plans: List[np.ndarray]) -> Dict[str, float]:
        """Calculate diversity metrics across multiple floor plans.
        
        Args:
            floor_plans: List of floor plan arrays.
            
        Returns:
            Dictionary containing diversity metrics.
        """
        if len(floor_plans) < 2:
            return {'diversity': 0.0, 'uniqueness': 0.0}
        
        # Calculate pairwise differences
        differences = []
        for i in range(len(floor_plans)):
            for j in range(i + 1, len(floor_plans)):
                diff = np.mean(floor_plans[i] != floor_plans[j])
                differences.append(diff)
        
        avg_difference = np.mean(differences)
        
        # Calculate uniqueness (fraction of unique floor plans)
        unique_plans = len(set(tuple(plan.flatten()) for plan in floor_plans))
        uniqueness = unique_plans / len(floor_plans)
        
        return {
            'diversity': avg_difference,
            'uniqueness': uniqueness,
            'num_unique': unique_plans,
            'total_samples': len(floor_plans)
        }
    
    def evaluate_floor_plan(self, floor_plan: np.ndarray) -> Dict[str, Any]:
        """Evaluate a single floor plan.
        
        Args:
            floor_plan: 2D numpy array representing the floor plan.
            
        Returns:
            Dictionary containing all metrics.
        """
        metrics = {}
        
        # Room statistics
        metrics.update(self.calculate_room_statistics(floor_plan))
        
        # Connectivity
        metrics.update(self.calculate_connectivity(floor_plan))
        
        # Compactness
        metrics.update(self.calculate_compactness(floor_plan))
        
        return metrics
    
    def evaluate_dataset(self, floor_plans: List[np.ndarray]) -> Dict[str, Any]:
        """Evaluate a dataset of floor plans.
        
        Args:
            floor_plans: List of floor plan arrays.
            
        Returns:
            Dictionary containing aggregated metrics.
        """
        if not floor_plans:
            return {}
        
        # Individual metrics
        individual_metrics = [self.evaluate_floor_plan(plan) for plan in floor_plans]
        
        # Aggregate metrics
        aggregated = {}
        for key in individual_metrics[0].keys():
            values = [metrics[key] for metrics in individual_metrics]
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
            aggregated[f'{key}_min'] = np.min(values)
            aggregated[f'{key}_max'] = np.max(values)
        
        # Diversity metrics
        diversity_metrics = self.calculate_diversity(floor_plans)
        aggregated.update(diversity_metrics)
        
        return aggregated


class FloorPlanQualityMetric(Metric):
    """PyTorch Lightning metric for floor plan quality."""
    
    def __init__(self, room_types: int = 5):
        """Initialize the metric.
        
        Args:
            room_types: Number of room types.
        """
        super().__init__()
        self.room_types = room_types
        self.add_state("scores", default=[], dist_reduce_fx="cat")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric state.
        
        Args:
            preds: Predicted floor plans.
            target: Target floor plans.
        """
        # Convert to numpy for evaluation
        preds_np = preds.cpu().numpy()
        target_np = target.cpu().numpy()
        
        scores = []
        for pred, targ in zip(preds_np, target_np):
            # Calculate quality score based on multiple factors
            score = self._calculate_quality_score(pred, targ)
            scores.append(score)
        
        self.scores.extend(scores)
    
    def compute(self) -> torch.Tensor:
        """Compute final metric value."""
        return torch.tensor(np.mean(self.scores), dtype=torch.float32)
    
    def _calculate_quality_score(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Calculate quality score for a single prediction.
        
        Args:
            pred: Predicted floor plan.
            target: Target floor plan.
            
        Returns:
            Quality score between 0 and 1.
        """
        # Normalize predictions to [0, room_types]
        pred = np.round(pred * self.room_types).astype(np.int32)
        pred = np.clip(pred, 0, self.room_types)
        
        # Calculate various quality factors
        metrics = ArchitecturalMetrics(self.room_types)
        
        # Room statistics
        pred_stats = metrics.calculate_room_statistics(pred)
        target_stats = metrics.calculate_room_statistics(target)
        
        # Connectivity
        pred_conn = metrics.calculate_connectivity(pred)
        target_conn = metrics.calculate_connectivity(target)
        
        # Compactness
        pred_comp = metrics.calculate_compactness(pred)
        target_comp = metrics.calculate_compactness(target)
        
        # Calculate similarity scores
        room_similarity = 1.0 - abs(pred_stats['total_rooms'] - target_stats['total_rooms']) / max(target_stats['total_rooms'], 1)
        coverage_similarity = 1.0 - abs(pred_stats['coverage_ratio'] - target_stats['coverage_ratio'])
        compactness_similarity = 1.0 - abs(pred_comp['compactness'] - target_comp['compactness'])
        
        # Overall quality score
        quality_score = (room_similarity + coverage_similarity + compactness_similarity) / 3.0
        
        return max(0.0, min(1.0, quality_score))
