import torch
import numpy as np
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        
        with torch.no_grad():
            stats = []
            for layer in model:
                x = layer(x)
                if isinstance(layer, nn.Linear):
                    mean = x.mean().item()
                    std = x.std().item()
                    dead_fraction = (x <= 0).all(dim=0).float().mean().item() 
                    stats.append({'mean': np.round(mean,4), 'std': np.round(std,4), 'dead_fraction': np.round(dead_fraction,4)})
            return stats

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()
        stats = []
        criterion = nn.MSELoss()

        # Forward pass
        y_pred = model(x)
        # Compute loss
        loss = criterion(y_pred, y)
        # Backward pass
        loss.backward()
        for layer in model:
            if isinstance(layer, nn.Linear):
                mean = layer.weight.grad.mean().item()
                std = layer.weight.grad.std().item()
                norm = torch.norm(layer.weight.grad).item()
                stats.append({'mean': np.round(mean,4), 'std': np.round(std,4), 'norm': np.round(norm,4)})
        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        for layer_stats in activation_stats:
            if layer_stats['dead_fraction'] > 0.5:
                return 'dead_neurons'

        for layer_stats in gradient_stats:
            if layer_stats['norm'] > 1000:
                return 'exploding_gradients'
        
        for layer_stats in gradient_stats:
            if layer_stats['norm'] < 1e-5:
                return 'vanishing_gradients'
        
        for layer_stats in activation_stats:
            if layer_stats['std'] < 0.1:
                return 'vanishing_gradients'
            if layer_stats['std'] > 10.0:
                return 'exploding_gradients'

        return 'healthy'

