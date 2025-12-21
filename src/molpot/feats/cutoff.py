"""Cutoff functions for smooth distance decay."""

import math

import torch
import torch.nn as nn


class CosineCutoff(nn.Module):
    """Cosine cutoff function.
    
    Smoothly decays to zero at cutoff radius:
    f(d) = 0.5 * (cos(pi * d / cutoff) + 1) if d < cutoff else 0
    
    Attributes:
        cutoff: Cutoff radius
    """
    
    def __init__(self, cutoff: float = 5.0):
        """Initialize cosine cutoff.
        
        Args:
            cutoff: Cutoff radius
        """
        super().__init__()
        self.register_buffer("cutoff", torch.tensor(cutoff))
    
    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """Apply cutoff function to distances.
        
        Args:
            distances: Pairwise distances [num_edges] or [batch, num_edges]
            
        Returns:
            Cutoff values in [0, 1], same shape as distances
        """
        # Cosine cutoff
        cutoff_values = 0.5 * (
            torch.cos(math.pi * distances / self.cutoff) + 1.0
        )
        
        # Zero out distances beyond cutoff
        cutoff_values = cutoff_values * (distances < self.cutoff).float()
        
        return cutoff_values
    
    def __repr__(self) -> str:
        return f"CosineCutoff(cutoff={self.cutoff.item()})"


class PolynomialCutoff(nn.Module):
    """Polynomial cutoff function.
    
    Smoothly decays to zero at cutoff with continuous derivatives:
    f(d) = 1 - 6*(d/cutoff)^5 + 15*(d/cutoff)^4 - 10*(d/cutoff)^3 if d < cutoff else 0
    
    Attributes:
        cutoff: Cutoff radius
    """
    
    def __init__(self, cutoff: float = 5.0):
        """Initialize polynomial cutoff.
        
        Args:
            cutoff: Cutoff radius
        """
        super().__init__()
        self.register_buffer("cutoff", torch.tensor(cutoff))
    
    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """Apply cutoff function to distances.
        
        Args:
            distances: Pairwise distances [num_edges] or [batch, num_edges]
            
        Returns:
            Cutoff values in [0, 1], same shape as distances
        """
        # Normalized distance
        x = distances / self.cutoff
        
        # Polynomial cutoff
        cutoff_values = 1.0 - 6.0 * x**5 + 15.0 * x**4 - 10.0 * x**3
        
        # Zero out distances beyond cutoff
        cutoff_values = cutoff_values * (distances < self.cutoff).float()
        
        return cutoff_values
    
    def __repr__(self) -> str:
        return f"PolynomialCutoff(cutoff={self.cutoff.item()})"
