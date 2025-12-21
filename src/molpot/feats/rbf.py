"""Radial basis functions for distance featurization."""

import math

import torch
import torch.nn as nn


class GaussianRBF(nn.Module):
    """Gaussian radial basis functions.
    
    Expands distances into a set of Gaussian basis functions:
    RBF_k(d) = exp(-gamma * (d - mu_k)^2)
    
    where mu_k are evenly spaced centers from 0 to cutoff.
    
    Attributes:
        num_rbf: Number of radial basis functions
        cutoff: Cutoff radius
        centers: RBF centers [num_rbf]
        gamma: Width parameter
    """
    
    def __init__(
        self,
        num_rbf: int = 50,
        cutoff: float = 5.0,
        learnable: bool = False,
    ):
        """Initialize Gaussian RBF.
        
        Args:
            num_rbf: Number of radial basis functions
            cutoff: Cutoff radius
            learnable: If True, centers and gamma are learnable parameters
        """
        super().__init__()
        
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        
        # Initialize centers evenly spaced from 0 to cutoff
        centers = torch.linspace(0, cutoff, num_rbf)
        
        # Initialize gamma (width parameter)
        # Use spacing between centers to set width
        gamma = 1.0 / (cutoff / num_rbf) ** 2
        
        if learnable:
            self.centers = nn.Parameter(centers)
            self.gamma = nn.Parameter(torch.tensor(gamma))
        else:
            self.register_buffer("centers", centers)
            self.register_buffer("gamma", torch.tensor(gamma))
    
    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """Compute RBF features from distances.
        
        Args:
            distances: Pairwise distances [num_edges] or [batch, num_edges]
            
        Returns:
            RBF features [num_edges, num_rbf] or [batch, num_edges, num_rbf]
        """
        # distances: [..., 1]
        # centers: [num_rbf]
        # output: [..., num_rbf]
        
        distances = distances.unsqueeze(-1)  # [..., 1]
        centers = self.centers.view(1, -1)  # [1, num_rbf]
        
        # Gaussian RBF
        rbf = torch.exp(-self.gamma * (distances - centers) ** 2)
        
        return rbf
    
    def __repr__(self) -> str:
        return f"GaussianRBF(num_rbf={self.num_rbf}, cutoff={self.cutoff})"
