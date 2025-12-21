"""Pooling and aggregation operations for atomic-to-molecular readout."""

import torch
import torch.nn as nn
from torch_scatter import scatter


class SumPooling(nn.Module):
    """Sum pooling: aggregate atomic features to molecular features.
    
    Sums atomic features belonging to the same molecule.
    """
    
    def __init__(self):
        """Initialize sum pooling."""
        super().__init__()
    
    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        dim_size: int = None,
    ) -> torch.Tensor:
        """Pool atomic features to molecular features via summation.
        
        Args:
            x: Atomic features [num_atoms, feature_dim]
            batch: Batch/molecule indices [num_atoms]
            dim_size: Number of molecules (optional, inferred if None)
            
        Returns:
            Molecular features [num_molecules, feature_dim]
        """
        if dim_size is None:
            dim_size = int(batch.max()) + 1
        
        return scatter(x, batch, dim=0, dim_size=dim_size, reduce="sum")
    
    def __repr__(self) -> str:
        return "SumPooling()"


class MeanPooling(nn.Module):
    """Mean pooling: aggregate atomic features to molecular features.
    
    Averages atomic features belonging to the same molecule.
    """
    
    def __init__(self):
        """Initialize mean pooling."""
        super().__init__()
    
    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        dim_size: int = None,
    ) -> torch.Tensor:
        """Pool atomic features to molecular features via averaging.
        
        Args:
            x: Atomic features [num_atoms, feature_dim]
            batch: Batch/molecule indices [num_atoms]
            dim_size: Number of molecules (optional, inferred if None)
            
        Returns:
            Molecular features [num_molecules, feature_dim]
        """
        if dim_size is None:
            dim_size = int(batch.max()) + 1
        
        return scatter(x, batch, dim=0, dim_size=dim_size, reduce="mean")
    
    def __repr__(self) -> str:
        return "MeanPooling()"


class MaxPooling(nn.Module):
    """Max pooling: aggregate atomic features to molecular features.
    
    Takes maximum of atomic features belonging to the same molecule.
    """
    
    def __init__(self):
        """Initialize max pooling."""
        super().__init__()
    
    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        dim_size: int = None,
    ) -> torch.Tensor:
        """Pool atomic features to molecular features via max operation.
        
        Args:
            x: Atomic features [num_atoms, feature_dim]
            batch: Batch/molecule indices [num_atoms]
            dim_size: Number of molecules (optional, inferred if None)
            
        Returns:
            Molecular features [num_molecules, feature_dim]
        """
        if dim_size is None:
            dim_size = int(batch.max()) + 1
        
        return scatter(x, batch, dim=0, dim_size=dim_size, reduce="max")
    
    def __repr__(self) -> str:
        return "MaxPooling()"
