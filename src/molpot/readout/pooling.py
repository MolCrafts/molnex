"""Pooling and aggregation operations for atomic-to-molecular readout.

Pure PyTorch implementation (no torch_scatter dependency).
"""

import torch
import torch.nn as nn


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
        
        # Pure PyTorch implementation using index_add_
        if x.dim() == 1:
            out = torch.zeros(dim_size, dtype=x.dtype, device=x.device)
        else:
            out = torch.zeros(dim_size, x.shape[1], dtype=x.dtype, device=x.device)
        
        out.index_add_(0, batch, x)
        return out
    
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
        
        # Sum pooling
        if x.dim() == 1:
            out_sum = torch.zeros(dim_size, dtype=x.dtype, device=x.device)
        else:
            out_sum = torch.zeros(dim_size, x.shape[1], dtype=x.dtype, device=x.device)
        out_sum.index_add_(0, batch, x)
        
        # Count atoms per molecule
        counts = torch.zeros(dim_size, dtype=x.dtype, device=x.device)
        ones = torch.ones_like(batch, dtype=x.dtype)
        counts.index_add_(0, batch, ones)
        
        # Average
        if x.dim() == 1:
            return out_sum / counts.clamp(min=1)
        else:
            return out_sum / counts.unsqueeze(-1).clamp(min=1)
    
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
        
        # Pure PyTorch implementation
        if x.dim() == 1:
            out = torch.full((dim_size,), float('-inf'), dtype=x.dtype, device=x.device)
        else:
            out = torch.full((dim_size, x.shape[1]), float('-inf'), dtype=x.dtype, device=x.device)
        
        # Scatter max using loop (not the most efficient, but pure PyTorch)
        for mol_idx in range(dim_size):
            mask = batch == mol_idx
            if mask.any():
                out[mol_idx] = x[mask].max(dim=0)[0] if x.dim() > 1 else x[mask].max()
        
        return out
    
    def __repr__(self) -> str:
        return "MaxPooling()"
