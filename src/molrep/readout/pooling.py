"""Pooling functions for aggregating atom features to graph-level features.

Provides masked pooling operations that handle variable-length molecules
with padding by using explicit masks.
"""

import torch


def masked_sum_pooling(
    features: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Sum pooling over valid atoms only.
    
    Aggregates atom-level features to graph-level features by summing
    over valid (non-padded) atoms.
    
    Args:
        features: Atom features [B, L, d_model]
        mask: Atom mask [B, L] (True = valid atom, False = padding)
        
    Returns:
        Graph-level features [B, d_model]
        
    Example:
        >>> features = torch.randn(2, 10, 128)
        >>> mask = torch.ones(2, 10, dtype=torch.bool)
        >>> mask[0, 7:] = False  # First molecule has 7 atoms
        >>> mask[1, 5:] = False  # Second molecule has 5 atoms
        >>> graph_features = masked_sum_pooling(features, mask)
        >>> graph_features.shape
        torch.Size([2, 128])
    """
    # Mask out padding positions
    masked_features = features * mask.unsqueeze(-1).float()  # [B, L, d_model]
    
    # Sum over atoms
    graph_features = masked_features.sum(dim=1)  # [B, d_model]
    
    return graph_features


def masked_mean_pooling(
    features: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean pooling over valid atoms only.
    
    Aggregates atom-level features to graph-level features by averaging
    over valid (non-padded) atoms.
    
    Args:
        features: Atom features [B, L, d_model]
        mask: Atom mask [B, L] (True = valid atom, False = padding)
        
    Returns:
        Graph-level features [B, d_model]
        
    Example:
        >>> features = torch.randn(2, 10, 128)
        >>> mask = torch.ones(2, 10, dtype=torch.bool)
        >>> mask[0, 7:] = False  # First molecule has 7 atoms
        >>> mask[1, 5:] = False  # Second molecule has 5 atoms
        >>> graph_features = masked_mean_pooling(features, mask)
        >>> graph_features.shape
        torch.Size([2, 128])
    """
    # Mask out padding positions
    masked_features = features * mask.unsqueeze(-1).float()  # [B, L, d_model]
    
    # Count valid atoms per molecule
    num_atoms = mask.sum(dim=1, keepdim=True).float()  # [B, 1]
    num_atoms = num_atoms.clamp(min=1)  # Avoid division by zero
    
    # Sum and normalize
    graph_features = masked_features.sum(dim=1) / num_atoms  # [B, d_model]
    
    return graph_features
