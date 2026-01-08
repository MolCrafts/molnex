"""Edge bias modules for graph-aware attention.

Provides modules for converting edge features (RBF) to attention bias:
- EdgeBiasMLP: Converts RBF features to per-head attention bias
- densify_edge_bias: Converts sparse edge bias to dense attention bias tensor
"""

import torch
import torch.nn as nn


class EdgeBiasMLP(nn.Module):
    """MLP for converting edge features to per-head attention bias.
    
    Takes RBF-expanded edge features and produces per-head attention bias
    values that will be added to the attention logits.
    
    Args:
        num_rbf: Number of RBF features
        num_heads: Number of attention heads
        hidden_dim: Hidden dimension for MLP (default: 64)
        
    Example:
        >>> mlp = EdgeBiasMLP(num_rbf=50, num_heads=4)
        >>> rbf = torch.randn(2, 100, 50)  # [B, E, K]
        >>> bias = mlp(rbf)  # [B, E, H]
        >>> bias.shape
        torch.Size([2, 100, 4])
    """
    
    def __init__(
        self,
        num_rbf: int,
        num_heads: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        
        self.num_rbf = num_rbf
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        # MLP: rbf → hidden → hidden → num_heads
        self.mlp = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads),
        )
    
    def forward(self, rbf: torch.Tensor) -> torch.Tensor:
        """Convert RBF features to per-head attention bias.
        
        Args:
            rbf: RBF features [B, E, K] or [E, K]
            
        Returns:
            Per-head attention bias [B, E, H] or [E, H]
        """
        return self.mlp(rbf)
    
    def __repr__(self) -> str:
        return (
            f"EdgeBiasMLP(num_rbf={self.num_rbf}, num_heads={self.num_heads}, "
            f"hidden_dim={self.hidden_dim})"
        )


def densify_edge_bias(
    edge_index: torch.Tensor,
    edge_bias: torch.Tensor,
    num_atoms: torch.Tensor,
    max_atoms: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert sparse edge bias to dense attention bias tensor.
    
    Takes sparse edge-list representation and converts to dense [B,H,L,L]
    attention bias tensor. Non-edge positions and padding are filled with -inf.
    
    Args:
        edge_index: Edge indices [2, E_total] (flattened across batch)
        edge_bias: Per-head edge bias [E_total, H]
        num_atoms: Number of atoms per molecule [B]
        max_atoms: Maximum number of atoms (L)
        mask: Optional atom mask [B, L] (True = valid atom)
        
    Returns:
        Dense attention bias [B, H, L, L]
        
    Example:
        >>> # Batch of 2 molecules: 3 atoms and 2 atoms
        >>> edge_index = torch.tensor([[0, 1, 3], [1, 2, 4]])  # 3 edges
        >>> edge_bias = torch.randn(3, 4)  # 4 heads
        >>> num_atoms = torch.tensor([3, 2])
        >>> attn_bias = densify_edge_bias(edge_index, edge_bias, num_atoms, max_atoms=3)
        >>> attn_bias.shape
        torch.Size([2, 4, 3, 3])
    """
    device = edge_index.device
    batch_size = num_atoms.size(0)
    num_heads = edge_bias.size(-1)
    
    # Initialize with -inf (no attention)
    attn_bias = torch.full(
        (batch_size, num_heads, max_atoms, max_atoms),
        float('-inf'),
        dtype=edge_bias.dtype,
        device=device,
    )
    
    # Determine batch index for each edge
    # edge_index contains global indices, need to map to (batch_idx, local_i, local_j)
    edge_i = edge_index[0]  # [E_total]
    edge_j = edge_index[1]  # [E_total]
    
    # Compute batch offsets
    offsets = torch.cat([torch.tensor([0], device=device), num_atoms.cumsum(0)[:-1]])
    
    # Find which batch each edge belongs to
    # For each edge, find the batch by comparing edge_i with cumulative atom counts
    cumsum = num_atoms.cumsum(0)  # [B]
    batch_idx = torch.searchsorted(cumsum, edge_i, right=False)  # [E_total]
    
    # Convert global indices to local indices
    local_i = edge_i - offsets[batch_idx]  # [E_total]
    local_j = edge_j - offsets[batch_idx]  # [E_total]
    
    # Fill in edge biases
    # attn_bias[batch_idx, :, local_i, local_j] = edge_bias
    for h in range(num_heads):
        attn_bias[batch_idx, h, local_i, local_j] = edge_bias[:, h]
    
    # Apply mask if provided (set padding positions to -inf)
    if mask is not None:
        # mask: [B, L], True = valid atom
        # Set attention from/to padding atoms to -inf
        # This is already done by initialization, but we can be explicit
        padding_mask = ~mask  # [B, L], True = padding
        attn_bias = attn_bias.masked_fill(
            padding_mask.unsqueeze(1).unsqueeze(2),  # [B, 1, 1, L]
            float('-inf')
        )
        attn_bias = attn_bias.masked_fill(
            padding_mask.unsqueeze(1).unsqueeze(3),  # [B, 1, L, 1]
            float('-inf')
        )
    
    return attn_bias
