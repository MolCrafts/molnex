"""Scalar prediction head for molrep.

Pools per-atom representations to per-molecule and predicts scalar properties.
"""

import torch
import torch.nn as nn
from tensordict import TensorDict


class ScalarHead(nn.Module):
    """Pool per-atom representations and predict scalar property.
    
    This head demonstrates how task-specific prediction can be attached to
    molrep representations. It:
    1. Pools variable-length per-atom features to fixed-size per-molecule
    2. Applies a simple MLP to predict a scalar value
    
    The head is REPLACEABLE - you can swap it for other tasks without
    modifying the encoder.
    
    Args:
        d_model: Input dimension (from encoder)
        hidden_dim: Hidden layer dimension
        pooling: Pooling method ('mean', 'sum', or 'max')
        
    Example:
        >>> head = ScalarHead(d_model=128, hidden_dim=64)
        >>> h_nt = torch.nested.nested_tensor([
        ...     torch.randn(5, 128),  # 5 atoms
        ...     torch.randn(3, 128),  # 3 atoms
        ... ])
        >>> td = TensorDict({("rep", "h"): h_nt}, batch_size=[])
        >>> td = head(td)
        >>> td["pred", "scalar"].shape
        torch.Size([2])
    """
    
    def __init__(
        self,
        d_model: int = 128,
        hidden_dim: int = 64,
        pooling: str = "mean",
    ):
        super().__init__()
        
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.pooling = pooling
        
        if pooling not in ["mean", "sum", "max"]:
            raise ValueError(f"Unknown pooling: {pooling}. Use 'mean', 'sum', or 'max'")
        
        # Simple MLP: d_model -> hidden_dim -> 1
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, td: TensorDict) -> TensorDict:
        """Forward pass: pool and predict.
        
        Args:
            td: TensorDict with:
                - ("rep", "h"): NestedTensor [B, (Li), d_model]
                
        Returns:
            TensorDict with:
                - ("pred", "scalar"): Tensor [B]
        """
        h_nt = td["rep", "h"]  # NestedTensor [B, (Li), d_model]
        
        # Pool per-atom features to per-molecule
        h_pooled = self._pool(h_nt)  # [B, d_model]
        
        # MLP prediction: [B, d_model] -> [B, 1] -> [B]
        scalar = self.mlp(h_pooled).squeeze(-1)
        
        td["pred", "scalar"] = scalar
        return td
    
    def _pool(self, h_nt: torch.Tensor) -> torch.Tensor:
        """Pool NestedTensor over atoms.
        
        Args:
            h_nt: NestedTensor [B, (Li), d_model]
            
        Returns:
            Pooled tensor [B, d_model]
        """
        # Convert NestedTensor to list for pooling
        h_list = h_nt.unbind()
        
        if self.pooling == "mean":
            # Mean pooling: average over atoms
            h_pooled = torch.stack([h.mean(dim=0) for h in h_list])
        elif self.pooling == "sum":
            # Sum pooling
            h_pooled = torch.stack([h.sum(dim=0) for h in h_list])
        elif self.pooling == "max":
            # Max pooling over atoms
            h_pooled = torch.stack([h.max(dim=0)[0] for h in h_list])
        
        return h_pooled
    
    def __repr__(self) -> str:
        return (
            f"ScalarHead(d_model={self.d_model}, "
            f"hidden_dim={self.hidden_dim}, pooling={self.pooling})"
        )


# Simple test
if __name__ == "__main__":
    print("=" * 60)
    print("ScalarHead Test")
    print("=" * 60)
    
    # Create module
    head = ScalarHead(d_model=128, hidden_dim=64, pooling="mean")
    print(f"\nModule: {head}")
    
    # Create test data (variable-length per-atom representations)
    h_nt = torch.nested.nested_tensor([
        torch.randn(5, 128),  # 5 atoms
        torch.randn(3, 128),  # 3 atoms
        torch.randn(7, 128),  # 7 atoms
    ])
    
    td = TensorDict({
        ("rep", "h"): h_nt,
    }, batch_size=[])
    
    print(f"\nInput:")
    print(f"  h_nt is nested: {h_nt.is_nested}")
    print(f"  Batch size: 3")
    print(f"  Atom counts: [5, 3, 7]")
    
    # Forward pass
    with torch.no_grad():
        td = head(td)
    
    scalar = td["pred", "scalar"]
    print(f"\nOutput:")
    print(f"  scalar shape: {scalar.shape}")
    print(f"  scalar values: {scalar}")
    
    print("\n✓ ScalarHead works!")
