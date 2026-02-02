import torch
import torch.nn as nn


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
    
    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass: pool and predict.
        
        Args:
            h: Padded tensor [B, L, d_model]
            mask: Boolean mask [B, L]
                
        Returns:
            Tensor [B]
        """
        # Pool per-atom features to per-molecule
        h_pooled = self._pool(h, mask)  # [B, d_model]
        
        # MLP prediction: [B, d_model] -> [B, 1] -> [B]
        scalar = self.mlp(h_pooled).squeeze(-1)
        
        return scalar
    
    def _pool(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Pool padded tensor over atoms using mask.
        
        Args:
            h: Padded tensor [B, L, d_model]
            mask: Boolean mask [B, L] (True = valid atom)
            
        Returns:
            Pooled tensor [B, d_model]
        """
        # Expand mask to match feature dimension: [B, L] -> [B, L, 1]
        mask_expanded = mask.unsqueeze(-1).float()  # [B, L, 1]
        
        if self.pooling == "mean":
            # Mean pooling: sum over valid atoms and divide by count
            h_masked = h * mask_expanded  # Zero out padded positions
            h_sum = h_masked.sum(dim=1)  # [B, d_model]
            atom_counts = mask_expanded.sum(dim=1)  # [B, 1]
            h_pooled = h_sum / atom_counts.clamp(min=1)  # Avoid division by zero
        elif self.pooling == "sum":
            # Sum pooling: sum over valid atoms
            h_masked = h * mask_expanded
            h_pooled = h_masked.sum(dim=1)  # [B, d_model]
        elif self.pooling == "max":
            # Max pooling: mask out padded positions with -inf before max
            h_masked = h.clone()
            h_masked[~mask] = float('-inf')
            h_pooled = h_masked.max(dim=1)[0]  # [B, d_model]
        
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
    
    # Create test data (padded per-atom representations)
    h = torch.randn(3, 7, 128)  # [B=3, L=7, d_model=128]
    mask = torch.zeros(3, 7, dtype=torch.bool)
    mask[0, :5] = True  # 5 atoms
    mask[1, :3] = True  # 3 atoms
    mask[2, :7] = True  # 7 atoms
    
    print(f"\nInput:")
    print(f"  h shape: {h.shape}")
    print(f"  Batch size: 3")
    print(f"  Atom counts: [5, 3, 7]")
    
    # Forward pass
    with torch.no_grad():
        scalar = head(h, mask)
    
    print(f"\nOutput:")
    print(f"  scalar shape: {scalar.shape}")
    print(f"  scalar values: {scalar}")
    
    print("\n✓ ScalarHead works!")
