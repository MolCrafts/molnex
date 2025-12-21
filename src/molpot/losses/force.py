"""Force loss function."""

import torch
import torch.nn as nn


class ForceLoss(nn.Module):
    """Mean squared error loss for force prediction.
    
    Attributes:
        reduction: Reduction method ("mean", "sum", "none")
    """
    
    def __init__(self, reduction: str = "mean"):
        """Initialize force loss.
        
        Args:
            reduction: How to reduce the loss ("mean", "sum", "none")
        """
        super().__init__()
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction=reduction)
    
    def forward(
        self,
        pred_forces: torch.Tensor,
        target_forces: torch.Tensor,
    ) -> torch.Tensor:
        """Compute force loss.
        
        Args:
            pred_forces: Predicted forces [num_atoms, 3]
            target_forces: Target forces [num_atoms, 3]
            
        Returns:
            Loss (scalar if reduction="mean" or "sum")
        """
        return self.mse(pred_forces, target_forces)
    
    def __repr__(self) -> str:
        return f"ForceLoss(reduction={self.reduction})"
