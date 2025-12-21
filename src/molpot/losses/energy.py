"""Energy loss function."""

import torch
import torch.nn as nn


class EnergyLoss(nn.Module):
    """Mean squared error loss for energy prediction.
    
    Attributes:
        reduction: Reduction method ("mean", "sum", "none")
    """
    
    def __init__(self, reduction: str = "mean"):
        """Initialize energy loss.
        
        Args:
            reduction: How to reduce the loss ("mean", "sum", "none")
        """
        super().__init__()
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction=reduction)
    
    def forward(
        self,
        pred_energy: torch.Tensor,
        target_energy: torch.Tensor,
    ) -> torch.Tensor:
        """Compute energy loss.
        
        Args:
            pred_energy: Predicted energies [batch]
            target_energy: Target energies [batch]
            
        Returns:
            Loss (scalar if reduction="mean" or "sum", else [batch])
        """
        return self.mse(pred_energy, target_energy)
    
    def __repr__(self) -> str:
        return f"EnergyLoss(reduction={self.reduction})"
