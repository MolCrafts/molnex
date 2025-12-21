"""Combined energy and force loss."""

import torch
import torch.nn as nn

from molpot.losses.energy import EnergyLoss
from molpot.losses.force import ForceLoss


class CombinedLoss(nn.Module):
    """Combined energy and force loss with configurable weights.
    
    L = w_energy * L_energy + w_force * L_force
    
    Attributes:
        energy_loss: Energy loss module
        force_loss: Force loss module
        energy_weight: Weight for energy loss
        force_weight: Weight for force loss
    """
    
    def __init__(
        self,
        energy_weight: float = 1.0,
        force_weight: float = 1.0,
        reduction: str = "mean",
    ):
        """Initialize combined loss.
        
        Args:
            energy_weight: Weight for energy loss
            force_weight: Weight for force loss
            reduction: Reduction method for individual losses
        """
        super().__init__()
        
        self.energy_loss = EnergyLoss(reduction=reduction)
        self.force_loss = ForceLoss(reduction=reduction)
        self.energy_weight = energy_weight
        self.force_weight = force_weight
    
    def forward(
        self,
        pred_energy: torch.Tensor,
        target_energy: torch.Tensor,
        pred_forces: torch.Tensor,
        target_forces: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined loss.
        
        Args:
            pred_energy: Predicted energies [batch]
            target_energy: Target energies [batch]
            pred_forces: Predicted forces [num_atoms, 3]
            target_forces: Target forces [num_atoms, 3]
            
        Returns:
            Combined loss (scalar)
        """
        loss_e = self.energy_loss(pred_energy, target_energy)
        loss_f = self.force_loss(pred_forces, target_forces)
        
        return self.energy_weight * loss_e + self.force_weight * loss_f
    
    def __repr__(self) -> str:
        return (
            f"CombinedLoss("
            f"energy_weight={self.energy_weight}, "
            f"force_weight={self.force_weight})"
        )
