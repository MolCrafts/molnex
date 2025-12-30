"""Loss functions as TensorDictModules.

All losses follow TensorDictModule pattern with explicit in_keys/out_keys.
Losses read predictions and targets from the same TensorDict.
"""

import torch
import torch.nn as nn
from tensordict import TensorDict


class EnergyLoss(nn.Module):
    """Energy loss (MSE).
    
    TensorDictModule that computes MSE loss between predicted and target energy.
    Reads both from the same TensorDict.
    
    in_keys: [("target", "energy")]  # Reads from both pred_td and true_td
    out_keys: [("loss",)]
    """
    
    def __init__(self, weight: float = 1.0):
        """Initialize energy loss.
        
        Args:
            weight: Loss weight for multi-task learning
        """
        super().__init__()
        self.weight = weight
        self.in_keys = [("target", "energy")]
        self.out_keys = [("loss",)]
    
    def forward(self, pred_td: TensorDict, true_td: TensorDict) -> TensorDict:
        """Compute energy loss.
        
        Args:
            pred_td: TensorDict with predicted target.energy
            true_td: TensorDict with true target.energy
            
        Returns:
            TensorDict with loss added
        """
        pred_energy = pred_td["target", "energy"]
        true_energy = true_td["target", "energy"]
        
        loss = self.weight * (pred_energy - true_energy).pow(2).mean()
        
        # Return new TensorDict with loss
        result = TensorDict({"loss": loss}, batch_size=[])
        return result


class ForceLoss(nn.Module):
    """Force loss (MSE).
    
    TensorDictModule that computes MSE loss between predicted and target forces.
    
    in_keys: [("atoms", "f")]  # Reads from both pred_td and true_td
    out_keys: [("loss",)]
    """
    
    def __init__(self, weight: float = 1.0):
        """Initialize force loss.
        
        Args:
            weight: Loss weight for multi-task learning
        """
        super().__init__()
        self.weight = weight
        self.in_keys = [("atoms", "f")]
        self.out_keys = [("loss",)]
    
    def forward(self, pred_td: TensorDict, true_td: TensorDict) -> TensorDict:
        """Compute force loss.
        
        Args:
            pred_td: TensorDict with predicted atoms.f
            true_td: TensorDict with true atoms.f
            
        Returns:
            TensorDict with loss added
        """
        pred_forces = pred_td["atoms", "f"]
        true_forces = true_td["atoms", "f"]
        
        loss = self.weight * (pred_forces - true_forces).pow(2).mean()
        
        result = TensorDict({"loss": loss}, batch_size=[])
        return result


class CombinedLoss(nn.Module):
    """Combined energy + force loss.
    
    TensorDictModule that combines multiple losses with weights.
    
    in_keys: [("target", "energy"), ("atoms", "f")]
    out_keys: [("loss",), ("loss_energy",), ("loss_force",)]
    """
    
    def __init__(self, energy_weight: float = 1.0, force_weight: float = 1.0):
        """Initialize combined loss.
        
        Args:
            energy_weight: Weight for energy loss
            force_weight: Weight for force loss
        """
        super().__init__()
        self.energy_loss = EnergyLoss(weight=energy_weight)
        self.force_loss = ForceLoss(weight=force_weight)
        self.in_keys = [("target", "energy"), ("atoms", "f")]
        self.out_keys = [("loss",), ("loss_energy",), ("loss_force",)]
    
    def forward(self, pred_td: TensorDict, true_td: TensorDict) -> TensorDict:
        """Compute combined loss.
        
        Args:
            pred_td: TensorDict with predictions
            true_td: TensorDict with targets
            
        Returns:
            TensorDict with loss, loss_energy, loss_force
        """
        # Compute individual losses
        energy_result = self.energy_loss(pred_td, true_td)
        force_result = self.force_loss(pred_td, true_td)
        
        loss_energy = energy_result["loss"]
        loss_force = force_result["loss"]
        loss_total = loss_energy + loss_force
        
        result = TensorDict({
            "loss": loss_total,
            "loss_energy": loss_energy,
            "loss_force": loss_force,
        }, batch_size=[])
        
        return result
