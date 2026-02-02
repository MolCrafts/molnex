"""Base class for molpot PyTorch potentials.

All molpot potentials inherit from BasePotential, which provides:
- PyTorch nn.Module functionality
- PotentialProtocol compliance (calc_energy, calc_forces)
- Automatic force computation via autograd
"""

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Union
from molix.data.atom_td import AtomTD


class BasePotential(nn.Module, ABC):
    """Base class for all molpot PyTorch potentials.
    
    Implements PotentialProtocol for compatibility with molpy ForceField.
    All molpot potentials (classic and ML) inherit from this class.
    
    Attributes:
        name: Potential name for registration (e.g., "lj126_torch", "pinet")
        type: Potential type for categorization (e.g., "pair", "bond", "ml")
    """
    
    name: str = "base"
    type: str = "unknown"
    
    def calc_energy(self, data=None, **kwargs) -> float:
        """Calculate energy (PotentialProtocol method).
        
        This method provides compatibility with molpy's Potential interface.
        It calls forward() and converts the result to a Python float.
        """
        energy_tensor = self.forward(data, **kwargs)
        return float(energy_tensor.item())
    
    def calc_forces(self, data=None, **kwargs) -> np.ndarray:
        """Calculate forces via autograd (PotentialProtocol method).
        
        Computes forces as F = -dE/dx using PyTorch autograd.
        """
        # Extract positions and enable gradients
        pos = self._get_positions(data, **kwargs)
        original_requires_grad = pos.requires_grad
        pos.requires_grad_(True)
        
        # Compute energy
        energy = self.forward(data, **kwargs)
        
        # Compute forces via autograd: F = -dE/dx
        forces = -torch.autograd.grad(
            energy,
            pos,
            create_graph=False,
            retain_graph=False,
        )[0]
        
        # Restore original requires_grad state
        pos.requires_grad_(original_requires_grad)
        
        return forces.detach().cpu().numpy()
    
    @abstractmethod
    def forward(self, data: Union[AtomTD, dict, None] = None, **kwargs) -> torch.Tensor:
        """Forward pass - must be implemented by subclasses.
        
        Args:
            data: Optional AtomTD or Frame (dict)
            **kwargs: Alternate way to pass explicit tensors (pos, atom_types, etc.)
            
        Returns:
            Energy as torch.Tensor (scalar)
        """
        pass
    
    def _get_positions(self, data=None, **kwargs) -> torch.Tensor:
        """Extract positions from data or kwargs."""
        pos = kwargs.get("pos")
        if pos is None and data is not None:
            if hasattr(data, "xyz"): # AtomTD
                pos = data.xyz
            elif isinstance(data, (dict, list)): # Frame/dict
                try:
                    pos = data["atoms"]["x"]
                except (KeyError, TypeError):
                    pos = data.get("x")
        
        if pos is None:
            raise ValueError("Could not extract positions from data or kwargs.")
            
        if isinstance(pos, np.ndarray):
            pos = torch.from_numpy(pos).float()
            
        return pos
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', type='{self.type}')"
