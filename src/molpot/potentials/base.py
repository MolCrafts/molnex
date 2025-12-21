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
    
    def calc_energy(self, data, **kwargs) -> float:
        """Calculate energy (PotentialProtocol method).
        
        This method provides compatibility with molpy's Potential interface.
        It calls forward() and converts the result to a Python float.
        
        Args:
            data: AtomicTD or Frame (both implement same protocol)
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Energy as Python float
        """
        energy_tensor = self.forward(data)
        return float(energy_tensor.item())
    
    def calc_forces(self, data, **kwargs) -> np.ndarray:
        """Calculate forces via autograd (PotentialProtocol method).
        
        Computes forces as F = -dE/dx using PyTorch autograd.
        
        Args:
            data: AtomicTD or Frame (both implement same protocol)
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Forces as numpy array [N, 3]
        """
        # Extract positions and enable gradients
        pos = self._get_positions(data)
        original_requires_grad = pos.requires_grad
        pos.requires_grad_(True)
        
        # Compute energy
        energy = self.forward(data)
        
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
    def forward(self, data) -> torch.Tensor:
        """Forward pass - must be implemented by subclasses.
        
        Args:
            data: AtomicTD or Frame (both support Frame protocol)
                  Expected fields depend on potential type:
                  - Classic: ("atoms", "x"), ("atoms", "type"), ("graph", "batch")
                  - ML: ("atoms", "x"), ("atoms", "z"), ("graph", "batch")
            
        Returns:
            Energy as torch.Tensor (scalar)
        """
        pass
    
    def _get_positions(self, data) -> torch.Tensor:
        """Extract positions from data (Frame or AtomicTD).
        
        Both Frame and AtomicTD support nested dict access:
        - Frame: data["atoms"]["x"] returns numpy array
        - AtomicTD: data["atoms"]["x"] returns torch.Tensor (TensorDict native)
        
        Args:
            data: AtomicTD or Frame
            
        Returns:
            Positions tensor [N, 3]
        """
        try:
            # Nested access works for both Frame and TensorDict
            pos = data["atoms"]["x"]
            
            # Convert numpy to torch if needed (for Frame compatibility)
            if isinstance(pos, np.ndarray):
                pos = torch.from_numpy(pos).float()
            
            return pos
        except (KeyError, TypeError) as e:
            raise ValueError(
                f"Could not extract positions from data. "
                f"Expected nested access data['atoms']['x']. "
                f"Error: {e}"
            )
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', type='{self.type}')"
