"""Base class for molpot PyTorch potentials.

All molpot potentials inherit from BasePotential, which provides:
- PyTorch nn.Module functionality
- PotentialProtocol compliance (calc_energy, calc_forces)
- Automatic force computation via ``torch.autograd.grad``
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from molpot.derivation.force import autograd_forces


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

    def calc_energy(self, data=None, **kwargs: Any) -> float:
        """Calculate energy (PotentialProtocol method).

        This method provides compatibility with molpy's Potential interface.
        It calls forward() and converts the result to a Python float.
        """
        energy_tensor = self.forward(data, **kwargs)
        return float(energy_tensor.item())

    def calc_forces(self, data=None, **kwargs: Any) -> np.ndarray:
        """Calculate forces (PotentialProtocol method).

        Computes forces as ``F = -∂E/∂x`` with ``torch.autograd.grad``: the
        energy is differentiated as a function of positions, which are passed
        back in through ``pos=`` so :meth:`forward` evaluates at the candidate
        geometry (``_get_positions`` resolves ``kwargs["pos"]`` first). Uses the
        universal autograd backend so this generic protocol method works for any
        potential, including cuEquivariance-fused ones — see
        :func:`molpot.derivation.force.autograd_forces`.
        """
        pos = self._get_positions(data, **kwargs).detach()

        def energy_fn(p: torch.Tensor) -> torch.Tensor:
            return self.forward(data, **{**kwargs, "pos": p}).sum()

        forces = autograd_forces(energy_fn, pos)

        return forces.detach().cpu().numpy()

    @abstractmethod
    def forward(self, data: dict[str, Any] | None = None, **kwargs: Any) -> torch.Tensor:
        """Forward pass - must be implemented by subclasses.

        Args:
            data: Optional dictionary with molecule fields
            **kwargs: Alternate way to pass explicit tensors such as positions
                or atom types.

        Returns:
            Energy as torch.Tensor (scalar)
        """
        pass

    def _get_positions(self, data=None, **kwargs: Any) -> torch.Tensor:
        """Extract positions from data or kwargs."""
        pos = kwargs.get("pos")
        if pos is None and data is not None:
            if isinstance(data, dict):
                pos = data.get("pos")
                if pos is None:
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
