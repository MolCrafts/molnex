"""Prediction heads for molecular property prediction.

Reusable TensorDictModuleBase heads for common prediction targets:
energy (scatter pooling), forces (autograd), and stress (autograd).

Example:
    >>> energy_head = EnergyHead(node_energy_key="node_energy")
    >>> force_head = ForceHead()
    >>> td = energy_head(td)  # pools node energies to molecular energy
    >>> td = force_head(td)   # computes forces via -dE/dpos
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EnergyHead(nn.Module):
    """Pool node-level energies to graph-level molecular energy.

    Performs scatter-based summation (or mean) of per-atom energies
    to produce per-molecule energies.

    Args:
        pooling: Pooling strategy (``"sum"`` or ``"mean"``).
    """

    def __init__(
        self,
        *,
        pooling: str = "sum",
    ):
        super().__init__()
        if pooling not in ("sum", "mean"):
            raise ValueError(f"pooling must be 'sum' or 'mean', got '{pooling}'")
        self.pooling = pooling

    def forward(self, node_energy: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Pool node energies to molecular energies.

        Args:
            node_energy: Per-atom energies [N].
            batch: Batch indices [N].

        Returns:
            Molecular energy [B].
        """
        num_graphs = int(batch.max().item()) + 1
        energy = torch.zeros(
            num_graphs, dtype=node_energy.dtype, device=node_energy.device
        )
        energy.index_add_(0, batch, node_energy)

        if self.pooling == "mean":
            counts = torch.zeros(
                num_graphs, dtype=node_energy.dtype, device=node_energy.device
            )
            counts.index_add_(
                0, batch, torch.ones_like(node_energy)
            )
            energy = energy / counts.clamp(min=1)

        return energy


class ForceHead(nn.Module):
    """Compute forces via autograd as ``-dE/dpos``.

    Positions must have ``requires_grad=True`` before the energy computation.
    """

    def __init__(self):
        super().__init__()

    def forward(self, energy: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Compute forces as negative gradient of energy w.r.t. positions.

        Args:
            energy: Molecular energy.
            pos: Atomic positions (``requires_grad=True``).

        Returns:
            Atomic forces.
        """
        forces = -torch.autograd.grad(
            energy.sum(),
            pos,
            create_graph=self.training,
            retain_graph=self.training,
        )[0]

        return forces


class StressHead(nn.Module):
    """Compute stress tensor via autograd as ``(1/V) * dE/dstrain``.

    Expects the upstream code to apply a symmetric strain displacement
    to positions before the energy computation. The strain tensor must
    have ``requires_grad=True``.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        energy: torch.Tensor,
        strain: torch.Tensor,
        cell: torch.Tensor,
    ) -> torch.Tensor:
        """Compute stress tensor from energy gradient w.r.t. strain.

        Args:
            energy: Molecular energy.
            strain: Strain tensor (``requires_grad=True``).
            cell: Unit cell.

        Returns:
            Stress tensor.
        """
        grad = torch.autograd.grad(
            energy.sum(),
            strain,
            create_graph=self.training,
            retain_graph=self.training,
        )[0]

        volume = torch.det(cell).abs()
        stress = grad / volume.view(-1, 1, 1)

        return stress
