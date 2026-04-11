"""Lennard-Jones 12-6 potential."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


def lj126_pair_energy(
    distance: torch.Tensor,
    epsilon_ij: torch.Tensor,
    sigma_ij: torch.Tensor,
) -> torch.Tensor:
    """LJ 12-6 pair energy kernel.

    ``E_ij = 4 * epsilon_ij * [(sigma_ij / r)^12 - (sigma_ij / r)^6]``
    """
    sr = sigma_ij / distance.clamp(min=1e-6)
    sr6 = sr.pow(6)
    sr12 = sr6.pow(2)
    return 4.0 * epsilon_ij * (sr12 - sr6)


def lorentz_berthelot(
    atom_params: dict[str, torch.Tensor],
    edge_index: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Lorentz-Berthelot mixing rules: per-atom → per-pair.

    Args:
        atom_params: Must contain ``"epsilon"`` ``(N,)`` and ``"sigma"`` ``(N,)``.
        edge_index: Edge indices ``(E, 2)``.

    Returns:
        Dict with ``"epsilon_ij"`` ``(E,)`` and ``"sigma_ij"`` ``(E,)``.
    """
    src, dst = edge_index[:, 0], edge_index[:, 1]
    epsilon = atom_params["epsilon"]
    sigma = atom_params["sigma"]
    return {
        "epsilon_ij": torch.sqrt(epsilon[src] * epsilon[dst] + 1e-12),
        "sigma_ij": 0.5 * (sigma[src] + sigma[dst]),
    }


class LJ126(nn.Module):
    """Lennard-Jones 12-6 potential.

    Pure physics: takes per-pair parameters and distances, returns energy.

    Args:
        mixing_fn: Converts per-atom params to per-pair params.
            Defaults to ``lorentz_berthelot``.
        bidirectional: Halve pair energies to avoid double-counting.
        energy_scale: Multiplicative energy scaling factor.
    """

    def __init__(
        self,
        mixing_fn: Callable[
            [dict[str, torch.Tensor], torch.Tensor],
            dict[str, torch.Tensor],
        ] = lorentz_berthelot,
        bidirectional: bool = True,
        energy_scale: float = 1.0,
    ):
        super().__init__()
        self.mixing_fn = mixing_fn
        self.bidirectional = bidirectional
        self.energy_scale = energy_scale

    def forward(
        self,
        *,
        distance: torch.Tensor,
        epsilon_ij: torch.Tensor,
        sigma_ij: torch.Tensor,
        edge_batch: torch.Tensor | None = None,
        num_graphs: int | None = None,
    ) -> torch.Tensor:
        """Compute LJ 12-6 energy.

        Args:
            distance: Pairwise distances ``(E,)``.
            epsilon_ij: Per-pair well depth ``(E,)``.
            sigma_ij: Per-pair zero-crossing distance ``(E,)``.
            edge_batch: Graph index per edge ``(E,)``.
                If None, returns summed scalar.
            num_graphs: Number of graphs (inferred from edge_batch if None).

        Returns:
            Per-graph energy ``(B,)`` or scalar if no batching.
        """
        pair_energy = lj126_pair_energy(distance, epsilon_ij, sigma_ij)
        pair_energy = pair_energy * self.energy_scale
        if self.bidirectional:
            pair_energy = 0.5 * pair_energy

        if edge_batch is None:
            return pair_energy.sum()

        if num_graphs is None:
            num_graphs = int(edge_batch.max().item()) + 1

        energy = torch.zeros(
            num_graphs, dtype=pair_energy.dtype, device=pair_energy.device
        )
        energy.index_add_(0, edge_batch, pair_energy)
        return energy
