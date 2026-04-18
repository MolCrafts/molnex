"""Energy aggregation: pool node-level energies to molecular energies.

Single responsibility: scatter-based summation (or mean) of per-atom
energies to produce per-molecule energies. Does NOT predict energies
(that is AtomicEnergyMLP's job) -- only aggregates them.

Example:
    >>> agg = EnergyAggregation(pooling="sum")
    >>> node_energy = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    >>> batch = torch.tensor([0, 0, 0, 1, 1])
    >>> mol_energy = agg(node_energy, batch)  # tensor([0.6, 0.9])
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EnergyAggregation(nn.Module):
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

    def forward(
        self,
        node_energy: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int | None = None,
    ) -> torch.Tensor:
        """Pool node energies to molecular energies.

        Args:
            node_energy: Per-atom energies ``(N,)``.
            batch: Batch indices ``(N,)``.
            num_graphs: Number of graphs in the batch. If None, inferred from batch.

        Returns:
            Molecular energy ``(B,)``.
        """
        if num_graphs is None:
            num_graphs = int(batch.max().item()) + 1

        energy = torch.zeros(num_graphs, dtype=node_energy.dtype, device=node_energy.device)
        energy.index_add_(0, batch, node_energy)

        if self.pooling == "mean":
            counts = torch.zeros(num_graphs, dtype=node_energy.dtype, device=node_energy.device)
            counts.index_add_(0, batch, torch.ones_like(node_energy))
            energy = energy / counts.clamp(min=1)

        return energy
