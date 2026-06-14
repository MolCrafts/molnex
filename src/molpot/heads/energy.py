"""Per-atom energy heads."""

import torch
import torch.nn as nn


class AtomicEnergyMLP(nn.Module):
    """MLP predicting per-atom energy from atomic features.

    Single responsibility: map atomic feature vectors to scalar
    per-atom energy values. Does NOT perform graph-level pooling.
    Use ``molpot.derivation.EnergyAggregation`` for pooling.

    Args:
        hidden_dim: Dimension of input hidden representation.
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, atoms_h: torch.Tensor) -> torch.Tensor:
        """Predict per-atom energies.

        Args:
            atoms_h: Atomic hidden states ``(N, D)``.

        Returns:
            Per-atom energies ``(N,)``.
        """
        return self.mlp(atoms_h).squeeze(-1)


class EnergyHead(nn.Module):
    """Predict molecular energy from atomic representations.

    Bundles an atomic MLP and sum pooling. For new code, prefer
    composing ``AtomicEnergyMLP`` + ``molpot.derivation.EnergyAggregation``
    explicitly.
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.atomic_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, atoms_h: torch.Tensor, graph_batch: torch.Tensor) -> torch.Tensor:
        """Predict molecular energy.

        Args:
            atoms_h: Atomic hidden states ``(N, D)``.
            graph_batch: Molecule indices ``(N,)``.

        Returns:
            Molecular energies ``(B,)``.
        """
        atomic_energies = self.atomic_mlp(atoms_h).squeeze(-1)

        num_molecules = int(graph_batch.max()) + 1
        molecular_energies = torch.zeros(
            num_molecules, dtype=atomic_energies.dtype, device=atomic_energies.device
        )
        molecular_energies.index_add_(0, graph_batch, atomic_energies)

        return molecular_energies
