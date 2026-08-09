"""Per-atom energy heads."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from molix import config


class AtomicReferenceEnergy(nn.Module):
    """Per-element reference (isolated-atom) energy ``E0``, additive & frozen.

    Returns a fixed per-atom baseline ``E0[Z_i]`` that is added to the model's
    predicted interaction energy. This is the counterpart of the official MACE
    ``AtomicEnergiesBlock`` (``mace.modules.blocks``), which computes
    ``one_hot(Z over the model's element table) @ atomic_energies``. Here the
    table is stored **Z-indexed** so the encoder can pass raw atomic numbers
    directly; a weight converter is responsible for scattering MACE's
    element-table-ordered ``atomic_energies`` into Z slots via the model's
    ``atomic_numbers`` (z-table).

    The ``atomic_energies`` buffer is persistent (appears in ``state_dict``)
    and **not** learnable, matching MACE.

    Args:
        atomic_energies: Either a 1D tensor/sequence of reference energies in
            element-table order (then ``atomic_numbers`` must be given), or a
            dense Z-indexed 1D tensor (``atomic_numbers=None``).
        atomic_numbers: Element table (z-table) mapping table position → atomic
            number ``Z``. When given, ``atomic_energies`` is scattered into a
            dense Z-indexed lookup.
        max_z: Optional explicit table size ``max_z + 1``; defaults to the
            largest ``Z`` present.

    Reference:
        Batatia et al. "MACE: Higher Order Equivariant Message Passing Neural
        Networks for Fast and Accurate Force Fields" NeurIPS 2022.
        https://arxiv.org/abs/2206.07697
    """

    def __init__(
        self,
        *,
        atomic_energies: torch.Tensor | Sequence[float],
        atomic_numbers: torch.Tensor | Sequence[int] | None = None,
        max_z: int | None = None,
    ) -> None:
        super().__init__()
        e0 = torch.as_tensor(atomic_energies, dtype=config.ftype).flatten()

        if atomic_numbers is not None:
            z = torch.as_tensor(atomic_numbers).flatten().to(torch.long)
            if z.shape != e0.shape:
                raise ValueError(
                    f"atomic_energies ({tuple(e0.shape)}) and atomic_numbers "
                    f"({tuple(z.shape)}) must have the same length."
                )
            size = (int(max_z) if max_z is not None else int(z.max().item())) + 1
            lookup = torch.zeros(size, dtype=config.ftype)
            lookup[z] = e0
        else:
            lookup = e0

        self.register_buffer("atomic_energies", lookup)
        self.atomic_energies: torch.Tensor

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """Return per-atom reference energy ``E0[Z]``.

        Args:
            Z: Atomic numbers ``(N,)`` (integer).

        Returns:
            Per-atom reference energies ``(N,)``.
        """
        return self.atomic_energies[Z.to(torch.long)]

    def extra_repr(self) -> str:
        """Render the Z-table size for ``repr(module)``."""
        return f"num_z={self.atomic_energies.shape[0]}"


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
            nn.Linear(hidden_dim, hidden_dim, dtype=config.ftype),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, dtype=config.ftype),
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
            nn.Linear(hidden_dim, hidden_dim, dtype=config.ftype),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, dtype=config.ftype),
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
