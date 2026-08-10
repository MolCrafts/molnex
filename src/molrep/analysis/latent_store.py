"""Store per-atom latent embeddings with molecule and optional type labels."""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["AtomLatentTable"]


@dataclass
class AtomLatentTable:
    """In-memory table of per-atom representations.

    Attributes:
        features: Embedding matrix ``(N, D)``.
        molecule_id: Per-atom molecule id strings length ``N``.
        ref_atom_type: Optional integer type labels ``(N,)``; ``-1`` = unlabeled.
    """

    features: torch.Tensor
    molecule_id: list[str]
    ref_atom_type: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.features.ndim != 2:
            raise ValueError(f"features must be (N, D), got {tuple(self.features.shape)}")
        n = self.features.shape[0]
        if len(self.molecule_id) != n:
            raise ValueError("molecule_id length must match N")
        if self.ref_atom_type is not None:
            if self.ref_atom_type.numel() != n:
                raise ValueError("ref_atom_type length must match N")
            self.ref_atom_type = self.ref_atom_type.long().reshape(-1)

    @property
    def n_atoms(self) -> int:
        return int(self.features.shape[0])

    @property
    def dim(self) -> int:
        return int(self.features.shape[1])
