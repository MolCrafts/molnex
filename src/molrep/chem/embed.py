"""Atom and bond continuous chemical embeddings.

Reuses :class:`~molrep.embedding.node.JointEmbedding` for discrete atomic
numbers (and optional continuous channels). Bond embeddings enforce endpoint
symmetry ``h_ij = h_ji`` by mean-pooling direction-specific MLPs.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from molix import config
from molrep.embedding.node import DiscreteEmbeddingSpec, JointEmbedding


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    """Two-layer SiLU MLP baked to project float dtype."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim, dtype=config.ftype),
        nn.SiLU(),
        nn.Linear(hidden_dim, out_dim, dtype=config.ftype),
    )


class AtomChemEmbedding(nn.Module):
    """Map atomic numbers (and optional discrete channels) to atom features.

    Inputs:
        ``Z`` of shape ``(N,)`` — atomic numbers.
        Optional keyword tensors for extra channels configured at construct
        time (future extension; v1 is Z-only via :class:`JointEmbedding`).

    Outputs:
        ``h_atom`` of shape ``(N, atom_dim)``.

    Args:
        atom_dim: Output feature dimension ``D_a``.
        num_elements: Vocabulary size for the atomic-number table (must cover
            the largest Z that will be looked up; default covers H–Og).
        emb_dim: Intermediate embedding width inside :class:`JointEmbedding`.
            Defaults to ``atom_dim``.
    """

    def __init__(
        self,
        *,
        atom_dim: int = 32,
        num_elements: int = 119,
        emb_dim: int | None = None,
    ) -> None:
        super().__init__()
        if atom_dim <= 0:
            raise ValueError(f"atom_dim must be positive, got {atom_dim}")
        if num_elements <= 0:
            raise ValueError(f"num_elements must be positive, got {num_elements}")
        self.atom_dim = int(atom_dim)
        self.num_elements = int(num_elements)
        z_emb = int(emb_dim) if emb_dim is not None else self.atom_dim
        self.joint = JointEmbedding(
            embedding_specs=[
                DiscreteEmbeddingSpec(
                    input_key="Z",
                    num_classes=self.num_elements,
                    emb_dim=z_emb,
                ),
            ],
            out_dim=self.atom_dim,
        )

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """Embed atomic numbers.

        Args:
            Z: Atomic numbers ``(N,)`` long.

        Returns:
            Atom features ``(N, atom_dim)``. Empty ``N=0`` returns
            ``(0, atom_dim)`` without a JointEmbedding call.
        """
        if Z.numel() == 0:
            return Z.new_zeros((0, self.atom_dim), dtype=config.ftype)
        return self.joint(Z=Z.long())


class BondChemEmbedding(nn.Module):
    """Endpoint-symmetric bond embedding from atom features.

    Implements ``h_ij = ½ (MLP([h_i, h_j]) + MLP([h_j, h_i]))`` so
    reversing bond endpoints leaves the feature unchanged.

    Args:
        atom_dim: Input atom feature dimension ``D_a``.
        bond_dim: Output bond feature dimension ``D_b``.
        hidden_dim: Hidden width of the direction MLP. Defaults to
            ``max(atom_dim * 2, bond_dim)``.
        num_bond_types: If ``> 0``, an optional discrete bond-type table is
            added (summed into both directions before the direction MLP).
            ``0`` disables type conditioning.
    """

    def __init__(
        self,
        *,
        atom_dim: int,
        bond_dim: int = 32,
        hidden_dim: int | None = None,
        num_bond_types: int = 0,
    ) -> None:
        super().__init__()
        if atom_dim <= 0 or bond_dim <= 0:
            raise ValueError("atom_dim and bond_dim must be positive")
        self.atom_dim = int(atom_dim)
        self.bond_dim = int(bond_dim)
        self.num_bond_types = int(num_bond_types)
        hid = int(hidden_dim) if hidden_dim is not None else max(self.atom_dim * 2, self.bond_dim)
        self.dir_mlp = _mlp(2 * self.atom_dim, hid, self.bond_dim)
        if self.num_bond_types > 0:
            self.type_emb = nn.Embedding(self.num_bond_types, self.bond_dim, dtype=config.ftype)
        else:
            self.type_emb = None

    def forward(
        self,
        h_atom: torch.Tensor,
        atomi: torch.Tensor,
        atomj: torch.Tensor,
        bond_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute symmetric bond features.

        Args:
            h_atom: Atom features ``(N, atom_dim)``.
            atomi: Source atom indices ``(N_bonds,)``.
            atomj: Target atom indices ``(N_bonds,)``.
            bond_type: Optional bond type indices ``(N_bonds,)`` when
                ``num_bond_types > 0``.

        Returns:
            Bond features ``(N_bonds, bond_dim)`` with ``h_ij == h_ji``.
        """
        n_bonds = int(atomi.shape[0])
        if n_bonds == 0:
            return h_atom.new_zeros((0, self.bond_dim))

        hi = h_atom[atomi.long()]
        hj = h_atom[atomj.long()]
        # Direction-specific then mean → endpoint symmetry
        fwd = self.dir_mlp(torch.cat([hi, hj], dim=-1))
        rev = self.dir_mlp(torch.cat([hj, hi], dim=-1))
        h_bond = 0.5 * (fwd + rev)

        if self.type_emb is not None and bond_type is not None:
            h_bond = h_bond + self.type_emb(bond_type.long())
        return h_bond
