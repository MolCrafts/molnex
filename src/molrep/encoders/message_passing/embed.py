"""Atom and bond continuous chemical embeddings.

Reuses :class:`~molrep.embedding.node.JointEmbedding` for discrete atomic
numbers. Extra integer columns of the batch — formal charge, total charge, ring
membership, bond order — enter as :class:`DiscreteChannels`, one embedding
table per column summed into the stream (ByteFF's per-feature embeddings,
``layers.py:132-216``). Bond embeddings enforce endpoint symmetry ``h_ij =
h_ji`` by mean-pooling direction-specific MLPs.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from molrep.embedding.mlp import DenseMLP
from molrep.embedding.node import DiscreteEmbeddingSpec, JointEmbedding
from molrep.encoders.message_passing.spec import DiscreteChannel

__all__ = ["AtomChemEmbedding", "BondChemEmbedding", "DiscreteChannel", "DiscreteChannels"]


class DiscreteChannels(nn.Module):
    """Embedding tables for a fixed set of :class:`DiscreteChannel`, summed.

    The plan (name, offset, size, per-graph) is snapshotted into plain tuples at
    construction: reading pydantic fields inside the forward is a graph break
    on a hot path.

    Args:
        channels: The columns, in order. May be empty, in which case
            :meth:`forward` returns ``None`` and the owner adds nothing.
        dim: Width of every table — the stream the sum is added to.
    """

    def __init__(
        self, channels: Sequence[DiscreteChannel], dim: int, *, dtype: torch.dtype
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        names = [channel.name for channel in channels]
        if len(set(names)) != len(names):
            raise ValueError(f"channel names must be unique, got {names}")
        self.dim = int(dim)
        self.tables = nn.ModuleDict(
            {
                channel.name: nn.Embedding(channel.num_classes, self.dim, dtype=dtype)
                for channel in channels
            }
        )
        self._plan: tuple[tuple[str, int, int, bool], ...] = tuple(
            (channel.name, int(channel.offset), int(channel.num_classes), channel.per == "graph")
            for channel in channels
        )

    @property
    def names(self) -> tuple[str, ...]:
        """The column names, in declaration order."""
        return tuple(name for name, _, _, _ in self._plan)

    @property
    def per_graph(self) -> tuple[str, ...]:
        """The names that are read from ``graphs`` and broadcast by ``atoms.batch``."""
        return tuple(name for name, _, _, graph in self._plan if graph)

    def __len__(self) -> int:
        return len(self._plan)

    def forward(
        self, batch: torch.Tensor | None = None, **columns: torch.Tensor
    ) -> torch.Tensor | None:
        """Sum of the channel embeddings, ``(rows, dim)``; ``None`` with no channels.

        Args:
            batch: ``(N,)`` graph index of each row, needed for per-graph
                columns.
            **columns: One integer (or integral float) tensor per channel name;
                ``(rows,)`` for node columns, ``(B,)`` for graph columns.
        """
        total: torch.Tensor | None = None
        for name, offset, num_classes, per_graph in self._plan:
            index = columns[name]
            if per_graph:
                if batch is None:
                    raise ValueError(
                        f"channel {name!r} is per graph and needs the atoms.batch index"
                    )
                index = index[batch.long()]
            if index.is_floating_point():
                index = torch.round(index)
            index = (index.long() + offset).clamp(0, num_classes - 1)
            embedded = self.tables[name](index)
            total = embedded if total is None else total + embedded
        return total


class AtomChemEmbedding(nn.Module):
    """Map atomic numbers to atom features, plus any extra channel sum.

    Inputs:
        ``Z`` of shape ``(N,)`` — atomic numbers; optionally ``extra``
        ``(N, atom_dim)``, the summed :class:`DiscreteChannels` of the batch's
        extra atom columns, added to the seed.

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
        dtype: torch.dtype,
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
            dtype=dtype,
        )
        # The empty-input answer, built once, at this module's own dtype. A
        # `forward` resolves a dtype from its tensors and nowhere else: a
        # process-wide config read on the traced path was a graph break — and
        # not a graceful one, it de-compiled the enclosing function (notes.md,
        # compile-no-graph-breaks, 2026-09-18).
        # The dtype is this module's own anyway: the empty result stands in for
        # what `self.joint` would have returned. Non-persistent, so it stays out
        # of the state dict but still follows `.to(device)` / `.to(dtype)`.
        self.register_buffer(
            "_no_atoms",
            torch.zeros(0, self.atom_dim, dtype=next(self.joint.parameters()).dtype),
            persistent=False,
        )

    def forward(self, Z: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
        """Embed atomic numbers.

        Args:
            Z: Atomic numbers ``(N,)`` long.
            extra: Optional ``(N, atom_dim)`` channel sum added to the seed.

        Returns:
            Atom features ``(N, atom_dim)``. Empty ``N=0`` returns
            ``(0, atom_dim)`` without a JointEmbedding call.
        """
        if Z.numel() == 0:
            return self._no_atoms
        h_atom = self.joint(Z=Z.long())
        return h_atom if extra is None else h_atom + extra


class BondChemEmbedding(nn.Module):
    """Endpoint-symmetric bond embedding from atom features.

    Seed (``h_bond is None``)::

        h_ij = ½ (MLP([h_i, h_j]) + MLP([h_j, h_i])) + extra_ij

    Iterative residual (``h_bond`` given)::

        h_ij ← pair + Linear_{D_b→D_b}(h_bond)

    Both forms are invariant under ``(i, j) ↔ (j, i)``; ``extra`` is a per-bond
    channel sum (bond order, ring membership) and so is too. Features are
    dimensionless learned embeddings (no unit conversion).

    Args:
        atom_dim: Input atom feature dimension ``D_a``.
        bond_dim: Output bond feature dimension ``D_b``.
        hidden_dim: Hidden width of the direction MLP. Defaults to
            ``max(atom_dim * 2, bond_dim)``.
    """

    def __init__(
        self,
        *,
        atom_dim: int,
        bond_dim: int = 32,
        hidden_dim: int | None = None,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if atom_dim <= 0 or bond_dim <= 0:
            raise ValueError("atom_dim and bond_dim must be positive")
        self.atom_dim = int(atom_dim)
        self.bond_dim = int(bond_dim)
        hid = int(hidden_dim) if hidden_dim is not None else max(self.atom_dim * 2, self.bond_dim)
        self.dir_mlp = DenseMLP(
            2 * self.atom_dim, hid, self.bond_dim, 2, activation="silu", dtype=dtype
        )
        self.mix = nn.Linear(self.bond_dim, self.bond_dim, dtype=dtype)

    def forward(
        self,
        h_atom: torch.Tensor,
        atomi: torch.Tensor,
        atomj: torch.Tensor,
        extra: torch.Tensor | None = None,
        h_bond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute symmetric bond features.

        Args:
            h_atom: Atom features ``(N, atom_dim)``.
            atomi: Source atom indices ``(N_bonds,)``.
            atomj: Target atom indices ``(N_bonds,)``.
            extra: Optional ``(N_bonds, bond_dim)`` channel sum of the batch's
                extra bond columns (:class:`DiscreteChannels`).
            h_bond: Optional previous-layer bond features ``(N_bonds, bond_dim)``.
                ``None`` keeps the seed formula bit-identical.

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
        pair = 0.5 * (fwd + rev)

        if extra is not None:
            pair = pair + extra
        if h_bond is not None:
            pair = pair + self.mix(h_bond)
        return pair
