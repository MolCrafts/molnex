"""Symmetry-aware interaction context builders for valence terms.

Contexts pool atom (and bond) embeddings into per-interaction feature
vectors used by continuous MM parameter heads. Each builder enforces the
documented reverse / outer-swap symmetry of its interaction class.

Symmetry contracts
------------------
- Bond: ``(i,j) ↔ (j,i)``
- Angle: ``(i,j,k) ↔ (k,j,i)``
- Proper: ``(i,j,k,l) ↔ (l,k,j,i)``
- Improper: outer-leg permutations with center fixed (``atomi`` = center,
  molrs ``[center, i, j, k]`` layout)

No energy, no molpot imports.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from molix import config
from molrep.chem.embed import BondChemEmbedding


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim, dtype=config.ftype),
        nn.SiLU(),
        nn.Linear(hidden_dim, out_dim, dtype=config.ftype),
    )


class BondContext(nn.Module):
    """Bond interaction context — symmetric endpoint pooling.

    Thin wrapper around :class:`~molrep.chem.embed.BondChemEmbedding` so all
    four valence contexts share a uniform ``forward(h_atom, *indices)`` surface.
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
        self.bond_dim = int(bond_dim)
        self.embed = BondChemEmbedding(
            atom_dim=atom_dim,
            bond_dim=bond_dim,
            hidden_dim=hidden_dim,
            num_bond_types=num_bond_types,
        )

    def forward(
        self,
        h_atom: torch.Tensor,
        atomi: torch.Tensor,
        atomj: torch.Tensor,
        bond_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build bond context features ``(N_bonds, bond_dim)``.

        Args:
            h_atom: Atom features ``(N, atom_dim)``.
            atomi: Endpoint i indices ``(N_bonds,)``.
            atomj: Endpoint j indices ``(N_bonds,)``.
            bond_type: Optional bond types ``(N_bonds,)``.

        Returns:
            Bond features invariant under ``(i,j) → (j,i)``.
        """
        return self.embed(h_atom, atomi, atomj, bond_type=bond_type)


class AngleContext(nn.Module):
    """Angle context with reverse symmetry ``(i,j,k) ↔ (k,j,i)``.

    Mean-pools direction-specific MLPs on ``[h_i, h_j, h_k]`` and
    ``[h_k, h_j, h_i]``.
    """

    def __init__(
        self,
        *,
        atom_dim: int,
        angle_dim: int = 32,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.atom_dim = int(atom_dim)
        self.angle_dim = int(angle_dim)
        hid = int(hidden_dim) if hidden_dim is not None else max(3 * self.atom_dim, self.angle_dim)
        self.dir_mlp = _mlp(3 * self.atom_dim, hid, self.angle_dim)

    def forward(
        self,
        h_atom: torch.Tensor,
        atomi: torch.Tensor,
        atomj: torch.Tensor,
        atomk: torch.Tensor,
    ) -> torch.Tensor:
        """Build angle features ``(N_angles, angle_dim)``.

        Args:
            h_atom: Atom features ``(N, atom_dim)``.
            atomi: Endpoint i ``(N_angles,)``.
            atomj: Central atom j ``(N_angles,)``.
            atomk: Endpoint k ``(N_angles,)``.

        Returns:
            Features invariant under ``(i,j,k) → (k,j,i)``.
        """
        n = int(atomi.shape[0])
        if n == 0:
            return h_atom.new_zeros((0, self.angle_dim))
        hi = h_atom[atomi.long()]
        hj = h_atom[atomj.long()]
        hk = h_atom[atomk.long()]
        fwd = self.dir_mlp(torch.cat([hi, hj, hk], dim=-1))
        rev = self.dir_mlp(torch.cat([hk, hj, hi], dim=-1))
        return 0.5 * (fwd + rev)


class ProperContext(nn.Module):
    """Proper-torsion context with reverse symmetry ``(i,j,k,l) ↔ (l,k,j,i)``.

    Mean-pools direction-specific MLPs on the four endpoint embeddings.
    """

    def __init__(
        self,
        *,
        atom_dim: int,
        proper_dim: int = 32,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.atom_dim = int(atom_dim)
        self.proper_dim = int(proper_dim)
        hid = int(hidden_dim) if hidden_dim is not None else max(4 * self.atom_dim, self.proper_dim)
        self.dir_mlp = _mlp(4 * self.atom_dim, hid, self.proper_dim)

    def forward(
        self,
        h_atom: torch.Tensor,
        atomi: torch.Tensor,
        atomj: torch.Tensor,
        atomk: torch.Tensor,
        atoml: torch.Tensor,
    ) -> torch.Tensor:
        """Build proper-torsion features ``(N_propers, proper_dim)``.

        Args:
            h_atom: Atom features ``(N, atom_dim)``.
            atomi: Atom i ``(N_propers,)``.
            atomj: Atom j ``(N_propers,)``.
            atomk: Atom k ``(N_propers,)``.
            atoml: Atom l ``(N_propers,)``.

        Returns:
            Features invariant under ``(i,j,k,l) → (l,k,j,i)``.
        """
        n = int(atomi.shape[0])
        if n == 0:
            return h_atom.new_zeros((0, self.proper_dim))
        hi = h_atom[atomi.long()]
        hj = h_atom[atomj.long()]
        hk = h_atom[atomk.long()]
        hl = h_atom[atoml.long()]
        fwd = self.dir_mlp(torch.cat([hi, hj, hk, hl], dim=-1))
        rev = self.dir_mlp(torch.cat([hl, hk, hj, hi], dim=-1))
        return 0.5 * (fwd + rev)


class ImproperContext(nn.Module):
    """Improper context invariant under outer-leg swaps (center fixed).

    Index convention (molrs / batch schema): ``atomi`` is the **center**,
    ``atomj`` / ``atomk`` / ``atoml`` are the three outer legs. Features are
    built as ``MLP([h_center, sum(h_outer)])``, which is fully symmetric in
    the outer legs.
    """

    def __init__(
        self,
        *,
        atom_dim: int,
        improper_dim: int = 32,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.atom_dim = int(atom_dim)
        self.improper_dim = int(improper_dim)
        hid = (
            int(hidden_dim) if hidden_dim is not None else max(2 * self.atom_dim, self.improper_dim)
        )
        # [center || sum(outer)] → improper_dim
        self.mlp = _mlp(2 * self.atom_dim, hid, self.improper_dim)

    def forward(
        self,
        h_atom: torch.Tensor,
        atomi: torch.Tensor,
        atomj: torch.Tensor,
        atomk: torch.Tensor,
        atoml: torch.Tensor,
    ) -> torch.Tensor:
        """Build improper features ``(N_impropers, improper_dim)``.

        Args:
            h_atom: Atom features ``(N, atom_dim)``.
            atomi: Center atom indices ``(N_impropers,)`` (molrs center-first).
            atomj: Outer leg j ``(N_impropers,)``.
            atomk: Outer leg k ``(N_impropers,)``.
            atoml: Outer leg l ``(N_impropers,)``.

        Returns:
            Features invariant under any permutation of ``(j,k,l)`` with
            center ``i`` fixed.
        """
        n = int(atomi.shape[0])
        if n == 0:
            return h_atom.new_zeros((0, self.improper_dim))
        h_c = h_atom[atomi.long()]
        h_outer = h_atom[atomj.long()] + h_atom[atomk.long()] + h_atom[atoml.long()]
        return self.mlp(torch.cat([h_c, h_outer], dim=-1))
