"""Symmetry-aware interaction context builders for valence terms.

Contexts pool atom and bond embeddings into per-interaction feature
vectors used by continuous MM parameter heads. Each builder enforces the
documented reverse / outer-swap symmetry of its interaction class, and each
reads the features of **the term's own bonds** interleaved with its atoms —
ByteFF's ``OutputWithBond`` (``layers.py:486-510``): an angle is
``[h_i, h_ij, h_j, h_jk, h_k]``, a proper ``[h_i, h_ij, h_j, h_jk, h_k, h_kl,
h_l]``, an improper the sum over its legs of ``[h_c, h_cl, h_l]``. The bond
rows are looked up by :class:`~molrep.chem.bond_table.BondTable`; the encoder
hands the gathered rows in.

Why the bonds are inputs and not inferred: the central bond order is the main
determinant of a torsion barrier, and a context that pools atom features only
has to decode it from a two-hop summary — on a vertex-transitive graph (a
Kekulé ring) the torsion about a single bond and the one about a double bond
got bit-identical features (probe, 2026-10-12).

Symmetry contracts
------------------
- Bond: ``(i,j) ↔ (j,i)``
- Angle: ``(i,j,k) ↔ (k,j,i)`` with ``h_ij ↔ h_jk``
- Proper: ``(i,j,k,l) ↔ (l,k,j,i)`` with ``h_ij ↔ h_kl``
- Improper: outer-leg permutations with center fixed (``atomi`` = center,
  molrs ``[center, i, j, k]`` layout), each leg carried with its own bond

No energy, no molpot imports.

Reference:
    Zheng et al. "Data-driven parametrization of molecular mechanics force
    fields for expansive chemical space coverage." (ByteFF)
    https://arxiv.org/abs/2408.12817
"""

from __future__ import annotations

import torch
import torch.nn as nn

from molrep.chem.embed import BondChemEmbedding


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, *, dtype: torch.dtype) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim, dtype=dtype),
        nn.SiLU(),
        nn.Linear(hidden_dim, out_dim, dtype=dtype),
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
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.bond_dim = int(bond_dim)
        self.embed = BondChemEmbedding(
            atom_dim=atom_dim,
            bond_dim=bond_dim,
            hidden_dim=hidden_dim,
            dtype=dtype,
        )

    def forward(
        self,
        h_atom: torch.Tensor,
        atomi: torch.Tensor,
        atomj: torch.Tensor,
        extra: torch.Tensor | None = None,
        h_bond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build bond context features ``(N_bonds, bond_dim)``.

        Args:
            h_atom: Atom features ``(N, atom_dim)``.
            atomi: Endpoint i indices ``(N_bonds,)``.
            atomj: Endpoint j indices ``(N_bonds,)``.
            extra: Optional ``(N_bonds, bond_dim)`` channel sum of the extra
                bond columns (bond order, ring membership).
            h_bond: Optional previous-layer bond features
                ``(N_bonds, bond_dim)``. Forwarded to
                :class:`~molrep.chem.embed.BondChemEmbedding` without
                rewriting ``(i,j) ↔ (j,i)``.

        Returns:
            Bond features invariant under ``(i,j) → (j,i)``.
        """
        return self.embed(h_atom, atomi, atomj, extra=extra, h_bond=h_bond)


class AngleContext(nn.Module):
    """Angle context with reverse symmetry ``(i,j,k) ↔ (k,j,i)``.

    Mean-pools direction-specific MLPs on ``[h_i, h_ij, h_j, h_jk, h_k]`` and
    ``[h_k, h_jk, h_j, h_ij, h_i]``.

    Args:
        atom_dim: Atom feature width.
        bond_dim: Bond feature width.
        angle_dim: Output width.
        hidden_dim: Hidden width of the MLP; defaults to the larger of the
            input and output widths.
    """

    def __init__(
        self,
        *,
        atom_dim: int,
        bond_dim: int,
        angle_dim: int = 32,
        hidden_dim: int | None = None,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.atom_dim = int(atom_dim)
        self.bond_dim = int(bond_dim)
        self.angle_dim = int(angle_dim)
        in_dim = 3 * self.atom_dim + 2 * self.bond_dim
        hid = int(hidden_dim) if hidden_dim is not None else max(in_dim, self.angle_dim)
        self.dir_mlp = _mlp(in_dim, hid, self.angle_dim, dtype=dtype)

    def forward(
        self,
        h_atom: torch.Tensor,
        atomi: torch.Tensor,
        atomj: torch.Tensor,
        atomk: torch.Tensor,
        h_ij: torch.Tensor,
        h_jk: torch.Tensor,
    ) -> torch.Tensor:
        """Build angle features ``(N_angles, angle_dim)``.

        Args:
            h_atom: Atom features ``(N, atom_dim)``.
            atomi: Endpoint i ``(N_angles,)``.
            atomj: Central atom j ``(N_angles,)``.
            atomk: Endpoint k ``(N_angles,)``.
            h_ij: ``(N_angles, bond_dim)`` features of the bond ``i-j``.
            h_jk: ``(N_angles, bond_dim)`` features of the bond ``j-k``.

        Returns:
            Features invariant under ``(i,j,k) → (k,j,i)`` with ``h_ij ↔ h_jk``.
        """
        n = int(atomi.shape[0])
        if n == 0:
            return h_atom.new_zeros((0, self.angle_dim))
        hi = h_atom[atomi.long()]
        hj = h_atom[atomj.long()]
        hk = h_atom[atomk.long()]
        fwd = self.dir_mlp(torch.cat([hi, h_ij, hj, h_jk, hk], dim=-1))
        rev = self.dir_mlp(torch.cat([hk, h_jk, hj, h_ij, hi], dim=-1))
        return 0.5 * (fwd + rev)


class ProperContext(nn.Module):
    """Proper-torsion context with reverse symmetry ``(i,j,k,l) ↔ (l,k,j,i)``.

    Mean-pools direction-specific MLPs on ``[h_i, h_ij, h_j, h_jk, h_k, h_kl,
    h_l]`` and its reverse.

    Args:
        atom_dim: Atom feature width.
        bond_dim: Bond feature width.
        proper_dim: Output width.
        hidden_dim: Hidden width of the MLP; defaults to the larger of the
            input and output widths.
    """

    def __init__(
        self,
        *,
        atom_dim: int,
        bond_dim: int,
        proper_dim: int = 32,
        hidden_dim: int | None = None,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.atom_dim = int(atom_dim)
        self.bond_dim = int(bond_dim)
        self.proper_dim = int(proper_dim)
        in_dim = 4 * self.atom_dim + 3 * self.bond_dim
        hid = int(hidden_dim) if hidden_dim is not None else max(in_dim, self.proper_dim)
        self.dir_mlp = _mlp(in_dim, hid, self.proper_dim, dtype=dtype)

    def forward(
        self,
        h_atom: torch.Tensor,
        atomi: torch.Tensor,
        atomj: torch.Tensor,
        atomk: torch.Tensor,
        atoml: torch.Tensor,
        h_ij: torch.Tensor,
        h_jk: torch.Tensor,
        h_kl: torch.Tensor,
    ) -> torch.Tensor:
        """Build proper-torsion features ``(N_propers, proper_dim)``.

        Args:
            h_atom: Atom features ``(N, atom_dim)``.
            atomi: Atom i ``(N_propers,)``.
            atomj: Atom j ``(N_propers,)``.
            atomk: Atom k ``(N_propers,)``.
            atoml: Atom l ``(N_propers,)``.
            h_ij: ``(N_propers, bond_dim)`` features of the bond ``i-j``.
            h_jk: ``(N_propers, bond_dim)`` features of the central bond ``j-k``.
            h_kl: ``(N_propers, bond_dim)`` features of the bond ``k-l``.

        Returns:
            Features invariant under ``(i,j,k,l) → (l,k,j,i)`` with ``h_ij ↔ h_kl``.
        """
        n = int(atomi.shape[0])
        if n == 0:
            return h_atom.new_zeros((0, self.proper_dim))
        hi = h_atom[atomi.long()]
        hj = h_atom[atomj.long()]
        hk = h_atom[atomk.long()]
        hl = h_atom[atoml.long()]
        fwd = self.dir_mlp(torch.cat([hi, h_ij, hj, h_jk, hk, h_kl, hl], dim=-1))
        rev = self.dir_mlp(torch.cat([hl, h_kl, hk, h_jk, hj, h_ij, hi], dim=-1))
        return 0.5 * (fwd + rev)


class ImproperContext(nn.Module):
    """Improper context invariant under outer-leg swaps (center fixed).

    Index convention (molrs / batch schema): ``atomi`` is the **center**,
    ``atomj`` / ``atomk`` / ``atoml`` are the three outer legs. Features are
    the mean over the legs of ``MLP([h_center, h_center-leg, h_leg])``, which
    is fully symmetric in the outer legs and keeps each leg with its own bond
    (ByteFF ``layers.py:497-501``).

    Args:
        atom_dim: Atom feature width.
        bond_dim: Bond feature width.
        improper_dim: Output width.
        hidden_dim: Hidden width of the MLP; defaults to the larger of the
            input and output widths.
    """

    def __init__(
        self,
        *,
        atom_dim: int,
        bond_dim: int,
        improper_dim: int = 32,
        hidden_dim: int | None = None,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.atom_dim = int(atom_dim)
        self.bond_dim = int(bond_dim)
        self.improper_dim = int(improper_dim)
        in_dim = 2 * self.atom_dim + self.bond_dim
        hid = int(hidden_dim) if hidden_dim is not None else max(in_dim, self.improper_dim)
        self.leg_mlp = _mlp(in_dim, hid, self.improper_dim, dtype=dtype)

    def forward(
        self,
        h_atom: torch.Tensor,
        atomi: torch.Tensor,
        atomj: torch.Tensor,
        atomk: torch.Tensor,
        atoml: torch.Tensor,
        h_cj: torch.Tensor,
        h_ck: torch.Tensor,
        h_cl: torch.Tensor,
    ) -> torch.Tensor:
        """Build improper features ``(N_impropers, improper_dim)``.

        Args:
            h_atom: Atom features ``(N, atom_dim)``.
            atomi: Center atom indices ``(N_impropers,)`` (molrs center-first).
            atomj: Outer leg j ``(N_impropers,)``.
            atomk: Outer leg k ``(N_impropers,)``.
            atoml: Outer leg l ``(N_impropers,)``.
            h_cj: ``(N_impropers, bond_dim)`` features of the bond centre-``j``.
            h_ck: ``(N_impropers, bond_dim)`` features of the bond centre-``k``.
            h_cl: ``(N_impropers, bond_dim)`` features of the bond centre-``l``.

        Returns:
            Features invariant under any permutation of ``(j,k,l)`` with
            center ``i`` fixed, each leg permuted together with its bond.
        """
        n = int(atomi.shape[0])
        if n == 0:
            return h_atom.new_zeros((0, self.improper_dim))
        h_c = h_atom[atomi.long()]
        legs = (
            self.leg_mlp(torch.cat([h_c, h_cj, h_atom[atomj.long()]], dim=-1))
            + self.leg_mlp(torch.cat([h_c, h_ck, h_atom[atomk.long()]], dim=-1))
            + self.leg_mlp(torch.cat([h_c, h_cl, h_atom[atoml.long()]], dim=-1))
        )
        return legs / 3.0
