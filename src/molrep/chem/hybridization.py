"""Hybridization read off the covalent graph: degree, π count, linear and trigonal centres.

A Class-I parameter head needs a reference geometry per centre before any
training signal arrives — an sp carbon starts at 180°, an sp² one at 120° —
and a torsion through a linear centre has no dihedral to speak of. Both facts
are functions of the bond list and the bond orders alone, so they are
computed here from ``bonds`` columns with two ``index_add`` calls on static
shapes: no coordinate, no host sync, no data-dependent shape.

Bond orders follow molpy's SMILES reader: ``bond_type`` ``1`` single, ``2``
double, ``3`` triple, ``4`` aromatic (order 1.5); ``0`` is a padded row.

Reference:
    Zheng et al. "Data-driven parametrization of molecular mechanics force
    fields for expansive chemical space coverage." (ByteFF) — linear-centre
    torsions are zeroed by a SMARTS match, ``mlparams.py:151-153``.
    https://arxiv.org/abs/2408.12817
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = ["Hybridization"]

#: Bond order by ``bonds.bond_type`` code; index 0 is a padded (dead) row.
_ORDER_OF_TYPE = (0.0, 1.0, 2.0, 3.0, 1.5)


class Hybridization(nn.Module):
    """Per-atom degree and π-bond count from the bond list.

    ``forward`` returns ``(degree, pi)``: ``degree`` is the number of live
    bonds on each atom, ``pi`` the summed bond order in excess of one
    (``order − 1`` per bond: 0 for single, 1 for double, 2 for triple, 0.5 for
    aromatic). From these, :meth:`linear` marks sp centres and
    :meth:`angle_reference` gives the VSEPR angle a harmonic angle head starts
    from.

    Args:
        linear_deg: Reference angle at a linear centre, degrees. **180°**: the
            harmonic angle ``K (θ − θ0)²`` is a smooth isotropic bend only when
            ``θ0`` is exactly 180° (molpot's angle kernel is C^∞ there); at any
            ``θ0 < 180°`` the linear geometry sits on top of a Mexican hat and
            the partial-Hessian fit drives ``K`` to zero (sp carbon with the old
            177.5° reference and a 179.5° ceiling).
        trigonal_deg: Reference angle at a trigonal centre, degrees.
        tetrahedral_deg: Reference angle everywhere else, degrees.
    """

    def __init__(
        self,
        *,
        linear_deg: float = 180.0,
        trigonal_deg: float = 120.0,
        tetrahedral_deg: float = 109.47,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "order_of_type", torch.tensor(_ORDER_OF_TYPE, dtype=torch.float64), persistent=False
        )
        self.linear_deg = float(linear_deg)
        self.trigonal_deg = float(trigonal_deg)
        self.tetrahedral_deg = float(tetrahedral_deg)

    def forward(
        self,
        bond_index: torch.Tensor,
        bond_type: torch.Tensor,
        bond_mask: torch.Tensor | None,
        n_atoms: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Degree and π count per atom.

        Args:
            bond_index: ``(2, N_bonds)`` endpoint indices.
            bond_type: ``(N_bonds,)`` bond-type codes (see module docstring).
            bond_mask: ``(N_bonds,)`` live-row mask, or ``None`` when every row
                is live.
            n_atoms: Atom rows ``N`` (a static shape, never read off a tensor).

        Returns:
            ``degree`` ``(N,)`` and ``pi`` ``(N,)``, both ``float64``.
        """
        i, j = bond_index[0].long(), bond_index[1].long()
        live = (
            bond_type.new_ones(bond_type.shape, dtype=torch.float64)
            if bond_mask is None
            else bond_mask.to(torch.float64)
        )
        order = self.order_of_type[bond_type.long().clamp(0, len(_ORDER_OF_TYPE) - 1)] * live
        degree = live.new_zeros(n_atoms).index_add(0, i, live).index_add(0, j, live)
        total_order = live.new_zeros(n_atoms).index_add(0, i, order).index_add(0, j, order)
        return degree, total_order - degree

    @staticmethod
    def linear(degree: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
        """``True`` on sp centres: two bonds carrying two π bonds (``C≡C``, ``C=C=C``)."""
        return (degree == 2) & (pi > 1.5)

    def angle_reference(self, degree: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
        """The VSEPR angle at each centre, radians.

        Linear centres get ``linear_deg``; a centre with at most three bonds and
        at least one π bond (sp²: carbonyl, aromatic, imine) gets
        ``trigonal_deg``; everything else — including an sp³ centre and an amide
        nitrogen, whose planarity the graph cannot see — gets ``tetrahedral_deg``.

        Args:
            degree: ``(N,)`` from :meth:`forward`.
            pi: ``(N,)`` from :meth:`forward`.

        Returns:
            ``(N,)`` radians, in ``degree``'s dtype.
        """
        tetrahedral = degree.new_full(degree.shape, math.radians(self.tetrahedral_deg))
        trigonal = degree.new_full(degree.shape, math.radians(self.trigonal_deg))
        linear = degree.new_full(degree.shape, math.radians(self.linear_deg))
        ref = torch.where((pi > 0.5) & (degree <= 3), trigonal, tetrahedral)
        return torch.where(self.linear(degree, pi), linear, ref)
