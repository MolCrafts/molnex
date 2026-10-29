"""Topological equivalence of atoms and bonds, as a module.

The published model averages features over graph-symmetry classes so that equivalent atoms
(the three hydrogens of a methyl, the two oxygens of a carboxylate) get equal
parameters by construction. The classes come from the batch
(``atoms.equi_index``, the index of each class's first atom); a bond's class is
the unordered pair of its endpoints' classes. Used by the covalent-graph
encoder (:mod:`molzoo.rondo`) and its bond-charge-correction head
(:mod:`molpot.classical`).

Reference:
    Zheng, T. et al. "Data-driven parametrization of molecular mechanics force
    fields for expansive chemical space coverage." Chem. Sci. 2025.
    https://arxiv.org/abs/2408.12817 — reference implementation
    ``ByteDance-Seed/byteff2@v1.1.0``,
    ``model/graph_block.py:equi_features``, ``utils/mol_utils.py:find_equivalent_index``.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["Equivalence"]


class Equivalence(nn.Module):
    """Topological equivalence: class-mean features, and bond classes from atom classes.

    The published model averages atom and bond features over graph-symmetry classes
    (``graph_block.py:equi_features``); a bond's class is the unordered pair of
    its endpoints' atom classes (``mol_utils.py:find_equivalent_index``). Both
    run on static shapes with no host sync.
    """

    def forward(self, features: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
        """Replace each row by the mean of its class.

        Args:
            features: Rows ``(M, D)``.
            classes: Class of each row ``(M,)``, an id in ``[0, M)``.

        Returns:
            ``(M, D)``; rows of one class are equal.
        """
        sums = torch.zeros_like(features).index_add(0, classes, features)
        ones = torch.ones_like(features[:, :1])
        counts = torch.zeros_like(ones).index_add(0, classes, ones)
        return (sums / counts.clamp(min=1.0))[classes]

    def bonds(self, i: torch.Tensor, j: torch.Tensor, atom_class: torch.Tensor) -> torch.Tensor:
        """The class of every bond ``(E,)``, ids in ``[0, E)``.

        Args:
            i: First endpoint of every bond ``(E,)``.
            j: Second endpoint ``(E,)``.
            atom_class: Class of every atom ``(N,)``, an atom index.
        """
        a, b = atom_class[i], atom_class[j]
        key = torch.minimum(a, b) * atom_class.shape[0] + torch.maximum(a, b)
        ordered, order = torch.sort(key)
        starts = torch.ones_like(ordered)
        starts[1:] = (ordered[1:] != ordered[:-1]).long()
        group = torch.cumsum(starts, dim=0) - 1
        return torch.empty_like(group).scatter(0, order, group)
