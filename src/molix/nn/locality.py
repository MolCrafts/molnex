"""
Module wrappers for locality operations (molix)
"""

import torch.nn as nn

from ..F import locality as F


class NeighborList(nn.Module):
    """Build neighbor pairs within a cutoff radius.

    A :class:`~torch.nn.Module` wrapper over
    :func:`molix.F.locality.get_neighbor_pairs`.

    Args:
        cutoff: Neighbor cutoff radius.
        pbc: Whether periodic boundary conditions apply.
    """

    def __init__(self, cutoff, pbc=True):
        super().__init__()
        self.cutoff = cutoff
        self.pbc = pbc

    def forward(self, positions, cell):
        """Return neighbor pairs for ``positions`` under the given cell.

        Args:
            positions: Atom positions ``(N, 3)``.
            cell: Box vectors ``(3, 3)`` defining the simulation cell.

        Returns:
            The neighbor-pair output of
            :func:`molix.F.locality.get_neighbor_pairs`.
        """
        return F.get_neighbor_pairs(positions, self.cutoff, box_vectors=cell)

    def extra_repr(self):
        """Render ``cutoff`` and ``pbc`` for ``repr(module)``."""
        return f"cutoff={self.cutoff}, pbc={self.pbc}"


__all__ = ["NeighborList"]
