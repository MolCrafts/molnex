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
        pbc: Whether periodic boundary conditions apply (``box_vectors`` used
            only when ``True``).
        max_num_pairs: Buffer size for the C++ kernel (``-1`` = all pairs).
    """

    def __init__(self, cutoff, pbc=True, max_num_pairs: int = -1):
        super().__init__()
        self.cutoff = cutoff
        self.pbc = pbc
        self.max_num_pairs = max_num_pairs

    def forward(self, positions, cell):
        """Return neighbor pairs for ``positions`` under the given cell.

        Args:
            positions: Atom positions ``(N, 3)``.
            cell: Box vectors ``(3, 3)`` defining the simulation cell.

        Returns:
            The neighbor-pair output of
            :func:`molix.F.locality.get_neighbor_pairs`.
        """
        box = cell if self.pbc else None
        return F.get_neighbor_pairs(
            positions,
            self.cutoff,
            max_num_pairs=self.max_num_pairs,
            box_vectors=box,
        )

    def extra_repr(self):
        """Render constructor args for ``repr(module)``."""
        return f"cutoff={self.cutoff}, pbc={self.pbc}, max_num_pairs={self.max_num_pairs}"


__all__ = ["NeighborList"]
