"""
Module wrappers for scatter operations (molix)
"""

import torch.nn as nn

from ..F import scatter as F


class ScatterSum(nn.Module):
    """Sum ``src`` rows into buckets given by an index tensor.

    A thin :class:`~torch.nn.Module` wrapper over
    :func:`molix.F.scatter.scatter_sum`.

    Args:
        dim: Axis along which to scatter-reduce.
    """

    def __init__(self, dim=0):
        super().__init__()
        self.dim = dim

    def forward(self, src, index, dim_size=None):
        """Sum ``src`` over ``dim`` into ``index`` buckets.

        Args:
            src: Source values to aggregate.
            index: Bucket index for each ``src`` slice along ``dim``.
            dim_size: Size of the output ``dim``; inferred from ``index`` if
                ``None``.

        Returns:
            The scatter-summed tensor.
        """
        return F.scatter_sum(src, index, self.dim, dim_size)

    def extra_repr(self):
        """Render the configured scatter ``dim`` for ``repr(module)``."""
        return f"dim={self.dim}"


class BatchAggregation(nn.Module):
    """Sum per-node values into per-graph totals using a ``batch`` index.

    A thin :class:`~torch.nn.Module` wrapper over
    :func:`molix.F.scatter.batch_add`, used to pool node features into
    graph-level quantities.
    """

    def __init__(self):
        super().__init__()

    def forward(self, src, batch, dim_size=None):
        """Sum ``src`` rows per graph as labelled by ``batch``.

        Args:
            src: Per-node values ``(N, ...)``.
            batch: Graph membership index ``(N,)`` mapping each row to a graph.
            dim_size: Number of graphs; inferred from ``batch`` if ``None``.

        Returns:
            Per-graph sums ``(num_graphs, ...)``.
        """
        return F.batch_add(src, batch, dim_size)


__all__ = ["ScatterSum", "BatchAggregation"]
