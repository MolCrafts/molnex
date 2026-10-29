"""Static entity maps for sharing one parameter prediction across conformers."""

from collections.abc import Mapping

import torch
from torch import nn


class ParameterMap(nn.Module):
    """Gather unique-topology tables into a packed conformer batch.

    Construct outside the compiled region using atom/interaction identity, never
    coordinate matching. The source sizes and indices are validated once here.
    Autograd sums all conformers' contributions back to the unique prediction.
    Recompute that prediction after each backward; this module caches only indices.
    """

    def __init__(self, indices: Mapping[str, torch.Tensor], sizes: Mapping[str, int]) -> None:
        super().__init__()
        if set(indices) != set(sizes):
            raise ValueError("each namespace needs indices and a source size")
        self.namespaces = tuple(indices)
        for name, index in indices.items():
            if index.dtype != torch.long or index.ndim != 1:
                raise ValueError(f"{name}: indices must be a one-dimensional int64 tensor")
            if sizes[name] < 0 or bool(((index < 0) | (index >= sizes[name])).any()):
                raise ValueError(f"{name}: parameter index outside source table")
            self.register_buffer(name, index.clone())

    def forward(self, tables):
        return {
            namespace: {
                name: value[self._buffers[namespace]] for name, value in tables[namespace].items()
            }
            for namespace in self.namespaces
        }
