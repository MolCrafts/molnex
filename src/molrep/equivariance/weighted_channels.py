"""Native Allegro weighted channels.

The module owns the path table and the CUDA op. It does not own the per-edge
weights. The caller passes them irrep-major, ``(edges, irreps, channels)``,
which is the layout after Allegro transposes its channel-major parameter.

On a CUDA tensor ``molnex_equivariance::weighted_channels`` evaluates the
table (an indexed sum is a deterministic two-pass sum by atom); on a CPU
tensor :mod:`.reference` does. A descriptor outside the uniform-channel family
raises at construction.

Reference:
    Musaelian et al. "Learning Local Equivariant Representations for
    Large-Scale Atomistic Dynamics" Nat. Commun. 14, 579 (2023).
    https://arxiv.org/abs/2204.05249
"""

from __future__ import annotations

import torch
from torch import nn

from . import reference
from .backend import NativeBackend
from .weighted_channels_terms import WeightedChannelsTerms


class WeightedChannels(nn.Module):
    """Weight each basis component by the channel row of its irrep.

    Args:
        irrep_dims: Dimension ``2l + 1`` of each irrep, in order.
        channels: Multiplicity ``u``.
        dtype: ``float32`` or ``float64``. Coefficients are stored in this dtype.
        device: Device of the term-table buffers.

    ``num_rows is None`` keeps one output row per edge. Otherwise ``index``
    sums the rows into ``num_rows`` atoms. ``weight`` is
    ``(edges, num_irreps, channels)`` or that matrix flattened.
    """

    def __init__(
        self,
        irrep_dims: list[int],
        channels: int,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("native weighted channels requires float32 or float64")
        tables = WeightedChannelsTerms(irrep_dims, channels)
        self.irrep_dims = [int(dim) for dim in irrep_dims]
        self.channels = tables.channels
        self.num_irreps = tables.num_irreps
        self.components = tables.components
        self.output_dim = self.components * self.channels
        self.weight_dim = self.num_irreps * self.channels
        self.register_buffer(
            "terms", torch.tensor(tables.terms, dtype=torch.int16, device=device), persistent=False
        )
        self.register_buffer(
            "coefficients",
            torch.tensor(tables.coefficients, dtype=dtype, device=device),
            persistent=False,
        )
        NativeBackend.load()

    def forward(
        self,
        basis: torch.Tensor,
        weight: torch.Tensor,
        index: torch.Tensor | None = None,
        num_rows: int | None = None,
    ) -> torch.Tensor:
        """Weight the basis, optionally summing rows by ``index``.

        Args:
            basis: ``(edges, components)``.
            weight: ``(edges, num_irreps, channels)`` or the flat irrep-major row.
            index: Atom each edge adds to. Omitted keeps one row per edge.
            num_rows: Output rows when ``index`` is given.

        Returns:
            ``(edges, components * channels)``, or ``(num_rows, components * channels)``.
        """
        if (index is None) != (num_rows is None):
            raise ValueError("index and num_rows come together")
        if basis.ndim != 2 or basis.shape[1] != self.components:
            raise ValueError("basis must be (edges, components)")
        if basis.dtype != self.coefficients.dtype or weight.dtype != self.coefficients.dtype:
            raise ValueError("basis and weight dtype must match the coefficient dtype")
        flat = self._weight(weight, basis.shape[0])
        if index is None:
            rows = -1
            ids = basis.new_empty((0,), dtype=torch.int32)
        else:
            rows = int(num_rows)
            if rows < 0:
                raise ValueError("num_rows must be non-negative")
            if index.ndim != 1 or index.shape[0] != basis.shape[0]:
                raise ValueError("index must have one entry per edge")
            if index.dtype not in (torch.int32, torch.int64):
                raise ValueError("index must be int32 or int64")
            ids = index.contiguous()
        if basis.is_cuda:
            return torch.ops.molnex_equivariance.weighted_channels(
                basis.contiguous(),
                flat,
                ids,
                self.terms,
                self.coefficients,
                self.channels,
                self.num_irreps,
                rows,
            )
        return reference.weighted_channels(
            basis,
            flat,
            self.coefficients,
            self.terms.long(),
            self.channels,
            self.num_irreps,
            None if index is None else ids,
            None if index is None else rows,
        )

    def _weight(self, weight: torch.Tensor, edges: int) -> torch.Tensor:
        matrix = (edges, self.num_irreps, self.channels)
        flat = (edges, self.weight_dim)
        if weight.shape == matrix or weight.shape == flat:
            return weight.contiguous()
        raise ValueError("weight must be (edges, irreps, channels) or the flat irrep-major row")
