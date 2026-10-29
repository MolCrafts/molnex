"""PiNet2 readout: the per-block energy output of the PiNet2 products.

The encoder's blocks are :mod:`molrep.encoders.pinet.layers`.

Reference:
    Li et al. "PiNN: Equivariant Neural Network Suite for Modeling
    Electrochemical Systems" J. Chem. Theory Comput. 21, 1382 (2025).
    https://doi.org/10.1021/acs.jctc.4c01570
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from molrep.embedding.mlp import FFLayer

__all__ = ["OutLayer"]


class OutLayer(nn.Module):
    """Per-block output with residual accumulation (PiNN ``OutLayer``).

    ``out_k = Dense_biasless(FFLayer(p_k)) + out_{k-1}``. PiNN builds the
    ``FFLayer`` with no activation (Keras ``Dense`` default), so the default
    here is linear; the final projection carries no bias — the energy zero
    point belongs to the atomic reference energy.

    Args:
        n_nodes: FF widths.
        in_dim: Width of the block feature.
        out_units: Output channels.
        activation: FF activation; ``None`` (default) matches PiNN.

    Raises:
        ValueError: If ``n_nodes`` is empty.
    """

    def __init__(
        self,
        n_nodes: Sequence[int],
        *,
        in_dim: int,
        out_units: int = 1,
        activation: str | type[nn.Module] | None = None,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if not n_nodes:
            raise ValueError("OutLayer requires at least one FF width.")
        self.ff_layer = FFLayer(
            n_nodes, in_dim=int(in_dim), activation=activation, use_bias=True, dtype=dtype
        )
        self.out_units = nn.Linear(int(n_nodes[-1]), int(out_units), bias=False, dtype=dtype)

    def forward(self, px: torch.Tensor, prev_output: torch.Tensor) -> torch.Tensor:
        """Args: ``px`` ``(N, in_dim)``; ``prev_output`` ``(N, out_units)``. Returns its shape."""
        return self.out_units(self.ff_layer(px)) + prev_output
