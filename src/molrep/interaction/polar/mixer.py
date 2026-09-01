"""Sum of per-layer equivariant linears (PolarMACE ``MultiLayerFeatureMixer``).

Official polar-1-M uses two layers of ``512x0e+512x1o`` (dim 2048).

Reference:
    Batatia et al. "MACE-POLAR-1: A Polarisable Electrostatic Foundation Model
    for Molecular Chemistry" arXiv:2602.19411.
"""

from __future__ import annotations

import cuequivariance as cue
import cuequivariance_torch as cuet
import torch
import torch.nn as nn

from molix import config


class MultiLayerFeatureMixer(nn.Module):
    """``tilde h = sum_t W^{(t)} h^{(t)}`` with one ``cuet.Linear`` per layer.

    Args:
        irreps: Node-feature irreps of every layer, e.g. ``"512x0e+512x1o"``.
        num_layers: Number of interaction layers ``T``. Must be >= 1.

    Forward:
        ``all_node_feats`` of shape ``(T, N, irreps.dim)``. ``T`` must equal
        ``num_layers``.
    """

    def __init__(self, *, irreps: str, num_layers: int) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.num_layers = num_layers
        ir = cue.Irreps("O3", irreps)
        self.irreps_dim = ir.dim
        self.linears = nn.ModuleList(
            [cuet.Linear(ir, ir, layout=cue.ir_mul, dtype=config.ftype) for _ in range(num_layers)]
        )

    def forward(self, all_node_feats: torch.Tensor) -> torch.Tensor:
        """Mix stacked per-layer features.

        Args:
            all_node_feats: ``(T, N, irreps.dim)``.

        Returns:
            Mixed features ``(N, irreps.dim)``.

        Raises:
            ValueError: If ``T != num_layers``.
        """
        n_stack = all_node_feats.shape[0]
        if n_stack != self.num_layers:
            raise ValueError(
                f"all_node_feats has {n_stack} layers but mixer was built with "
                f"num_layers={self.num_layers}"
            )
        mixed = self.linears[0](all_node_feats[0])
        for i in range(1, self.num_layers):
            mixed = mixed + self.linears[i](all_node_feats[i])
        return mixed
