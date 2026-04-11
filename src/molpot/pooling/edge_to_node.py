"""Aggregate edge features to destination nodes.

Single responsibility: convert edge-centric encoder outputs (e.g. Allegro)
to per-node features suitable for downstream parameter heads.

Example:
    >>> pool = EdgeToNodePooling("mean")
    >>> edge_features = torch.randn(20, 64)  # 20 edges, 64 features
    >>> edge_index = torch.randint(0, 10, (20, 2))
    >>> node_features = pool(edge_features, edge_index, num_nodes=10)  # (10, 64)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EdgeToNodePooling(nn.Module):
    """Aggregate edge features to destination nodes.

    Converts edge-centric encoder outputs (e.g. Allegro) to per-node
    features suitable for parameter heads.

    Args:
        reduction: Aggregation strategy (``"mean"`` or ``"sum"``).
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in ("mean", "sum"):
            raise ValueError(
                f"reduction must be 'mean' or 'sum', got '{reduction}'"
            )
        self.reduction = reduction

    def forward(
        self,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Aggregate edge features to nodes.

        Args:
            edge_features: Per-edge features ``(E, D)``.
            edge_index: Edge indices ``(E, 2)``.
            num_nodes: Total number of nodes.

        Returns:
            Per-node features ``(N, D)``.
        """
        dst = edge_index[:, 1]
        feat_dim = edge_features.shape[-1]

        node_features = torch.zeros(
            num_nodes, feat_dim,
            dtype=edge_features.dtype,
            device=edge_features.device,
        )
        node_features.index_add_(0, dst, edge_features)

        if self.reduction == "mean":
            counts = torch.zeros(
                num_nodes, dtype=edge_features.dtype, device=edge_features.device
            )
            counts.index_add_(0, dst, torch.ones_like(dst, dtype=edge_features.dtype))
            node_features = node_features / counts.clamp(min=1.0).unsqueeze(-1)

        return node_features

    def __repr__(self) -> str:
        return f"EdgeToNodePooling(reduction='{self.reduction}')"
