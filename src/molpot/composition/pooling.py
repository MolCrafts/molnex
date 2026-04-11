"""Feature pooling modules for encoder output preprocessing."""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerPooling(nn.Module):
    """Pool multi-layer encoder features along the layer axis.

    Handles both 2D ``(N, D)`` and 3D ``(N, L, D)`` inputs.
    2D inputs pass through unchanged.

    Args:
        reduction: Pooling strategy (``"mean"``, ``"sum"``, or ``"last"``).
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in ("mean", "sum", "last"):
            raise ValueError(
                f"reduction must be 'mean', 'sum', or 'last', got '{reduction}'"
            )
        self.reduction = reduction

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Pool layer dimension.

        Args:
            features: Encoder features ``(N, L, D)`` or ``(N, D)``.

        Returns:
            Pooled features ``(N, D)``.
        """
        if features.ndim == 2:
            return features
        if features.ndim != 3:
            raise ValueError(f"Expected 2D or 3D tensor, got {features.ndim}D.")
        if self.reduction == "mean":
            return features.mean(dim=1)
        if self.reduction == "sum":
            return features.sum(dim=1)
        return features[:, -1]  # "last"

    def __repr__(self) -> str:
        return f"LayerPooling(reduction='{self.reduction}')"


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
