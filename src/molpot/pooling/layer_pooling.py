"""Pool multi-layer encoder features along the layer axis.

Single responsibility: reduce the layer dimension of multi-layer encoder
outputs. Handles both 2D (N, D) pass-through and 3D (N, L, D) reduction.

Example:
    >>> pool = LayerPooling("mean")
    >>> features = torch.randn(10, 3, 64)  # 10 nodes, 3 layers, 64 features
    >>> pooled = pool(features)             # (10, 64)
"""

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
