"""Graph-level pooling: aggregate atomic features to molecular features.

Single responsibility: scatter-based node-to-graph aggregation via
sum, mean, or max operations.

Example:
    >>> pool = SumPooling()
    >>> x = torch.randn(5, 64)                   # 5 atoms, 64 features
    >>> batch = torch.tensor([0, 0, 0, 1, 1])
    >>> mol_features = pool(x, batch)             # (2, 64)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SumPooling(nn.Module):
    """Sum pooling: aggregate atomic features to molecular features."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        dim_size: int | None = None,
    ) -> torch.Tensor:
        """Pool atomic features to molecular features via summation.

        Args:
            x: Atomic features ``(N,)`` or ``(N, D)``.
            batch: Batch/molecule indices ``(N,)``.
            dim_size: Number of molecules (optional, inferred if None).

        Returns:
            Molecular features ``(B,)`` or ``(B, D)``.
        """
        if dim_size is None:
            dim_size = int(batch.max()) + 1

        if x.dim() == 1:
            out = torch.zeros(dim_size, dtype=x.dtype, device=x.device)
        else:
            out = torch.zeros(dim_size, x.shape[1], dtype=x.dtype, device=x.device)

        out.index_add_(0, batch, x)
        return out

    def __repr__(self) -> str:
        return "SumPooling()"


class MeanPooling(nn.Module):
    """Mean pooling: aggregate atomic features to molecular features."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        dim_size: int | None = None,
    ) -> torch.Tensor:
        """Pool atomic features to molecular features via averaging.

        Args:
            x: Atomic features ``(N,)`` or ``(N, D)``.
            batch: Batch/molecule indices ``(N,)``.
            dim_size: Number of molecules (optional, inferred if None).

        Returns:
            Molecular features ``(B,)`` or ``(B, D)``.
        """
        if dim_size is None:
            dim_size = int(batch.max()) + 1

        if x.dim() == 1:
            out_sum = torch.zeros(dim_size, dtype=x.dtype, device=x.device)
        else:
            out_sum = torch.zeros(dim_size, x.shape[1], dtype=x.dtype, device=x.device)
        out_sum.index_add_(0, batch, x)

        counts = torch.zeros(dim_size, dtype=x.dtype, device=x.device)
        ones = torch.ones_like(batch, dtype=x.dtype)
        counts.index_add_(0, batch, ones)

        if x.dim() == 1:
            return out_sum / counts.clamp(min=1)
        else:
            return out_sum / counts.unsqueeze(-1).clamp(min=1)

    def __repr__(self) -> str:
        return "MeanPooling()"


class MaxPooling(nn.Module):
    """Max pooling: aggregate atomic features to molecular features."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        dim_size: int | None = None,
    ) -> torch.Tensor:
        """Pool atomic features to molecular features via max operation.

        Args:
            x: Atomic features ``(N,)`` or ``(N, D)``.
            batch: Batch/molecule indices ``(N,)``.
            dim_size: Number of molecules (optional, inferred if None).

        Returns:
            Molecular features ``(B,)`` or ``(B, D)``.
        """
        if dim_size is None:
            dim_size = int(batch.max()) + 1

        if x.dim() == 1:
            out = torch.full((dim_size,), float("-inf"), dtype=x.dtype, device=x.device)
        else:
            out = torch.full((dim_size, x.shape[1]), float("-inf"), dtype=x.dtype, device=x.device)

        for mol_idx in range(dim_size):
            mask = batch == mol_idx
            if mask.any():
                out[mol_idx] = x[mask].max(dim=0)[0] if x.dim() > 1 else x[mask].max()

        return out

    def __repr__(self) -> str:
        return "MaxPooling()"
