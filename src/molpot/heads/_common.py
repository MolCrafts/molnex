"""Shared building blocks for molpot head modules."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from molix import config
from molix.F.scatter import scatter_sum


def scalar_mlp(
    in_dim: int, hidden: Sequence[int], out_dim: int, *, dtype: torch.dtype | None = None
) -> nn.Sequential:
    """``[Linear → SiLU] × len(hidden) → Linear`` at ``dtype`` (no activation on the last)."""
    dtype = config.ftype if dtype is None else dtype  # transitional: goes with molix.config (2c)
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers.append(nn.Linear(prev, h, dtype=dtype))
        layers.append(nn.SiLU())
        prev = h
    layers.append(nn.Linear(prev, out_dim, dtype=dtype))
    return nn.Sequential(*layers)


def graph_counts(batch: torch.Tensor, num_graphs: int, *, dtype: torch.dtype) -> torch.Tensor:
    """Per-graph atom counts (clamped to ``>= 1``) for safe per-graph means.

    ``dtype`` is the caller's charges' dtype: this runs on the hot path, where a
    dtype is read off a tensor and never resolved from process-wide state.
    """
    ones = torch.ones(batch.shape[0], dtype=dtype, device=batch.device)
    return scatter_sum(ones, batch, dim=0, dim_size=num_graphs).clamp(min=1.0)
