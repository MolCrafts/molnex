"""Shared building blocks for molpot head modules."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from molix import config
from molix.F.scatter import scatter_sum


def scalar_mlp(in_dim: int, hidden: Sequence[int], out_dim: int) -> nn.Sequential:
    """``[Linear → SiLU] × len(hidden) → Linear`` (no activation on final layer)."""
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers.append(nn.Linear(prev, h, dtype=config.ftype))
        layers.append(nn.SiLU())
        prev = h
    layers.append(nn.Linear(prev, out_dim, dtype=config.ftype))
    return nn.Sequential(*layers)


def graph_counts(batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """Per-graph atom counts (clamped to ``>= 1``) for safe per-graph means."""
    ones = torch.ones(batch.shape[0], dtype=config.ftype, device=batch.device)
    return scatter_sum(ones, batch, dim=0, dim_size=num_graphs).clamp(min=1.0)
