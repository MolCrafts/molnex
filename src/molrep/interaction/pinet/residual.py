"""Residual channel update for PiNet state tracks."""

from __future__ import annotations

import torch
import torch.nn as nn

from molix import config


class ResUpdate(nn.Module):
    """Residual update with optional biasless channel projection."""

    def __init__(self, *, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        if self.in_dim == self.out_dim:
            self.transform: nn.Module = nn.Identity()
        else:
            self.transform = nn.Linear(self.in_dim, self.out_dim, bias=False, dtype=config.ftype)

    def forward(self, old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        return self.transform(old) + new
