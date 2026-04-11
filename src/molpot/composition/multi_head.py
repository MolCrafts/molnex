"""MultiHead: run multiple parameter heads and merge outputs."""

from __future__ import annotations

import torch
import torch.nn as nn


class MultiHead(nn.Module):
    """Run multiple parameter heads and merge their output dicts.

    Each head must produce a dict with disjoint keys.
    Extra keyword arguments (e.g. ``batch``, ``Z``) are forwarded
    to every head that accepts them.

    Args:
        heads: Named parameter heads.
    """

    def __init__(self, heads: dict[str, nn.Module]):
        super().__init__()
        if not heads:
            raise ValueError("MultiHead requires at least one head.")
        self.heads = nn.ModuleDict(heads)

    def forward(self, node_features: torch.Tensor, **extra) -> dict[str, torch.Tensor]:
        """Run all heads and merge outputs.

        Args:
            node_features: Per-node features ``(N, D)``.
            **extra: Forwarded to each head (e.g. ``batch``, ``Z``).

        Returns:
            Merged dict of per-atom parameters (disjoint keys).
        """
        merged: dict[str, torch.Tensor] = {}
        for name, head in self.heads.items():
            out = head(node_features, **extra)
            for key, val in out.items():
                if key in merged:
                    raise ValueError(
                        f"Duplicate key '{key}' from head '{name}'. "
                        "All heads must produce disjoint keys."
                    )
                merged[key] = val
        return merged
