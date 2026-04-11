"""Shared mixing utilities for combining per-atom parameters into per-pair parameters."""

from __future__ import annotations

import torch


def geometric_arithmetic_mixing(
    atom_params: dict[str, torch.Tensor],
    edge_index: torch.Tensor,
    geometric_keys: list[str],
    arithmetic_keys: list[str],
) -> dict[str, torch.Tensor]:
    """Apply geometric and arithmetic mixing rules to per-atom parameters.

    Geometric mean: ``p_ij = sqrt(p_i * p_j)`` (for energy-like parameters).
    Arithmetic mean: ``p_ij = 0.5 * (p_i + p_j)`` (for distance-like parameters).

    Args:
        atom_params: Per-atom parameter dict, each value ``(N,)``.
        edge_index: Edge indices ``(E, 2)``.
        geometric_keys: Parameter names to combine via geometric mean.
        arithmetic_keys: Parameter names to combine via arithmetic mean.

    Returns:
        Dict with ``"key_ij"`` entries for each input key ``(E,)``.
    """
    src, dst = edge_index[:, 0], edge_index[:, 1]
    result: dict[str, torch.Tensor] = {}

    for key in geometric_keys:
        p = atom_params[key]
        result[f"{key}_ij"] = torch.sqrt(p[src] * p[dst] + 1e-12)

    for key in arithmetic_keys:
        p = atom_params[key]
        result[f"{key}_ij"] = 0.5 * (p[src] + p[dst])

    return result
