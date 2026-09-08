"""Shared mixing utilities for combining per-atom parameters into per-pair parameters."""

from __future__ import annotations

import torch

from molpot.potentials.vdw.lj126 import geometric_mean


def geometric_arithmetic_mixing(
    atom_params: dict[str, torch.Tensor],
    edge_index: torch.Tensor,
    geometric_keys: list[str],
    arithmetic_keys: list[str],
) -> dict[str, torch.Tensor]:
    """Apply geometric and arithmetic mixing rules to per-atom parameters.

    Geometric mean: ``p_ij = sqrt(p_i * p_j)`` (for energy-like parameters).
    Arithmetic mean: ``p_ij = 0.5 * (p_i + p_j)`` (for distance-like parameters).

    The geometric branch is the **exact**
    :func:`~molpot.potentials.vdw.lj126.geometric_mean`: nothing is added under
    the root. Until 2026-09-08 it computed ``sqrt(p_i p_j + 1e-12)``, which is
    not the rule this docstring states — the bias is ``1.25e-11`` relative at
    ``p = 0.2`` and 41 % at a learnable ``p = 1e-6``, i.e. largest exactly where
    a fitted parameter is smallest (ledger debt ``F10``; the same guard left
    :func:`~molpot.potentials.vdw.lj126.lorentz_berthelot` a day earlier). The
    zero it was protecting is handled by masking instead: ``p_i p_j == 0`` maps
    to ``0`` with a finite backward pass, where the unmasked ``d sqrt(x)/dx``
    diverges. A negative product is a broken parameterisation and still yields
    ``NaN``.

    Args:
        atom_params: Per-atom parameter dict, each value ``(N,)``.
        edge_index: Edge indices ``(E, 2)``.
        geometric_keys: Parameter names to combine via geometric mean.
        arithmetic_keys: Parameter names to combine via arithmetic mean.

    Returns:
        Dict with ``"key_ij"`` entries for each input key ``(E,)``, each in the
        unit of the parameter it was mixed from.
    """
    src, dst = edge_index[:, 0], edge_index[:, 1]
    result: dict[str, torch.Tensor] = {}

    for key in geometric_keys:
        p = atom_params[key]
        result[f"{key}_ij"] = geometric_mean(p[src], p[dst])

    for key in arithmetic_keys:
        p = atom_params[key]
        result[f"{key}_ij"] = 0.5 * (p[src] + p[dst])

    return result
