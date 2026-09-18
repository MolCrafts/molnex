"""Molecule-level index split helpers for multi-conformer energy training."""

from __future__ import annotations

import random
from collections.abc import Sequence

__all__ = ["group_split_indices"]


def group_split_indices(
    group_ids: Sequence[str | int],
    *,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> tuple[list[int], list[int], list[int]]:
    """Split sample indices by unique *group_ids* (no molecule leakage).

    Args:
        group_ids: Per-sample molecule / group identity (length N).
        ratios: ``(train, val, test)`` fractions summing to 1.
        seed: RNG seed for shuffling unique groups.

    Returns:
        ``(train_idx, val_idx, test_idx)`` lists of parent indices.
    """
    if not group_ids:
        raise ValueError("group_ids must be non-empty")
    train_r, val_r, test_r = ratios
    if abs(train_r + val_r + test_r - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")

    by_g: dict[str, list[int]] = {}
    order: list[str] = []
    for i, g in enumerate(group_ids):
        key = str(g)
        if key not in by_g:
            by_g[key] = []
            order.append(key)
        by_g[key].append(i)

    groups = list(order)
    rng = random.Random(seed)
    rng.shuffle(groups)
    n = len(groups)
    n_train = int(n * train_r)
    n_val = int(n * val_r)
    train_g = groups[:n_train]
    val_g = groups[n_train : n_train + n_val]
    test_g = groups[n_train + n_val :]

    def expand(gs: list[str]) -> list[int]:
        out: list[int] = []
        for g in gs:
            out.extend(by_g[g])
        return out

    train_idx, val_idx, test_idx = expand(train_g), expand(val_g), expand(test_g)
    # Leakage guard
    train_set = {group_ids[i] for i in train_idx}
    val_set = {group_ids[i] for i in val_idx}
    test_set = {group_ids[i] for i in test_idx}
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError("group_split_indices produced molecule leakage")
    return train_idx, val_idx, test_idx
