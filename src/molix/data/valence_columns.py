"""Kernel-local stack helpers: valence column TensorDicts → COO index tables.

The post-collate batch schema is **column form** under nested namespaces
(``batch["angles"]["atomi"]`` etc.). Some Class-I potential kernels still take
packed COO ``[arity, N]`` indices. These helpers stack columns only at the
call site — they are **not** part of the collate / cache schema.

See ``.claude/notes/learnable-classical-ff.md`` and spec
``learnable-classical-ff-02-valence-topology``.

Impropers follow molrs center-first: ``atomi`` is the center.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch


def _stack_columns(
    block: Mapping[str, torch.Tensor],
    cols: tuple[str, ...],
) -> torch.Tensor:
    """Stack 1-D long columns into COO ``[len(cols), N]``.

    Args:
        block: Mapping / TensorDict with the named 1-D columns.
        cols: Column names in row order.

    Returns:
        Long tensor of shape ``[len(cols), N]``.

    Raises:
        KeyError: a required column is missing.
        ValueError: column lengths differ.
    """
    tensors = [torch.as_tensor(block[c]).long().reshape(-1) for c in cols]
    n = int(tensors[0].shape[0])
    for name, t in zip(cols, tensors):
        if int(t.shape[0]) != n:
            raise ValueError(
                f"valence columns must share length; {cols[0]}={n}, {name}={t.shape[0]}"
            )
    return torch.stack(tensors, dim=0)


def stack_angle_index(angles_td: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Stack ``atomi`` / ``atomj`` / ``atomk`` into COO ``[3, N]``.

    Args:
        angles_td: Angles namespace (``atomi``, ``atomj`` central, ``atomk``).

    Returns:
        Long tensor ``[3, N_angles]`` for kernel call sites.
    """
    return _stack_columns(angles_td, ("atomi", "atomj", "atomk"))


def stack_proper_index(propers_td: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Stack ``atomi``..``atoml`` into COO ``[4, N]`` for proper torsions.

    Args:
        propers_td: Propers namespace with four atom-index columns.

    Returns:
        Long tensor ``[4, N_propers]``.
    """
    return _stack_columns(propers_td, ("atomi", "atomj", "atomk", "atoml"))


def stack_improper_index(impropers_td: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Stack improper columns into COO ``[4, N]`` (molrs center-first).

    Row 0 is ``atomi`` = **center**. OpenFF trefoil reorder is an export
    adapter, not a second internal convention.

    Args:
        impropers_td: Impropers namespace with four atom-index columns.

    Returns:
        Long tensor ``[4, N_impropers]`` with center at row 0.
    """
    return _stack_columns(impropers_td, ("atomi", "atomj", "atomk", "atoml"))
