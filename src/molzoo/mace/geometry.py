"""Edge displacement / length for the MACE family.

MACE's edge vector is ``r_ij = pos[target] - pos[source] + S_ij`` with an
*additive* periodic shift ``S_ij = n_ij · h`` (``unit_shifts @ cell``), exactly
as the upstream ``MACE`` models consume a neighbour list. ``S_ij`` is constant
with respect to ``pos``, so ``∂r_ij/∂pos`` is the open-system one and forces
stay exact.

Why MACE owns a geometry module of its own, next to
``molzoo.pinet.geometry``: PiNet's ``edge_bond_diff``
(``src/molzoo/pinet/geometry.py:28-49``) computes
``detach(imaged) + (raw - detach(raw))`` — the *minimum-image* value with a
straight-through gradient. The two share a gradient (``∂raw/∂pos``) but not a
value: minimum image versus an explicit integer translation are different
mathematical objects, and folding them into one helper would smuggle PiNet's
straight-through semantics into MACE's periodic path. Composition of the two
functions below is the caller's job; there is no one-step façade.

Reference:
    Batatia et al. "MACE: Higher Order Equivariant Message Passing Neural
    Networks for Fast and Accurate Force Fields" NeurIPS 2022.
    https://arxiv.org/abs/2206.07697
"""

from __future__ import annotations

import torch


def edge_vectors(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    edge_diff: torch.Tensor | None = None,
    shifts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Source→target edge displacements from live ``pos``.

    Args:
        pos: Atom positions ``(N, 3)`` in Å.
        edge_index: Source/target pairs ``(E, 2)``; ``[:, 0]`` = source,
            ``[:, 1]`` = target (the repo-wide edge convention).
        edge_diff: Optional already-imaged displacements ``(E, 3)`` in Å.
            Used as the *value*; the gradient is ``∂(pos[t]-pos[s])/∂pos``.
        shifts: Rejected. Raises ``TypeError`` — PBC is not applied here.

    Returns:
        Edge displacement ``(E, 3)`` in Å.

    Raises:
        TypeError: If ``shifts`` is passed.
    """
    if shifts is not None:
        raise TypeError("edge_vectors no longer takes shifts; pass edge_diff=")
    raw = pos[edge_index[:, 1]] - pos[edge_index[:, 0]]
    if edge_diff is None:
        return raw
    if edge_diff.requires_grad:
        # Live imaged geometry (strained ``edge_diff`` for the virial). Trust
        # the caller; STE would add a second unimaged ``∂/∂ε`` and double it.
        return edge_diff
    # Frozen neighbour-list buffer: value from the image, ``∂/∂pos`` from raw.
    return edge_diff + (raw - raw.detach())


def edge_lengths(vectors: torch.Tensor, *, keepdim: bool = False) -> torch.Tensor:
    """Euclidean length of each edge displacement.

    Args:
        vectors: Edge displacements ``(E, 3)`` in Å, e.g. from
            :func:`edge_vectors`.
        keepdim: Keep the contracted axis, giving ``(E, 1)`` instead of
            ``(E,)``.

    Returns:
        Edge lengths ``d_ij = ‖r_ij‖`` — ``(E,)``, or ``(E, 1)`` when
        ``keepdim`` — in Å.
    """
    return torch.linalg.norm(vectors, dim=-1, keepdim=keepdim)
