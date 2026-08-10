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
    shifts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Source→target edge displacements, with optional periodic shifts.

    Args:
        pos: Atom positions ``(N, 3)`` in Å.
        edge_index: Source/target pairs ``(E, 2)``; ``[:, 0]`` = source,
            ``[:, 1]`` = target (the repo-wide edge convention).
        shifts: Optional periodic shift vectors ``(E, 3)`` in Å
            (``unit_shifts @ cell``), added to the raw displacement.

    Returns:
        Edge displacement ``r_ij = pos[target] - pos[source] (+ S_ij)``
        ``(E, 3)`` in Å.
    """
    vectors = pos[edge_index[:, 1]] - pos[edge_index[:, 0]]
    if shifts is not None:
        vectors = vectors + shifts
    return vectors


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
