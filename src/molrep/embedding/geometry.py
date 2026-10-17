"""Edge displacement and length, differentiable with respect to ``pos``.

The one owner of the edge geometry every molzoo model reads (MACE, PiNet,
Allegro). An edge vector is ``r_ij = pos[target] - pos[source]`` — the
repo-wide convention, ``edge_index[:, 0]`` = source. A neighbour list that
already imaged the displacement (``pairs.pair_diff``) supplies the *value*;
the gradient always comes from the raw positions, which is exact because an
image translation is constant in ``pos``.

Composition of the two functions is the caller's job; there is no one-step
façade.
"""

from __future__ import annotations

import torch


def edge_vectors(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    edge_diff: torch.Tensor | None = None,
) -> torch.Tensor:
    """Source→target edge displacements from live ``pos``.

    Three cases:

    * ``edge_diff`` absent — open system: ``pos[t] - pos[s]``.
    * ``edge_diff`` frozen (a neighbour-list buffer) — its imaged value with
      the gradient of the raw displacement (straight-through term).
    * ``edge_diff`` live (``requires_grad``: rebuilt from the position leaf by
      :class:`molix.autograd.Derivative`, strained under a virial) — returned
      as is; a straight-through term would add a second, unimaged ``∂/∂ε``.

    Args:
        pos: Atom positions ``(N, 3)`` in Å.
        edge_index: Source/target pairs ``(E, 2)``; ``[:, 0]`` = source,
            ``[:, 1]`` = target.
        edge_diff: Optional already-imaged displacements ``(E, 3)`` in Å.

    Returns:
        Edge displacement ``(E, 3)`` in Å.
    """
    raw = pos[edge_index[:, 1]] - pos[edge_index[:, 0]]
    if edge_diff is None:
        return raw
    if edge_diff.requires_grad:
        return edge_diff
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
