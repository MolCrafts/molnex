"""Edge geometry helpers used by the PiNet encoder and property heads.

Kept separate from the encoder so force-path STE logic has a single owner and
can be unit-tested without constructing a full model.
"""

from __future__ import annotations

import torch


def compute_d5(d3: torch.Tensor) -> torch.Tensor:
    """Five-component symmetric-traceless rank-5 direction basis."""
    x, y, z = d3[:, 0], d3[:, 1], d3[:, 2]
    x2, y2, z2 = x.square(), y.square(), z.square()
    return torch.stack(
        [
            (2.0 / 3.0) * x2 - (1.0 / 3.0) * y2 - (1.0 / 3.0) * z2,
            (2.0 / 3.0) * y2 - (1.0 / 3.0) * x2 - (1.0 / 3.0) * z2,
            x * y,
            x * z,
            y * z,
        ],
        dim=1,
    )


def edge_bond_diff(edges, pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Source→target edge displacement, PBC-correct and differentiable.

    Open systems: recompute ``pos[target] - pos[source]`` so forces flow to ``pos``.
    Periodic systems: use the neighbour list's minimum-image ``edge_diff`` value
    but route the gradient through the raw displacement via a straight-through
    term — exact, because ``∂(imaged diff)/∂pos = ∂raw/∂pos``.

    Args:
        edges: The batch's ``edges`` sub-TensorDict (may carry ``edge_diff``).
        pos: Atom positions ``(N, 3)``.
        edge_index: Source/target pairs ``(E, 2)``; ``[:,0]``=source, ``[:,1]``=target.

    Returns:
        Edge displacement ``(E, 3)`` with periodic-correct value and exact gradient.
    """
    raw = pos[edge_index[:, 1]] - pos[edge_index[:, 0]]
    if "edge_diff" in edges.keys():
        # Detach the supplied value so gradient always flows solely through
        # ``raw`` (never double-counts a live supplied tensor).
        return edges["edge_diff"].detach() + (raw - raw.detach())
    return raw
