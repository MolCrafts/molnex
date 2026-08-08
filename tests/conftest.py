"""Generic symmetry-test helpers for molecular ML modules.

Three physical symmetries every encoder + property head must satisfy:

    * **Translation invariance** — features / energy / scalar moments
      unchanged under a rigid shift of all positions; vector moments
      (μ, F) are also invariant *for neutral systems* because their
      origin-dependent piece (Σ q_i) vanishes.
    * **Rotation invariance / equivariance** — scalar features and
      energy are invariant; vector outputs (forces, dipoles) rotate
      with the same rotation matrix.
    * **Permutation equivariance** — relabelling atoms permutes
      per-atom outputs accordingly; per-graph scalars / vectors
      are invariant.

This module provides three things:

    1. :func:`make_graph_batch` — build a ``TensorDict`` from raw
       tensors, computing ``edge_diff`` / ``edge_dist`` from positions.
    2. :func:`translate_graph` / :func:`rotate_graph` /
       :func:`permute_graph` — the three input transforms.
    3. :func:`recompute_edge_geometry` — call inside a forward pass so
       autograd can trace ``∂E/∂pos`` for force / equivariance tests.

The transforms are intentionally small — concrete tests live next to
the module they exercise (encoders in ``test_molzoo``, heads in
``test_molpot``).
"""

from __future__ import annotations

import torch
from tensordict import TensorDict


def make_graph_batch(
    pos: torch.Tensor,
    Z: torch.Tensor,
    edge_index: torch.Tensor,
    batch: torch.Tensor,
    *,
    graphs: dict[str, torch.Tensor] | None = None,
    shifts: torch.Tensor | None = None,
) -> TensorDict:
    """Build a ``TensorDict`` from raw tensors.

    Args:
        pos: ``(N, 3)`` atomic positions.
        Z: ``(N,)`` atomic numbers.
        edge_index: ``(E, 2)`` source/target index pairs.
        batch: ``(N,)`` graph membership index per atom.
        graphs: Optional per-graph fields (e.g. ``{"total_charge": tensor}``)
            written to the ``"graphs"`` sub-tensordict. ``num_atoms`` is
            auto-derived from ``batch`` and always present.
        shifts: Optional PBC shift vectors ``(E, 3)`` (``unit_shifts @ cell``),
            written to ``edges.shifts`` and folded additively into
            ``edge_diff``. ``S_ij`` is constant w.r.t. ``pos``, so the
            position gradient of the edge geometry is unchanged.

    Returns:
        A fully-formed ``TensorDict`` with
        ``edge_diff = pos[dst] - pos[src] (+ shifts)`` and
        ``edge_dist = ‖edge_diff‖`` recomputed from ``pos``.
    """
    edge_diff = pos[edge_index[:, 1]] - pos[edge_index[:, 0]]
    if shifts is not None:
        edge_diff = edge_diff + shifts
    edge_dist = edge_diff.norm(dim=-1).clamp(min=1e-6)
    n_atoms = pos.shape[0]
    n_edges = edge_index.shape[0]
    n_graphs = int(batch.max().item()) + 1 if n_atoms > 0 else 0

    num_atoms_per_graph = torch.zeros(n_graphs, dtype=torch.long)
    num_atoms_per_graph.scatter_add_(0, batch, torch.ones_like(batch))
    graph_data = TensorDict(num_atoms=num_atoms_per_graph, batch_size=[n_graphs])
    if graphs is not None:
        for k, v in graphs.items():
            graph_data[k] = v

    edge_data = TensorDict(
        edge_index=edge_index,
        edge_diff=edge_diff,
        edge_dist=edge_dist,
        batch_size=[n_edges],
    )
    if shifts is not None:
        edge_data["shifts"] = shifts

    return TensorDict(
        atoms=TensorDict(Z=Z, pos=pos, batch=batch, batch_size=[n_atoms]),
        edges=edge_data,
        graphs=graph_data,
        batch_size=[],
    )


def translate_graph(batch: TensorDict, t: torch.Tensor) -> TensorDict:
    """Shift all atomic positions by ``t``. Edge geometry recomputes from pos.

    ``edges.shifts`` (if present) is carried unchanged: a rigid translation
    does not change the box, so ``S = n · h`` is untouched.
    """
    pos = batch["atoms", "pos"] + t
    extras = _extract_graph_extras(batch)
    return make_graph_batch(
        pos=pos,
        Z=batch["atoms", "Z"],
        edge_index=batch["edges", "edge_index"],
        batch=batch["atoms", "batch"],
        graphs=extras,
        shifts=_extract_shifts(batch),
    )


def rotate_graph(batch: TensorDict, R: torch.Tensor) -> TensorDict:
    """Rotate all atomic positions by ``R`` (3×3 rotation matrix).

    ``edges.shifts`` (if present) rotates with the positions: ``S = n · h``
    is a lattice vector, so rotating the box rotates ``S`` by the same ``R``
    while the integer image ``n`` stays put.
    """
    pos = batch["atoms", "pos"] @ R.T
    extras = _extract_graph_extras(batch)
    shifts = _extract_shifts(batch)
    return make_graph_batch(
        pos=pos,
        Z=batch["atoms", "Z"],
        edge_index=batch["edges", "edge_index"],
        batch=batch["atoms", "batch"],
        graphs=extras,
        shifts=None if shifts is None else shifts @ R.T,
    )


def permute_graph(batch: TensorDict, perm: torch.Tensor) -> TensorDict:
    """Relabel atoms by ``perm``; edge_index is remapped, per-graph fields kept.

    ``edges.shifts`` is per-edge and the edge *rows* keep their order (only the
    node labels inside ``edge_index`` are rewritten), so shifts stay aligned
    and are carried unchanged.
    """
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(len(perm))
    pos = batch["atoms", "pos"][perm]
    Z = batch["atoms", "Z"][perm]
    batch_idx = batch["atoms", "batch"][perm]
    edge_index = inv_perm[batch["edges", "edge_index"]]
    extras = _extract_graph_extras(batch)
    return make_graph_batch(
        pos=pos,
        Z=Z,
        edge_index=edge_index,
        batch=batch_idx,
        graphs=extras,
        shifts=_extract_shifts(batch),
    )


def recompute_edge_geometry(batch: TensorDict) -> TensorDict:
    """Re-derive ``edge_diff`` / ``edge_dist`` from ``pos`` in-place.

    Call this at the start of a pipeline forward when ``pos`` carries
    ``requires_grad`` so autograd can trace ``∂E/∂pos`` for force tests.
    ``edges.shifts`` (if present) is folded back in, exactly as
    :func:`make_graph_batch` does.
    """
    pos = batch["atoms", "pos"]
    edge_index = batch["edges", "edge_index"]
    edge_diff = pos[edge_index[:, 1]] - pos[edge_index[:, 0]]
    shifts = _extract_shifts(batch)
    if shifts is not None:
        edge_diff = edge_diff + shifts
    edge_dist = edge_diff.norm(dim=-1).clamp(min=1e-6)
    batch["edges", "edge_diff"] = edge_diff
    batch["edges", "edge_dist"] = edge_dist
    return batch


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


_RESERVED_GRAPH_KEYS = {"num_atoms"}


def _extract_graph_extras(batch: TensorDict) -> dict[str, torch.Tensor] | None:
    """Lift user-set per-graph fields off ``batch`` so make_graph_batch can
    re-attach them after a transform. ``num_atoms`` is auto-derived and
    therefore skipped."""
    if "graphs" not in batch.keys():
        return None
    graphs = batch["graphs"]
    extras = {k: graphs[k] for k in graphs.keys() if k not in _RESERVED_GRAPH_KEYS}
    return extras or None


def _extract_shifts(batch: TensorDict) -> torch.Tensor | None:
    """Return ``edges.shifts`` if the batch carries PBC shift vectors."""
    if ("edges", "shifts") not in batch.keys(include_nested=True):
        return None
    return batch["edges", "shifts"]
