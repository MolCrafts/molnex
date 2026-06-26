"""Graph-aware collation producing nested TensorDict batches.

Collates a list of single-molecule sample dicts into a nested
``TensorDict`` with per-level batch sizes: atoms (N), edges (E), graphs (B).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from molix.data.dataset import PackedView

# ---------------------------------------------------------------------------
# Target schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetSchema:
    """Declares how targets are collated.

    ``graph_level`` targets (e.g. energy) are per-molecule scalars → ``(B,)``.
    ``atom_level`` targets (e.g. forces) are per-atom tensors → ``(N_total, ...)``.

    Generic defaults are intentionally minimal: data-source classes that
    ship with molix expose schemas as class attributes
    (e.g. :attr:`molix.datasets.QM9Source.TARGET_SCHEMA`) that workflows
    pass explicitly to :class:`DataModule`. :class:`DataModule` also
    checks ``getattr(dataset, "target_schema", ...)`` as a fallback for
    any subclass that declares its own.
    """

    graph_level: frozenset[str] = field(default_factory=lambda: frozenset({"energy"}))
    atom_level: frozenset[str] = field(default_factory=lambda: frozenset({"forces"}))


DEFAULT_TARGET_SCHEMA = TargetSchema()


# ---------------------------------------------------------------------------
# Edge normalisation
# ---------------------------------------------------------------------------


# Atom-index offset registry — the lazy declarative subset of PyTorch
# Geometric's ``Data.__inc__`` / ``__cat_dim__`` contract. Each connectivity
# key maps to ``(cat_dim, index_axis)``: ``cat_dim`` is the axis along which
# per-molecule tensors concatenate (the "count" axis), and ``index_axis`` is
# the axis carrying atom indices, over which the running ``atom_offset``
# broadcasts when rebasing local indices into global ones on multi-molecule
# batching. Only ``edge_index`` is produced by a task today; the others are
# declared-but-unproduced, reserved for sub-spec 03 / future bonded terms so
# the differing-axis path is exercised before a real producer exists.
INDEX_KEYS: dict[str, tuple[int, int]] = {
    "edge_index": (0, 1),  # [E, 2] — count axis 0, both columns are atom indices
    "bond_index": (1, 0),  # [2, N] COO — count axis 1, both rows are atom indices
    "angle_index": (1, 0),  # [3, N]
    "dihedral_index": (1, 0),  # [4, N]
}


def rebase(tensor: torch.Tensor, offset: int | torch.Tensor, key: str) -> torch.Tensor:
    """Shift a registered atom-index tensor into global coordinates.

    Args:
        tensor: A connectivity tensor whose entries are *local* atom indices
            (e.g. ``edge_index`` ``(E, 2)`` or ``bond_index`` ``(2, N)``).
        offset: Either a scalar (the running ``atom_offset`` int in
            :func:`collate_molecules`), broadcast over the whole tensor, or a
            per-count-element ``(count,)`` tensor (the gathered segment offsets
            in :func:`collate_packed`), broadcast over ``key``'s ``index_axis``.
        key: A key in :data:`INDEX_KEYS` selecting the ``index_axis``.

    Returns:
        ``tensor`` with each atom index rebased by ``offset``.
    """
    if isinstance(offset, torch.Tensor) and offset.ndim >= 1:
        index_axis = INDEX_KEYS[key][1]
        return tensor + offset.unsqueeze(index_axis)
    return tensor + offset


def _normalize_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    """Normalize edge_index to canonical ``(E, 2)`` format."""
    if edge_index.ndim != 2:
        raise ValueError(f"edge_index must be 2D, got shape {tuple(edge_index.shape)}")
    if edge_index.shape[1] == 2:
        return edge_index.long()
    if edge_index.shape[0] == 2:
        return edge_index.t().contiguous().long()
    raise ValueError(f"edge_index must have shape (E, 2) or (2, E), got {tuple(edge_index.shape)}")


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


def collate_molecules(
    samples: list[dict],
    target_schema: TargetSchema = DEFAULT_TARGET_SCHEMA,
) -> TensorDict:
    """Collate molecule samples into a nested TensorDict.

    Each sample is a plain dict with at least ``Z`` and ``pos`` keys.
    Optional: ``edge_index``, ``bond_diff``, ``bond_dist``, ``targets``.

    Args:
        samples: List of single-molecule sample dicts.
        target_schema: Declares which targets are graph-level vs atom-level.

    Returns:
        Nested ``TensorDict`` with ``atoms``, ``edges``, ``graphs`` namespaces.
    """
    if not samples:
        raise ValueError("Cannot collate an empty sample list")

    z_all: list[torch.Tensor] = []
    pos_all: list[torch.Tensor] = []
    batch_all: list[torch.Tensor] = []
    num_atoms: list[int] = []

    edge_all: list[torch.Tensor] = []
    diff_all: list[torch.Tensor] = []
    dist_all: list[torch.Tensor] = []

    graph_targets: dict[str, list[torch.Tensor]] = {}
    atom_targets: dict[str, list[torch.Tensor]] = {}

    atom_offset = 0

    for graph_idx, sample in enumerate(samples):
        if "Z" not in sample or "pos" not in sample:
            raise KeyError("Each sample must contain 'Z' and 'pos'")

        z = sample["Z"].long()
        pos = sample["pos"]
        n_atoms = int(z.shape[0])

        z_all.append(z)
        pos_all.append(pos)
        batch_all.append(torch.full((n_atoms,), graph_idx, dtype=torch.long, device=z.device))
        num_atoms.append(n_atoms)

        if "edge_index" in sample and sample["edge_index"] is not None:
            edge_index = _normalize_edge_index(sample["edge_index"])
            edge_all.append(rebase(edge_index, atom_offset, "edge_index"))

            if "bond_diff" in sample and sample["bond_diff"] is not None:
                diff_all.append(sample["bond_diff"])
            if "bond_dist" in sample and sample["bond_dist"] is not None:
                dist_all.append(sample["bond_dist"])

        for name, value in sample.get("targets", {}).items():
            value = value if isinstance(value, torch.Tensor) else torch.tensor(value)
            if name in target_schema.atom_level:
                atom_targets.setdefault(name, []).append(value)
            else:
                graph_targets.setdefault(name, []).append(value.reshape(-1))

        atom_offset += n_atoms

    # --- Build atom-level TensorDict ---
    atoms_dict: dict[str, torch.Tensor] = {
        "Z": torch.cat(z_all, dim=0),
        "pos": torch.cat(pos_all, dim=0),
        "batch": torch.cat(batch_all, dim=0),
    }
    for name, vals in atom_targets.items():
        atoms_dict[name] = torch.cat(vals, dim=0)

    n_total = atoms_dict["Z"].shape[0]
    atoms = TensorDict(atoms_dict, batch_size=[n_total])

    # --- Build edge-level TensorDict ---
    if edge_all:
        edges_dict: dict[str, torch.Tensor] = {
            "edge_index": torch.cat(edge_all, dim=0),
        }
        if diff_all:
            edges_dict["bond_diff"] = torch.cat(diff_all, dim=0)
        if dist_all:
            edges_dict["bond_dist"] = torch.cat(dist_all, dim=0)
        e_total = edges_dict["edge_index"].shape[0]
        edges = TensorDict(edges_dict, batch_size=[e_total])
    else:
        # Empty edge data
        edges = TensorDict(
            edge_index=torch.zeros(0, 2, dtype=torch.long),
            bond_diff=torch.zeros(0, 3),
            bond_dist=torch.zeros(0),
            batch_size=[0],
        )

    # --- Build graph-level TensorDict ---
    num_graphs = len(samples)
    graphs_dict: dict[str, torch.Tensor] = {
        "num_atoms": torch.tensor(num_atoms, dtype=torch.long),
    }
    for name, vals in graph_targets.items():
        graphs_dict[name] = torch.cat(vals, dim=0)

    graphs = TensorDict(graphs_dict, batch_size=[num_graphs])

    # --- Assemble top-level TensorDict ---
    return TensorDict(
        atoms=atoms,
        edges=edges,
        graphs=graphs,
        batch_size=[],
    )


# ---------------------------------------------------------------------------
# Packed-aware fast path
# ---------------------------------------------------------------------------

_TARGET_PREFIX = "targets."


def _gather_indices(ptr: torch.Tensor, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Row gather-index and per-sample counts for slicing a packed bucket.

    Given a cumsum pointer ``ptr`` ``(n_samples + 1,)`` and selected sample
    indices ``idx`` ``(B,)``, returns ``(gather, counts)`` where ``counts``
    ``(B,)`` is each selected sample's element count and ``gather``
    ``(sum(counts),)`` indexes the packed concat tensor in sample-major
    order — so ``packed[gather]`` equals concatenating per-sample slices.
    """
    counts = ptr[idx + 1] - ptr[idx]
    starts = ptr[idx]
    total = int(counts.sum().item())
    seg = torch.repeat_interleave(torch.arange(idx.numel()), counts)
    new_offsets = torch.cumsum(counts, 0) - counts
    gather = starts[seg] + (torch.arange(total) - new_offsets[seg])
    return gather, counts


def collate_packed(
    view: "PackedView",
    indices: Sequence[int],
    target_schema: TargetSchema = DEFAULT_TARGET_SCHEMA,
) -> TensorDict:
    """Collate a batch directly from packed cache tensors (no per-sample dicts).

    Functionally identical to running :func:`collate_molecules` over
    ``[dataset[i] for i in indices]``, but builds the nested
    ``atoms`` / ``edges`` / ``graphs`` :class:`~tensordict.TensorDict` by
    slicing the packed concat tensors and vectorizing the ``edge_index``
    rebase — bypassing the unpack→repack round trip. Per-key routing and
    dtypes mirror :func:`collate_molecules` exactly (it is the equivalence
    oracle).

    Args:
        view: A :class:`~molix.data.dataset.PackedView` onto the cache
            payload; ``view.map_indices`` resolves *indices* to packed
            rows.
        indices: Sample indices in the view's local coordinate. Must be
            non-empty.
        target_schema: Routes target keys — ``atom_level`` names become
            per-atom ``(N, ...)`` leaves under ``atoms``; all others
            become per-graph leaves under ``graphs``.

    Returns:
        Nested ``TensorDict`` with ``atoms`` ``(N,)`` / ``edges`` ``(E,)``
        / ``graphs`` ``(B,)`` namespaces — leaf-for-leaf equal to
        :func:`collate_molecules`.

    Raises:
        ValueError: *indices* is empty, or the cache schema lacks ``Z`` or
            ``pos`` per-atom keys.
    """
    if len(indices) == 0:
        raise ValueError("Cannot collate an empty sample list")

    payload = view.payload
    packed = view.map_indices(indices)
    idx = torch.as_tensor(packed, dtype=torch.long)
    n_graphs = idx.numel()

    atoms_bucket: Mapping[str, torch.Tensor] = payload.get("atoms", {})
    edges_bucket: Mapping[str, torch.Tensor] = payload.get("edges", {})
    graphs_bucket: Mapping[str, torch.Tensor] = payload.get("graphs", {})
    scalars_bucket: Mapping[str, list[Any]] = payload.get("scalars", {})

    for required in ("Z", "pos"):
        if required not in atoms_bucket:
            raise ValueError(
                f"packed cache has no per-atom '{required}' key — collate_packed "
                f"requires both 'Z' and 'pos'. Rebuild the cache from samples "
                f"that carry '{required}'."
            )

    atom_ptr = payload["atom_ptr"]
    a_gather, counts = _gather_indices(atom_ptr, idx)
    new_atom_offsets = torch.cumsum(counts, 0) - counts
    seg = torch.repeat_interleave(torch.arange(n_graphs), counts)

    # --- atom level ---
    atoms_dict: dict[str, torch.Tensor] = {
        "Z": atoms_bucket["Z"][a_gather].long(),
        "pos": atoms_bucket["pos"][a_gather],
        "batch": seg.long(),
    }

    # --- graph-level accumulator (filled by target routing below) ---
    graphs_dict: dict[str, torch.Tensor] = {
        "num_atoms": counts.long(),
    }

    def _route_target(key: str, value: torch.Tensor) -> None:
        """Place a target key under atoms (atom_level) or graphs, oracle-style.

        *value* is the per-sample values stacked/concatenated in sample-major
        order. atom_level targets keep the raw concat (oracle cats raw);
        graph targets are flattened, which equals the oracle's per-sample
        ``reshape(-1)`` then ``cat`` for sample-major data.
        """
        name = key[len(_TARGET_PREFIX) :]
        if name in target_schema.atom_level:
            atoms_dict[name] = value
        else:
            graphs_dict[name] = value.reshape(-1)

    # atom-bucket targets
    for key in atoms_bucket:
        if key.startswith(_TARGET_PREFIX):
            _route_target(key, atoms_bucket[key][a_gather])

    # --- edge level ---
    if "edge_index" in edges_bucket:
        edge_ptr = payload["edge_ptr"]
        e_gather, e_counts = _gather_indices(edge_ptr, idx)
        e_seg = torch.repeat_interleave(torch.arange(n_graphs), e_counts)
        edge_index = edges_bucket["edge_index"][e_gather].long()
        edge_index = rebase(edge_index, new_atom_offsets[e_seg], "edge_index")
        edges_dict: dict[str, torch.Tensor] = {"edge_index": edge_index}
        if "bond_diff" in edges_bucket:
            edges_dict["bond_diff"] = edges_bucket["bond_diff"][e_gather]
        if "bond_dist" in edges_bucket:
            edges_dict["bond_dist"] = edges_bucket["bond_dist"][e_gather]
        for key in edges_bucket:
            if key.startswith(_TARGET_PREFIX):
                _route_target(key, edges_bucket[key][e_gather])
        e_total = int(edge_index.shape[0])
        edges = TensorDict(edges_dict, batch_size=[e_total])
    else:
        edges = TensorDict(
            edge_index=torch.zeros(0, 2, dtype=torch.long),
            bond_diff=torch.zeros(0, 3),
            bond_dist=torch.zeros(0),
            batch_size=[0],
        )

    # graph-bucket and scalar-bucket targets
    for key in graphs_bucket:
        if key.startswith(_TARGET_PREFIX):
            _route_target(key, graphs_bucket[key][idx])
    for key in scalars_bucket:
        if key.startswith(_TARGET_PREFIX):
            name = key[len(_TARGET_PREFIX) :]
            vals = [scalars_bucket[key][i] for i in packed]
            graphs_dict[name] = torch.tensor(vals).reshape(-1)

    n_total = int(atoms_dict["Z"].shape[0])
    atoms = TensorDict(atoms_dict, batch_size=[n_total])
    graphs = TensorDict(graphs_dict, batch_size=[n_graphs])

    return TensorDict(atoms=atoms, edges=edges, graphs=graphs, batch_size=[])
