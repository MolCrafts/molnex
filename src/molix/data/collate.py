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
    "angle_index": (1, 0),  # [3, N]  (kernel-local COO only; not collate schema)
    "dihedral_index": (1, 0),  # [4, N] (kernel-local COO only; not collate schema)
}

# Valence connectivity lives as 1-D atom-index columns under nested namespaces
# (molpy/molrs Frame style). Required columns per family; optional ``type``.
# Impropers: atomi is the **center** (molrs center-first).
_VALENCE_REQUIRED: dict[str, tuple[str, ...]] = {
    "angles": ("atomi", "atomj", "atomk"),
    "propers": ("atomi", "atomj", "atomk", "atoml"),
    "impropers": ("atomi", "atomj", "atomk", "atoml"),
}
_VALENCE_OPTIONAL: tuple[str, ...] = ("type",)
_ATOM_INDEX_COLS: frozenset[str] = frozenset({"atomi", "atomj", "atomk", "atoml"})


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


def _sample_has_valence(sample: Mapping[str, Any], family: str) -> bool:
    """Return True if *sample* carries a non-None nested *family* block."""
    block = sample.get(family)
    return block is not None and isinstance(block, Mapping)


def _collate_valence_family(
    samples: list[dict],
    family: str,
    required: tuple[str, ...],
    atom_offsets: list[int],
) -> TensorDict | None:
    """Collate one valence namespace as 1-D columns rebased by atom offset.

    Pre-collate form (preferred)::

        sample["angles"] = {
            "atomi": Long[N], "atomj": Long[N], "atomk": Long[N], "type"?: Long[N]
        }

    All-or-none: every sample must carry the family or none may. Atom-index
    columns (``atomi``/``atomj``/``atomk``/``atoml``) are shifted by the
    sample's atom offset; optional ``type`` is concatenated without offset.

    Returns:
        Nested :class:`~tensordict.TensorDict` with ``batch_size=[N_terms]``,
        or ``None`` when no sample carries the family.

    Raises:
        ValueError: mixed presence, missing required columns, or length mismatch.
        TypeError: family value is not a mapping of tensors.
    """
    present = [_sample_has_valence(s, family) for s in samples]
    if not any(present):
        return None
    if not all(present):
        raise ValueError(
            f"{family!r} must be present in all samples or none "
            f"(got presence={present})."
        )

    cols: dict[str, list[torch.Tensor]] = {c: [] for c in required}
    optional_lists: dict[str, list[torch.Tensor]] = {c: [] for c in _VALENCE_OPTIONAL}
    have_optional = {c: True for c in _VALENCE_OPTIONAL}

    for sample, offset in zip(samples, atom_offsets):
        block = sample[family]
        if not isinstance(block, Mapping):
            raise TypeError(
                f"sample[{family!r}] must be a mapping of column tensors, "
                f"got {type(block).__name__}"
            )
        for col in required:
            if col not in block or block[col] is None:
                raise ValueError(
                    f"sample[{family!r}] missing required column {col!r}"
                )
            t = torch.as_tensor(block[col]).long().reshape(-1)
            if col in _ATOM_INDEX_COLS:
                t = t + offset
            cols[col].append(t)
        n_terms = int(cols[required[0]][-1].shape[0])
        for col in required[1:]:
            if int(cols[col][-1].shape[0]) != n_terms:
                raise ValueError(
                    f"sample[{family!r}] column lengths differ: "
                    f"{col}={cols[col][-1].shape[0]} vs {required[0]}={n_terms}"
                )
        for col in _VALENCE_OPTIONAL:
            val = block.get(col)
            if val is None:
                have_optional[col] = False
            else:
                ot = torch.as_tensor(val).long().reshape(-1)
                if int(ot.shape[0]) != n_terms:
                    raise ValueError(
                        f"sample[{family!r}] optional {col!r} length "
                        f"{ot.shape[0]} != n_terms={n_terms}"
                    )
                optional_lists[col].append(ot)

    out_dict: dict[str, torch.Tensor] = {
        col: torch.cat(ts, dim=0) for col, ts in cols.items()
    }
    for col, ts in optional_lists.items():
        if have_optional[col] and ts:
            out_dict[col] = torch.cat(ts, dim=0)
    n_total = int(out_dict[required[0]].shape[0])
    return TensorDict(out_dict, batch_size=[n_total])


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


def collate_molecules(
    samples: list[dict],
    target_schema: TargetSchema = DEFAULT_TARGET_SCHEMA,
) -> TensorDict:
    """Collate molecule samples into a nested TensorDict.

    Each sample is a plain dict with at least ``Z`` and ``pos`` keys.
    Optional: ``edge_index``, ``edge_diff``, ``edge_dist``, ``targets``,
    flat ``bond_index`` / ``bond_types``, and nested valence blocks
    ``angles`` / ``propers`` / ``impropers`` with 1-D atom-index columns
    (``atomi`` / ``atomj`` / …, optional ``type``).

    Pre-collate valence form (preferred)::

        {
            "Z": ..., "pos": ...,
            "angles": {"atomi": Long[N_a], "atomj": Long[N_a],
                       "atomk": Long[N_a], "type"?: Long[N_a]},
            "propers": {"atomi": ..., "atomj": ..., "atomk": ..., "atoml": ...},
            "impropers": {"atomi": ..., ...},  # atomi = center (molrs)
        }

    Post-collate the same namespaces are nested TensorDicts with columns
    rebased by cumulative atom offset. Packed COO ``angle_index [3, N]`` is
    **not** the collate schema (see :mod:`molix.datasets._valence_columns`
    for optional kernel-local stacks).

    Args:
        samples: List of single-molecule sample dicts.
        target_schema: Declares which targets are graph-level vs atom-level.

    Returns:
        Nested ``TensorDict`` with ``atoms``, ``edges``, ``graphs`` namespaces
        and optional ``bonds`` / ``angles`` / ``propers`` / ``impropers``.
    """
    if not samples:
        raise ValueError("Cannot collate an empty sample list")

    z_all: list[torch.Tensor] = []
    pos_all: list[torch.Tensor] = []
    batch_all: list[torch.Tensor] = []
    num_atoms: list[int] = []
    atom_offsets: list[int] = []

    edge_all: list[torch.Tensor] = []
    diff_all: list[torch.Tensor] = []
    dist_all: list[torch.Tensor] = []

    bond_all: list[torch.Tensor] = []
    btype_all: list[torch.Tensor] = []

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
        atom_offsets.append(atom_offset)

        if "edge_index" in sample and sample["edge_index"] is not None:
            edge_index = _normalize_edge_index(sample["edge_index"])
            edge_all.append(rebase(edge_index, atom_offset, "edge_index"))

            if "edge_diff" in sample and sample["edge_diff"] is not None:
                diff_all.append(sample["edge_diff"])
            if "edge_dist" in sample and sample["edge_dist"] is not None:
                dist_all.append(sample["edge_dist"])

        # Covalent bonds: bond_index is COO [2, n_bonds] — offset both rows by
        # atom_offset (registry "bond_index", index_axis 0) and concat on dim 1.
        if "bond_index" in sample and sample["bond_index"] is not None:
            bond_all.append(rebase(sample["bond_index"].long(), atom_offset, "bond_index"))
            if "bond_types" in sample and sample["bond_types"] is not None:
                btype_all.append(sample["bond_types"])

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
            edges_dict["edge_diff"] = torch.cat(diff_all, dim=0)
        if dist_all:
            edges_dict["edge_dist"] = torch.cat(dist_all, dim=0)
        e_total = edges_dict["edge_index"].shape[0]
        edges = TensorDict(edges_dict, batch_size=[e_total])
    else:
        # Empty edge data
        edges = TensorDict(
            edge_index=torch.zeros(0, 2, dtype=torch.long),
            edge_diff=torch.zeros(0, 3),
            edge_dist=torch.zeros(0),
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
    out = TensorDict(
        atoms=atoms,
        edges=edges,
        graphs=graphs,
        batch_size=[],
    )

    # --- Optional covalent-bond namespace (only when samples carry bonds) ---
    # batch_size=[] because bond_index is COO [2, N_bonds] (leading dim 2, not
    # N_bonds) and so cannot share a batch axis with bond_types [N_bonds] — the
    # same deliberate [2, N] vs [E, 2] layout split that guards edge != bond.
    if bond_all:
        bonds_dict: dict[str, torch.Tensor] = {"bond_index": torch.cat(bond_all, dim=1)}
        if btype_all:
            bonds_dict["bond_types"] = torch.cat(btype_all, dim=0)
        out["bonds"] = TensorDict(bonds_dict, batch_size=[])

    # --- Optional valence column namespaces (angles / propers / impropers) ---
    # Primary schema is 1-D columns under nested TensorDict, not packed COO.
    # Prefer batch_size=[N_terms] when every leaf shares length N_terms.
    for family, required in _VALENCE_REQUIRED.items():
        td = _collate_valence_family(samples, family, required, atom_offsets)
        if td is not None:
            out[family] = td

    return out


# ---------------------------------------------------------------------------
# Packed-aware fast path
# ---------------------------------------------------------------------------

_TARGET_PREFIX = "targets."


def _gather_indices(
    ptr: torch.Tensor, idx: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Row gather-index, counts, segment ids and offsets for a packed bucket.

    Given a cumsum pointer ``ptr`` ``(n_samples + 1,)`` and selected sample
    indices ``idx`` ``(B,)``, returns ``(gather, counts, seg, offsets)``:

    * ``counts`` ``(B,)`` — each selected sample's element count.
    * ``gather`` ``(sum(counts),)`` — indexes the packed concat tensor in
      sample-major order, so ``packed[gather]`` equals concatenating the
      per-sample slices.
    * ``seg`` ``(sum(counts),)`` — owning sample position per gathered row
      (i.e. the batch vector for the atom bucket).
    * ``offsets`` ``(B,)`` — exclusive-cumsum start of each sample in the
      gathered output, used to rebase local atom indices.

    ``seg`` and ``offsets`` fall out of building ``gather`` and every caller
    needs at least one of them, so they are returned rather than recomputed
    (a second ``repeat_interleave`` per bucket, per batch, per worker).
    """
    counts = ptr[idx + 1] - ptr[idx]
    starts = ptr[idx]
    total = int(counts.sum().item())
    seg = torch.repeat_interleave(torch.arange(idx.numel()), counts)
    offsets = torch.cumsum(counts, 0) - counts
    gather = starts[seg] + (torch.arange(total) - offsets[seg])
    return gather, counts, seg, offsets


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
    a_gather, counts, seg, new_atom_offsets = _gather_indices(atom_ptr, idx)

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
        e_gather, _e_counts, e_seg, _ = _gather_indices(edge_ptr, idx)
        edge_index = edges_bucket["edge_index"][e_gather].long()
        edge_index = rebase(edge_index, new_atom_offsets[e_seg], "edge_index")
        edges_dict: dict[str, torch.Tensor] = {"edge_index": edge_index}
        if "edge_diff" in edges_bucket:
            edges_dict["edge_diff"] = edges_bucket["edge_diff"][e_gather]
        if "edge_dist" in edges_bucket:
            edges_dict["edge_dist"] = edges_bucket["edge_dist"][e_gather]
        for key in edges_bucket:
            if key.startswith(_TARGET_PREFIX):
                _route_target(key, edges_bucket[key][e_gather])
        e_total = int(edge_index.shape[0])
        edges = TensorDict(edges_dict, batch_size=[e_total])
    else:
        edges = TensorDict(
            edge_index=torch.zeros(0, 2, dtype=torch.long),
            edge_diff=torch.zeros(0, 3),
            edge_dist=torch.zeros(0),
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

    out = TensorDict(atoms=atoms, edges=edges, graphs=graphs, batch_size=[])

    # --- covalent-bond level (mirror of collate_molecules' bonds namespace) ---
    bonds_bucket: Mapping[str, torch.Tensor] = payload.get("bonds", {})
    if "bond_index" in bonds_bucket and payload.get("bond_ptr") is not None:
        b_gather, _b_counts, b_seg, _ = _gather_indices(payload["bond_ptr"], idx)
        # bond_index is COO [2, N]: gather columns, offset both rows by the
        # owning sample's atom base (registry "bond_index", index_axis 0).
        bond_index = rebase(
            bonds_bucket["bond_index"][:, b_gather].long(), new_atom_offsets[b_seg], "bond_index"
        )
        bonds_dict: dict[str, torch.Tensor] = {"bond_index": bond_index}
        if "bond_types" in bonds_bucket:
            bonds_dict["bond_types"] = bonds_bucket["bond_types"][b_gather]
        out["bonds"] = TensorDict(bonds_dict, batch_size=[])

    # --- valence column namespaces (mirror of collate_molecules) ---
    # Packed payload stores concatenated 1-D columns + angle_ptr / proper_ptr /
    # improper_ptr. Gather rows, rebase atom-index columns by segment atom base.
    _FAMILY_PTR = {
        "angles": "angle_ptr",
        "propers": "proper_ptr",
        "impropers": "improper_ptr",
    }
    for family, required in _VALENCE_REQUIRED.items():
        ptr_key = _FAMILY_PTR[family]
        bucket: Mapping[str, torch.Tensor] = payload.get(family, {})
        ptr = payload.get(ptr_key)
        if not bucket or ptr is None:
            continue
        v_gather, _v_counts, v_seg, _ = _gather_indices(ptr, idx)
        seg_offsets = new_atom_offsets[v_seg]
        fam_dict: dict[str, torch.Tensor] = {}
        for col, tensor in bucket.items():
            gathered = tensor[v_gather].long()
            if col in _ATOM_INDEX_COLS:
                gathered = gathered + seg_offsets
            fam_dict[col] = gathered
        # Ensure required columns are present (defensive for corrupt caches).
        for col in required:
            if col not in fam_dict:
                raise ValueError(
                    f"packed cache {family!r} bucket missing required column {col!r}"
                )
        n_terms = int(fam_dict[required[0]].shape[0])
        out[family] = TensorDict(fam_dict, batch_size=[n_terms])

    return out
