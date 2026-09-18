"""Graph-aware collation producing nested TensorDict batches.

Collates a list of molpy ``Frame`` objects into a nested ``TensorDict`` with
per-level batch sizes: atoms (N), edges (E), graphs (B). A frame's ``box``
becomes ``graphs.cell``, compute float columns keep their precision unless a
``dtype`` is asked for, and its ``x``/``y``/``z``, ``fx``/``fy``/``fz`` and
``vx``/``vy``/``vz`` triples stack into ``atoms.pos`` / ``forces`` / ``vel``.
Protected graph totals at :data:`~molix.schema.FLOAT64_LABEL_KEYS` stay
IEEE-754 binary64; units are the Frame's.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase

from molix.frame import TensorFrame, as_tensor_frame
from molix.schema import FLOAT64_LABEL_KEYS

# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoleculeCollator:
    """Picklable callable that packs heterogeneous molecules into one batch.

    The emitted :class:`TensorDict` has globally rebased connectivity,
    ``atoms.batch`` graph membership, a ``graphs`` namespace, and packed
    bond/angle/proper/improper namespaces. Where a column sits in the frame
    decides where it lands; the one thing to configure is the floating dtype
    compute leaves are built in. Graph totals at
    :data:`~molix.schema.FLOAT64_LABEL_KEYS` stay IEEE-754 binary64.

    Args:
        dtype: Floating dtype compute float leaves are built in, or ``None``
            to keep each frame column's own precision (see
            :func:`collate_molecules`).
    """

    dtype: torch.dtype | None = None

    def __call__(self, samples: Sequence[Any]) -> TensorDict:
        return collate_molecules(samples, dtype=self.dtype)


# ---------------------------------------------------------------------------
# Edge normalisation
# ---------------------------------------------------------------------------


# Atom-index offset registry — the lazy declarative subset of PyTorch
# Geometric's ``Data.__inc__`` / ``__cat_dim__`` contract. Each connectivity
# key maps to ``(cat_dim, index_axis)``: ``cat_dim`` is the axis along which
# per-molecule tensors concatenate (the "count" axis), and ``index_axis`` is
# the axis carrying atom indices, over which the running ``atom_offset``
# broadcasts when rebasing local indices into global ones on multi-molecule
# batching. Only ``pair_index`` is produced by a task today; the others are
# declared-but-unproduced, reserved for sub-spec 03 / future bonded terms so
# the differing-axis path is exercised before a real producer exists.
INDEX_KEYS: dict[str, tuple[int, int]] = {
    "pair_index": (0, 1),  # [E, 2] — count axis 0, both columns are atom indices
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
#: Per-atom *vector* leaves and the three frame columns each is stacked from.
#: ``pos`` is required; the others are carried when every frame has all three
#: components. A per-atom vector has no other spelling in a frame, so a new one
#: is a new row here and nothing else.
_VECTORS: dict[str, tuple[str, str, str]] = {
    "pos": ("x", "y", "z"),
    "forces": ("fx", "fy", "fz"),
    "vel": ("vx", "vy", "vz"),
}
#: Frame ``atoms`` columns that become structure, not a per-atom scalar leaf.
_ATOM_STRUCTURAL = frozenset({"number", *(c for cols in _VECTORS.values() for c in cols)})

_ATOM_INDEX_COLS: frozenset[str] = frozenset({"atomi", "atomj", "atomk", "atoml"})


def rebase(tensor: torch.Tensor, offset: int | torch.Tensor, key: str) -> torch.Tensor:
    """Shift a registered atom-index tensor into global coordinates.

    Args:
        tensor: A connectivity tensor whose entries are *local* atom indices
            (e.g. ``pair_index`` ``(E, 2)`` or ``bond_index`` ``(2, N)``).
        offset: Either a scalar (the running ``atom_offset`` int in
            :func:`collate_molecules`), broadcast over the whole tensor, or a
            per-count-element ``(count,)`` tensor (the gathered segment offsets
            in the collate payload), broadcast over
            ``key``'s ``index_axis``.
        key: A key in :data:`INDEX_KEYS` selecting the ``index_axis``.

    Returns:
        ``tensor`` with each atom index rebased by ``offset``.
    """
    if isinstance(offset, torch.Tensor) and offset.ndim >= 1:
        index_axis = INDEX_KEYS[key][1]
        return tensor + offset.unsqueeze(index_axis)
    return tensor + offset


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


def _column(values: Any, dtype: torch.dtype | None) -> torch.Tensor:
    """One frame column as a tensor: integers stay integral, floats keep or take *dtype*.

    Args:
        values: The column, anything ``np.asarray`` accepts.
        dtype: Floating dtype to build a float column in, or ``None`` to keep
            the column's own (a molpy ``Frame`` holds float columns at fp64).

    Returns:
        The column as a 1-D tensor.
    """
    tensor = torch.as_tensor(np.asarray(values))
    if dtype is not None and tensor.is_floating_point():
        return tensor.to(dtype)
    return tensor


def _meta_scalar(value: Any, dtype: torch.dtype | None) -> torch.Tensor:
    """One frame ``meta`` scalar as a 1-element tensor, keeping integers integral.

    Args:
        value: The unwrapped meta value (``bool`` / ``int`` / ``float`` / any
            object with ``__int__`` or ``__float__``).
        dtype: Floating dtype for a float scalar, or ``None`` for ``float64`` —
            a Python float is a double, and a frame at rest is fp64.

    Returns:
        ``(1,)`` tensor — ``torch.long`` for an integral value, otherwise a
        float at *dtype*.
    """
    if isinstance(value, bool):
        return torch.as_tensor([int(value)], dtype=torch.long)
    if isinstance(value, (int, np.integer)):
        return torch.as_tensor([int(value)], dtype=torch.long)
    return torch.as_tensor([float(value)], dtype=dtype or torch.float64)


def collate_molecules(frames: Sequence[Any], *, dtype: torch.dtype | None = None) -> TensorFrame:
    """Collate molpy ``Frame`` objects into the typed batch the model reads.

    A ``Frame`` is the unit of data at rest and a :class:`~molix.data.TensorFrame`
    is the unit the model sees; this is the only conversion between them, and it
    happens **here**, once, on the data path.

    The return is the *typed* view, not a bare ``TensorDict``. A model that
    receives a bare batch has to wrap it to read it, and
    ``TypedTensorDict.from_tensordict`` on a fresh batch object is a dynamo
    graph break (``typedtensordict.py``) — once per step, inside whatever region
    the caller compiled. The conversion belongs to the data path, not to the
    forward: molpy on the definition side, torch and molnex from collate
    onward. Nothing is declared:
    where a column sits in the frame decides where it lands in the batch, so a
    new target is a new frame column and nothing else.

    * ``atoms`` block — ``number`` becomes ``atoms.Z`` and ``x``/``y``/``z``
      become ``atoms.pos``; ``fx``/``fy``/``fz`` stack into ``atoms.forces`` and
      ``vx``/``vy``/``vz`` into ``atoms.vel`` (:data:`_VECTORS`); every
      remaining column is carried through as ``atoms.<name>``.
    * ``meta`` — every scalar becomes a per-graph leaf ``graphs.<name>``.
    * ``box`` — a frame's lattice becomes ``graphs.cell`` ``(B, 3, 3)`` with the
      lattice vectors as **rows** (the convention every reader of ``graphs.cell``
      uses: ``fractional @ cell``); molpy's ``Box.matrix`` holds them as columns
      and is transposed here. Either every frame carries a box or none does.
    * ``bonds`` block — ``atomi``/``atomj`` become the COO ``bonds.bond_index``
      (rebased into batch space), ``type`` becomes ``bonds.bond_types``.
    * any further block whose columns are ``atomi``…``atoml`` (``angles`` /
      ``propers`` / ``impropers``) becomes its own namespace of 1-D columns,
      each atom index rebased.

    ``pairs`` comes back empty: the cutoff graph is a function of positions and
    a cutoff, not frame content, so it is built downstream by
    :class:`~molix.neighbor.NeighborList`.

    **Precision is the frame's.** A molpy ``Frame`` holds its float columns at
    fp64, and by default that is what the batch carries — the collate narrows
    nothing on its own, because a caller that prices a golden at 1e-10 cannot
    cast single precision's rounding of the geometry back out. The floating
    dtype a *pipeline* wants is a policy of the loader that serves it:
    :class:`~molix.data.datamodule.DataModule` passes its ``ftype`` as
    *dtype*, and compute float leaves (columns, stacked vectors, other meta
    floats, the cell) are then built at that dtype directly. Graph totals at
    :data:`~molix.schema.FLOAT64_LABEL_KEYS` stay IEEE-754 binary64 regardless
    of *dtype* — a Python float is a double and the Frame at rest is fp64.
    Units are the Frame's; this seam does not convert. Integer columns are
    never touched either way.

    Args:
        frames: Frames, or sequences of frames, in batch order.
        dtype: Floating dtype to build compute float leaves in, or ``None`` to
            keep each column's own precision. Graph totals at
            :data:`~molix.schema.FLOAT64_LABEL_KEYS` are always binary64.

    Returns:
        Nested ``TensorDict`` with ``atoms``, ``pairs``, ``graphs`` and any
        ``bonds`` / ``angles`` / ``propers`` / ``impropers`` namespace the
        frames carry.

    Raises:
        ValueError: *frames* is empty, or some frames carry a ``box`` and
            others do not — a batch is heterogeneous in size, never in schema.
        KeyError: A frame has no ``atoms`` block.
    """
    flat: list[Any] = []
    for item in frames:
        if hasattr(item, "keys") and "atoms" in item:
            flat.append(item)
        else:
            flat.extend(item)
    if not flat:
        raise ValueError("Cannot collate an empty frame list")

    z_all: list[torch.Tensor] = []
    pos_all: list[torch.Tensor] = []
    batch_all: list[torch.Tensor] = []
    num_atoms: list[int] = []
    atom_extras: dict[str, list[torch.Tensor]] = {}
    graph_leaves: dict[str, list[torch.Tensor]] = {}
    cells: list[torch.Tensor] = []
    boxless: list[int] = []
    bond_index: list[torch.Tensor] = []
    bond_types: list[torch.Tensor] = []
    terms: dict[str, dict[str, list[torch.Tensor]]] = {}

    offset = 0
    for graph_idx, frame in enumerate(flat):
        if "atoms" not in frame:
            raise KeyError("every frame needs an atoms block")
        atoms = frame["atoms"]
        z = torch.as_tensor(np.asarray(atoms["number"]), dtype=torch.long)
        n_atoms = int(z.shape[0])
        columns = set(atoms.keys())
        pos = torch.stack([_column(atoms[axis], dtype) for axis in _VECTORS["pos"]], dim=-1)
        z_all.append(z)
        pos_all.append(pos)
        batch_all.append(torch.full((n_atoms,), graph_idx, dtype=torch.long))
        num_atoms.append(n_atoms)

        for name, components in _VECTORS.items():
            if name != "pos" and all(axis in columns for axis in components):
                atom_extras.setdefault(name, []).append(
                    torch.stack([_column(atoms[axis], dtype) for axis in components], dim=-1)
                )
        for name in sorted(columns):
            if name in _ATOM_STRUCTURAL:
                continue
            array = np.asarray(atoms[name])
            # OMol25 / molpy frames carry ``element`` (and other labels) as
            # unicode. Meta already skips strings; atom columns must too —
            # torch cannot hold numpy.str_.
            if array.dtype.kind in {"U", "S", "O"}:
                continue
            atom_extras.setdefault(name, []).append(_column(array, dtype))

        box = getattr(frame, "box", None)
        if box is None:
            boxless.append(graph_idx)
        else:
            # molpy holds the lattice vectors as the columns of ``Box.matrix``;
            # ``graphs.cell`` holds them as rows.
            matrix = np.asarray(box.matrix, dtype=np.float64).T
            cells.append(_column(matrix, dtype).reshape(3, 3))

        for name, raw in (getattr(frame, "meta", None) or {}).items():
            # molrs >= 0.14 wraps meta entries in MetaValue; .value unwraps.
            value = getattr(raw, "value", raw)
            if isinstance(value, str):
                continue
            # An integer scalar stays integral. A task code is the reason this
            # matters: ``TaskRouter`` refuses a floating label rather than
            # truncate one silently, so casting every meta scalar to float here
            # would break multi-task routing for every frame-fed batch.
            # Graph totals: NestedKey ("graphs", name), not the leaf "energy".
            leaf = _meta_scalar(
                value,
                dtype=None if ("graphs", name) in FLOAT64_LABEL_KEYS else dtype,
            )
            graph_leaves.setdefault(name, []).append(leaf)

        if "bonds" in frame:
            block = frame["bonds"]
            bond_index.append(
                torch.stack(
                    [
                        torch.as_tensor(np.asarray(block[col]), dtype=torch.long)
                        for col in ("atomi", "atomj")
                    ],
                    dim=0,
                )
                + offset
            )
            if "type_id" in block.keys():
                bond_types.append(torch.as_tensor(np.asarray(block["type_id"]), dtype=torch.long))

        for name in sorted(frame.keys()):
            if name in {"atoms", "bonds"}:
                continue
            block = frame[name]
            columns = [col for col in block.keys() if col in _ATOM_INDEX_COLS]
            if not columns:
                continue
            bucket = terms.setdefault(name, {})
            for col in sorted(block.keys()):
                value = torch.as_tensor(np.asarray(block[col]), dtype=torch.long)
                bucket.setdefault(col, []).append(
                    value + offset if col in _ATOM_INDEX_COLS else value
                )

        offset += n_atoms

    atoms_dict: dict[str, torch.Tensor] = {
        "Z": torch.cat(z_all),
        "pos": torch.cat(pos_all),
        "batch": torch.cat(batch_all),
    }
    # A leaf exists only when every frame carries it: a batch is heterogeneous
    # in size, never in schema, so a column present in some frames is dropped
    # rather than padded with a value nobody measured.
    for name, values in atom_extras.items():
        if len(values) == len(flat):
            atoms_dict[name] = torch.cat(values)

    graphs_dict: dict[str, torch.Tensor] = {"num_atoms": torch.tensor(num_atoms, dtype=torch.long)}
    for name in sorted(graph_leaves):
        values = graph_leaves[name]
        if len(values) == len(flat):
            graphs_dict[name] = torch.cat(values)
    if cells and boxless:
        raise ValueError(
            f"{len(cells)} of {len(flat)} frames carry a box and frames {boxless} do not; a batch "
            "is heterogeneous in size, never in schema — give every frame a box or none"
        )
    if cells:
        graphs_dict["cell"] = torch.stack(cells)

    out = TensorDict(
        atoms=TensorDict(atoms_dict, batch_size=[int(atoms_dict["Z"].shape[0])]),
        pairs=TensorDict(
            pair_index=torch.zeros(0, 2, dtype=torch.long),
            pair_diff=torch.zeros(0, 3),
            pair_dist=torch.zeros(0),
            batch_size=[0],
        ),
        graphs=TensorDict(graphs_dict, batch_size=[len(flat)]),
        batch_size=[],
    )
    if bond_index:
        bonds: dict[str, torch.Tensor] = {"bond_index": torch.cat(bond_index, dim=1)}
        if bond_types:
            bonds["bond_types"] = torch.cat(bond_types)
        out["bonds"] = TensorDict(bonds, batch_size=[])
    # A term namespace exists when *any* frame declares it: a molecule with no
    # angles contributes zero angle rows, which is not the same as the batch
    # having no angles at all. (Per-atom leaves cannot be zero-filled that way,
    # so they still require every frame — see above.)
    for name, bucket in terms.items():
        columns = {col: torch.cat(values) for col, values in bucket.items()}
        out[name] = TensorDict(columns, batch_size=[int(next(iter(columns.values())).shape[0])])
    return as_tensor_frame(out)


#: Preferred ``graphs`` key order when compiling a collator schema.
#: Dynamo guards ``dict.get`` on key *order* (GH200 2367461).
GRAPH_COLUMNS: tuple[str, ...] = (
    "num_atoms",
    "energy",
    "total_charge",
    "energy_residual",
    "e0",
    "task",
    "group",
    "cell",
)


def graph_schema_from(keys: object) -> tuple[str, ...]:
    """Canonical ``graphs`` key order for a compiled collator schema.

    The *set* is the first batch's; the *order* is :data:`GRAPH_COLUMNS`
    then sorted leftovers so DataLoader workers with the same set emit
    the same ``dict.keys()`` without sharing insertion order.
    """
    present = set(keys)
    preferred = tuple(name for name in GRAPH_COLUMNS if name in present)
    extras = tuple(sorted(present.difference(GRAPH_COLUMNS)))
    return preferred + extras


def freeze_graph_columns(batch: TensorDictBase, schema: tuple[str, ...]) -> TensorDictBase:
    """Rebuild ``graphs`` to *schema*: missing names raise, extras drop."""
    graphs = batch.get("graphs")
    if graphs is None:
        return batch
    backend = getattr(graphs, "_tensordict", None)
    if not isinstance(backend, dict):
        return batch
    missing = [name for name in schema if name not in backend]
    if missing:
        raise ValueError(
            "graphs schema compiled from the first batch requires "
            f"{list(schema)}; this batch is missing {missing}"
        )
    frozen = {name: backend[name] for name in schema}
    batch["graphs"] = TensorDict(frozen, batch_size=graphs.batch_size, device=graphs.device)
    return batch
