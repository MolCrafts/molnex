"""Graph-aware collation producing nested TensorDict batches.

Collates a list of single-molecule sample dicts into a nested
``TensorDict`` with per-level batch sizes: atoms (N), edges (E), graphs (B).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Target schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoleculeCollator:
    """Picklable callable that packs heterogeneous molecules into one batch.

    The emitted :class:`TensorDict` has globally rebased connectivity,
    ``atoms.batch`` graph membership, a ``graphs`` namespace, and packed
    bond/angle/proper/improper namespaces. Stateless: where a column sits in
    the frame decides where it lands, so there is nothing to configure.
    """

    def __call__(self, samples: Sequence[Any]) -> TensorDict:
        return collate_molecules(samples)


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
#: Frame ``atoms`` columns that become structure, not a per-atom leaf.
_XYZ = ("x", "y", "z")
_FORCES = ("fx", "fy", "fz")
_ATOM_STRUCTURAL = frozenset({"number", *_XYZ, *_FORCES})

_ATOM_INDEX_COLS: frozenset[str] = frozenset({"atomi", "atomj", "atomk", "atoml"})


def rebase(tensor: torch.Tensor, offset: int | torch.Tensor, key: str) -> torch.Tensor:
    """Shift a registered atom-index tensor into global coordinates.

    Args:
        tensor: A connectivity tensor whose entries are *local* atom indices
            (e.g. ``edge_index`` ``(E, 2)`` or ``bond_index`` ``(2, N)``).
        offset: Either a scalar (the running ``atom_offset`` int in
            :func:`collate_molecules`), broadcast over the whole tensor, or a
            per-count-element ``(count,)`` tensor (the gathered segment offsets
            in :func:`~molix.store.frame_store_view.payload_to_batch`), broadcast over
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


def _meta_scalar(value: Any) -> torch.Tensor:
    """One frame ``meta`` scalar as a 1-element tensor, keeping integers integral.

    Args:
        value: The unwrapped meta value (``bool`` / ``int`` / ``float`` / any
            object with ``__int__`` or ``__float__``).

    Returns:
        ``(1,)`` tensor — ``torch.long`` for an integral value, otherwise
        ``torch.float32``.
    """
    if isinstance(value, bool):
        return torch.as_tensor([int(value)], dtype=torch.long)
    if isinstance(value, (int, np.integer)):
        return torch.as_tensor([int(value)], dtype=torch.long)
    return torch.as_tensor([float(value)], dtype=torch.float32)


def collate_molecules(frames: Sequence[Any]) -> TensorDict:
    """Collate molpy ``Frame`` objects into the batch ``TensorDict`` the model reads.

    A ``Frame`` is the unit of data at rest and a ``TensorDict`` is the unit the
    model sees; this is the only conversion between them. Nothing is declared:
    where a column sits in the frame decides where it lands in the batch, so a
    new target is a new frame column and nothing else.

    * ``atoms`` block — ``number`` becomes ``atoms.Z`` and ``x``/``y``/``z``
      become ``atoms.pos``; ``fx``/``fy``/``fz`` stack into ``atoms.forces``;
      every remaining column is carried through as ``atoms.<name>``.
    * ``meta`` — every scalar becomes a per-graph leaf ``graphs.<name>``.
    * ``bonds`` block — ``atomi``/``atomj`` become the COO ``bonds.bond_index``
      (rebased into batch space), ``type`` becomes ``bonds.bond_types``.
    * any further block whose columns are ``atomi``…``atoml`` (``angles`` /
      ``propers`` / ``impropers``) becomes its own namespace of 1-D columns,
      each atom index rebased.

    ``edges`` comes back empty: the cutoff graph is a function of positions and
    a cutoff, not frame content, so it is built downstream by
    :class:`~molix.batch.neighbor.NeighborList`.

    Args:
        frames: Frames, or sequences of frames, in batch order.

    Returns:
        Nested ``TensorDict`` with ``atoms``, ``edges``, ``graphs`` and any
        ``bonds`` / ``angles`` / ``propers`` / ``impropers`` namespace the
        frames carry.

    Raises:
        ValueError: *frames* is empty.
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
        pos = torch.stack(
            [torch.as_tensor(np.asarray(atoms[axis]), dtype=torch.float32) for axis in _XYZ],
            dim=-1,
        )
        z_all.append(z)
        pos_all.append(pos)
        batch_all.append(torch.full((n_atoms,), graph_idx, dtype=torch.long))
        num_atoms.append(n_atoms)

        if all(axis in atoms.keys() for axis in _FORCES):
            atom_extras.setdefault("forces", []).append(
                torch.stack(
                    [
                        torch.as_tensor(np.asarray(atoms[axis]), dtype=torch.float32)
                        for axis in _FORCES
                    ],
                    dim=-1,
                )
            )
        for name in sorted(atoms.keys()):
            if name in _ATOM_STRUCTURAL:
                continue
            atom_extras.setdefault(name, []).append(
                torch.as_tensor(np.asarray(atoms[name]), dtype=torch.float32)
            )

        for name, raw in (getattr(frame, "meta", None) or {}).items():
            # molrs >= 0.14 wraps meta entries in MetaValue; .value unwraps.
            value = getattr(raw, "value", raw)
            if isinstance(value, str):
                continue
            # An integer scalar stays integral. A task code is the reason this
            # matters: ``TaskRouter`` refuses a floating label rather than
            # truncate one silently, so casting every meta scalar to float here
            # would break multi-task routing for every frame-fed batch.
            leaf = _meta_scalar(value)
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
    for name, values in graph_leaves.items():
        if len(values) == len(flat):
            graphs_dict[name] = torch.cat(values)

    out = TensorDict(
        atoms=TensorDict(atoms_dict, batch_size=[int(atoms_dict["Z"].shape[0])]),
        edges=TensorDict(
            edge_index=torch.zeros(0, 2, dtype=torch.long),
            edge_diff=torch.zeros(0, 3),
            edge_dist=torch.zeros(0),
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
    return out


# ---------------------------------------------------------------------------
# Packed-aware fast path
# ---------------------------------------------------------------------------

_TARGET_PREFIX = "targets."
