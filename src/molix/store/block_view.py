"""Read-only dataset view over an ordered sequence of sealed READY blocks.

A training epoch consumes many sealed blocks, but :class:`molix.store.block.BlockReader`
reads exactly one directory and :class:`molix.store.dataset.SubsetDataset` wraps
exactly one packed payload. :class:`BlockDataset` is the one level up: an ordered,
read-only view whose global index is ``(block, local frame)`` **in the given block
order**, presenting the concatenated metadata surface a batch sampler needs and an
order-preserving, cross-block :meth:`BlockDataset.frame_payload`. It moves no
numbers, converts no units and re-exposes each block's columns verbatim (``pos`` in
Å, ``energy`` in kcal mol⁻¹, ``forces`` in kcal mol⁻¹ Å⁻¹, as recorded in every
manifest's ``provenance.units``); its whole domain content is index algebra plus the
molecule-level grouping discipline the block format already encodes.

Index algebra
-------------

For blocks ``b = 0 … B-1`` in caller order, with ``F_b = BlockReader.frame_count``
and ``M_b = BlockManifest.units``::

    frame_ptr[0] = 0,  frame_ptr[b+1] = frame_ptr[b] + F_b        # (B+1,) long
    unit_ptr[0]  = 0,  unit_ptr[b+1]  = unit_ptr[b]  + M_b        # (B+1,) long

and for a global frame ``g ∈ [0, frame_ptr[B])``::

    b     = searchsorted(frame_ptr, g, right=True) - 1     # owning block
    local = g - frame_ptr[b]                               # frame row inside b
    atom_counts[g] = BlockReader(b).atom_counts[local]                    # (F,) long
    group_codes[g] = BlockReader(b).group_codes[local] + unit_ptr[b]      # (F,) long
    task_codes[g]  = global_of[b][ BlockReader(b).task_codes[local] ]     # (F,) long

``M_b`` is read from the manifest (authoritative once sealed); ``F_b`` is not a
manifest field — ``rows`` carries a ``frame`` entry only when a frame-space column
such as ``energy`` is declared — so the frame count is read from
:class:`~molix.store.block.BlockReader` through the bounded pool, which maps the
small count columns only.

Why ``locate`` is vectorized only
---------------------------------

:meth:`BlockDataset.locate` takes a whole frame list and returns two ``(B,)``
tensors. That is the shape the plan's collate route calls it in — one
``searchsorted`` over ``(B+1,)`` pointers per batch instead of ``len(frames)``
Python-level lookups — and a scalar API would invite exactly the per-frame loop the
pointer layout exists to avoid. A single lookup is spelled ``locate([idx])``, which
costs one tensor and reads as what it is: a batch of one.

Why there is no wrap indexing
-----------------------------

A plan speaks in absolute frame rows, so a negative row is a producer bug, never a
request for the tail: ``view[-1]`` would silently read a frame from the *last*
block of the plan while its author meant "frame −1 of this block". Both
:meth:`BlockDataset.locate` and :meth:`BlockDataset.frame_payload` therefore
range-check ``[0, len(view))`` and refuse anything outside it, and
:meth:`BlockDataset.__getitem__` inherits that refusal by routing through
``frame_payload``, rather than deferring to the wrap-around
:func:`_unpack_packed_sample` performs on a packed payload.

Scope of the not-ready guard
----------------------------

A planned block that is missing, unparsable or unsealed raises
:class:`BlockNotReadyError` at construction: it is never skipped and never
substituted by a neighbour, so a hole in the plan stalls the view instead of
silently shortening the epoch. The same signal wraps a mid-epoch failure to reopen a
block. That guard is scoped honestly: POSIX mmap survives ``unlink``, so a block
whose columns are already mapped keeps serving reads after its directory is deleted
— the guard fires on a **cold** block only, i.e. one never opened or already evicted
from the bounded reader pool. Detecting deletion under a live mapping would need a
per-read ``stat``, which is a cost the read path must not pay.

Why ``edge_counts`` refuses
---------------------------

Blocks deliberately store no neighbour-graph columns (see the
:mod:`molix.store.block` module docstring for the ~10x payload argument), so this
view delegates :attr:`BlockDataset.edge_counts` to the block store's own refusal and
fabricates nothing: handing a token-budget sampler a per-frame count of nothing at
all would be a false budget for a cutoff graph that will exist downstream, built by
:class:`molix.batch.NeighborList` or the MD bind path. Samplers over this view are
constructed with ``max_edges=None`` and budget on :attr:`BlockDataset.atom_counts`.

Why group codes are an offset, not a ``(block, code)`` pair
-----------------------------------------------------------

Block ``b``'s per-block group codes are exactly its unit rows ``[0, M_b)`` — one
unit is one group — so the offset images ``code + unit_ptr[b]`` partition
``[0, unit_ptr[B])`` with no overlap and no gap, and the invariant "a group never
straddles a batch" survives concatenation by construction. Keeping the flat integer
also keeps the table the 1-D ``(F,)`` shape the samplers accept: a ``(block, code)``
pair would force either an ``(F, 2)`` table they reject or a table of Python tuples,
which is un-tensorable, slower to build and no longer sortable in an audit.

Layering: this module imports stdlib + torch + :mod:`molix.store` only. It must not
import ``tensordict`` or any collate symbol — batch construction stays the caller's
composition, ``collate_packed(PackedView(view.frame_payload(frames)), ...)``.

References:
    Wang, Fass & Chodera. "End-to-end differentiable construction of molecular
    mechanics force fields." Chem. Sci. 13, 12016 (2022). DOI:10.1039/D2SC02739A
    — molecule-level grouping before configuration sampling; the invariant the
    globally unique group codes preserve across blocks.
    Fey & Lenssen. "Fast Graph Representation Learning with PyTorch Geometric."
    arXiv:1903.02428 — concatenated per-key tensors + slice index, the shape of the
    merged payload.
    Apache Arrow columnar format, variable-size list layout (offsets + values).
    https://arrow.apache.org/docs/format/Columnar.html
"""

from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from functools import cached_property
from pathlib import Path
from typing import Any, Final, NoReturn

import torch

from molix.store.block import (
    _FAMILY_BUCKETS,
    _MANIFEST_NAME,
    _META_ROUTE_PREFIX,
    BlockManifest,
    BlockReader,
    _exclusive_cumsum,
)
from molix.store.dataset import BaseDataset

__all__ = [
    "BlockDataset",
    "BlockNotReadyError",
    "BlockSchemaMismatchError",
    "BlockSplitMismatchError",
]

#: Covalent-bond keys, handled outside the per-atom / per-frame row model.
_BOND_INDEX_KEY: Final[str] = "bond_index"
_BOND_TYPES_KEY: Final[str] = "bond_types"

#: Valence family -> its cumsum pointer key in a packed payload. Impropers keep
#: molpy's center-first ``atomi``.
_VALENCE_PTR_KEY: Final[dict[str, str]] = {
    "angles": "angle_ptr",
    "propers": "proper_ptr",
    "impropers": "improper_ptr",
}

#: Valence buckets :func:`_unpack_packed_sample` can route — derived from the
#: unpacker's own tables (``bond_ptr`` plus :data:`_VALENCE_PTR_KEY`) rather
#: than restated, so wiring a further family downstream relaxes this guard
#: automatically instead of leaving a stale copy behind.
_ROUTABLE_BUCKETS: Final[frozenset[str]] = frozenset({"bonds", *_VALENCE_PTR_KEY})


def _unflatten(flat: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild a nested sample dict from dotted keys."""
    out: dict[str, Any] = {}
    for key, val in flat.items():
        parts = key.split(".")
        cur = out
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = val
    return out


def _unpack_packed_sample(payload: Mapping[str, Any], idx: int) -> dict[str, Any]:
    """Reconstruct the ``idx``-th sample dict from a packed payload.

    Slices into the shared concat tensors — no storage copy when the payload
    came from a memory-mapped read. The payload shape is the one
    :func:`~molix.batch.collate.collate_packed` consumes: ``atoms`` / ``edges``
    / ``graphs`` / ``scalars`` buckets plus the cumsum pointers that locate
    each sample's rows.

    Args:
        payload: Packed payload mapping.
        idx: Sample index; negative indices count from the end.

    Returns:
        The flat (nested-dict) sample.

    Raises:
        IndexError: *idx* is outside ``[-n_samples, n_samples)``.
    """
    n = payload["n_samples"]
    if idx < 0:
        idx += n
    if not 0 <= idx < n:
        raise IndexError(f"sample index {idx} out of range for n_samples={n}")

    flat: dict[str, Any] = {}

    atom_ptr = payload.get("atom_ptr")
    if atom_ptr is not None:
        a0, a1 = int(atom_ptr[idx]), int(atom_ptr[idx + 1])
        for key, tensor in payload["atoms"].items():
            flat[key] = tensor[a0:a1]

    edge_ptr = payload.get("edge_ptr")
    if edge_ptr is not None:
        e0, e1 = int(edge_ptr[idx]), int(edge_ptr[idx + 1])
        for key, tensor in payload["edges"].items():
            flat[key] = tensor[e0:e1]

    bond_ptr = payload.get("bond_ptr")
    if bond_ptr is not None:
        b0, b1 = int(bond_ptr[idx]), int(bond_ptr[idx + 1])
        bonds = payload.get("bonds", {})
        if _BOND_INDEX_KEY in bonds:
            flat[_BOND_INDEX_KEY] = bonds[_BOND_INDEX_KEY][:, b0:b1]  # COO dim 1
        if _BOND_TYPES_KEY in bonds:
            flat[_BOND_TYPES_KEY] = bonds[_BOND_TYPES_KEY][b0:b1]

    for family, ptr_key in _VALENCE_PTR_KEY.items():
        v_ptr = payload.get(ptr_key)
        if v_ptr is None:
            continue
        v0, v1 = int(v_ptr[idx]), int(v_ptr[idx + 1])
        for col, tensor in payload.get(family, {}).items():
            flat[f"{family}.{col}"] = tensor[v0:v1]

    for key, tensor in payload["graphs"].items():
        flat[key] = tensor[idx]

    for key, values in payload["scalars"].items():
        flat[key] = values[idx]

    return _unflatten(flat)


class BlockNotReadyError(ValueError):
    """A planned block is missing, unreadable or unsealed (invariant I8).

    Raised at construction for a root with no ``manifest.json``, an unparsable
    manifest, a manifest of another ``schema_version``, or ``sealed == False``; and
    again mid-epoch when a **cold** (never-opened or LRU-evicted) block cannot be
    reopened. The block is never skipped and never substituted by a neighbour — a
    hole in the plan stalls the view rather than silently shortening the epoch.

    Subclasses :class:`ValueError` so a scheduler can branch on the failure kind
    without string matching while every existing ``except ValueError`` handler keeps
    working.
    """


class BlockSplitMismatchError(ValueError):
    """The planned blocks do not share one manifest split (invariant I7).

    Raised when the roots of one view carry distinct ``manifest.split`` values.
    Mixing is never inferred and never silently allowed: construct one
    :class:`BlockDataset` per split. Subclasses :class:`ValueError`.
    """


class BlockSchemaMismatchError(ValueError):
    """The planned blocks are not concatenable.

    Raised for the three ways concatenation would lie: a block whose routed
    (non-``meta.``) column-name set differs from block 0's, so a cross-block batch
    would fail mid-epoch; a block carrying a ``task`` code beyond its manifest's
    ``task_names``, so the code table and the name table disagree in shape; and a
    block declaring a valence family with no downstream route (``internal``), which
    a view that accepted it would drop silently. Subclasses :class:`ValueError`.
    """


class BlockDataset(BaseDataset):
    """Ordered, read-only view over several sealed blocks as one frame index.

    Global frame ``g`` is ``(block, local)`` in the **given** block order (see the
    module docstring for the index algebra). The view holds no column data of its
    own: every tensor it hands out is produced by advanced indexing inside
    :meth:`~molix.store.block.BlockReader.frame_payload`, which copies, so an
    eviction from the bounded reader pool can never invalidate a live batch or a
    cached table and results are independent of *max_open*.

    Construction is manifest-driven — readiness, split, the declared column set and
    the task dictionary are read from ``manifest.json``, and only the small per-unit
    count columns are mapped — so a plan is validated before a single frame is
    faulted in. The three refusals (:class:`BlockNotReadyError`,
    :class:`BlockSplitMismatchError`, :class:`BlockSchemaMismatchError`) all fire
    here, at construction, rather than mid-epoch.

    ``split`` is a **property** on this type, deliberately shadowing
    :meth:`~molix.store.dataset.BaseDataset.split`: a block view's split is a
    manifest fact (invariant I7 forbids mixing), and a seeded random re-split of its
    frames would break the molecule-level grouping discipline the block layout
    encodes. Split views over blocks are built by planning different roots, not by
    re-partitioning frames.

    Composition stays the caller's::

        view = BlockDataset(roots)
        batch = collate_packed(PackedView(view.frame_payload(frames)), range(len(frames)))

    Args:
        roots: Sealed block directories, in visit order. The order *is* the global
            index order; it is consumed as given (which blocks and in what order is
            the plan's decision, not this view's).
        max_open: Bound ``K`` on the reader pool — at most this many
            :class:`~molix.store.block.BlockReader` instances (and their column
            mappings) stay resident, evicted least-recently-used. Trades address
            space and file descriptors against reopen churn; it never changes a
            result.

    Raises:
        ValueError: *roots* is empty, or ``max_open < 1``.
        BlockNotReadyError: A root is missing its manifest, unreadable, or unsealed.
        BlockSplitMismatchError: The roots carry more than one manifest split.
        BlockSchemaMismatchError: The roots are not concatenable (differing routed
            column sets, a task code beyond its dictionary, or an unroutable
            valence family).

    Reference:
        Wang, Fass & Chodera. "End-to-end differentiable construction of molecular
        mechanics force fields." Chem. Sci. 13, 12016 (2022).
        DOI:10.1039/D2SC02739A
    """

    def __init__(
        self,
        roots: Sequence[str | os.PathLike[str]],
        *,
        max_open: int = 4,
    ) -> None:
        blocks = tuple(Path(root) for root in roots)
        if not blocks:
            raise ValueError(
                "BlockDataset needs at least one block root: an empty view indexes no "
                "frame, so it can only hide the fact that the plan resolved to nothing."
            )
        if max_open < 1:
            raise ValueError(
                f"max_open must be >= 1, got {max_open}: the reader pool has to hold at "
                "least the block it is currently reading."
            )
        self._blocks: Final[tuple[Path, ...]] = blocks
        self._max_open: Final[int] = int(max_open)
        self._pool: OrderedDict[int, BlockReader] = OrderedDict()

        manifests = self._ready_manifests()
        self._split = self._single_split(manifests)
        self._refuse_unroutable_families(manifests)
        self._refuse_column_set_drift(manifests)
        self._task_names, self._task_lookup = self._merged_task_dictionary(manifests)
        frame_counts, max_atoms = self._scan_blocks(manifests)
        self._frame_ptr = _exclusive_cumsum(torch.tensor(frame_counts, dtype=torch.long))
        self._unit_ptr = _exclusive_cumsum(
            torch.tensor([manifest.units for manifest in manifests], dtype=torch.long)
        )
        self._max_atoms = max_atoms

    # -- directory discovery -----------------------------------------------

    @classmethod
    def load_from_disk(
        cls,
        directory: str | os.PathLike[str],
        *,
        max_open: int = 4,
    ) -> BlockDataset:
        """Open a directory of sealed block roots as one view.

        The consuming half of the cross-repo seam: molhub's ``prepare`` writes a
        directory of sealed ``molix.block/1`` roots and this alternate constructor
        reads it, so the only thing the two repos agree on is a directory path. It is
        the ``load_from_disk`` spelling of the HF-style dataset surface (operator
        directive, 2026-08-15).

        Discovery is the whole of its domain content — *which* roots, and *in what
        order*. The roots are the immediate children of *directory* carrying a
        ``manifest.json``, **sorted by child name**, so one prepared directory has
        exactly one frame index whatever order the filesystem happens to report;
        children without a manifest (a ``README.md``, a scratch directory) are not
        roots and are ignored. A *directory* that carries ``manifest.json`` itself is
        taken as the single root of a one-block view. Every refusal below block
        construction stays :meth:`__init__`'s and is not re-checked here: a
        discovered-but-unsealed root is refused by :class:`BlockNotReadyError`, never
        quietly skipped, which would shorten the epoch by exactly the frames still
        being written.

        Args:
            directory: A ``prepare`` output directory of block roots, or one sealed
                block root.
            max_open: Reader-pool bound, forwarded verbatim to :class:`BlockDataset`.

        Returns:
            A view over the discovered roots, in sorted child-name order.

        Raises:
            ValueError: *directory* holds no ``manifest.json`` root at either level.
                An empty view indexes no frame, so it could only hide a mistyped path
                or a ``prepare`` run that wrote nothing.
            BlockNotReadyError: A discovered root is unreadable or unsealed.
            BlockSplitMismatchError: The discovered roots carry more than one split.
            BlockSchemaMismatchError: The discovered roots are not concatenable.
        """
        root = Path(directory)
        children = sorted(root.iterdir(), key=lambda child: child.name) if root.is_dir() else []
        discovered = [child for child in children if (child / _MANIFEST_NAME).is_file()]
        if not discovered and (root / _MANIFEST_NAME).is_file():
            discovered = [root]
        if not discovered:
            raise ValueError(
                f"no block root found at {root}: neither that directory nor any of its "
                f"immediate children carries a {_MANIFEST_NAME}. Refusing rather than "
                "returning an empty view, which would hide a mistyped path or a prepare "
                "run that wrote nothing."
            )
        return cls(discovered, max_open=max_open)

    # -- construction guards -----------------------------------------------

    @staticmethod
    def _readiness(root: Path) -> tuple[BlockManifest | None, str]:
        """Return the sealed manifest of *root*, or why it is not READY.

        Args:
            root: Block directory.

        Returns:
            ``(manifest, "")`` when *root* is a sealed block, else
            ``(None, state)`` with *state* one of ``missing manifest``,
            ``unreadable manifest`` or ``unsealed — the writer has not called
            seal()``.
        """
        if not (root / _MANIFEST_NAME).is_file():
            return None, "missing manifest"
        try:
            manifest = BlockManifest.read(root)
        except ValueError:
            return None, "unreadable manifest"
        if not manifest.sealed:
            return None, "unsealed — the writer has not called seal()"
        return manifest, ""

    def _not_ready(self, row: int, root: Path, state: str) -> BlockNotReadyError:
        """Build the I8 refusal naming the block row, its path and its state."""
        return BlockNotReadyError(
            f"block row {row} of {len(self._blocks)} at {root} is not READY: {state}. "
            "A planned block is never skipped and never substituted by a neighbour, so "
            "this view stalls here instead of silently shortening the epoch; seal the "
            "block, restore it, or re-plan without it."
        )

    def _ready_manifests(self) -> tuple[BlockManifest, ...]:
        """Read every planned manifest in order, refusing the first not-READY block."""
        manifests: list[BlockManifest] = []
        for row, root in enumerate(self._blocks):
            manifest, state = self._readiness(root)
            if manifest is None:
                raise self._not_ready(row, root, state)
            manifests.append(manifest)
        return tuple(manifests)

    def _single_split(self, manifests: Sequence[BlockManifest]) -> str:
        """Return the one split of every planned block, or raise the I7 refusal."""
        splits = [manifest.split for manifest in manifests]
        if len(set(splits)) > 1:
            listing = ", ".join(f"{root} -> {split!r}" for root, split in zip(self._blocks, splits))
            raise BlockSplitMismatchError(
                f"one BlockDataset holds exactly one manifest split, but the planned "
                f"blocks carry {sorted(set(splits))}: {listing}. Mixing splits is never "
                "inferred — construct one view per split."
            )
        return splits[0]

    @staticmethod
    def _declared_families(manifest: BlockManifest) -> tuple[str, ...]:
        """Return the valence families *manifest* declares, in sorted order."""
        return tuple(
            sorted(
                {
                    str(spec["space"]).split(":", 1)[1]
                    for spec in manifest.columns.values()
                    if str(spec["space"]).startswith("molecule_term:")
                }
            )
        )

    def _refuse_unroutable_families(self, manifests: Sequence[BlockManifest]) -> None:
        """Refuse a block declaring a valence family nothing downstream can route.

        Checked before the column-set comparison so a single-root view refuses on
        the family itself rather than on a cross-block difference that a one-block
        plan cannot even have.
        """
        for row, manifest in enumerate(manifests):
            unroutable = [
                family
                for family in self._declared_families(manifest)
                if _FAMILY_BUCKETS[family] not in _ROUTABLE_BUCKETS
            ]
            if unroutable:
                raise BlockSchemaMismatchError(
                    f"block row {row} at {self._blocks[row]} declares the valence "
                    f"{'families' if len(unroutable) > 1 else 'family'} "
                    f"{unroutable}, which has no downstream route: "
                    "the packed unpacker and collate_molecules route only "
                    f"{sorted(_ROUTABLE_BUCKETS)}, so a view that accepted this block "
                    "would drop that family silently. Refusing is the honest behaviour; "
                    "routing the family is a collate-side change."
                )

    @staticmethod
    def _routed_columns(manifest: BlockManifest) -> frozenset[str]:
        """Return the payload-routed (non-``meta.``) column names of *manifest*."""
        return frozenset(
            name
            for name, spec in manifest.columns.items()
            if not str(spec["route"]).startswith(_META_ROUTE_PREFIX)
        )

    def _refuse_column_set_drift(self, manifests: Sequence[BlockManifest]) -> None:
        """Refuse blocks whose routed column sets differ from block 0's."""
        reference = self._routed_columns(manifests[0])
        for row, manifest in enumerate(manifests[1:], start=1):
            columns = self._routed_columns(manifest)
            if columns == reference:
                continue
            raise BlockSchemaMismatchError(
                f"block row {row} at {self._blocks[row]} is not concatenable with block "
                f"row 0 at {self._blocks[0]}: their routed column sets differ by "
                f"{sorted(columns ^ reference)}. A cross-block batch would fail "
                "mid-epoch, so the disagreement is refused at construction — re-convert "
                "the odd block or plan it into its own view."
            )

    @staticmethod
    def _merged_task_dictionary(
        manifests: Sequence[BlockManifest],
    ) -> tuple[tuple[str, ...], tuple[torch.Tensor, ...]]:
        """Merge the per-block task dictionaries in first-appearance order.

        Args:
            manifests: The planned manifests, in view order.

        Returns:
            ``(task_names, lookup)`` where *task_names* is the merged dictionary and
            ``lookup[b]`` is the ``(K_b,) torch.long`` local-code → global-code map
            of block ``b``. Equal names map to one global code by construction, so a
            block's local ``(0, 1)`` may legitimately become global ``(1, 2)``.
        """
        names: list[str] = []
        code_of: dict[str, int] = {}
        lookup: list[torch.Tensor] = []
        for manifest in manifests:
            local: list[int] = []
            for name in manifest.task_names:
                if name not in code_of:
                    code_of[name] = len(names)
                    names.append(name)
                local.append(code_of[name])
            lookup.append(torch.tensor(local, dtype=torch.long))
        return tuple(names), tuple(lookup)

    def _scan_blocks(self, manifests: Sequence[BlockManifest]) -> tuple[list[int], int]:
        """Open every block once for its frame count, atom bound and code check.

        The frame count is not a manifest field, so it is read through the bounded
        pool — which maps the per-unit count columns only, never a frame column.

        Returns:
            ``(frame_counts, max_atoms)`` over the planned blocks.
        """
        frame_counts: list[int] = []
        max_atoms = 0
        for row, manifest in enumerate(manifests):
            reader = self.reader(row)
            frame_counts.append(reader.frame_count)
            max_atoms = max(max_atoms, reader.max_atoms)
            codes = reader.task_codes
            if codes.numel() and int(codes.max()) >= len(manifest.task_names):
                raise BlockSchemaMismatchError(
                    f"block row {row} at {self._blocks[row]} carries the task code "
                    f"{int(codes.max())} but its manifest task_names holds "
                    f"{len(manifest.task_names)} name(s), so the code table and the name "
                    "table disagree in shape and the code cannot be decoded. Re-convert "
                    "the block with the dictionary its task column was coded against."
                )
        return frame_counts, max_atoms

    # -- bounded-K reader pool ---------------------------------------------

    def reader(self, block: int) -> BlockReader:
        """Return the reader of one block row, opening it if it is not resident.

        At most :attr:`max_open` readers stay live in an LRU pool: a hit moves the
        row to the most-recent end, an overflow drops the least-recent one and
        unmaps its columns. Reopening parses ``manifest.json`` and re-maps the small
        count columns only, and the file pages stay in the OS page cache, so a warm
        reopen re-reads no bytes.

        Args:
            block: Block row in ``[0, len(blocks))``.

        Returns:
            The live :class:`~molix.store.block.BlockReader` of that row.

        Raises:
            IndexError: *block* is outside the planned block rows.
            BlockNotReadyError: The block is cold (never opened or evicted) and can
                no longer be opened — e.g. its directory was deleted under the view.
        """
        row = int(block)
        if not 0 <= row < len(self._blocks):
            raise IndexError(
                f"block row {row} is outside [0, {len(self._blocks)}) — this view plans "
                f"{len(self._blocks)} block(s)."
            )
        resident = self._pool.get(row)
        if resident is not None:
            self._pool.move_to_end(row)
            return resident
        root = self._blocks[row]
        try:
            opened = BlockReader(root)
        except (ValueError, OSError) as exc:
            _, state = self._readiness(root)
            raise self._not_ready(row, root, state or str(exc)) from exc
        self._pool[row] = opened
        while len(self._pool) > self._max_open:
            self._pool.popitem(last=False)
        return opened

    @property
    def open_blocks(self) -> tuple[int, ...]:
        """Resident block rows, least-recently-used first (at most :attr:`max_open`)."""
        return tuple(self._pool)

    @property
    def max_open(self) -> int:
        """The reader-pool bound ``K``, fixed at construction.

        Read-only on purpose: it is also the plan's resident-window size, and one
        number with one name cannot drift between the view and its conveyor.
        """
        return self._max_open

    # -- planned blocks ----------------------------------------------------

    @property
    def blocks(self) -> tuple[Path, ...]:
        """The planned block directories, in the given (index) order."""
        return self._blocks

    @property
    def block_ids(self) -> tuple[str, ...]:
        """Block identity strings — the directory basenames, in view order.

        The producer names a block root deterministically, so the basename is the
        identity a visit-order contract can be stated in without reading a manifest.
        """
        return tuple(root.name for root in self._blocks)

    @property
    def split(self) -> str:
        """The single manifest split every planned block carries (invariant I7)."""
        return self._split

    @property
    def max_atoms(self) -> int:
        """Largest unit atom count ``max_b max(N_m)`` over the planned blocks."""
        return self._max_atoms

    # -- global index map --------------------------------------------------

    def __len__(self) -> int:
        """Total frame count ``frame_ptr[B]`` across the planned blocks."""
        return int(self._frame_ptr[-1])

    def _resolve(self, caller: str, frames: Sequence[int] | torch.Tensor) -> torch.Tensor:
        """Validate *frames* as global frame rows of this view.

        Args:
            caller: Method name, for the message.
            frames: Global frame rows, as a sequence or a 1-D tensor.

        Returns:
            ``(B,) torch.long`` copy of *frames*, in the given order.

        Raises:
            ValueError: *frames* is not 1-D, or names a row outside
                ``[0, len(self))`` — negative rows included, since a plan speaks in
                absolute rows and this view never wraps.
        """
        selected = torch.as_tensor(frames, dtype=torch.long)
        if selected.ndim != 1:
            raise ValueError(
                f"{caller} takes a 1-D list of global frame rows, got shape "
                f"{tuple(selected.shape)}; a scalar lookup is {caller}([idx])."
            )
        if selected.numel():
            low, high = int(selected.min()), int(selected.max())
            if low < 0 or high >= len(self):
                raise ValueError(
                    f"{caller} got frame rows in [{low}, {high}] but this view holds "
                    f"{len(self)} frames across {len(self._blocks)} blocks. Frame rows "
                    "are absolute and are never wrapped: a negative row is a plan bug, "
                    "not a request for the tail."
                )
        return selected

    def locate(self, frames: Sequence[int] | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map global frame rows to their owning block and local row.

        Vectorized by contract — one ``searchsorted`` over the ``(B+1,)`` frame
        pointer for the whole request; a single lookup is ``locate([idx])``.

        Args:
            frames: Global frame rows in ``[0, len(self))``, as a sequence or a 1-D
                tensor.

        Returns:
            ``(blocks, locals)``, both ``(B,) torch.long``: the block ordinal in the
            planned order, and the frame row inside that block.

        Raises:
            ValueError: *frames* is not 1-D, or names a row outside the view.
        """
        selected = self._resolve("locate", frames)
        blocks = torch.searchsorted(self._frame_ptr, selected, right=True) - 1
        return blocks, selected - self._frame_ptr[blocks]

    # -- concatenated sampler tables ---------------------------------------

    @cached_property
    def atom_counts(self) -> torch.Tensor:
        """Atoms per frame ``(F,) torch.long``, concatenated in planned order.

        Pointer arithmetic only — each block contributes
        :attr:`~molix.store.block.BlockReader.atom_counts`, a gathered copy — so a
        token-budget sampler consumes the whole view without unpacking a frame.
        """
        return torch.cat([self.reader(row).atom_counts for row in range(len(self._blocks))])

    @cached_property
    def group_codes(self) -> torch.Tensor:
        """Globally unique group id per frame ``(F,) torch.long``.

        Block ``b``'s per-block codes (its unit rows ``[0, M_b)``) are offset by
        ``unit_ptr[b]``, so the images partition ``[0, unit_ptr[B])`` — group ids
        are injective across the view and a group never straddles a batch.
        """
        return torch.cat(
            [
                self.reader(row).group_codes.to(torch.long) + int(self._unit_ptr[row])
                for row in range(len(self._blocks))
            ]
        )

    @cached_property
    def task_codes(self) -> torch.Tensor:
        """Task code per frame ``(F,) torch.long`` in the merged dictionary.

        Each block's local codes are remapped through its first-appearance
        ``local -> global`` lookup, so equal task names share one global code even
        when the blocks ordered their dictionaries differently. Decode with
        ``[view.task_names[c] for c in view.task_codes.tolist()]``.
        """
        return torch.cat(
            [
                self._task_lookup[row][self.reader(row).task_codes.to(torch.long)]
                for row in range(len(self._blocks))
            ]
        )

    @property
    def task_names(self) -> tuple[str, ...]:
        """The merged task dictionary, in first-appearance order across blocks."""
        return self._task_names

    @property
    def edge_counts(self) -> NoReturn:
        """Never returns — blocks carry no neighbour-graph columns by design.

        Delegates to :attr:`~molix.store.block.BlockReader.edge_counts` so the
        refusal keeps one copy of its rationale and one message, which already names
        :class:`molix.batch.NeighborList` as the downstream builder. Budget batches
        on :attr:`atom_counts` and construct samplers with ``max_edges=None``.

        Raises:
            ValueError: Always.
        """
        return self.reader(0).edge_counts

    # -- payload assembly --------------------------------------------------

    def frame_payload(self, frames: Sequence[int] | torch.Tensor) -> dict[str, Any]:
        """Assemble one packed-cache-v1 payload for *frames*, in caller order.

        Same name, same contract and same return shape as
        :meth:`~molix.store.block.BlockReader.frame_payload`, one level up, so the
        caller composition is byte-identical to the single-block one::

            collate_packed(PackedView(view.frame_payload(frames)), range(len(frames)))

        Frames are grouped by owning block in first-appearance order and each group
        is materialised by its own reader. A request touching one block returns that
        reader's payload unchanged (zero merge cost); a cross-block request is
        concatenated block-major and restored to caller order in one gather.
        Topology atom indices stay molecule-local, so the collate's own ``rebase``
        is unchanged.

        Args:
            frames: Global frame rows in ``[0, len(self))``, in the order they
                should appear in the batch.

        Returns:
            Payload mapping with ``n_samples``, ``atom_ptr`` ``(B+1,)``, an ``atoms``
            bucket (``Z`` ``(N,)``, ``pos`` ``(N, 3)`` in Å, dotted ``targets.*``
            leaves such as ``forces`` in kcal mol⁻¹ Å⁻¹), a ``graphs`` bucket
            (dotted ``targets.*`` leaves such as ``energy`` in kcal mol⁻¹), and one
            bucket + ``<family>_ptr`` per declared valence family (``bonds`` carries
            the stacked ``(2, T)`` ``bond_index``).

        Raises:
            ValueError: *frames* is empty, not 1-D, or names a row outside the view.
            BlockNotReadyError: An owning block is cold and can no longer be opened.
        """
        selected = self._resolve("frame_payload", frames)
        if selected.numel() == 0:
            raise ValueError("frame_payload needs at least one frame — cannot build an empty batch")
        blocks, locals_ = self.locate(selected)

        slots_of_block: dict[int, list[int]] = {}
        for slot, row in enumerate(blocks.tolist()):
            slots_of_block.setdefault(row, []).append(slot)

        local_rows = locals_.tolist()
        if len(slots_of_block) == 1:
            (row,) = slots_of_block
            return self.reader(row).frame_payload(local_rows)

        payloads: list[Mapping[str, Any]] = []
        block_major: list[int] = []
        for row, slots in slots_of_block.items():
            payloads.append(self.reader(row).frame_payload([local_rows[s] for s in slots]))
            block_major.extend(slots)
        perm = torch.argsort(torch.tensor(block_major, dtype=torch.long))
        return self._merge(payloads, perm)

    @staticmethod
    def _gather(counts: torch.Tensor, perm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Plan one block-major → caller-order gather of a ragged row space.

        The arithmetic :meth:`~molix.store.block.BlockReader.frame_payload` already
        uses for its own ragged spaces, applied to the concatenated payloads.

        Args:
            counts: ``(B,)`` rows per sample in **block-major** order.
            perm: ``(B,)`` caller slot → block-major sample.

        Returns:
            ``(ptr, rows)``: the caller-order ``(B+1,)`` exclusive-cumsum pointer,
            and the ``(sum(counts),)`` block-major row index to gather with.
        """
        block_major_ptr = _exclusive_cumsum(counts)
        ordered = counts[perm]
        ptr = _exclusive_cumsum(ordered)
        segment = torch.repeat_interleave(torch.arange(perm.numel()), ordered)
        within = torch.arange(int(ptr[-1])) - ptr[segment]
        return ptr, block_major_ptr[perm][segment] + within

    def _merge(self, payloads: Sequence[Mapping[str, Any]], perm: torch.Tensor) -> dict[str, Any]:
        """Concatenate per-block payloads and restore caller order in one gather.

        Args:
            payloads: One payload per owning block, in block-major order.
            perm: ``(B,)`` caller slot → block-major sample row.

        Returns:
            One payload of the same shape, its samples in caller order.
        """
        atom_counts = torch.cat(
            [payload["atom_ptr"][1:] - payload["atom_ptr"][:-1] for payload in payloads]
        )
        atom_ptr, atom_rows = self._gather(atom_counts, perm)
        merged: dict[str, Any] = {
            "n_samples": int(perm.numel()),
            "atom_ptr": atom_ptr,
            "atoms": {
                leaf: torch.cat([payload["atoms"][leaf] for payload in payloads])[atom_rows]
                for leaf in payloads[0]["atoms"]
            },
            "graphs": {
                leaf: torch.cat([payload["graphs"][leaf] for payload in payloads])[perm]
                for leaf in payloads[0]["graphs"]
            },
        }
        for family, bucket in _FAMILY_BUCKETS.items():
            ptr_key = f"{family}_ptr"
            if ptr_key not in payloads[0]:
                continue
            counts = torch.cat(
                [payload[ptr_key][1:] - payload[ptr_key][:-1] for payload in payloads]
            )
            merged[ptr_key], rows = self._gather(counts, perm)
            merged[bucket] = {
                leaf: (
                    torch.cat([payload[bucket][leaf] for payload in payloads], dim=1)[:, rows]
                    if leaf == "bond_index"
                    else torch.cat([payload[bucket][leaf] for payload in payloads])[rows]
                )
                for leaf in payloads[0][bucket]
            }
        return merged

    def __getitem__(self, idx: int) -> dict:
        """Return one frame as a flat pre-collate sample ``dict``.

        The one-frame payload run through
        :func:`_unpack_packed_sample` — there is no second
        route into the data. The ``"scalars"`` adapter is empty because the block
        column registry has no Python-scalar space while the unpacker indexes that
        bucket unconditionally.

        **Negative indexing is unsupported**: a plan speaks in absolute frame rows,
        so ``view[-1]`` is refused exactly like ``view[len(view)]`` rather than
        silently resolving to the last frame of the last planned block (the range
        check inside :meth:`frame_payload` is the enforcement).

        Args:
            idx: Global frame row in ``[0, len(self))``.

        Returns:
            Flat sample ``dict`` with ``Z`` ``(N,)``, ``pos`` ``(N, 3)`` in Å,
            nested ``targets`` (e.g. ``energy`` in kcal mol⁻¹, ``forces`` in
            kcal mol⁻¹ Å⁻¹), ``bond_index`` ``(2, T)`` with ``bond_types`` ``(T,)``,
            and one nested dict per further declared valence family.

        Raises:
            ValueError: *idx* is outside ``[0, len(self))``.
        """
        return _unpack_packed_sample({**self.frame_payload([idx]), "scalars": {}}, 0)
