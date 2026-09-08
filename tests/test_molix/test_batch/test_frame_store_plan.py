"""Unit tests for :mod:`molix.batch.frame_store_plan` — the block-aware batch plan.

Spec: ``.claude/specs/streaming-data-tiers-04-scheduler-integration-03-plan.md``

Layout mirror: ``src/molix/batch/frame_store_plan.py`` → this file, one class per public
type (:class:`FrameStoreCoverage`, :class:`FrameStoreView`, :class:`FrameStorePlan`,
:class:`FrameStorePlanSource`).

Pinned value-object contract — these tests **are** the RED specification of the
constructor surface, since the spec fixes the members but not their spelling::

    FrameStorePlan(
        batches=(list[int], ...),       # index batches, emission order
        blocks=(str, ...),              # owning block identity per batch, parallel
        coverage=(FrameStoreCoverage, ...),  # one row per block the plan touches
        split=str, seed=int, window=int, boundary_splits=int,
    )
    FrameStoreCoverage(block=str, units=int, frames=int, batches=int)

Planning is pointer arithmetic over metadata, so every planning fixture is a
literal-table stub view — no block reaches the disk except in
:class:`TestFrameStoreView`, whose contract-seam probes **call** the real
:class:`~molix.store.frame_store_view.FrameStoreDataset` (a ``runtime_checkable``
``isinstance`` only proves member presence, never a signature).

Hand-computed fixtures
----------------------

**Pure view** — three blocks, one task per block, ``max_atoms=12``::

    block    frames  units (frames; atoms/frame; task)          unit costs / batches
    block-a  0..3    g0 (0,1; 3; equilibrium)  g1 (2,3; 3)      6 + 6 = 12 -> 1
    block-b  4..7    g2 (4,5; 4; torsion)      g3 (6,7; 4)      8 | 8      -> 2
    block-c  8..11   g4 (8,9; 2; relax)        g5 (10,11; 2)    4 + 4 = 8  -> 1

    mixture {equilibrium: .25, torsion: .5, relax: .25}: epoch length 4, slots
    apportioned 1 / 2 / 1, every task in supply — so for **every** seed the plan
    holds 4 block-pure batches covering all 12 frames exactly once, and
    ``boundary_splits == 0``.

**Straddle view** — two blocks, ``torsion`` split across both, ``max_atoms=12``::

    block    frames  units (frames; atoms/frame; task)          unit costs / batches
    block-x  0..3    g0 (0,1; 4; relax)   g1 (2,3; 2; torsion)  relax 8 | 8   -> 2
    block-y  4..7    g2 (4,5; 2; torsion) g3 (6,7; 4; relax)    torsion 4+4=8 -> 1

    mixture {relax: .5, torsion: .5}: epoch length 2, one relax batch (block-pure)
    plus the single torsion batch, which straddles both blocks for every seed —
    so ``boundary_splits == 1`` and division emits 3 batches.

Every fixture float is a multiple of ``0.5`` (exactly representable in float32)
and every planning quantity is an integer count, so **no FP tolerance appears in
this file**.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import NoReturn

import pytest
import torch

from molix.batch.frame_store_plan import (
    FrameStoreCoverage,
    FrameStorePlan,
    FrameStorePlanSource,
)
from molix.batch.sampler import TaskMixtureBatchSampler
from molix.store.frame_store import FrameStoreWriter
from molix.store.frame_store_view import FrameStoreDataset, FrameStoreNotReadyError
from tests.conftest import frame

# ---------------------------------------------------------------------------
# Hand-computed goldens
# ---------------------------------------------------------------------------

#: Planning seed. At this seed the pure view's planned block sequence is
#: ``(b, a, c, b)`` — block-b appears twice, non-contiguously — so the ``K = 1``
#: and ``1 < K < n_blocks`` orderings are observably different from plan order.
_SEED = 11

_PURE_MAX_ATOMS = 12
_PURE_MIXTURE: dict[str, float] = {"equilibrium": 0.25, "torsion": 0.5, "relax": 0.25}
_PURE_BLOCK_IDS: tuple[str, ...] = ("block-a", "block-b", "block-c")
_PURE_FRAME_PTR: tuple[int, ...] = (0, 4, 8, 12)
_PURE_ATOM_COUNTS: tuple[int, ...] = (3, 3, 3, 3, 4, 4, 4, 4, 2, 2, 2, 2)
_PURE_TASK_CODES: tuple[int, ...] = (0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2)
_PURE_GROUP_CODES: tuple[int, ...] = (0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5)
_PURE_TASK_NAMES: tuple[str, ...] = ("equilibrium", "torsion", "relax")
_PURE_FRAMES = 12
_PURE_BATCHES = 4

_STRADDLE_MAX_ATOMS = 12
_STRADDLE_MIXTURE: dict[str, float] = {"relax": 0.5, "torsion": 0.5}
_STRADDLE_BLOCK_IDS: tuple[str, ...] = ("block-x", "block-y")
_STRADDLE_FRAME_PTR: tuple[int, ...] = (0, 4, 8)
_STRADDLE_ATOM_COUNTS: tuple[int, ...] = (4, 4, 2, 2, 2, 2, 4, 4)
_STRADDLE_TASK_CODES: tuple[int, ...] = (0, 0, 1, 1, 1, 1, 0, 0)
_STRADDLE_GROUP_CODES: tuple[int, ...] = (0, 0, 1, 1, 2, 2, 3, 3)
_STRADDLE_TASK_NAMES: tuple[str, ...] = ("relax", "torsion")
#: One planned batch straddles the two blocks, so division emits one extra batch.
_STRADDLE_PLANNED_BATCHES = 2
_STRADDLE_EMITTED_BATCHES = 3
_STRADDLE_BOUNDARY_SPLITS = 1

# Literal :class:`FrameStorePlan` of the value-object tests: block-b is visited, left
# for block-a, and never returned to — so first appearance is *not* sorted order.
_LITERAL_BATCHES: tuple[list[int], ...] = ([0, 1], [8, 9], [2, 3])
_LITERAL_BLOCKS: tuple[str, ...] = ("block-b", "block-a", "block-b")
_LITERAL_VISIT_ORDER: tuple[str, ...] = ("block-b", "block-a")
_LITERAL_SPLIT = "train"
_LITERAL_SEED = 43
_LITERAL_WINDOW = 2
_LITERAL_BOUNDARY_SPLITS = 1
_LITERAL_COVERAGE_ROWS: tuple[dict[str, object], ...] = (
    {"block": "block-b", "units": 2, "frames": 4, "batches": 2},
    {"block": "block-a", "units": 1, "frames": 2, "batches": 1},
)

_VISIT_ORDER_FORMAT = "molix.visit-order/1"

# ---------------------------------------------------------------------------
# Stub view — the ``FrameStoreView`` surface, literal tables, no disk
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class _StubFrameStoreReader:
    """The sliver of :class:`~molix.store.frame_store.FrameStoreReader` a stub can supply.

    Planning never opens a reader (residency is the view's concern), so this
    exists only to make ``view.reader(b)`` answerable at all — and to make the
    "planning opened a reader" failure mode observable.
    """

    task_codes: torch.Tensor


class _StubFrameStoreView:
    """In-memory :class:`~molix.batch.frame_store_plan.FrameStoreView` over literal tables.

    Holds the exact members the plan consumes — ``block_ids`` as directory-basename
    strings, one ``max_open`` (``K``), and a **vectorized** ``locate`` — so a
    planning test never writes a block (a unit commit costs ~18 fsyncs).

    Args:
        frame_ptr: ``(B+1,)`` exclusive-cumsum frame pointer over the blocks.
        atom_counts: Atoms per frame, ``F`` long (a dimensionless count).
        task_codes: Task code per frame, ``F`` long.
        group_codes: Sampling-unit code per frame, ``F`` long — globally unique.
        task_names: Bounded task dictionary.
        block_ids: Block identity strings, ordinal-indexed.
        max_open: The reader-pool bound ``K``, reused as the plan window.
        split: The single split of the view.
        unavailable: Block ordinals whose ``reader()`` refuses, as a conveyor
            would for a block that is not READY.
    """

    def __init__(
        self,
        *,
        frame_ptr: Sequence[int],
        atom_counts: Sequence[int],
        task_codes: Sequence[int],
        group_codes: Sequence[int],
        task_names: tuple[str, ...],
        block_ids: tuple[str, ...],
        max_open: int,
        split: str = "train",
        unavailable: frozenset[int] = frozenset(),
    ) -> None:
        self._frame_ptr = torch.tensor(list(frame_ptr), dtype=torch.long)
        self.atom_counts = torch.tensor(list(atom_counts), dtype=torch.long)
        self.task_codes = torch.tensor(list(task_codes), dtype=torch.long)
        self.group_codes = torch.tensor(list(group_codes), dtype=torch.long)
        self.task_names = task_names
        self.block_ids = block_ids
        self.max_open = int(max_open)
        self.split = split
        self._unavailable = unavailable
        #: Block ordinals whose reader was requested, in request order.
        self.opened: list[int] = []

    def __len__(self) -> int:
        """Global frame count ``F``."""
        return int(self._frame_ptr[-1])

    @property
    def edge_counts(self) -> NoReturn:
        """Never returns — a block stores no neighbour-graph column.

        Raises:
            ValueError: Always, naming ``edge_counts`` as the block store does.
        """
        raise ValueError(
            "a block stores no edge columns by design, so edge_counts is "
            "unavailable — budget batches on atom_counts."
        )

    def locate(self, frames: Sequence[int] | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map global frame rows to ``(block ordinal, block-local row)``.

        Args:
            frames: Global frame rows, as a sequence or a 1-D tensor.

        Returns:
            ``(blocks, locals)``, both ``(B,) torch.long``.
        """
        selected = torch.as_tensor(frames, dtype=torch.long)
        blocks = torch.searchsorted(self._frame_ptr, selected, right=True) - 1
        return blocks, selected - self._frame_ptr[blocks]

    def reader(self, block: int) -> _StubFrameStoreReader:
        """Return one block's reader, refusing the ordinals declared unavailable.

        Args:
            block: Block ordinal in the planned order.

        Returns:
            A :class:`_StubFrameStoreReader` over that block's local task codes.

        Raises:
            FrameStoreNotReadyError: *block* was declared unavailable.
        """
        row = int(block)
        self.opened.append(row)
        if row in self._unavailable:
            raise FrameStoreNotReadyError(
                f"block row {row} at {self.block_ids[row]} is not READY: missing manifest."
            )
        start, stop = int(self._frame_ptr[row]), int(self._frame_ptr[row + 1])
        return _StubFrameStoreReader(task_codes=self.task_codes[start:stop])


def _pure_view(*, max_open: int, unavailable: frozenset[int] = frozenset()) -> _StubFrameStoreView:
    """The three-block, task-per-block fixture (module docstring)."""
    return _StubFrameStoreView(
        frame_ptr=_PURE_FRAME_PTR,
        atom_counts=_PURE_ATOM_COUNTS,
        task_codes=_PURE_TASK_CODES,
        group_codes=_PURE_GROUP_CODES,
        task_names=_PURE_TASK_NAMES,
        block_ids=_PURE_BLOCK_IDS,
        max_open=max_open,
        unavailable=unavailable,
    )


def _straddle_view() -> _StubFrameStoreView:
    """The two-block fixture whose ``torsion`` batch straddles both blocks."""
    return _StubFrameStoreView(
        frame_ptr=_STRADDLE_FRAME_PTR,
        atom_counts=_STRADDLE_ATOM_COUNTS,
        task_codes=_STRADDLE_TASK_CODES,
        group_codes=_STRADDLE_GROUP_CODES,
        task_names=_STRADDLE_TASK_NAMES,
        block_ids=_STRADDLE_BLOCK_IDS,
        max_open=len(_STRADDLE_BLOCK_IDS),
    )


def _pure_source() -> FrameStorePlanSource:
    """The source under test over the pure fixture's mixture and atom budget."""
    return FrameStorePlanSource(mixture=_PURE_MIXTURE, max_atoms=_PURE_MAX_ATOMS)


def _straddle_source() -> FrameStorePlanSource:
    """The source under test over the straddle fixture's mixture and budget."""
    return FrameStorePlanSource(mixture=_STRADDLE_MIXTURE, max_atoms=_STRADDLE_MAX_ATOMS)


def _direct_plan(
    view: _StubFrameStoreView,
    *,
    mixture: dict[str, float],
    max_atoms: int,
    seed: int = _SEED,
) -> list[list[int]]:
    """The planning authority, built directly — the equality oracle.

    One :class:`~molix.batch.sampler.TaskMixtureBatchSampler` over the view's own
    int-code tables, the same mixture, the same atoms-only budget and the same
    seed the source is handed.
    """
    sampler = TaskMixtureBatchSampler(
        view,
        indices=list(range(len(view))),
        task_labels=view.task_codes,
        group_ids=view.group_codes,
        task_names=view.task_names,
        mixture=mixture,
        max_atoms=max_atoms,
        max_edges=None,
        seed=seed,
    )
    return [list(batch) for batch in sampler]


def _block_id(view: _StubFrameStoreView, batch: Sequence[int]) -> str:
    """The identity string of the single block *batch* lives in."""
    blocks, _ = view.locate(list(batch))
    assert blocks.unique().numel() == 1, "helper takes block-pure batches only"
    return view.block_ids[int(blocks[0])]


def _emitted_blocks(plan: FrameStorePlan) -> list[str]:
    """The block identity of every emitted batch, in emission order."""
    return [plan.block_of(position) for position in range(len(plan))]


def _windows(order: Sequence[str], size: int) -> list[tuple[str, ...]]:
    """Partition a visit order into consecutive windows of at most *size* blocks."""
    return [tuple(order[start : start + size]) for start in range(0, len(order), size)]


def _literal_plan(
    *,
    coverage: tuple[FrameStoreCoverage, ...] | None = None,
) -> FrameStorePlan:
    """A hand-written plan — the value-object tests never run the source."""
    rows = coverage
    if rows is None:
        rows = (
            FrameStoreCoverage(block="block-b", units=2, frames=4, batches=2),
            FrameStoreCoverage(block="block-a", units=1, frames=2, batches=1),
        )
    return FrameStorePlan(
        batches=tuple(list(batch) for batch in _LITERAL_BATCHES),
        blocks=_LITERAL_BLOCKS,
        coverage=rows,
        split=_LITERAL_SPLIT,
        seed=_LITERAL_SEED,
        window=_LITERAL_WINDOW,
        boundary_splits=_LITERAL_BOUNDARY_SPLITS,
    )


# ---------------------------------------------------------------------------
# Real-block fixture — contract seam only (a unit commit costs ~18 fsyncs)
# ---------------------------------------------------------------------------

_SEAM_PROVENANCE: dict[str, object] = {
    "source": {"name": "molnex-unit-test", "coordinate": "n/a"},
    "generator": {"tool": "test_frame_store_plan.py", "version": "1", "commit": "0"},
    "created_at": "2026-08-14T00:00:00+00:00",
    "units": {"pos": "angstrom", "energy": "kcal/mol", "forces": "kcal/mol/angstrom"},
}

_SEAM_LEN = 4
_SEAM_ATOM_COUNTS: tuple[int, ...] = (3, 3, 2, 2)
_SEAM_TASK_CODES: tuple[int, ...] = (0, 0, 1, 1)
_SEAM_GROUP_CODES: tuple[int, ...] = (0, 0, 1, 1)
_SEAM_TASK_NAMES: tuple[str, ...] = ("equilibrium", "torsion")
_SEAM_MAX_OPEN = 2


def _halves(count: int, start: int) -> torch.Tensor:
    """*count* distinct multiples of 0.5 — exactly representable in float32."""
    return (torch.arange(count, dtype=torch.float32) + float(start)) * 0.5


def _seam_unit(key: str, task: str, n_atoms: int, n_frames: int, offset: int) -> tuple:
    """One molecule's frames (pos in Å, energy in kcal/mol) as ``(frames, key, task)``."""
    frames = [
        frame(
            Z=[8] + [1] * (n_atoms - 1),
            pos=_halves(n_atoms * 3, offset + k * n_atoms * 3).reshape(n_atoms, 3).numpy(),
            energy=float(_halves(1, offset + 3 + k)[0]),
        )
        for k in range(n_frames)
    ]
    return frames, key, task


@pytest.fixture
def real_view(tmp_path: Path) -> FrameStoreDataset:
    """One sealed ``split="train"`` block: 2 units × 2 frames, opened at ``K = 2``."""
    root = tmp_path / "block-seam"
    writer = FrameStoreWriter(root, split="train", provenance=_SEAM_PROVENANCE)
    for unit_frames, key, task in (
        _seam_unit("u0", "equilibrium", 3, 2, 0),
        _seam_unit("u1", "torsion", 2, 2, 40),
    ):
        writer.append(unit_frames, key=key, task=task)
    writer.seal()
    return FrameStoreDataset([root], max_open=_SEAM_MAX_OPEN)


# ---------------------------------------------------------------------------
# FrameStoreCoverage
# ---------------------------------------------------------------------------


class TestFrameStoreCoverage:
    """One block's accounting row: identity plus three dimensionless counts."""

    def test_records_the_block_identity_and_its_unit_frame_batch_counts(self):
        row = FrameStoreCoverage(block="block-a", units=2, frames=4, batches=1)
        assert row.block == "block-a"
        assert row.units == 2
        assert row.frames == 4
        assert row.batches == 1

    def test_is_frozen(self):
        row = FrameStoreCoverage(block="block-a", units=2, frames=4, batches=1)
        with pytest.raises(dataclasses.FrozenInstanceError):
            row.frames = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# FrameStorePlan
# ---------------------------------------------------------------------------


class TestFrameStorePlan:
    """The value object handed to ``DataLoader(batch_sampler=...)``."""

    def test_iteration_yields_the_stored_index_batches(self):
        plan = _literal_plan()
        emitted: list[list[int]] = list(plan)
        assert emitted == [list(batch) for batch in _LITERAL_BATCHES]
        assert all(isinstance(batch, list) for batch in emitted)

    def test_len_is_the_batch_count(self):
        assert len(_literal_plan()) == len(_LITERAL_BATCHES)

    def test_visit_order_is_the_first_appearance_order_of_the_blocks(self):
        # block-b appears at positions 0 and 2, block-a at 1: first appearance
        # is (block-b, block-a) — neither sorted nor de-duplicated by position.
        assert _literal_plan().visit_order() == _LITERAL_VISIT_ORDER

    def test_block_of_returns_the_owning_block_identity_of_one_batch(self):
        plan = _literal_plan()
        assert [plan.block_of(position) for position in range(len(plan))] == list(_LITERAL_BLOCKS)

    def test_write_visit_order_reparses_as_the_visit_order_document(self, tmp_path: Path):
        plan = _literal_plan()
        sink = tmp_path / "v.json"

        plan.write_visit_order(sink)

        document = json.loads(sink.read_text())
        assert document["format"] == _VISIT_ORDER_FORMAT
        assert document["split"] == _LITERAL_SPLIT
        assert document["seed"] == _LITERAL_SEED
        assert document["window"] == _LITERAL_WINDOW
        assert document["order"] == list(_LITERAL_VISIT_ORDER)

    def test_write_visit_order_coverage_rows_carry_block_units_frames_batches(self, tmp_path: Path):
        sink = tmp_path / "v.json"
        _literal_plan().write_visit_order(sink)

        document = json.loads(sink.read_text())
        assert document["coverage"] == [dict(row) for row in _LITERAL_COVERAGE_ROWS]

    def test_rejects_coverage_whose_frames_do_not_total_the_batches(self):
        # 4 + 2 == 6 planned frames; this record claims 99 + 2.
        with pytest.raises(ValueError, match="(?i)coverage"):
            _literal_plan(
                coverage=(
                    FrameStoreCoverage(block="block-b", units=2, frames=99, batches=2),
                    FrameStoreCoverage(block="block-a", units=1, frames=2, batches=1),
                )
            )

    def test_rejects_coverage_naming_a_block_no_batch_belongs_to(self):
        with pytest.raises(ValueError, match="(?i)coverage"):
            _literal_plan(
                coverage=(
                    FrameStoreCoverage(block="block-b", units=2, frames=4, batches=2),
                    FrameStoreCoverage(block="block-z", units=1, frames=2, batches=1),
                )
            )

    def test_is_frozen(self):
        plan = _literal_plan()
        with pytest.raises(dataclasses.FrozenInstanceError):
            plan.window = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# FrameStoreView — the contract seam against the real -02 view
# ---------------------------------------------------------------------------


class TestFrameStoreView:
    """``runtime_checkable`` presence **plus** behavioural probes of each member.

    ``isinstance`` against a ``Protocol`` checks member *names* only, so a
    signature drift in :mod:`molix.store.frame_store_view` (a scalar ``locate``, a
    ``resident_window`` rename, path-valued ``block_ids``) would sail through it
    and fail on the first production collate instead. Each member is therefore
    called or read here.
    """

    def test_real_block_dataset_locate_is_vectorized(self, real_view: FrameStoreDataset):
        blocks, locals_ = real_view.locate([0])
        assert tuple(blocks.shape) == (1,)
        assert tuple(locals_.shape) == (1,)
        assert blocks.dtype is torch.long
        assert locals_.dtype is torch.long
        assert (int(blocks[0]), int(locals_[0])) == (0, 0)

    def test_real_block_dataset_block_ids_are_directory_basename_strings(
        self, real_view: FrameStoreDataset
    ):
        assert real_view.block_ids == ("block-seam",)

    def test_real_block_dataset_max_open_is_the_window(self, real_view: FrameStoreDataset):
        assert real_view.max_open == _SEAM_MAX_OPEN

    def test_real_block_dataset_reports_one_split(self, real_view: FrameStoreDataset):
        assert real_view.split == "train"

    def test_real_block_dataset_code_tables_are_frame_length(self, real_view: FrameStoreDataset):
        assert len(real_view) == _SEAM_LEN
        assert real_view.atom_counts.tolist() == list(_SEAM_ATOM_COUNTS)
        assert real_view.task_codes.tolist() == list(_SEAM_TASK_CODES)
        assert real_view.group_codes.tolist() == list(_SEAM_GROUP_CODES)
        assert real_view.task_names == _SEAM_TASK_NAMES


# ---------------------------------------------------------------------------
# FrameStorePlanSource
# ---------------------------------------------------------------------------


class TestFrameStorePlanSource:
    """Plans once globally, emits in block-windowed order (invariant I8)."""

    # -- the plan is the authority --------------------------------

    def test_plan_equals_the_direct_task_mixture_sampler_at_full_window(self):
        view = _pure_view(max_open=len(_PURE_BLOCK_IDS))

        plan = _pure_source().batch_sampler(view, seed=_SEED)

        assert plan.boundary_splits == 0
        assert [list(batch) for batch in plan] == _direct_plan(
            view, mixture=_PURE_MIXTURE, max_atoms=_PURE_MAX_ATOMS
        )

    def test_plan_covers_every_frame_of_the_view_exactly_once(self):
        view = _pure_view(max_open=len(_PURE_BLOCK_IDS))

        plan = _pure_source().batch_sampler(view, seed=_SEED)

        assert len(plan) == _PURE_BATCHES
        assert sorted(frame for batch in plan for frame in batch) == list(range(_PURE_FRAMES))

    def test_records_the_split_and_the_epoch_seed(self):
        view = _pure_view(max_open=len(_PURE_BLOCK_IDS))

        plan = _pure_source().batch_sampler(view, seed=_SEED)

        assert plan.split == view.split
        assert plan.seed == _SEED

    def test_records_the_views_max_open_as_the_window(self):
        view = _pure_view(max_open=2)

        plan = _pure_source().batch_sampler(view, seed=_SEED)

        assert plan.window == 2

    def test_planning_never_opens_a_block_reader(self):
        # Every block refuses to open: planning is metadata-only, so the plan
        # must still be built (residency is the view's / conveyor's concern).
        view = _pure_view(max_open=3, unavailable=frozenset({0, 1, 2}))

        plan = _pure_source().batch_sampler(view, seed=_SEED)

        assert view.opened == []
        assert len(plan) == _PURE_BATCHES

    # -- block-pure division --------------------------------------

    def test_every_emitted_batch_maps_to_exactly_one_block(self):
        view = _straddle_view()

        plan = _straddle_source().batch_sampler(view, seed=_SEED)

        for position, batch in enumerate(plan):
            blocks, _ = view.locate(list(batch))
            assert blocks.unique().numel() == 1
            assert plan.block_of(position) == view.block_ids[int(blocks[0])]

    def test_boundary_splits_counts_the_divided_planned_batches(self):
        view = _straddle_view()
        planned = _direct_plan(view, mixture=_STRADDLE_MIXTURE, max_atoms=_STRADDLE_MAX_ATOMS)
        assert len(planned) == _STRADDLE_PLANNED_BATCHES  # fixture guard

        plan = _straddle_source().batch_sampler(view, seed=_SEED)

        assert plan.boundary_splits == _STRADDLE_BOUNDARY_SPLITS
        assert len(plan) == _STRADDLE_EMITTED_BATCHES

    def test_division_is_a_multiset_permutation_of_the_planned_batches(self):
        view = _straddle_view()
        planned = _direct_plan(view, mixture=_STRADDLE_MIXTURE, max_atoms=_STRADDLE_MAX_ATOMS)

        plan = _straddle_source().batch_sampler(view, seed=_SEED)

        assert sorted(frame for batch in plan for frame in batch) == sorted(
            frame for batch in planned for frame in batch
        )

    def test_division_never_splits_a_sampling_unit(self):
        view = _straddle_view()

        plan = _straddle_source().batch_sampler(view, seed=_SEED)

        owner: dict[int, int] = {}
        for position, batch in enumerate(plan):
            for row in batch:
                code = int(view.group_codes[row])
                assert owner.setdefault(code, position) == position

    # -- K-window ordering ----------------------------------------

    def test_window_of_one_is_strict_block_major(self):
        view = _pure_view(max_open=1)
        planned_blocks = [
            _block_id(view, batch)
            for batch in _direct_plan(view, mixture=_PURE_MIXTURE, max_atoms=_PURE_MAX_ATOMS)
        ]
        # Fixture guard: the plan itself is *not* block-major, so contiguity
        # below cannot pass vacuously.
        assert len([key for key, _ in groupby(planned_blocks)]) > len(set(planned_blocks))

        plan = _pure_source().batch_sampler(view, seed=_SEED)

        emitted = _emitted_blocks(plan)
        runs = [key for key, _ in groupby(emitted)]
        assert len(runs) == len(set(emitted))

    def test_visit_order_is_the_plans_first_appearance_block_order(self):
        view = _pure_view(max_open=1)
        planned_blocks = [
            _block_id(view, batch)
            for batch in _direct_plan(view, mixture=_PURE_MIXTURE, max_atoms=_PURE_MAX_ATOMS)
        ]
        expected: tuple[str, ...] = tuple(dict.fromkeys(planned_blocks))

        plan = _pure_source().batch_sampler(view, seed=_SEED)

        assert plan.visit_order() == expected

    def test_window_below_the_block_count_is_consumed_window_by_window(self):
        view = _pure_view(max_open=2)

        plan = _pure_source().batch_sampler(view, seed=_SEED)

        order = plan.visit_order()
        assert len(order) == len(_PURE_BLOCK_IDS)  # windowing is not a no-op here
        windows = _windows(order, view.max_open)
        assert all(len(window) <= view.max_open for window in windows)
        window_of = {block: index for index, window in enumerate(windows) for block in window}
        visited = [window_of[block] for block in _emitted_blocks(plan)]
        assert visited == sorted(visited)

    def test_window_below_the_block_count_preserves_plan_order_inside_a_window(self):
        view = _pure_view(max_open=2)
        planned = _direct_plan(view, mixture=_PURE_MIXTURE, max_atoms=_PURE_MAX_ATOMS)

        plan = _pure_source().batch_sampler(view, seed=_SEED)

        assert plan.boundary_splits == 0  # division is the identity on this fixture
        emitted = [list(batch) for batch in plan]
        for window in _windows(plan.visit_order(), view.max_open):
            got = [
                batch for position, batch in enumerate(emitted) if plan.block_of(position) in window
            ]
            want = [batch for batch in planned if _block_id(view, batch) in window]
            assert got == want

    # -- coverage --------------------------------------------------

    def test_coverage_covers_exactly_the_blocks_the_plan_touches(self):
        view = _pure_view(max_open=len(_PURE_BLOCK_IDS))

        plan = _pure_source().batch_sampler(view, seed=_SEED)

        blocks = [row.block for row in plan.coverage]
        assert len(blocks) == len(set(blocks))
        assert set(blocks) == set(_emitted_blocks(plan))

    def test_coverage_frames_and_batches_total_the_plan(self):
        view = _pure_view(max_open=len(_PURE_BLOCK_IDS))

        plan = _pure_source().batch_sampler(view, seed=_SEED)

        assert sum(row.frames for row in plan.coverage) == _PURE_FRAMES
        assert sum(row.batches for row in plan.coverage) == len(plan)

    def test_coverage_units_count_the_distinct_group_codes_of_a_block(self):
        view = _pure_view(max_open=len(_PURE_BLOCK_IDS))

        plan = _pure_source().batch_sampler(view, seed=_SEED)

        for row in plan.coverage:
            planned_frames = [
                frame
                for position, batch in enumerate(plan)
                if plan.block_of(position) == row.block
                for frame in batch
            ]
            assert row.frames == len(planned_frames)
            assert row.units == len({int(view.group_codes[frame]) for frame in planned_frames})

    # -- eager refusals -------------------------------------------

    def test_max_edges_is_refused_eagerly_naming_edge_counts(self):
        with pytest.raises(ValueError, match="edge_counts"):
            FrameStorePlanSource(mixture=_PURE_MIXTURE, max_atoms=_PURE_MAX_ATOMS, max_edges=64)

    def test_batch_sampler_refuses_a_dataset_that_is_not_a_block_view(self):
        with pytest.raises(TypeError):
            _pure_source().batch_sampler(object(), seed=_SEED)  # type: ignore[arg-type]
