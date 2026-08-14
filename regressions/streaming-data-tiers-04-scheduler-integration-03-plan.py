"""Public-API regression for the block-aware plan behind ``batch_sampler_source``.

Spec: ``streaming-data-tiers-04-scheduler-integration-03-plan`` (acceptance ac-011).

One minimal scenario, public API only: three small single-task blocks are written
with :class:`~molix.store.BlockWriter` into a
:class:`tempfile.TemporaryDirectory` — two ``split="train"`` blocks opened as one
:class:`~molix.store.BlockDataset` (the plan's view) and one ``split="valid"``
block opened as its own single-block view (validation sets are hot-tier and
permanently resident, so the val loader keeps its fixed ``batch_size``; strategy
§ 6). :class:`~molix.batch.BlockPlanSource` plans the train epoch once at a pinned
seed, and a full :class:`~molix.batch.DataModule` epoch is iterated **offline**
(``num_workers=0``, no network, no subprocess) through both loaders. Every number
below is a hard-coded literal of this script, so the plan order, the block
windowing, the ``("graphs", "task")`` stamp, the coverage accounting and the
MolHub visit-order handoff all fail loudly the moment any of them drifts.

Fixture (deterministic — no RNG in the data, and every stored float is a multiple
of ``0.5``, hence exact in ``float32``, so **no FP tolerance applies anywhere in
this script**; ``pos`` Å, ``energy`` kcal mol⁻¹)::

    view    block          unit    task         N  frames  halves  global frames
    ----------------------------------------------------------------------------
    train   block-train-a  eq-0    equilibrium  4    2         0   0, 1
                           eq-1    equilibrium  4    2        40   2, 3
            block-train-b  tor-0   torsion      3    2        80   4, 5
                           tor-1   torsion      3    2       120   6, 7
    valid   block-valid    rel-0   relax        2    2       200   0, 1
                           rel-1   relax        2    2       240   2, 3

One unit is one sampling group (the block layout *is* the grouping), and each
block carries exactly one task — which is what makes the fixture **discriminating
for the global-vs-local task code**: ``block-train-b``'s own manifest dictionary
is ``("torsion",)``, so its block-local code column is all ``0``, while the view's
merged dictionary is ``("equilibrium", "torsion")`` and its global code is ``1``.
A collate that stamped ``reader.task_codes`` would label every torsion batch
``0`` — i.e. route it to the *equilibrium* ``TaskSpec`` of
:class:`~molix.core.steps.multitask.TaskRouter` — and legs 1 and 4 below both
fail on it (verified by patching the stamp back to the block-local column).

Hand-derived from that table alone — every golden is closed-form integer or
halves arithmetic, re-derived in :func:`_check_fixture_conventions` so a typo in a
literal cannot redefine the golden it is compared against:

1. **View tables.** ``len(train view) = 8``, ``atom_counts = (4,4,4,4,3,3,3,3)``,
   ``group_codes = (0,0,1,1,2,2,3,3)`` (globally unique), ``task_codes =
   (0,0,0,0,1,1,1,1)`` in the merged dictionary ``("equilibrium", "torsion")``,
   and the per-block readers report the local ``(0,0,0,0)`` column of leg 1's
   discriminating pair.
2. **Plan.** Unit costs are ``frames × N``: equilibrium ``8 | 8``, torsion
   ``6 | 6``. With ``max_atoms=8`` no two units of a task share a batch, so each
   task supplies **2** batches; mixture ``{equilibrium: 0.5, torsion: 0.5}``
   gives the natural epoch ``min(floor(2/0.5), floor(2/0.5)) = 4`` slots,
   apportioned ``2 / 2`` — the whole supply, so the epoch covers all 8 frames
   exactly once. At ``seed=1`` the plan is ``[2,3] [6,7] [0,1] [4,5]``, i.e. the
   blocks ``a, b, a, b``: with ``K = max_open = 2 >= n_blocks`` the windowing is
   the identity, so the **emitted** order is the plan order — blocks are revisited
   rather than regrouped block-major, which is exactly the ac-001 / ac-003
   identity claim. No planned batch straddles a block, so ``boundary_splits = 0``.
3. **Coverage** (invariant I10), in visit order: ``block-train-a`` 2 units /
   4 frames / 2 batches and ``block-train-b`` likewise — the frame total is the
   planned frame total and the batch total is ``len(plan)``.
4. **Batches.** Every batch is block-pure and single-task; ``("graphs", "task")``
   is a ``(B,) int64`` leaf carrying the **view-global** code (``0, 1, 0, 1``
   over the epoch), ``graphs.num_atoms`` is ``(4,4)`` for an equilibrium batch
   and ``(3,3)`` for a torsion one, and ``graphs.energy`` (post-collate, the
   ``targets.`` prefix is consumed by the routing — the leaf is at
   ``("graphs", "energy")``, not ``("graphs", "targets", "energy")``) is
   ``(k + offset + 3) * 0.5`` for the ``k``-th frame of its unit. Blocks store no
   neighbour graph, so ``edges.edge_index`` is the empty ``(0, 2)`` fallback and
   the cutoff graph is built downstream.
5. **Visit-order export.** ``molix.visit-order/1``, re-parsed with stdlib
   :mod:`json`: ``order`` is the visit-order tuple as a list and ``coverage``
   carries the ``block / units / frames / batches`` rows. MolHub builds its own
   ``VisitOrder(order, cursor)`` from this file; neither repo imports the other.
6. **Exit criterion.** The emitted plan equals a directly constructed
   :class:`~molix.batch.sampler.TaskMixtureBatchSampler` over the same metadata,
   mixture, budget and seed, batch for batch — the strategy's "the eager plan
   drives consumption unchanged" claim, made executable.

**Sharp edge this script pins deliberately.** A task code is the code of *its own
view's* merged dictionary. The train view's dictionary here is ``("equilibrium",
"torsion")`` while the independently constructed val view's is ``("relax",)``, so
the val batches are stamped ``0`` — the same integer the train side spends on
``equilibrium``. Codes are comparable **within** a view, not across two views, so
pairing a train and a val loader under one
:class:`~molix.core.steps.multitask.TaskRouter` is only sound when the caller
checks that name → code agrees on both sides. Leg 4 asserts the observed value so
this stays visible rather than being discovered by a mis-routed validation loss.

    capture: derived by hand from the fixture table above — every golden is
             closed-form integer / halves arithmetic over a deterministic
             fixture, re-derived in ``_check_fixture_conventions``, except the
             seeded batch *order* of leg 2, which is the plan of
             ``TaskMixtureBatchSampler`` at ``seed=1`` and is pinned both as a
             literal and against a directly constructed sampler (leg 6). No tool
             was run to produce a number; nothing was captured from an external
             implementation.
    verify:  PYTHONPATH=src python \\
             regressions/streaming-data-tiers-04-scheduler-integration-03-plan.py
    commit:  bc75a9d (+ working tree implementing
             streaming-data-tiers-04-scheduler-integration-03-plan)
    torch:   2.12.1+cpu
    device:  cpu (planning is integer pointer arithmetic; the float32 payload is
             exact halves)
    date:    2026-08-14
    oracle:  none — no third-party package is imported, subprocessed or consulted,
             at capture time or at run time. ``torch`` / ``tensordict`` are the
             tensor and batch containers the feature is used *with*, and the
             direct ``TaskMixtureBatchSampler`` of leg 6 is this repo's own
             planning authority, not an external reference.
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader

from molix.batch import BlockPlan, BlockPlanSource, DataModule, TargetSchema
from molix.batch.sampler import TaskMixtureBatchSampler
from molix.store import BlockDataset, BlockReader, BlockWriter, SamplingUnit

# ---------------------------------------------------------------------------
# Fixture tables (see module docstring)
# ---------------------------------------------------------------------------


class _UnitSpec(NamedTuple):
    """One sampling unit of the fixture — the whole data generator, by hand.

    Attributes:
        key: Unit identity, stored in the block's unit-key dictionary.
        task: Task label, stored as a dictionary code in the ``task`` column.
        n_atoms: Atoms ``N`` of the topology.
        n_frames: Frames ``n`` of that topology.
        offset: Integer offset feeding :func:`_halves`, distinct per unit, so a
            frame's payload identifies the unit it came from by value alone.
    """

    key: str
    task: str
    n_atoms: int
    n_frames: int
    offset: int


#: The two ``split="train"`` blocks, in **view order** — the plan's index order.
_TRAIN_BLOCKS: Mapping[str, tuple[_UnitSpec, ...]] = {
    "block-train-a": (
        _UnitSpec("eq-0", "equilibrium", 4, 2, 0),
        _UnitSpec("eq-1", "equilibrium", 4, 2, 40),
    ),
    "block-train-b": (
        _UnitSpec("tor-0", "torsion", 3, 2, 80),
        _UnitSpec("tor-1", "torsion", 3, 2, 120),
    ),
}
#: The single hot-tier ``split="valid"`` block behind the fixed-batch val loader.
_VALID_BLOCK: Mapping[str, tuple[_UnitSpec, ...]] = {
    "block-valid": (
        _UnitSpec("rel-0", "relax", 2, 2, 200),
        _UnitSpec("rel-1", "relax", 2, 2, 240),
    ),
}
#: Opaque, caller-supplied provenance — fixed literals, presence-validated only.
_PROVENANCE: Mapping[str, object] = {
    "source": {"name": "molnex-regression", "coordinate": "n/a"},
    "generator": {
        "tool": "regressions/streaming-data-tiers-04-scheduler-integration-03-plan.py",
        "version": "1",
        "commit": "bc75a9d",
    },
    "created_at": "2026-08-14T00:00:00+00:00",
    "units": {"pos": "angstrom", "energy": "kcal/mol"},
}

# ---------------------------------------------------------------------------
# Hard-coded goldens — leg 1, the train view's tables
# ---------------------------------------------------------------------------

#: Block identity strings of the train view, in plan order (directory basenames).
_TRAIN_BLOCK_IDS: tuple[str, ...] = ("block-train-a", "block-train-b")
#: Global frame count ``F`` of the train view.
_TRAIN_LEN = 8
#: The single manifest split every planned block carries (invariant I7).
_TRAIN_SPLIT = "train"
#: Reader-pool bound ``K``; ``K >= n_blocks``, so the plan window is the identity.
_TRAIN_MAX_OPEN = 2
#: Atoms per frame ``(F,)`` — the only budget a block plan can spend.
_TRAIN_ATOM_COUNTS: tuple[int, ...] = (4, 4, 4, 4, 3, 3, 3, 3)
#: Sampling-unit code per frame ``(F,)``, globally unique across the blocks.
_TRAIN_GROUP_CODES: tuple[int, ...] = (0, 0, 1, 1, 2, 2, 3, 3)
#: Task code per frame ``(F,)`` in the view's **merged** dictionary.
_TRAIN_TASK_CODES: tuple[int, ...] = (0, 0, 0, 0, 1, 1, 1, 1)
#: The merged dictionary, first-appearance order across the two blocks.
_TRAIN_TASK_NAMES: tuple[str, ...] = ("equilibrium", "torsion")
#: The discriminating pair: block-train-b's own dictionary and local code column.
#: Local ``0`` is global ``1`` — a local stamp would mis-route every torsion batch.
_LOCAL_TASK_NAMES_B: tuple[str, ...] = ("torsion",)
_LOCAL_TASK_CODES_B: tuple[int, ...] = (0, 0, 0, 0)
_GLOBAL_TASK_CODE_B = 1

# ---------------------------------------------------------------------------
# Hard-coded goldens — legs 2, 3 and 6, the plan
# ---------------------------------------------------------------------------

#: Planning seed. ``DataModule(seed=_SEED)`` at epoch 0 hands exactly this to the
#: source (``seed + epoch``), so the loader's plan is the one asserted here.
_SEED = 1
_EPOCH = 0
#: Per-batch atom-token budget. No edge budget exists over a block by design.
_MAX_ATOMS = 8
#: Batch shares; both tasks supply two batches, so the epoch consumes everything.
_MIXTURE: Mapping[str, float] = {"equilibrium": 0.5, "torsion": 0.5}
#: The emitted batches, in consumption order (hard-coded golden).
_PLAN_BATCHES: tuple[tuple[int, ...], ...] = ((2, 3), (6, 7), (0, 1), (4, 5))
#: The owning block of every emitted batch, parallel to :data:`_PLAN_BATCHES`.
_PLAN_BLOCKS: tuple[str, ...] = (
    "block-train-a",
    "block-train-b",
    "block-train-a",
    "block-train-b",
)
#: Blocks in the order the conveyor must make them READY (first appearance).
_PLAN_VISIT_ORDER: tuple[str, ...] = ("block-train-a", "block-train-b")
#: One coverage row per visited block: ``(block, units, frames, batches)``.
_PLAN_COVERAGE: tuple[tuple[str, int, int, int], ...] = (
    ("block-train-a", 2, 4, 2),
    ("block-train-b", 2, 4, 2),
)
#: No planned batch straddles a block, so nothing had to be divided.
_PLAN_BOUNDARY_SPLITS = 0
#: The document version tag MolHub's conveyor parses.
_VISIT_ORDER_FORMAT = "molix.visit-order/1"


class _BatchGolden(NamedTuple):
    """Hand-derived post-collate goldens of one emitted batch.

    Attributes:
        frames: The batch's global frame rows, in batch order.
        block: Identity of the single block the batch is drawn from.
        task: The ``("graphs", "task")`` code — the **view-global** one.
        num_atoms: ``graphs.num_atoms`` ``(B,)``, a count of atoms per graph.
        energy: ``graphs.energy`` ``(B,)`` in kcal mol⁻¹, batch order.
        atomic_numbers: ``atoms.Z`` ``(N,)``, batch order.
        membership: ``atoms.batch`` ``(N,)`` — the owning graph of every atom row.
        head_pos: First ``pos`` row ``(3,)`` in Å of each graph, batch order.
    """

    frames: tuple[int, ...]
    block: str
    task: int
    num_atoms: tuple[int, ...]
    energy: tuple[float, ...]
    atomic_numbers: tuple[int, ...]
    membership: tuple[int, ...]
    head_pos: tuple[tuple[float, float, float], ...]


#: The four train batches of the epoch, in emission order.
_TRAIN_BATCHES: tuple[_BatchGolden, ...] = (
    _BatchGolden(
        frames=(2, 3),
        block="block-train-a",
        task=0,
        num_atoms=(4, 4),
        energy=(21.5, 22.0),
        atomic_numbers=(8, 1, 1, 1, 8, 1, 1, 1),
        membership=(0, 0, 0, 0, 1, 1, 1, 1),
        head_pos=((20.0, 20.5, 21.0), (26.0, 26.5, 27.0)),
    ),
    _BatchGolden(
        frames=(6, 7),
        block="block-train-b",
        task=1,
        num_atoms=(3, 3),
        energy=(61.5, 62.0),
        atomic_numbers=(8, 1, 1, 8, 1, 1),
        membership=(0, 0, 0, 1, 1, 1),
        head_pos=((60.0, 60.5, 61.0), (64.5, 65.0, 65.5)),
    ),
    _BatchGolden(
        frames=(0, 1),
        block="block-train-a",
        task=0,
        num_atoms=(4, 4),
        energy=(1.5, 2.0),
        atomic_numbers=(8, 1, 1, 1, 8, 1, 1, 1),
        membership=(0, 0, 0, 0, 1, 1, 1, 1),
        head_pos=((0.0, 0.5, 1.0), (6.0, 6.5, 7.0)),
    ),
    _BatchGolden(
        frames=(4, 5),
        block="block-train-b",
        task=1,
        num_atoms=(3, 3),
        energy=(41.5, 42.0),
        atomic_numbers=(8, 1, 1, 8, 1, 1),
        membership=(0, 0, 0, 1, 1, 1),
        head_pos=((40.0, 40.5, 41.0), (44.5, 45.0, 45.5)),
    ),
)

# ---------------------------------------------------------------------------
# Hard-coded goldens — leg 4, the hot-tier val loader
# ---------------------------------------------------------------------------

#: Fixed samples per val batch — validation is resident, so no plan is needed.
_VAL_BATCH_SIZE = 2
_VALID_BLOCK_ID = "block-valid"
_VALID_SPLIT = "valid"
_VALID_LEN = 4
#: The val view's **own** merged dictionary: one block, one task, code 0. This is
#: *not* comparable with the train view's code 0 (``equilibrium``) — see the
#: "Sharp edge" note in the module docstring.
_VALID_TASK_NAMES: tuple[str, ...] = ("relax",)
_VALID_TASK_CODE = 0
#: The two fixed-size val batches, in loader order (never shuffled).
_VAL_BATCHES: tuple[_BatchGolden, ...] = (
    _BatchGolden(
        frames=(0, 1),
        block=_VALID_BLOCK_ID,
        task=_VALID_TASK_CODE,
        num_atoms=(2, 2),
        energy=(101.5, 102.0),
        atomic_numbers=(8, 1, 8, 1),
        membership=(0, 0, 1, 1),
        head_pos=((100.0, 100.5, 101.0), (103.0, 103.5, 104.0)),
    ),
    _BatchGolden(
        frames=(2, 3),
        block=_VALID_BLOCK_ID,
        task=_VALID_TASK_CODE,
        num_atoms=(2, 2),
        energy=(121.5, 122.0),
        atomic_numbers=(8, 1, 8, 1),
        membership=(0, 0, 1, 1),
        head_pos=((120.0, 120.5, 121.0), (123.0, 123.5, 124.0)),
    ),
)

#: Targets routed per graph; ``task`` must never be atom-level on this route.
_SCHEMA = TargetSchema(graph_level=frozenset({"energy"}), atom_level=frozenset())


def _require(condition: bool, message: str) -> None:
    """Raise :class:`AssertionError` with *message* unless *condition* holds."""
    if not condition:
        raise AssertionError(message)


def _require_equal_tensors(actual: torch.Tensor, expected: torch.Tensor, what: str) -> None:
    """Assert bit-exact tensor equality (dtype included) — never a tolerance."""
    _require(
        actual.dtype == expected.dtype,
        f"{what}: dtype drift {actual.dtype} vs {expected.dtype}",
    )
    _require(
        tuple(actual.shape) == tuple(expected.shape),
        f"{what}: shape drift {tuple(actual.shape)} vs {tuple(expected.shape)}",
    )
    _require(torch.equal(actual, expected), f"{what}: values drift")


def _cumsum(counts: Sequence[int]) -> tuple[int, ...]:
    """Exclusive cumsum with the total appended — the pointer rule, by hand."""
    out = [0]
    for count in counts:
        out.append(out[-1] + count)
    return tuple(out)


# ---------------------------------------------------------------------------
# Deterministic fixture (no RNG — closed-form integers and halves)
# ---------------------------------------------------------------------------


def _halves(count: int, start: int) -> torch.Tensor:
    """``count`` consecutive multiples of 0.5 — exactly representable in float32.

    Args:
        count: Number of values.
        start: Integer offset applied before scaling, so each unit occupies its
            own distinguishable value range.

    Returns:
        ``(count,)`` float32 tensor ``[start, start+1, ...] * 0.5``.
    """
    return (torch.arange(count, dtype=torch.float32) + float(start)) * 0.5


def _unit(spec: _UnitSpec) -> SamplingUnit:
    """Build one sampling unit: one topology plus ``spec.n_frames`` frames of it.

    Args:
        spec: The unit's row of the fixture table.

    Returns:
        The validated :class:`~molix.store.SamplingUnit`: ``pos`` ``(n*N, 3)`` Å
        and ``energy`` ``(n,)`` kcal mol⁻¹. No edge column exists — blocks store
        no neighbour graph by design.
    """
    rows = spec.n_frames * spec.n_atoms
    return SamplingUnit(
        key=spec.key,
        task=spec.task,
        topology={"Z": torch.tensor([8] + [1] * (spec.n_atoms - 1), dtype=torch.int16)},
        frames={
            "pos": _halves(rows * 3, spec.offset).reshape(rows, 3),
            "energy": _halves(spec.n_frames, spec.offset + 3),
        },
    )


def _write_block(root: Path, specs: Sequence[_UnitSpec], *, split: str) -> Path:
    """Write *specs* into *root* one unit at a time and seal the block.

    Args:
        root: Block directory to create; its basename is the block identity.
        specs: Unit rows of the fixture table, in append order.
        split: The block's single manifest split.

    Returns:
        *root*.
    """
    writer = BlockWriter(root, split=split, provenance=_PROVENANCE)
    for spec in specs:
        writer.append_unit(_unit(spec))
    writer.seal()
    return root


def _write_fixture(workdir: Path) -> tuple[list[Path], Path]:
    """Write the two train blocks and the valid block.

    Args:
        workdir: Scratch directory holding the three block directories.

    Returns:
        ``(train roots in view order, valid root)``.
    """
    train = [
        _write_block(workdir / name, specs, split=_TRAIN_SPLIT)
        for name, specs in _TRAIN_BLOCKS.items()
    ]
    valid = [
        _write_block(workdir / name, specs, split=_VALID_SPLIT)
        for name, specs in _VALID_BLOCK.items()
    ]
    return train, valid[0]


# ---------------------------------------------------------------------------
# Fixture derivations — the goldens must mean what the header says
# ---------------------------------------------------------------------------


class _Frame(NamedTuple):
    """One global frame of a view, derived in closed form from the unit table."""

    block: int
    unit: int
    task: str
    n_atoms: int
    energy: float
    head_pos: tuple[float, float, float]


def _frame_table(blocks: Mapping[str, tuple[_UnitSpec, ...]]) -> tuple[_Frame, ...]:
    """Re-derive a view's whole per-frame index from its unit table.

    Args:
        blocks: The view's blocks, in view order, each with its unit rows.

    Returns:
        One :class:`_Frame` per global frame, in view order. ``unit`` is the
        running unit row across blocks (= the globally unique group code),
        ``energy`` is ``(k + offset + 3) * 0.5`` for the ``k``-th frame of its
        unit and ``head_pos`` is ``(3*k*N + offset + j) * 0.5`` for ``j`` in
        ``(0, 1, 2)`` — the first ``pos`` row of that frame.
    """
    frames: list[_Frame] = []
    unit_row = 0
    for block, specs in enumerate(blocks.values()):
        for spec in specs:
            for k in range(spec.n_frames):
                head = 3 * k * spec.n_atoms + spec.offset
                frames.append(
                    _Frame(
                        block=block,
                        unit=unit_row,
                        task=spec.task,
                        n_atoms=spec.n_atoms,
                        energy=(k + spec.offset + 3) * 0.5,
                        head_pos=(head * 0.5, (head + 1) * 0.5, (head + 2) * 0.5),
                    )
                )
            unit_row += 1
    return tuple(frames)


def _check_batch_goldens(
    label: str,
    table: Sequence[_Frame],
    names: Sequence[str],
    goldens: Sequence[_BatchGolden],
    block_ids: Sequence[str],
) -> None:
    """Re-derive every per-batch literal from the frame table of its view.

    Args:
        label: View name, for the failure messages.
        table: The view's derived per-frame index.
        names: The view's merged task dictionary, first-appearance order.
        goldens: The per-batch literals to check.
        block_ids: Block identity strings in view order.
    """
    typo = "a typo in the literal must not redefine the golden it is compared against"
    for golden in goldens:
        rows = [table[frame] for frame in golden.frames]
        derived = {
            "block": block_ids[rows[0].block],
            "task": names.index(rows[0].task),
            "num_atoms": tuple(row.n_atoms for row in rows),
            "energy": tuple(row.energy for row in rows),
            "atomic_numbers": tuple(z for row in rows for z in (8, *([1] * (row.n_atoms - 1)))),
            "membership": tuple(slot for slot, row in enumerate(rows) for _ in range(row.n_atoms)),
            "head_pos": tuple(row.head_pos for row in rows),
        }
        for field, value in derived.items():
            literal = getattr(golden, field)
            _require(
                literal == value,
                f"{label} batch {list(golden.frames)}: {field} literal {literal!r} is not "
                f"what the unit table implies ({value!r}) — {typo}",
            )
        _require(
            len({row.block for row in rows}) == 1,
            f"{label} batch {list(golden.frames)} spans several blocks — every emitted "
            "batch of this fixture is block-pure",
        )
        _require(
            len({row.task for row in rows}) == 1,
            f"{label} batch {list(golden.frames)} mixes tasks — TaskRouter reads one "
            'code per batch at ("graphs", "task")',
        )


def _check_fixture_conventions() -> None:
    """Pin every literal to the unit tables before it is used as a golden.

    A golden compares numbers against numbers, so it cannot catch a literal that
    was typed wrong in the first place, nor a fixture that quietly stopped being
    exact in ``float32``. Both are re-derived here from :data:`_TRAIN_BLOCKS` /
    :data:`_VALID_BLOCK` alone, in closed form.
    """
    for blocks in (_TRAIN_BLOCKS, _VALID_BLOCK):
        for specs in blocks.values():
            for spec in specs:
                for name, tensor in _unit(spec).frames.items():
                    doubled = tensor * 2.0
                    _require(
                        torch.equal(doubled, doubled.round()),
                        f"{spec.key}.{name} is not a multiple of 0.5 — the fixture must be "
                        "exactly representable in float32 so no tolerance is ever needed",
                    )

    table = _frame_table(_TRAIN_BLOCKS)
    names: list[str] = []
    for frame in table:
        if frame.task not in names:
            names.append(frame.task)

    typo = "a typo in the literal must not redefine the golden it is compared against"
    for name, count, derived in (
        ("_TRAIN_LEN", _TRAIN_LEN, len(table)),
        ("_VALID_LEN", _VALID_LEN, len(_frame_table(_VALID_BLOCK))),
        ("_TRAIN_MAX_OPEN", _TRAIN_MAX_OPEN, len(_TRAIN_BLOCKS)),
        ("_PLAN_BOUNDARY_SPLITS", _PLAN_BOUNDARY_SPLITS, 0),
    ):
        _require(
            count == derived,
            f"{name} = {count} is not what the unit table implies ({derived}) — {typo}",
        )
    for name, vector, derived_vector in (
        ("_TRAIN_BLOCK_IDS", _TRAIN_BLOCK_IDS, tuple(_TRAIN_BLOCKS)),
        ("_TRAIN_ATOM_COUNTS", _TRAIN_ATOM_COUNTS, tuple(f.n_atoms for f in table)),
        ("_TRAIN_GROUP_CODES", _TRAIN_GROUP_CODES, tuple(f.unit for f in table)),
        ("_TRAIN_TASK_CODES", _TRAIN_TASK_CODES, tuple(names.index(f.task) for f in table)),
        ("_TRAIN_TASK_NAMES", _TRAIN_TASK_NAMES, tuple(names)),
        ("_VALID_TASK_NAMES", _VALID_TASK_NAMES, ("relax",)),
        ("_PLAN_BLOCKS", _PLAN_BLOCKS, tuple(b.block for b in _TRAIN_BATCHES)),
        ("_PLAN_BATCHES", _PLAN_BATCHES, tuple(b.frames for b in _TRAIN_BATCHES)),
        ("_PLAN_VISIT_ORDER", _PLAN_VISIT_ORDER, tuple(dict.fromkeys(_PLAN_BLOCKS))),
    ):
        _require(
            vector == derived_vector,
            f"{name} = {vector} is not what the unit table implies ({derived_vector}) — {typo}",
        )

    # The local-vs-global discriminator is a property of the *second* block's
    # own dictionary: its single task is the view's second, so local 0 != global 1.
    _require(
        _LOCAL_TASK_NAMES_B == (_TRAIN_TASK_NAMES[_GLOBAL_TASK_CODE_B],)
        and _GLOBAL_TASK_CODE_B != 0,
        f"the fixture no longer discriminates a block-local task code from the global "
        f"one: block-train-b declares {_LOCAL_TASK_NAMES_B} and the merged dictionary is "
        f"{_TRAIN_TASK_NAMES}",
    )

    # Coverage rows must be the per-block totals of the golden batches themselves.
    frames_of: dict[str, list[int]] = {}
    batches_of: dict[str, int] = {}
    for golden in _TRAIN_BATCHES:
        frames_of.setdefault(golden.block, []).extend(golden.frames)
        batches_of[golden.block] = batches_of.get(golden.block, 0) + 1
    derived_rows = tuple(
        (
            block,
            len({table[frame].unit for frame in frames}),
            len(frames),
            batches_of[block],
        )
        for block, frames in frames_of.items()
    )
    _require(
        _PLAN_COVERAGE == derived_rows,
        f"_PLAN_COVERAGE {_PLAN_COVERAGE} is not what the golden batches imply "
        f"({derived_rows}) — {typo}",
    )
    planned = sorted(frame for golden in _TRAIN_BATCHES for frame in golden.frames)
    _require(
        planned == list(range(_TRAIN_LEN)),
        f"the golden epoch covers {planned}, not every train frame exactly once",
    )
    _require(
        all(
            sum(table[frame].n_atoms for frame in golden.frames) <= _MAX_ATOMS
            for golden in _TRAIN_BATCHES
        ),
        f"a golden batch exceeds the atom budget {_MAX_ATOMS} without being an oversize "
        "whole unit — this fixture has no oversize unit by construction",
    )

    _check_batch_goldens("train", table, names, _TRAIN_BATCHES, _TRAIN_BLOCK_IDS)
    _check_batch_goldens(
        "valid",
        _frame_table(_VALID_BLOCK),
        list(_VALID_TASK_NAMES),
        _VAL_BATCHES,
        tuple(_VALID_BLOCK),
    )


# ---------------------------------------------------------------------------
# Leg 1 — the train view's tables, and the local-vs-global task code
# ---------------------------------------------------------------------------


def _check_view_tables(view: BlockDataset, roots: Sequence[Path]) -> None:
    """Leg 1: the metadata a plan reads, plus the discriminating code pair.

    Args:
        view: The two-block train view.
        roots: The train block roots, re-opened one at a time as an independent
            per-block source of the block-local task dictionary.
    """
    _require(len(view) == _TRAIN_LEN, f"len(view) = {len(view)}, hand-computed {_TRAIN_LEN}")
    _require(
        view.block_ids == _TRAIN_BLOCK_IDS,
        f"block_ids = {view.block_ids}, expected the directory basenames {_TRAIN_BLOCK_IDS}",
    )
    _require(view.split == _TRAIN_SPLIT, f"split = {view.split!r}, expected {_TRAIN_SPLIT!r}")
    _require(
        view.max_open == _TRAIN_MAX_OPEN,
        f"max_open = {view.max_open}, expected the plan window {_TRAIN_MAX_OPEN}",
    )
    for name, actual, expected in (
        ("atom_counts", view.atom_counts, _TRAIN_ATOM_COUNTS),
        ("group_codes", view.group_codes, _TRAIN_GROUP_CODES),
        ("task_codes", view.task_codes, _TRAIN_TASK_CODES),
    ):
        _require_equal_tensors(actual, torch.tensor(expected, dtype=torch.long), f"view.{name}")
    _require(
        view.task_names == _TRAIN_TASK_NAMES,
        f"task_names = {view.task_names}, hand-computed merge {_TRAIN_TASK_NAMES}",
    )

    # The reason the ("graphs", "task") stamp must read the *view's* table: the
    # second block's own dictionary codes its single task 0, which the merge
    # remaps to 1. Stamping the local column would label every torsion batch 0 —
    # the equilibrium TaskSpec's code.
    reader = BlockReader(roots[1])
    _require(
        reader.task_names == _LOCAL_TASK_NAMES_B,
        f"block-train-b declares {reader.task_names}, expected {_LOCAL_TASK_NAMES_B}",
    )
    _require(
        tuple(reader.task_codes.tolist()) == _LOCAL_TASK_CODES_B,
        f"block-train-b local task codes {reader.task_codes.tolist()}, expected "
        f"{list(_LOCAL_TASK_CODES_B)}",
    )
    blocks, _ = view.locate(list(range(len(view))))
    owned = [frame for frame, block in enumerate(blocks.tolist()) if block == 1]
    global_codes = sorted(set(view.task_codes[torch.tensor(owned)].tolist()))
    _require(
        global_codes == [_GLOBAL_TASK_CODE_B],
        f"block-train-b's frames {owned} carry global task codes {global_codes}, expected "
        f"[{_GLOBAL_TASK_CODE_B}] — the merged dictionary must remap the block-local 0",
    )


# ---------------------------------------------------------------------------
# Leg 2 — the plan, its visit order and its coverage
# ---------------------------------------------------------------------------


def _check_plan(plan: BlockPlan) -> None:
    """Legs 2 and 3: emitted batches, block ownership, visit order, coverage.

    Args:
        plan: The epoch's plan, as the ``batch_sampler`` the loader consumes.
    """
    emitted = tuple(tuple(batch) for batch in plan)
    _require(
        emitted == _PLAN_BATCHES,
        f"emitted batches {list(emitted)}, hand-computed {list(_PLAN_BATCHES)}",
    )
    _require(
        len(plan) == len(_PLAN_BATCHES),
        f"len(plan) = {len(plan)}, hand-computed {len(_PLAN_BATCHES)}",
    )
    owners = tuple(plan.block_of(position) for position in range(len(plan)))
    _require(owners == _PLAN_BLOCKS, f"owning blocks {list(owners)}, expected {list(_PLAN_BLOCKS)}")
    _require(
        plan.visit_order() == _PLAN_VISIT_ORDER,
        f"visit_order() = {plan.visit_order()}, hand-computed {_PLAN_VISIT_ORDER}",
    )
    _require(
        plan.split == _TRAIN_SPLIT and plan.seed == _SEED and plan.window == _TRAIN_MAX_OPEN,
        f"plan records split={plan.split!r} seed={plan.seed} window={plan.window}, expected "
        f"{_TRAIN_SPLIT!r} / {_SEED} / {_TRAIN_MAX_OPEN}",
    )
    _require(
        plan.boundary_splits == _PLAN_BOUNDARY_SPLITS,
        f"boundary_splits = {plan.boundary_splits}, hand-computed {_PLAN_BOUNDARY_SPLITS} — "
        "no unit of this fixture straddles a block, so no planned batch is divided",
    )

    rows = tuple((row.block, row.units, row.frames, row.batches) for row in plan.coverage)
    _require(
        rows == _PLAN_COVERAGE, f"coverage rows {list(rows)}, hand-computed {list(_PLAN_COVERAGE)}"
    )
    _require(
        sum(row.frames for row in plan.coverage) == _TRAIN_LEN,
        f"coverage accounts {sum(row.frames for row in plan.coverage)} frames, but the epoch "
        f"plans all {_TRAIN_LEN}",
    )
    _require(
        sum(row.batches for row in plan.coverage) == len(plan),
        f"coverage accounts {sum(row.batches for row in plan.coverage)} batches, but the plan "
        f"emits {len(plan)}",
    )
    _require(
        tuple(row.block for row in plan.coverage) == _PLAN_VISIT_ORDER,
        f"coverage is ordered {[row.block for row in plan.coverage]}, expected visit order "
        f"{list(_PLAN_VISIT_ORDER)}",
    )


# ---------------------------------------------------------------------------
# Leg 4 — one full DataModule epoch, offline, through both loaders
# ---------------------------------------------------------------------------


def _check_batch(label: str, batch: TensorDict, golden: _BatchGolden) -> None:
    """One collated batch equals its hand-derived post-collate goldens.

    Args:
        label: Loader + position, for the failure messages.
        batch: The nested ``atoms`` / ``edges`` / ``graphs`` batch.
        golden: The literals of that batch.
    """
    label_leaf = batch["graphs", "task"]
    _require(
        label_leaf.dtype is torch.int64,
        f"{label}: ('graphs','task') dtype {label_leaf.dtype}, expected torch.int64 — "
        "batch_to casts floating leaves only, so the routing label must survive",
    )
    _require_equal_tensors(
        label_leaf,
        torch.full((len(golden.frames),), golden.task, dtype=torch.int64),
        f"{label}: ('graphs','task') global code",
    )
    _require_equal_tensors(
        batch["graphs", "num_atoms"],
        torch.tensor(golden.num_atoms, dtype=torch.long),
        f"{label}: graphs.num_atoms",
    )
    # ``targets.energy`` rides the target route, so the leaf lands at
    # ("graphs", "energy") — not ("graphs", "targets", "energy").
    _require(
        ("graphs", "energy") in batch.keys(include_nested=True, leaves_only=True),
        f"{label}: no ('graphs','energy') leaf; keys are "
        f"{sorted(batch.keys(include_nested=True, leaves_only=True))}",
    )
    _require_equal_tensors(
        batch["graphs", "energy"],
        torch.tensor(golden.energy, dtype=torch.float32),
        f"{label}: graphs.energy (kcal/mol)",
    )
    _require_equal_tensors(
        batch["atoms", "Z"],
        torch.tensor(golden.atomic_numbers, dtype=torch.long),
        f"{label}: atoms.Z",
    )
    _require_equal_tensors(
        batch["atoms", "batch"],
        torch.tensor(golden.membership, dtype=torch.long),
        f"{label}: atoms.batch (graph membership)",
    )
    heads = batch["atoms", "pos"][torch.tensor(_cumsum(golden.num_atoms)[:-1], dtype=torch.long)]
    _require_equal_tensors(
        heads,
        torch.tensor(golden.head_pos, dtype=torch.float32),
        f"{label}: first pos row (angstrom) of every graph",
    )
    # A block stores no neighbour graph: the empty-edge fallback is the design,
    # and the cutoff graph is built downstream (NeighborList / the MD bind path).
    _require(
        tuple(batch["edges", "edge_index"].shape) == (0, 2),
        f"{label}: edges.edge_index has shape {tuple(batch['edges', 'edge_index'].shape)}, "
        "expected the empty (0, 2) fallback — blocks carry no edge column",
    )


def _check_epoch(loader: DataLoader, label: str, goldens: Sequence[_BatchGolden]) -> None:
    """Iterate one loader to exhaustion and check every batch against its golden.

    Args:
        loader: The train or val loader, built with ``num_workers=0``.
        label: Loader name, for the failure messages.
        goldens: The expected batches, in loader order.
    """
    seen = 0
    for position, batch in enumerate(loader):
        _require(
            position < len(goldens),
            f"{label} emitted batch {position}, but the epoch is {len(goldens)} batches long",
        )
        golden = goldens[position]
        _check_batch(f"{label} batch {position} {list(golden.frames)}", batch, golden)
        seen += 1
    _require(
        seen == len(goldens),
        f"{label} emitted {seen} batches, hand-computed {len(goldens)}",
    )


# ---------------------------------------------------------------------------
# Leg 5 — the MolHub visit-order handoff
# ---------------------------------------------------------------------------


def _check_visit_order_document(plan: BlockPlan, path: Path) -> None:
    """Leg 5: the exported document re-parses with stdlib :mod:`json`.

    Args:
        plan: The plan that wrote the document.
        path: Destination the document was written to.
    """
    plan.write_visit_order(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    _require(
        document["format"] == _VISIT_ORDER_FORMAT,
        f"document format {document['format']!r}, expected {_VISIT_ORDER_FORMAT!r}",
    )
    for field, expected in (
        ("split", _TRAIN_SPLIT),
        ("seed", _SEED),
        ("window", _TRAIN_MAX_OPEN),
        ("order", list(_PLAN_VISIT_ORDER)),
    ):
        _require(
            document[field] == expected,
            f"document {field} = {document[field]!r}, expected {expected!r}",
        )
    _require(
        document["order"] == list(plan.visit_order()),
        f"document order {document['order']} is not the plan's visit order "
        f"{list(plan.visit_order())}",
    )
    rows = [
        (row["block"], row["units"], row["frames"], row["batches"]) for row in document["coverage"]
    ]
    _require(
        tuple(rows) == _PLAN_COVERAGE,
        f"document coverage {rows}, hand-computed {list(_PLAN_COVERAGE)}",
    )


# ---------------------------------------------------------------------------
# Leg 6 — the strategy exit criterion
# ---------------------------------------------------------------------------


def _direct_plan(view: BlockDataset) -> list[list[int]]:
    """The planning authority, built directly — the exit-criterion oracle.

    One :class:`~molix.batch.sampler.TaskMixtureBatchSampler` over the view's own
    int-code tables, the same mixture, the same atoms-only budget and the same
    seed :class:`~molix.batch.BlockPlanSource` is handed.

    Args:
        view: The two-block train view.

    Returns:
        The planned batches, in plan order.
    """
    sampler = TaskMixtureBatchSampler(
        view,
        indices=list(range(len(view))),
        task_labels=view.task_codes,
        group_ids=view.group_codes,
        task_names=view.task_names,
        mixture=_MIXTURE,
        max_atoms=_MAX_ATOMS,
        max_edges=None,
        seed=_SEED,
    )
    return [list(batch) for batch in sampler]


def _check_exit_criterion(view: BlockDataset, plan: BlockPlan) -> None:
    """Leg 6: at ``K >= n_blocks`` the emitted plan **is** the eager plan.

    Args:
        view: The two-block train view.
        plan: The block plan the loader consumed.
    """
    direct = _direct_plan(view)
    _require(
        [list(batch) for batch in plan] == direct,
        f"the emitted plan {[list(b) for b in plan]} differs from the directly constructed "
        f"TaskMixtureBatchSampler {direct} at the same seed — with max_open "
        f"({view.max_open}) >= n_blocks ({len(view.block_ids)}) and no boundary split, "
        "block windowing is the identity, so the eager plan must drive consumption "
        "unchanged",
    )
    _require(
        plan.boundary_splits == 0,
        f"boundary_splits = {plan.boundary_splits}: the equality above only holds when no "
        "planned batch had to be divided at a block boundary",
    )


# ---------------------------------------------------------------------------


def _check_determinism(view: BlockDataset, plan: BlockPlan) -> None:
    """A second, independently built source plans the same epoch at the same seed.

    Args:
        view: The two-block train view.
        plan: The plan to reproduce.
    """
    again = BlockPlanSource(mixture=_MIXTURE, max_atoms=_MAX_ATOMS).batch_sampler(view, seed=_SEED)
    _require(
        [list(batch) for batch in again] == [list(batch) for batch in plan]
        and again.blocks == plan.blocks
        and again.coverage == plan.coverage,
        "two identical BlockPlanSource constructions at the same seed disagree — the plan "
        "is not deterministic, so epoch k would not be re-derivable on resume",
    )
    _require(
        [list(batch) for batch in plan] == [list(batch) for batch in plan],
        "iterating one plan twice must yield the same batches — the plan is a value object",
    )


def main() -> int:
    _check_fixture_conventions()

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        train_roots, valid_root = _write_fixture(workdir)

        train_view = BlockDataset(train_roots, max_open=_TRAIN_MAX_OPEN)
        val_view = BlockDataset([valid_root], max_open=1)
        _require(
            val_view.split == _VALID_SPLIT and len(val_view) == _VALID_LEN,
            f"the val view reports split {val_view.split!r} over {len(val_view)} frames, "
            f"expected {_VALID_SPLIT!r} over {_VALID_LEN}",
        )
        _require(
            val_view.task_names == _VALID_TASK_NAMES,
            f"the val view's dictionary is {val_view.task_names}, expected "
            f"{_VALID_TASK_NAMES} — its codes are its own, not the train view's",
        )

        _check_view_tables(train_view, train_roots)

        # Composition stays the caller's: view + source + DataModule, no façade.
        datamodule = DataModule(
            train_view,
            val_view,
            target_schema=_SCHEMA,
            batch_sampler_source=BlockPlanSource(mixture=_MIXTURE, max_atoms=_MAX_ATOMS),
            batch_size=_VAL_BATCH_SIZE,
            num_workers=0,
            pin_memory=False,
            seed=_SEED - _EPOCH,
        )
        datamodule.on_epoch_start(_EPOCH)
        train_loader = datamodule.train_dataloader()

        plan = train_loader.batch_sampler
        _require(
            isinstance(plan, BlockPlan),
            f"the train loader's batch_sampler is {type(plan).__name__}, expected the "
            "BlockPlan the source returned",
        )
        _check_plan(plan)
        _check_exit_criterion(train_view, plan)
        _check_determinism(train_view, plan)
        _check_epoch(train_loader, "train", _TRAIN_BATCHES)

        val_loader = datamodule.val_dataloader()
        _require(
            val_loader.batch_size == _VAL_BATCH_SIZE,
            f"the val loader batches {val_loader.batch_size} samples, expected the fixed "
            f"{_VAL_BATCH_SIZE} of a hot-tier, permanently resident validation set",
        )
        _check_epoch(val_loader, "valid", _VAL_BATCHES)

        _check_visit_order_document(plan, workdir / "visit-order.json")

    print(
        "streaming-data-tiers-04-scheduler-integration-03-plan: two sealed train blocks "
        f"(4+4 frames, 2+2 units, tasks {_TRAIN_TASK_NAMES}) -> one BlockDataset with "
        f"atom_counts {_TRAIN_ATOM_COUNTS}, group_codes {_TRAIN_GROUP_CODES}, task_codes "
        f"{_TRAIN_TASK_CODES} (block-train-b's local {list(_LOCAL_TASK_CODES_B)} remapped to "
        f"the global {_GLOBAL_TASK_CODE_B}); BlockPlanSource(mixture 0.5/0.5, max_atoms="
        f"{_MAX_ATOMS}, seed={_SEED}) emits {[list(b) for b in _PLAN_BATCHES]} over blocks "
        f"{list(_PLAN_BLOCKS)} with boundary_splits {_PLAN_BOUNDARY_SPLITS}, visit order "
        f"{list(_PLAN_VISIT_ORDER)} and coverage {[list(row) for row in _PLAN_COVERAGE]} — "
        "batch-for-batch equal to a direct TaskMixtureBatchSampler at the same seed; one "
        "offline DataModule epoch (num_workers=0) stamps ('graphs','task') int64 "
        f"{[b.task for b in _TRAIN_BATCHES]} with num_atoms "
        f"{[list(b.num_atoms) for b in _TRAIN_BATCHES]} and energy "
        f"{[list(b.energy) for b in _TRAIN_BATCHES]} kcal/mol over empty (0, 2) edges; the "
        f"hot-tier valid block feeds {len(_VAL_BATCHES)} fixed batch_size={_VAL_BATCH_SIZE} "
        f"val batches at its own code {_VALID_TASK_CODE}; visit-order document re-parsed as "
        f"{_VISIT_ORDER_FORMAT} OK"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
