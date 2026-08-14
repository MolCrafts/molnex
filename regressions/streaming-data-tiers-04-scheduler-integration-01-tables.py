"""Public-API regression for the scheduler-integration tables over a sealed block.

Spec: ``streaming-data-tiers-04-scheduler-integration-01-tables`` (acceptance ac-011).

One minimal scenario, public API only: a small block is written unit-by-unit with
:class:`~molix.store.BlockWriter` into a :class:`tempfile.TemporaryDirectory`,
sealed, and opened with :class:`~molix.store.BlockReader`; the same
:class:`~molix.batch.TaskMixtureBatchSampler` is then built **twice over that
reader at the same seed**, once from the legacy per-frame Python label sequences
and once from the block's own ``(F,)`` integer code columns::

    (a) task_labels=[reader.task_names[c] for c in reader.task_codes.tolist()]
        group_ids=[int(g) for g in reader.group_codes.tolist()]
    (b) task_labels=reader.task_codes, group_ids=reader.group_codes,
        task_names=reader.task_names

Both must produce the *same* hard-coded golden plan and the same hard-coded
audit dicts — that equality is the whole point of the generalization: a block
consumer stops paying for ``F`` Python label objects and the plan does not move
by one index. The sealed manifest's typed accessors ``units`` / ``rows`` are
pinned to hand-computed literals in the same run.

Fixture — 7 sampling units across 3 tasks, ``F = 17`` frames (no RNG anywhere;
every stored float is a multiple of ``0.5``, hence exact in ``float32``, so no
FP tolerance applies anywhere in this script)::

    unit  key    task         N   n_frames  frames        atoms = n_frames*N
    -----------------------------------------------------------------------
    0     eq-a   equilibrium  3      2      0,1                    6
    1     eq-b   equilibrium  3      2      2,3                    6
    2     eq-c   equilibrium  4      1      4                      4
    3     rel-a  relax        4      3      5,6,7                 12
    4     rel-b  relax        5      2      8,9                   10
    5     tor-a  torsion      6      4      10,11,12,13           24
    6     tor-b  torsion      6      3      14,15,16              18
    -----------------------------------------------------------------------
                                    F = 17                  sum = 80

One unit is one group (the block layout *is* the grouping), so ``group_codes``
is the owning unit row and every unit above is exactly one sampling unit.

Hand-derived from that table — everything except the *within-batch index order*
and the *slot order*, which are the seeded permutations the goldens capture:

1. **Manifest tables.** ``units = 7`` (``M``); ``rows`` counts one entry per row
   space of the declared columns: ``molecule = M = 7``, ``molecule_atom =
   sum(N) = 3+3+4+4+5+6+6 = 31``, ``frame_atom = sum(n_frames*N) = 80``,
   ``frame = F = 17`` (present only because this block declares the frame-space
   column ``energy``; a ``pos``-only block would carry no ``"frame"`` key and
   ``F`` would stay a ``BlockReader.frame_count`` fact), and ``bytes =
   sum(len(key)) = 3*4 + 4*5 = 32`` for the UTF-8 unit-key dictionary.
2. **Per-task supply under ``max_atoms=24``** (next-fit, one open bin, whole
   units only, aggregate unit cost from the last column above) — equilibrium:
   ``6+6+4 = 16 <= 24`` in any order = **1** batch (frames 0..4); relax:
   ``12+10 = 22 <= 24`` = **1** batch (frames 5..9); torsion: ``24`` and ``18``
   never co-fit (``42 > 24``) and neither exceeds the budget on its own
   (``24 <= 24``, so no oversize warning) = **2** batches, ``{10,11,12,13}`` and
   ``{14,15,16}``. Every task's supply is therefore permutation-independent;
   only the order of the frames inside a batch is not.
3. **Natural epoch length** ``min_{p>0} floor(len(batches_t) / p_t)``
   = ``min(floor(1/0.25)=4, floor(1/0.25)=4, floor(2/0.5)=4)`` = **4**.
4. **Largest-remainder apportionment** over 4 slots: shares are equilibrium
   1.00, relax 1.00, torsion 2.00 -> bases ``(1, 1, 2)``, no leftover, no
   rescue -> ``{equilibrium: 1, relax: 1, torsion: 2}``, which exactly consumes
   the supply of step 2 (no task runs short, so no warning fires and the plan
   covers all 17 frames).
5. **Frame counts** follow: equilibrium 5, relax 5, torsion 4+3 = 7. The
   configured mixture governs *batch* shares only; frame shares are audit.

Only the index *order* in ``_GOLDEN_PLAN`` (and the order of the 4 batches) come
from running this repo's own deterministic plan; the batch contents, the counts
and the manifest literals above are hand-computed, so a mismatch means the block
store or the sampler deviated from the spec algorithm, not that the goldens went
stale.

    capture: PYTHONPATH=src python \\
             regressions/streaming-data-tiers-04-scheduler-integration-01-tables.py
    commit:  11361eb (+ working tree implementing
             streaming-data-tiers-04-scheduler-integration-01-tables)
    torch:   2.12.1+cpu
    device:  cpu (the plan is integer-only; float32 payload is exact halves)
    date:    2026-08-14
    oracle:  none — no third-party package is imported, subprocessed or consulted,
             at capture time or at run time; ``torch`` is the tensor library the
             feature is used *with*, not an oracle.
"""

from __future__ import annotations

import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

import torch

from molix.batch import TaskMixtureBatchSampler
from molix.store import BlockManifest, BlockReader, BlockWriter, SamplingUnit

# ---------------------------------------------------------------------------
# Fixture tables (see module docstring)
# ---------------------------------------------------------------------------

#: ``(unit key, task, n_atoms, n_frames)`` in append order — the whole fixture.
_UNITS: tuple[tuple[str, str, int, int], ...] = (
    ("eq-a", "equilibrium", 3, 2),
    ("eq-b", "equilibrium", 3, 2),
    ("eq-c", "equilibrium", 4, 1),
    ("rel-a", "relax", 4, 3),
    ("rel-b", "relax", 5, 2),
    ("tor-a", "torsion", 6, 4),
    ("tor-b", "torsion", 6, 3),
)
#: Opaque, caller-supplied provenance — fixed literals, presence-validated only.
_PROVENANCE: Mapping[str, object] = {
    "source": {"name": "molnex-regression", "coordinate": "n/a"},
    "generator": {
        "tool": "regressions/streaming-data-tiers-04-scheduler-integration-01-tables.py",
        "version": "1",
        "commit": "11361eb",
    },
    "created_at": "2026-08-14T00:00:00+00:00",
    "units": {"pos": "angstrom", "energy": "kcal/mol"},
}
#: Task mixture (batch shares), sums to 1.0.
_MIXTURE: Mapping[str, float] = {"equilibrium": 0.25, "relax": 0.25, "torsion": 0.5}
#: Per-batch atom-token budget. Edge budgets are unavailable over a block by
#: design (``BlockReader.edge_counts`` raises), so atoms are the only cost.
_MAX_ATOMS = 24
#: Plan seed, shared by both table representations.
_SEED = 20260814

# ---------------------------------------------------------------------------
# Hard-coded goldens
# ---------------------------------------------------------------------------

#: ``BlockManifest.units`` of the sealed block (hard-coded golden) = ``M``.
_GOLDEN_UNITS = 7
#: ``BlockManifest.rows`` of the sealed block (hard-coded golden, step 1).
_GOLDEN_ROWS: Mapping[str, int] = {
    "bytes": 32,
    "frame": 17,
    "frame_atom": 80,
    "molecule": 7,
    "molecule_atom": 31,
}
#: ``list(sampler)`` at seed 20260814 — identical through both table forms
#: (hard-coded golden).
_GOLDEN_PLAN: list[list[int]] = [
    [8, 9, 5, 6, 7],  # relax: units rel-b (10 atoms) + rel-a (12) = 22 <= 24
    [10, 11, 12, 13],  # torsion: unit tor-a alone, 24 atoms — exactly at budget
    [14, 15, 16],  # torsion: unit tor-b alone, 18 atoms (24 + 18 = 42 > 24)
    [0, 1, 4, 2, 3],  # equilibrium: units eq-a (6) + eq-c (4) + eq-b (6) = 16
]
#: Apportioned batch shares (hard-coded golden, step 4).
_GOLDEN_BATCHES_PER_TASK: dict[str, int] = {"equilibrium": 1, "relax": 1, "torsion": 2}
#: Frame counts implied by the plan (hard-coded golden, step 5).
_GOLDEN_FRAMES_PER_TASK: dict[str, int] = {"equilibrium": 5, "relax": 5, "torsion": 7}
#: Epoch length (hard-coded golden, step 3).
_GOLDEN_EPOCH_LEN = 4
#: Frames of every unit, in block order (hard-coded golden; also the batch
#: contents of step 2 once units are grouped by task).
_GOLDEN_UNIT_FRAMES: tuple[tuple[int, ...], ...] = (
    (0, 1),
    (2, 3),
    (4,),
    (5, 6, 7),
    (8, 9),
    (10, 11, 12, 13),
    (14, 15, 16),
)


def _require(condition: bool, message: str) -> None:
    """Raise :class:`AssertionError` with *message* unless *condition* holds."""
    if not condition:
        raise AssertionError(message)


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


def _unit(key: str, task: str, n_atoms: int, n_frames: int, offset: int) -> SamplingUnit:
    """Build one sampling unit: one topology plus *n_frames* frames of it.

    Args:
        key: Unit identity, stored UTF-8 in ``unit_key_bytes``.
        task: Task label, stored as a dictionary code in the ``task`` column.
        n_atoms: Atoms ``N`` of the topology.
        n_frames: Frames ``n`` of that topology.
        offset: Integer offset feeding :func:`_halves`, distinct per unit.

    Returns:
        The validated :class:`~molix.store.SamplingUnit`: one heavy atom plus
        hydrogens, ``pos`` ``(n*N, 3)`` Å and ``energy`` ``(n,)`` kcal/mol.
    """
    rows = n_frames * n_atoms
    return SamplingUnit(
        key=key,
        task=task,
        topology={"Z": torch.tensor([6] + [1] * (n_atoms - 1), dtype=torch.int16)},
        frames={
            "pos": _halves(rows * 3, offset).reshape(rows, 3),
            "energy": _halves(n_frames, offset + 1),
        },
    )


def _write_sealed_block(root: Path) -> Path:
    """Write the 7-unit fixture into *root* one unit at a time and seal it.

    Args:
        root: Block directory to create.

    Returns:
        *root*, sealed — a sealed manifest is the only witness of READY, and
        :class:`~molix.store.BlockReader` refuses anything else.
    """
    writer = BlockWriter(root, split="train", provenance=_PROVENANCE)
    for row, (key, task, n_atoms, n_frames) in enumerate(_UNITS):
        index = writer.append_unit(_unit(key, task, n_atoms, n_frames, 10 * row))
        _require(index == row, f"unit {key!r} committed at index {index}, expected {row}")
    writer.seal()
    return root


# ---------------------------------------------------------------------------
# Leg 1 — the typed manifest accessors
# ---------------------------------------------------------------------------


def _check_manifest_tables(root: Path, reader: BlockReader) -> None:
    """Leg 1: ``units`` / ``rows`` equal the hand-computed literals of step 1.

    Args:
        root: The sealed block directory.
        reader: A reader over the same block, for the cross-checks that ``F``
            stays a reader fact and ``M`` agrees with the unit count.
    """
    manifest = BlockManifest.read(root)
    _require(manifest.sealed, "the fixture block is not sealed — units/rows would be stale")
    _require(
        manifest.units == _GOLDEN_UNITS,
        f"manifest.units drift: {manifest.units} vs hand-computed {_GOLDEN_UNITS}",
    )
    rows = dict(manifest.rows)
    _require(rows == dict(_GOLDEN_ROWS), f"manifest.rows drift: {rows} vs {dict(_GOLDEN_ROWS)}")
    _require(
        manifest.units == reader.unit_count,
        f"manifest.units {manifest.units} != reader.unit_count {reader.unit_count}",
    )
    _require(
        rows["frame"] == reader.frame_count,
        f"rows['frame'] {rows['frame']} != reader.frame_count {reader.frame_count}",
    )


# ---------------------------------------------------------------------------
# Leg 2 — one plan, two table representations
# ---------------------------------------------------------------------------


def _sampler_from_labels(reader: BlockReader) -> TaskMixtureBatchSampler:
    """Build the sampler from decoded Python label sequences (the legacy form).

    Args:
        reader: The sealed block, which is also the count source: it exposes
            ``__len__`` and ``atom_counts`` (pointer arithmetic), the only
            dataset surface this sampler touches.

    Returns:
        The eagerly planned sampler.
    """
    return TaskMixtureBatchSampler(
        reader,
        indices=list(range(reader.frame_count)),
        task_labels=[reader.task_names[code] for code in reader.task_codes.tolist()],
        group_ids=[int(group) for group in reader.group_codes.tolist()],
        mixture=_MIXTURE,
        max_atoms=_MAX_ATOMS,
        seed=_SEED,
    )


def _sampler_from_codes(reader: BlockReader) -> TaskMixtureBatchSampler:
    """Build the same sampler from the block's ``(F,)`` code columns, unchanged.

    Args:
        reader: The sealed block; ``task_codes`` / ``group_codes`` are handed
            over verbatim, with ``task_names`` as their bounded dictionary — no
            per-frame Python object is materialized anywhere.

    Returns:
        The eagerly planned sampler.
    """
    return TaskMixtureBatchSampler(
        reader,
        indices=list(range(reader.frame_count)),
        task_labels=reader.task_codes,
        group_ids=reader.group_codes,
        task_names=reader.task_names,
        mixture=_MIXTURE,
        max_atoms=_MAX_ATOMS,
        seed=_SEED,
    )


def _check_golden_plan(sampler: TaskMixtureBatchSampler, label: str) -> list[list[int]]:
    """Leg 2a: the plan, its length and its audit dicts match the goldens.

    Args:
        sampler: A planned sampler.
        label: Which representation built it, for the messages.

    Returns:
        The plan, for the cross-representation comparison.
    """
    plan = [list(batch) for batch in sampler]
    _require(plan == _GOLDEN_PLAN, f"{label}: plan drift {plan} vs {_GOLDEN_PLAN}")
    _require(
        len(sampler) == _GOLDEN_EPOCH_LEN,
        f"{label}: epoch length drift {len(sampler)} vs {_GOLDEN_EPOCH_LEN}",
    )
    _require(
        [list(batch) for batch in sampler] == plan,
        f"{label}: iterating twice must yield the same plan",
    )

    batches = sampler.batches_per_task()
    frames = sampler.frames_per_task()
    _require(
        batches == _GOLDEN_BATCHES_PER_TASK,
        f"{label}: batches_per_task drift {batches} vs {_GOLDEN_BATCHES_PER_TASK}",
    )
    _require(
        frames == _GOLDEN_FRAMES_PER_TASK,
        f"{label}: frames_per_task drift {frames} vs {_GOLDEN_FRAMES_PER_TASK}",
    )
    _require(
        sum(batches.values()) == len(sampler),
        f"{label}: batch shares {batches} must sum to len(sampler) = {len(sampler)}",
    )
    return plan


def _check_plan_structure(plan: list[list[int]], reader: BlockReader) -> None:
    """Leg 2b: single-task batches, budget compliance, whole units, full epoch.

    Args:
        plan: The golden plan as built.
        reader: The sealed block, read for its per-frame task codes and atom
            counts — the same facts the docstring's step 2 derived by hand.
    """
    task_codes = reader.task_codes.tolist()
    atom_counts = reader.atom_counts.tolist()

    flat = [frame for batch in plan for frame in batch]
    _require(len(flat) == len(set(flat)), f"an epoch must not repeat a frame: {flat}")
    _require(
        sorted(flat) == list(range(reader.frame_count)),
        f"the plan covers {sorted(flat)}, not all {reader.frame_count} frames",
    )

    for batch in plan:
        _require(bool(batch), "every batch must be non-empty")
        tasks = {task_codes[frame] for frame in batch}
        _require(len(tasks) == 1, f"batch {batch} mixes task codes {sorted(tasks)}")
        total = sum(atom_counts[frame] for frame in batch)
        _require(
            total <= _MAX_ATOMS,
            f"batch {batch} sums to {total} atoms > budget {_MAX_ATOMS}",
        )

    # Unit integrity: every unit is drawn whole, inside exactly one batch — a
    # torsion scan is never split and never partially sampled.
    for row, members in enumerate(_GOLDEN_UNIT_FRAMES):
        holding = [b for b, batch in enumerate(plan) if set(members) & set(batch)]
        _require(len(holding) == 1, f"unit {row} straddles batches {holding}")
        missing = sorted(set(members) - set(plan[holding[0]]))
        _require(not missing, f"unit {row} sampled partially; missing frames {missing}")


# ---------------------------------------------------------------------------
# Fixture self-check — the literals must mean what the header says
# ---------------------------------------------------------------------------


def _check_fixture_conventions(reader: BlockReader) -> None:
    """Re-derive the hand-typed literals from the unit table, in closed form.

    A golden compares numbers against numbers, so it cannot catch a literal
    that was typed wrong in the first place, nor a fixture that quietly stopped
    being exact in float32. Both are re-derived here from :data:`_UNITS` alone.

    Args:
        reader: The sealed block, for the frame-space cross-checks.
    """
    n_atoms = [unit[2] for unit in _UNITS]
    n_frames = [unit[3] for unit in _UNITS]
    derived: dict[str, int] = {
        "molecule": len(_UNITS),
        "molecule_atom": sum(n_atoms),
        "frame_atom": sum(n * f for n, f in zip(n_atoms, n_frames, strict=True)),
        "frame": sum(n_frames),
        "bytes": sum(len(unit[0].encode("utf-8")) for unit in _UNITS),
    }
    _require(
        dict(_GOLDEN_ROWS) == derived,
        f"_GOLDEN_ROWS = {dict(_GOLDEN_ROWS)} is not what the unit table implies ({derived}) — "
        "a typo in the literal must not redefine the golden it is compared against",
    )
    _require(
        _GOLDEN_UNITS == len(_UNITS),
        f"_GOLDEN_UNITS = {_GOLDEN_UNITS} != {len(_UNITS)} appended units",
    )

    frames_of_unit: list[tuple[int, ...]] = []
    start = 0
    for count in n_frames:
        frames_of_unit.append(tuple(range(start, start + count)))
        start += count
    _require(
        list(_GOLDEN_UNIT_FRAMES) == frames_of_unit,
        f"_GOLDEN_UNIT_FRAMES = {_GOLDEN_UNIT_FRAMES} is not what the unit table implies "
        f"({frames_of_unit})",
    )
    _require(
        [tuple(int(f) for f in reader.unit_frames(row).tolist()) for row in range(len(_UNITS))]
        == frames_of_unit,
        "the block's unit_frames disagree with the hand-tabulated frame ranges",
    )

    for row, (key, task, atoms, frames) in enumerate(_UNITS):
        unit = _unit(key, task, atoms, frames, 10 * row)
        for name, tensor in unit.frames.items():
            doubled = tensor * 2.0
            _require(
                torch.equal(doubled, doubled.round()),
                f"{key}.{name} is not a multiple of 0.5 — the fixture must be exactly "
                "representable in float32 so no tolerance is ever needed",
            )

    per_task_frames: dict[str, int] = {}
    for _key, task, _atoms, frames in _UNITS:
        per_task_frames[task] = per_task_frames.get(task, 0) + frames
    _require(
        per_task_frames == _GOLDEN_FRAMES_PER_TASK,
        f"_GOLDEN_FRAMES_PER_TASK = {_GOLDEN_FRAMES_PER_TASK} disagrees with the unit table "
        f"({per_task_frames}) — this plan draws every unit exactly once",
    )
    _require(
        sum(_GOLDEN_BATCHES_PER_TASK.values()) == _GOLDEN_EPOCH_LEN,
        f"_GOLDEN_BATCHES_PER_TASK {_GOLDEN_BATCHES_PER_TASK} does not sum to the epoch "
        f"length {_GOLDEN_EPOCH_LEN}",
    )


# ---------------------------------------------------------------------------


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = _write_sealed_block(Path(tmp) / "block-train")
        reader = BlockReader(root)

        _check_fixture_conventions(reader)
        _check_manifest_tables(root, reader)

        from_labels = _check_golden_plan(_sampler_from_labels(reader), "decoded string labels")
        from_codes = _check_golden_plan(_sampler_from_codes(reader), "block code columns")
        _require(
            from_labels == from_codes,
            f"the two table representations disagree: {from_labels} vs {from_codes}",
        )
        _check_plan_structure(from_codes, reader)

    print(
        "streaming-data-tiers-04-scheduler-integration-01-tables: sealed block of 7 units / "
        "3 tasks / 17 frames -> manifest.units 7, manifest.rows "
        f"{dict(_GOLDEN_ROWS)}; TaskMixtureBatchSampler(mixture 0.25/0.25/0.50, max_atoms=24, "
        f"seed=20260814) built from decoded string labels and from reader.task_codes/"
        f"group_codes/task_names yields the same plan {_GOLDEN_PLAN}, batches_per_task "
        f"{_GOLDEN_BATCHES_PER_TASK}, frames_per_task {_GOLDEN_FRAMES_PER_TASK}, every batch "
        "single-task, every unit whole in exactly one batch, all 17 frames covered OK"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
