#!/usr/bin/env python3
"""Measure what a per-frame Python label table costs a plan, against the code form.

Settles risk **R1** of ``streaming-data-tiers-04-scheduler-integration-01-tables``
§ "Domain basis": the sub-spec's footprint arithmetic *predicts* that per-frame
Python label sequences (``task_labels: list[str]`` / ``group_ids: list[str]``)
and the block-native code form (``(F,) int32`` columns plus a bounded
``task_names`` tuple) are only ~2x apart in steady state, and that the plan —
not the table — dominates at every frame count. That is a hand-derivation, not
a measurement. This bench plans the *same* synthetic split through both
representations of :class:`~molix.data.sampler.TaskMixtureBatchSampler` at
F in {1e6, 1e7, 1e8} and reports the numbers ``02-view`` / ``03-plan`` are
allowed to design against.

Method
------

* **One subprocess per cell.** A cell is one ``(F, representation)`` pair. It is
  measured in a freshly spawned copy of this script (``--cell-frames`` /
  ``--cell-form``) so ``/proc/self/status:VmHWM`` is that cell's own high-water
  resident set and not a mark left by an earlier, larger cell. The child prints
  one JSON line; the parent parses it. ``VmHWM`` right after the imports is
  reported as ``base MB`` so the interpreter + torch offset stays visible.
* **The dataset is counts only.** :class:`CountOnlyDataset` exposes exactly the
  two members the sampler touches — ``len(dataset)`` in ``_validate_tables``
  (``src/molix/data/sampler.py``) and ``n_atoms[frame_mol]`` in
  ``_frame_budgets`` (``:833``) — and raises from ``__getitem__``, so "no sample
  is unpacked during planning" is enforced by construction rather than trusted.
* **No divergence between representations.** :class:`SyntheticSplit` derives
  atom counts, unit membership (``--frames-per-unit``, default 10) and task
  assignment from closed-form integer arithmetic over ``--seed`` — no
  wall-clock, no RNG, no filesystem — so the list-form and code-form cells plan
  byte-identical input. Both plans are hashed (sha256 over ``repr`` of each
  batch, incrementally, so hashing itself costs no peak) and the run **aborts**
  if the two digests disagree: without that guard the two timings would not be
  comparable.
* **Projected-footprint guard.** Each cell's peak is projected *before* it is
  spawned, from the hand model below, and a cell whose projection exceeds
  ``--rss-budget-gb`` (default 6.0) is skipped with its projection printed
  instead of being run — this is a shared login node. The bench exits non-zero
  if the F=1e6 or F=1e7 cell was skipped, so the crossover cannot pass
  trivially unmeasured.

Projection model (CPython 3.13/3.14, 64-bit)
--------------------------------------------

Per-unit terms are amortized over ``--frames-per-unit`` frames. Per frame,
shared by both representations: the ``int64`` ``n_atoms`` column
(8 B); the parent-index tuple the sampler materialises, pointer plus uncached
``int`` object (36 B); the two ``codes_at`` Python lists, two pointers plus the
uncached group ``int`` (44 B); one pointer per frame in its unit's member list
and one in its plan batch (8 B each). Per unit, shared: the bucketing ``dict``
entry with its ``(task, group)`` tuple key (156 B) and the member list header
with next-fit over-allocation (120 B). The list form adds, per frame, the two
label-list pointers (16 B), the transient ``from_labels`` code list (36 B) and
the factorized ``int64`` columns (16 B), plus, per unit, the distinct 8-char id
string (56 B) and its factorize-lookup entry (108 B). The code form adds only
the two ``int32`` columns (8 B/frame). Ratio predicted by the model: ~1.5x, i.e.
below the 2x crossover the verdict line looks for — the bench exists to say
whether that holds.

Usage::

    OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/batch/bench_plan_tables.py
    OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/batch/bench_plan_tables.py \\
        --frames 1000000 --rss-budget-gb 2.0

Default grid: 3 frame counts x 2 representations = 6 cells, run sequentially,
single-threaded. The F=1e8 cells take minutes when the budget admits them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from molix.data.sampler import TaskMixtureBatchSampler  # noqa: E402

_MB = 1e6
_GB = 1e9
_KIB = 1024

# The bounded task set of a universal Class-I collection, in first-appearance
# order, and a mixture whose shares match the unit assignment below so an epoch
# consumes essentially every packed batch (no task starves the epoch length).
_TASK_NAMES = ("equilibrium", "torsion", "relax", "electrostatic")
_UNIT_TASK_DECILE = (0, 0, 0, 0, 1, 1, 1, 2, 2, 3)
_MIXTURE = {"equilibrium": 0.4, "torsion": 0.3, "relax": 0.2, "electrostatic": 0.1}

_FORMS = ("list", "code")
_REQUIRED_FRAMES = (1_000_000, 10_000_000)

# Hand model of the peak footprint, in bytes; see the module docstring.
_FRAME_TERMS: dict[str, tuple[tuple[str, int], ...]] = {
    "shared": (
        ("n_atoms int64 column", 8),
        ("parent index tuple: pointer + int object", 36),
        ("codes_at lists: 2 pointers + uncached group int", 44),
        ("unit member list: one pointer per frame", 8),
        ("plan batch list: one pointer per frame", 8),
    ),
    "list": (
        ("task_labels: pointer to a shared label str", 8),
        ("group_ids: pointer to its unit's id str", 8),
        ("from_labels code list: pointer + uncached int", 36),
        ("factorized int64 code columns (task+group)", 16),
    ),
    "code": (("(F,) int32 task + group code columns", 8),),
}
_UNIT_TERMS: dict[str, tuple[tuple[str, int], ...]] = {
    "shared": (
        ("unit dict entry + (task, group) tuple key", 156),
        ("unit member list header + over-allocation", 120),
    ),
    "list": (
        ("one distinct 8-char group id str", 56),
        ("factorize lookup entry + names pointer", 108),
    ),
    "code": (),
}


# ---------------------------------------------------------------------------
# process peak
# ---------------------------------------------------------------------------


class PeakRss:
    """This process's own high-water resident set.

    ``VmHWM`` is monotonic and un-resettable from user space, which is exactly
    why every cell runs in its own process: the mark a 1e8 cell leaves would
    otherwise be reported as a 1e6 cell's peak.

    Args:
        path: Status file to read; the default is this process's own.
    """

    __slots__ = ("_path",)

    def __init__(self, path: str | os.PathLike[str] = "/proc/self/status") -> None:
        self._path = Path(path)

    def mb(self) -> float:
        """Return the high-water resident set in MB (decimal, 1 MB = 1e6 B).

        Returns:
            Peak resident set of this process so far.

        Raises:
            SystemExit: The kernel exposes no ``VmHWM`` (non-Linux host); this
                bench's whole claim is a peak measurement, so it refuses to
                report a substitute.
        """
        for line in self._path.read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * _KIB / _MB
        raise SystemExit(f"{self._path} exposes no VmHWM: peak RSS is unmeasurable on this host.")


# ---------------------------------------------------------------------------
# the synthetic split
# ---------------------------------------------------------------------------


class CountOnlyDataset:
    """The minimal object :class:`TaskMixtureBatchSampler` actually consumes.

    The sampler touches a dataset exactly twice — ``len(dataset)`` when it
    validates the label table against the parent index space
    (``src/molix/data/sampler.py``) and
    :func:`~molix.data.nrows.nrows` (this object's ``n_atoms``
    column at ``frame_mol``). It never indexes one, which is the invariant
    ``test_no_sample_is_unpacked_during_planning`` pins; ``__getitem__`` here
    raises so this bench would fail loudly rather than quietly measure an
    unpacking path.

    Args:
        n_atoms: ``(F,)`` ``int64`` per-frame ``atoms.nrows``.
    """

    __slots__ = ("_n_atoms", "frame_mol")

    def __init__(self, n_atoms: torch.Tensor) -> None:
        self._n_atoms = n_atoms
        self.frame_mol = torch.arange(n_atoms.shape[0], dtype=torch.long)

    def column(self, name: str) -> torch.Tensor:
        """Per-frame ``n_atoms`` column, ``(F,)`` for this synthetic split."""
        if name != "n_atoms":
            raise KeyError(name)
        return self._n_atoms

    def __len__(self) -> int:
        """Frame count ``F`` of the parent index space."""
        return int(self._n_atoms.shape[0])

    def __getitem__(self, index: int) -> dict:
        """Never called during planning.

        Args:
            index: Parent-space frame index.

        Raises:
            AssertionError: Always — planning is pointer arithmetic.
        """
        raise AssertionError(
            f"CountOnlyDataset[{index}]: planning must not unpack a sample; "
            "the sampler is only allowed len() and n_atoms[frame_mol]."
        )


class SyntheticSplit:
    """A count-only stand-in for one split of a READY-tier block.

    Everything is closed-form integer arithmetic over ``seed``: no RNG, no
    wall-clock, no filesystem. That is what makes the list-form and code-form
    cells comparable — they plan bit-identical input, so any difference in the
    reported numbers is the representation and nothing else.

    Frame ``i`` belongs to unit ``i // frames_per_unit``; a unit is one
    molecule's conformers, so every frame of a unit carries the same atom count
    (8..32). Unit ``u`` takes its task from ``u % 10`` through
    :data:`_UNIT_TASK_DECILE`, which reproduces :data:`_MIXTURE`'s shares.

    Args:
        frames: Frame count ``F`` of the parent index space.
        frames_per_unit: Frames per sampling unit; ``frames`` must be a
            multiple of it.
        seed: Seed of the atom-count pattern.

    Raises:
        ValueError: *frames* is not a positive multiple of *frames_per_unit*.
    """

    __slots__ = ("_frames", "_frames_per_unit", "_seed")

    def __init__(self, *, frames: int, frames_per_unit: int, seed: int) -> None:
        if frames <= 0 or frames_per_unit <= 0 or frames % frames_per_unit:
            raise ValueError(
                f"frames={frames} must be a positive multiple of "
                f"frames_per_unit={frames_per_unit}; a partial unit would give "
                "the two representations different unit counts."
            )
        self._frames = int(frames)
        self._frames_per_unit = int(frames_per_unit)
        self._seed = int(seed)

    @property
    def units(self) -> int:
        """Number of sampling units ``M = F / frames_per_unit``."""
        return self._frames // self._frames_per_unit

    def dataset(self) -> CountOnlyDataset:
        """Build the count-only dataset stand-in.

        Returns:
            A :class:`CountOnlyDataset` over ``(F,)`` ``int64`` atom counts,
            constant within a unit and in ``[8, 32]``.
        """
        units = torch.arange(self.units, dtype=torch.int64)
        per_unit = 8 + (units * 2_654_435_761 + self._seed) % 25
        return CountOnlyDataset(per_unit.repeat_interleave(self._frames_per_unit))

    def labels(self) -> tuple[list[str], list[str]]:
        """Build the legacy per-frame Python label sequences.

        The bounded task labels are shared objects (one ``str`` per task) and
        every unit owns exactly one 8-character group id shared by its frames —
        the *cheapest honest* list form, matching the spec's footprint
        arithmetic. A per-frame ``f"u{i // 10:07d}"`` would allocate one string
        per frame and stack the deck.

        Returns:
            ``(task_labels, group_ids)``, both ``F`` long.
        """
        span = range(self._frames_per_unit)
        tasks = self._unit_tasks().tolist()
        ids = [f"u{unit:07d}" for unit in range(self.units)]
        task_labels = [_TASK_NAMES[task] for task in tasks for _ in span]
        group_ids = [group for group in ids for _ in span]
        return task_labels, group_ids

    def codes(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the block-native code columns.

        Returns:
            ``(task_codes, group_codes)``, both ``(F,)`` ``int32`` exactly as
            per-frame ``task_codes`` /
            ``group_codes`` emit them. Codes match :meth:`labels`'
            first-appearance factorization frame for frame.
        """
        task_codes = self._unit_tasks().repeat_interleave(self._frames_per_unit)
        groups = torch.arange(self.units, dtype=torch.int32)
        return task_codes.to(torch.int32), groups.repeat_interleave(self._frames_per_unit)

    def _unit_tasks(self) -> torch.Tensor:
        """Return the ``(M,)`` task code of every unit."""
        units = torch.arange(self.units, dtype=torch.int64)
        return torch.tensor(_UNIT_TASK_DECILE, dtype=torch.int64)[units % 10]


# ---------------------------------------------------------------------------
# projection and results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Footprint:
    """Hand-derived peak-RSS projection of one cell.

    Attributes:
        frames: Frame count ``F`` of the cell.
        frames_per_unit: Frames per sampling unit (amortizes the per-unit terms).
        form: ``"list"`` or ``"code"``.
        baseline_bytes: Interpreter + torch import offset a child inherits,
            taken from this process's own ``VmHWM``.
    """

    frames: int
    frames_per_unit: int
    form: str
    baseline_bytes: float

    @property
    def bytes_per_frame(self) -> float:
        """Projected bytes per frame: shared terms plus this representation's."""
        per_frame = sum(size for _, size in _FRAME_TERMS["shared"] + _FRAME_TERMS[self.form])
        per_unit = sum(size for _, size in _UNIT_TERMS["shared"] + _UNIT_TERMS[self.form])
        return per_frame + per_unit / self.frames_per_unit

    @property
    def gb(self) -> float:
        """Projected peak resident set in GB (decimal), baseline included."""
        return (self.baseline_bytes + self.frames * self.bytes_per_frame) / _GB


@dataclass(frozen=True)
class CellResult:
    """One measured ``(F, representation)`` cell, as reported by its subprocess.

    Attributes:
        frames: Frame count ``F``.
        form: ``"list"`` or ``"code"``.
        table_seconds: Wall time of building the label tables alone.
        plan_seconds: Wall time of ``TaskMixtureBatchSampler(...)``, i.e.
            validation + bucketing + packing + apportionment + interleave.
        peak_rss_mb: ``VmHWM`` of the cell's own process, in MB.
        baseline_rss_mb: ``VmHWM`` of that process right after the imports.
        batches: Batches in the resulting plan.
        plan_digest: sha256 over the plan, batch by batch.
    """

    frames: int
    form: str
    table_seconds: float
    plan_seconds: float
    peak_rss_mb: float
    baseline_rss_mb: float
    batches: int
    plan_digest: str

    @property
    def net_rss_mb(self) -> float:
        """Peak above the interpreter + torch baseline, in MB."""
        return self.peak_rss_mb - self.baseline_rss_mb

    def as_json(self) -> str:
        """Serialize this result as the one line a child process prints."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, payload: str) -> CellResult:
        """Parse a child process's line.

        Args:
            payload: The JSON line :meth:`as_json` produced.

        Returns:
            The reconstructed result.
        """
        return cls(**json.loads(payload))


@dataclass(frozen=True)
class SkippedCell:
    """A cell that was projected over budget and therefore never spawned.

    Attributes:
        frames: Frame count ``F``.
        form: ``"list"`` or ``"code"``.
        projected_gb: Projected peak resident set in GB.
        budget_gb: The ``--rss-budget-gb`` it exceeded.
    """

    frames: int
    form: str
    projected_gb: float
    budget_gb: float


class CellRun:
    """Measure one ``(F, representation)`` cell inside the current process.

    Args:
        split: The synthetic split to plan.
        form: ``"list"`` for per-frame Python label sequences, ``"code"`` for
            the block-native ``(F,)`` int32 columns plus ``task_names``.
        max_atoms: Per-batch atom budget.
        seed: Sampler seed (unit permutations and slot order).
    """

    __slots__ = ("_form", "_max_atoms", "_seed", "_split")

    def __init__(self, split: SyntheticSplit, *, form: str, max_atoms: int, seed: int) -> None:
        self._split = split
        self._form = form
        self._max_atoms = int(max_atoms)
        self._seed = int(seed)

    def measure(self) -> CellResult:
        """Build the tables, build the plan, and report the cell's own peak.

        Returns:
            The measured :class:`CellResult`. The plan digest is folded batch
            by batch so hashing cannot itself raise the peak.
        """
        rss = PeakRss()
        baseline_mb = rss.mb()
        dataset = self._split.dataset()

        start = time.perf_counter()
        tables = self._tables()
        table_seconds = time.perf_counter() - start

        start = time.perf_counter()
        sampler = TaskMixtureBatchSampler(
            dataset,
            indices=range(len(dataset)),
            mixture=_MIXTURE,
            max_atoms=self._max_atoms,
            seed=self._seed,
            **tables,
        )
        plan_seconds = time.perf_counter() - start

        digest = hashlib.sha256()
        batches = 0
        for batch in sampler:
            digest.update(repr(batch).encode())
            batches += 1
        return CellResult(
            frames=len(dataset),
            form=self._form,
            table_seconds=table_seconds,
            plan_seconds=plan_seconds,
            peak_rss_mb=rss.mb(),
            baseline_rss_mb=baseline_mb,
            batches=batches,
            plan_digest=digest.hexdigest(),
        )

    def _tables(self) -> dict[str, object]:
        """Return this representation's label-table keyword arguments."""
        if self._form == "list":
            task_labels, group_ids = self._split.labels()
            return {"task_labels": task_labels, "group_ids": group_ids}
        task_codes, group_codes = self._split.codes()
        return {"task_labels": task_codes, "group_ids": group_codes, "task_names": _TASK_NAMES}


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------


def _table(title: str, header: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    """Print one compact fixed-width table.

    Args:
        title: Table caption.
        header: Column headings.
        rows: Pre-formatted cells, one tuple per row.
    """
    widths = [max(len(header[i]), *(len(row[i]) for row in rows)) for i in range(len(header))]
    line = "  ".join("-" * width for width in widths)
    print(f"\n{title}")
    print("  ".join(cell.ljust(width) for cell, width in zip(header, widths)))
    print(line)
    for row in rows:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)))


def _frames_label(frames: int) -> str:
    """Render a frame count as ``1e7`` when it is a power of ten, else in full."""
    exponent = round(math.log10(frames)) if frames > 0 else 0
    return f"1e{exponent}" if frames == 10**exponent else str(frames)


def _verdict(results: list[CellResult], frames_grid: tuple[int, ...]) -> str:
    """Return the one-line crossover verdict.

    Args:
        results: Every measured cell.
        frames_grid: The full requested frame grid, measured or not.

    Returns:
        A sentence naming the smallest ``F`` at which the list form's peak RSS
        reaches 2x the code form's, or stating that no crossover was found
        below the grid's largest ``F``.
    """
    ratios: list[tuple[int, float]] = []
    for frames in sorted(frames_grid):
        peaks = {r.form: r.peak_rss_mb for r in results if r.frames == frames}
        if len(peaks) == len(_FORMS) and peaks["code"]:
            ratios.append((frames, peaks["list"] / peaks["code"]))
    crossover = next(((f, ratio) for f, ratio in ratios if ratio >= 2.0), None)
    if crossover is not None:
        return (
            f"plan-table verdict — list-form peak RSS reaches 2x code-form at "
            f"F={_frames_label(crossover[0])} ({crossover[1]:.2f}x)"
        )
    largest = max(ratios, key=lambda pair: pair[1]) if ratios else None
    seen = (
        f"largest measured ratio {largest[1]:.2f}x at F={_frames_label(largest[0])}"
        if largest is not None
        else "no F measured in both representations"
    )
    return (
        f"plan-table verdict — no crossover below F={_frames_label(max(frames_grid))} "
        f"in peak RSS ({seen})"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: Argument vector; ``None`` reads ``sys.argv``.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--frames",
        type=int,
        nargs="+",
        default=[1_000_000, 10_000_000, 100_000_000],
        help="frame counts F to measure (default: 1000000 10000000 100000000)",
    )
    parser.add_argument(
        "--frames-per-unit", type=int, default=10, help="frames per sampling unit (default: 10)"
    )
    parser.add_argument(
        "--max-atoms", type=int, default=4096, help="per-batch atom budget (default: 4096)"
    )
    parser.add_argument(
        "--rss-budget-gb",
        type=float,
        default=6.0,
        help=(
            "skip a cell whose projected peak resident set exceeds this many GB, "
            "printing the projection instead of running it (default: 6.0)"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260814, help="sampler and pattern seed")
    parser.add_argument(
        "--cell-frames",
        type=int,
        default=None,
        help=argparse.SUPPRESS,  # child mode: F
    )
    parser.add_argument(
        "--cell-form",
        choices=_FORMS,
        default=None,
        help=argparse.SUPPRESS,  # child mode: representation
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Measure every cell in its own process, print the tables and the verdict.

    Args:
        argv: Argument vector; ``None`` reads ``sys.argv``.

    Returns:
        Process exit status: non-zero when a required cell (F=1e6 or F=1e7) was
        skipped, so an unmeasured crossover can never look like a pass.
    """
    args = parse_args(argv)
    torch.set_num_threads(1)
    if args.cell_form is not None:
        return _run_cell(args)

    grid = tuple(sorted(args.frames))
    print(f"host        {platform.node()}")
    print(f"torch       {torch.__version__}")
    print(f"python      {platform.python_version()}")
    print(f"date        {time.strftime('%Y-%m-%d %H:%M:%S %z')}")
    print(f"grid        F={' '.join(_frames_label(f) for f in grid)} x {'/'.join(_FORMS)}-form")
    print(f"split       {args.frames_per_unit} frames/unit, tasks {_MIXTURE}")
    print(
        f"budget      max_atoms={args.max_atoms}, rss {args.rss_budget_gb:.1f} GB, seed {args.seed}"
    )

    results, skipped = _run_grid(grid, args)
    _report(results, skipped, grid, args)

    blocked = [cell for cell in skipped if cell.frames in _REQUIRED_FRAMES]
    if blocked:
        print()
        for cell in blocked:
            print(
                f"REQUIRED CELL SKIPPED — F={_frames_label(cell.frames)} {cell.form}-form "
                f"projected {cell.projected_gb:.2f} GB > budget {cell.budget_gb:.1f} GB; "
                "raise --rss-budget-gb and re-run: the crossover is unmeasured."
            )
        return 1
    return 0


def _run_cell(args: argparse.Namespace) -> int:
    """Measure the single cell this child process was spawned for.

    Args:
        args: Parsed command line carrying ``--cell-frames`` / ``--cell-form``.

    Returns:
        ``0``; the result is printed as one JSON line on stdout.
    """
    split = SyntheticSplit(
        frames=args.cell_frames, frames_per_unit=args.frames_per_unit, seed=args.seed
    )
    run = CellRun(split, form=args.cell_form, max_atoms=args.max_atoms, seed=args.seed)
    print(run.measure().as_json())
    return 0


def _run_grid(
    grid: tuple[int, ...], args: argparse.Namespace
) -> tuple[list[CellResult], list[SkippedCell]]:
    """Project, spawn and collect every cell, smallest ``F`` first.

    Args:
        grid: Frame counts to measure, ascending.
        args: Parsed command line.

    Returns:
        ``(results, skipped)`` — measured cells and over-budget ones.

    Raises:
        SystemExit: A child failed, printed nothing parseable, or the two
            representations produced different plans at the same ``F``.
    """
    baseline_bytes = PeakRss().mb() * _MB
    _table(
        "projected peak (hand model; baseline = this process's VmHWM "
        f"{baseline_bytes / _GB:.2f} GB)",
        ("F", "form", "B/frame", "projected GB", "decision"),
        [
            (
                _frames_label(frames),
                form,
                f"{footprint.bytes_per_frame:.1f}",
                f"{footprint.gb:.2f}",
                "run" if footprint.gb <= args.rss_budget_gb else "skip (over budget)",
            )
            for frames in grid
            for form in _FORMS
            for footprint in [Footprint(frames, args.frames_per_unit, form, baseline_bytes)]
        ],
    )

    results: list[CellResult] = []
    skipped: list[SkippedCell] = []
    for frames in grid:
        digests: dict[str, str] = {}
        for form in _FORMS:
            footprint = Footprint(frames, args.frames_per_unit, form, baseline_bytes)
            if footprint.gb > args.rss_budget_gb:
                skipped.append(SkippedCell(frames, form, footprint.gb, args.rss_budget_gb))
                print(
                    f"\nskip  F={_frames_label(frames)} {form}-form — projected "
                    f"{footprint.gb:.2f} GB ({footprint.bytes_per_frame:.1f} B/frame x "
                    f"{frames} + {baseline_bytes / _GB:.2f} GB baseline) "
                    f"> budget {args.rss_budget_gb:.1f} GB"
                )
                continue
            print(f"\nrun   F={_frames_label(frames)} {form}-form ...", flush=True)
            result = _spawn(frames, form, args)
            print(
                f"      {result.table_seconds:.2f} s table, {result.plan_seconds:.2f} s plan, "
                f"{result.peak_rss_mb:.0f} MB peak, {result.batches} batches"
            )
            digests[form] = result.plan_digest
            results.append(result)
        if len(digests) == len(_FORMS) and len(set(digests.values())) != 1:
            raise SystemExit(
                f"PLAN MISMATCH at F={_frames_label(frames)}: the two representations "
                f"planned different batches {digests}. The timings above are not "
                "comparable; fix FrameLabelTable before trusting any number here."
            )
    return results, skipped


def _spawn(frames: int, form: str, args: argparse.Namespace) -> CellResult:
    """Run one cell in a fresh process so its ``VmHWM`` is its own.

    Args:
        frames: Frame count ``F``.
        form: ``"list"`` or ``"code"``.
        args: Parsed command line, forwarded to the child.

    Returns:
        The child's measurement.

    Raises:
        SystemExit: The child exited non-zero or printed no parseable line.
    """
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--cell-frames",
        str(frames),
        "--cell-form",
        form,
        "--frames-per-unit",
        str(args.frames_per_unit),
        "--max-atoms",
        str(args.max_atoms),
        "--seed",
        str(args.seed),
    ]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    completed = subprocess.run(command, capture_output=True, text=True, env=env, check=False)
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode != 0 or not lines:
        raise SystemExit(
            f"cell F={_frames_label(frames)} {form}-form failed "
            f"(exit {completed.returncode}):\n{completed.stderr.strip()}"
        )
    return CellResult.from_json(lines[-1])


def _report(
    results: list[CellResult],
    skipped: list[SkippedCell],
    grid: tuple[int, ...],
    args: argparse.Namespace,
) -> None:
    """Print the measured cells, the per-``F`` ratios and the verdict.

    Args:
        results: Measured cells.
        skipped: Over-budget cells.
        grid: The requested frame grid.
        args: Parsed command line.
    """
    if results:
        _table(
            f"measured cells (own-subprocess VmHWM, max_atoms={args.max_atoms}, "
            f"{args.frames_per_unit} frames/unit)",
            (
                "F",
                "form",
                "table s",
                "plan s",
                "peak MB",
                "base MB",
                "net MB",
                "batches",
                "proj GB",
                "meas/proj",
            ),
            [
                (
                    _frames_label(result.frames),
                    result.form,
                    f"{result.table_seconds:.2f}",
                    f"{result.plan_seconds:.2f}",
                    f"{result.peak_rss_mb:.0f}",
                    f"{result.baseline_rss_mb:.0f}",
                    f"{result.net_rss_mb:.0f}",
                    str(result.batches),
                    f"{projection.gb:.2f}",
                    f"{result.peak_rss_mb * _MB / (projection.gb * _GB):.2f}",
                )
                for result in results
                for projection in [
                    Footprint(
                        result.frames,
                        args.frames_per_unit,
                        result.form,
                        result.baseline_rss_mb * _MB,
                    )
                ]
            ],
        )
    else:
        print("\nno cell was measured — every cell exceeded --rss-budget-gb")

    ratios = [
        (
            frames,
            {result.form: result for result in results if result.frames == frames},
        )
        for frames in sorted(grid)
    ]
    rows = [
        (
            _frames_label(frames),
            f"{cells['list'].peak_rss_mb / cells['code'].peak_rss_mb:.2f}",
            f"{cells['list'].net_rss_mb / cells['code'].net_rss_mb:.2f}",
            f"{cells['list'].table_seconds / cells['code'].table_seconds:.1f}",
            f"{cells['list'].plan_seconds / cells['code'].plan_seconds:.2f}",
            str(cells["list"].batches),
            cells["list"].plan_digest[:12],
        )
        for frames, cells in ratios
        if len(cells) == len(_FORMS)
    ]
    if rows:
        _table(
            "list-form / code-form (>1 means the list form costs more)",
            ("F", "peak", "net peak", "table s", "plan s", "batches", "plan sha256"),
            rows,
        )
    if skipped:
        _table(
            f"skipped cells (projection > --rss-budget-gb {args.rss_budget_gb:.1f})",
            ("F", "form", "projected GB"),
            [
                (_frames_label(cell.frames), cell.form, f"{cell.projected_gb:.2f}")
                for cell in skipped
            ],
        )
    print()
    print(_verdict(results, grid))


if __name__ == "__main__":
    raise SystemExit(main())
