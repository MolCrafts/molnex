"""Public-API scenario for `DatasetProfiler` (spec `dataset-profiler-salvage`, ac-007).

The scenario a user actually performs, in one process, with nothing mocked:

    five sample dicts  →  PackedCache(tmp).save(...)  →  CachedDataset(sink)
                       →  DatasetProfiler(n_samples=5).run(ds)

and then every non-timing field of the returned `DatasetResult` is compared to
a literal computed by hand from the five samples written out below. The five
samples are **literals in this file**, not a fixture, a generator or an RNG
draw, so each golden is arithmetic anyone can redo on the page: 3 + 4 + 5 + 6 +
2 = 20 atoms over 5 records is a mean of 4.00, and 4 + 6 + 8 + 10 + 2 = 30
edges over those 20 atoms is `avg_num_neighbors` = 1.50, exactly.

What is pinned, and why those things

* **Sizes** — `n_total`, `atom_stats.mean`, `max_atoms`, `max_edges`,
  `avg_num_neighbors`. These are the claim the salvage rests on: with a
  packed cache behind it the profiler reads `atom_ptr` / `edge_ptr` and
  reports **all five** records exactly, rather than extrapolating from the
  sampled ones. `counts_exact is True` and an empty `warnings` list are pinned
  alongside them, because a silent fall-back to the sampled path would still
  produce these same numbers here (`n_samples=5` of 5) and must not pass.
* **Field layout** — the sorted key list with each key's axis, dtype and
  trailing shape, straight off the packed `payload["schema"]`
  (`fields_exact is True`). Pins the atom / edge / graph classification of a
  realistic key set, including that `targets.U0` `(1,)` is *not* mistaken for
  a per-atom or per-edge column.
* **Targets** — `targets.U0` mean / min / max. Every literal target value is
  dyadic (exactly representable in float32 **and** float64), so the mean is
  exact arithmetic rather than a captured measurement: (-1.5) + (-0.5) + 0.25
  + 2.0 + 3.75 = 4.0, over 5 records, is 0.80.
* **Footprint** — `sample_bytes.mean` and `est_total_mb`. Not required by
  ac-007, but they are analytic (Σ numel × element_size over the leaves) and
  they are the one golden here that notices a dtype change: widening `pos` to
  float64 moves 204.0 B/record and nothing else in this file.
* **Package re-export** — `from molix.profiler import DatasetProfiler,
  DatasetResult` resolves to the very objects `molix.profiler.dataset`
  defines, and `molix.profiler.__all__` lists them and stays alphabetised.
  The unit tests import from the submodule, so this file is the only thing
  holding the documented package-level import path.

What is deliberately **not** pinned: anything timed. `access_ms`,
`cold_access_ms` and the whole Access section of the report are wall-clock
measurements, non-deterministic by construction; asserting a magnitude on them
would make this file fail on a busy node rather than on a regression. They are
exercised (the sampled path runs) and then ignored.

Goldens
-------
    capture command : none. There is no oracle and nothing was captured — every
                      literal below is arithmetic over the five sample dicts in
                      this file, and the derivation is written next to each one.
                      Confirmed against the implementation by running
                      `PYTHONPATH=src python regressions/dataset-profiler-salvage.py`.
    commit          : e94c35e (e94c35eb02a0bf961d0676addc5467e3ccd62623), with
                      the uncommitted `dataset-profiler-salvage` working tree on
                      top of it (`src/molix/profiler/dataset.py` is new there).
    torch           : 2.12.1+cpu   (python 3.14.5, numpy via molix.profiler._utils)
    date            : 2026-08-09
    device / dtype  : CPU throughout. `pos` / `edge_dist` / `targets.U0` are
                      float32 and `Z` / `edge_index` are int64, fixed by the
                      literals below rather than by `molix.config` — this file
                      never touches the global precision singleton, so it is
                      insensitive to it.
    oracle          : none. No third-party package, no network, no subprocess,
                      no RNG, no wall-clock value. The only filesystem use is a
                      `tempfile.TemporaryDirectory` that is deleted on exit.
    tolerance       : `math.isclose(rel_tol=1e-12)` on floats, exact equality on
                      counts, dtypes, axes and keys. 1e-12 is slack, not need:
                      every float golden here is a dyadic rational or a
                      correctly-rounded quotient of two exact integers, so the
                      observed values match to the last bit.

Run:
    PYTHONPATH=src python regressions/dataset-profiler-salvage.py
"""

from __future__ import annotations

import io
import math
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import torch

from molix import profiler as profiler_package
from molix.data.cache import PackedCache
from molix.data.dataset import CachedDataset
from molix.profiler import DatasetProfiler, DatasetResult
from molix.profiler import dataset as dataset_module

# ---------------------------------------------------------------------------
# The five samples, written out in full.
#
# Each is a linear chain: atoms on the x axis, bidirectional nearest-neighbour
# edges (source, target), so `edge_dist` is the spacing and every literal is a
# dyadic rational — exact in float32, exact in float64, and hand-checkable
# against `pos`. Chain of n atoms → n-1 bonds → 2(n-1) edges. The `edge_index`
# rows follow the repo convention: column 0 is the source, column 1 the target.
#
# The atom counts 3 / 4 / 5 / 6 / 2 are deliberately not monotone and the last
# record's atom count (2) coincides with its edge count, which is exactly the
# ambiguity `molix.data.cache._infer_schema_across` resolves by scanning all
# records: a single-record inference could classify `edge_index` as per-atom.
# ---------------------------------------------------------------------------

SAMPLES: list[dict[str, object]] = [
    {  # 0 — 3 atoms, spacings 1.0 / 1.5 → 4 edges
        "Z": torch.tensor([8, 1, 1], dtype=torch.long),
        "pos": torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.5, 0.0, 0.0]], dtype=torch.float32
        ),
        "edge_index": torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1]], dtype=torch.long),
        "edge_dist": torch.tensor([1.0, 1.0, 1.5, 1.5], dtype=torch.float32),
        "targets": {"U0": torch.tensor([-1.5], dtype=torch.float32)},
    },
    {  # 1 — 4 atoms, spacings 1.0 / 1.0 / 1.5 → 6 edges
        "Z": torch.tensor([6, 1, 1, 1], dtype=torch.long),
        "pos": torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.5, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        "edge_index": torch.tensor(
            [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]], dtype=torch.long
        ),
        "edge_dist": torch.tensor([1.0, 1.0, 1.0, 1.0, 1.5, 1.5], dtype=torch.float32),
        "targets": {"U0": torch.tensor([-0.5], dtype=torch.float32)},
    },
    {  # 2 — 5 atoms, spacings 1.5 / 1.0 / 1.0 / 1.5 → 8 edges
        "Z": torch.tensor([6, 6, 1, 1, 1], dtype=torch.long),
        "pos": torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [2.5, 0.0, 0.0],
                [3.5, 0.0, 0.0],
                [5.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        "edge_index": torch.tensor(
            [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3]], dtype=torch.long
        ),
        "edge_dist": torch.tensor([1.5, 1.5, 1.0, 1.0, 1.0, 1.0, 1.5, 1.5], dtype=torch.float32),
        "targets": {"U0": torch.tensor([0.25], dtype=torch.float32)},
    },
    {  # 3 — 6 atoms, spacings 1.0 ×4 / 1.5 → 10 edges (the largest record)
        "Z": torch.tensor([6, 6, 6, 1, 1, 1], dtype=torch.long),
        "pos": torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [5.5, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        "edge_index": torch.tensor(
            [
                [0, 1],
                [1, 0],
                [1, 2],
                [2, 1],
                [2, 3],
                [3, 2],
                [3, 4],
                [4, 3],
                [4, 5],
                [5, 4],
            ],
            dtype=torch.long,
        ),
        "edge_dist": torch.tensor(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 1.5], dtype=torch.float32
        ),
        "targets": {"U0": torch.tensor([2.0], dtype=torch.float32)},
    },
    {  # 4 — 2 atoms, spacing 1.25 → 2 edges (n_edges == n_atoms here)
        "Z": torch.tensor([1, 1], dtype=torch.long),
        "pos": torch.tensor([[0.0, 0.0, 0.0], [1.25, 0.0, 0.0]], dtype=torch.float32),
        "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        "edge_dist": torch.tensor([1.25, 1.25], dtype=torch.float32),
        "targets": {"U0": torch.tensor([3.75], dtype=torch.float32)},
    },
]

#: How many records the sampled slow path may read. Equal to `len(SAMPLES)`, so
#: the sampled and exact paths cover the same records — which is why
#: `counts_exact` has to be asserted separately from the numbers themselves.
N_SAMPLES = 5

#: Relative tolerance for the float goldens. See the docstring: slack, not need.
REL_TOL = 1e-12

# ---------------------------------------------------------------------------
# Goldens — analytic, derived above each literal.
# ---------------------------------------------------------------------------

#: `len(ds)`.
N_TOTAL = 5

#: (3 + 4 + 5 + 6 + 2) / 5 = 20 / 5.
ATOM_MEAN = 4.0

#: Population std (numpy default, ddof=0) of 3, 4, 5, 6, 2 about the mean 4:
#: (1 + 0 + 1 + 4 + 4) / 5 = 2, so √2. Pins the ddof convention of `ValueStat`.
ATOM_STD = 1.4142135623730951

#: (4 + 6 + 8 + 10 + 2) / 5 = 30 / 5. Non-`None` here: the cache has an
#: `edge_ptr`, which is the branch the "no NeighborList" warning path forgoes.
EDGE_MEAN = 6.0

#: Record 3, the 6-atom chain.
MAX_ATOMS = 6

#: Record 3 again: 2 × (6 - 1).
MAX_EDGES = 10

#: 30 total edges / 20 total atoms. Bidirectional edges, so this is the mean
#: neighbour count per atom — the MACE / Allegro normalisation constant.
AVG_NUM_NEIGHBORS = 1.5

#: Σ numel × element_size per record, over the five records:
#:   n=3, e=4  → 3·8 + 3·3·4 + 4·2·8 + 4·4 + 1·4 = 144 B
#:   n=4, e=6  → 32 + 48 +  96 + 24 + 4           = 204 B
#:   n=5, e=8  → 40 + 60 + 128 + 32 + 4           = 264 B
#:   n=6, e=10 → 48 + 72 + 160 + 40 + 4           = 324 B
#:   n=2, e=2  → 16 + 24 +  32 +  8 + 4           =  84 B
#: 1020 B over 5 records.
SAMPLE_BYTES_MEAN = 204.0

#: 204.0 B/record × 5 records / 1e6.
EST_TOTAL_MB = 0.00102

#: The packed `payload["schema"]`, as `(key, axis, dtype, extra_shape)` sorted
#: by key. `extra_shape` is the shape after the packing axis for atom / edge
#: fields — `(3,)` for `pos` `(N, 3)`, `()` for `Z` `(N,)` — and the full
#: per-record shape for a graph field, hence `(1,)` for `targets.U0`.
FIELDS: tuple[tuple[str, str, str, tuple[int, ...]], ...] = (
    ("Z", "atom", "int64", ()),
    ("edge_dist", "edge", "float32", ()),
    ("edge_index", "edge", "int64", (2,)),
    ("pos", "atom", "float32", (3,)),
    ("targets.U0", "graph", "float32", (1,)),
)

#: The one label column: `targets.U0`, dotted exactly as the packed schema
#: names it (nested `{"targets": {"U0": ...}}` in the raw sample).
TARGET_KEY = "targets.U0"

#: (-1.5) + (-0.5) + 0.25 + 2.0 + 3.75 = 4.0, over 5 records.
TARGET_MEAN = 0.8

#: Record 0 / record 4.
TARGET_MIN = -1.5
TARGET_MAX = 3.75

#: No `nan` / `inf` among the five literals, so the non-finite warning path
#: stays silent and `warnings` is empty overall.
TARGET_NONFINITE = 0

#: Section labels `DatasetResult.print_report` must emit. The atom-count skew
#: is p95/p50 = 5.8/4 = 1.45× here, well under the 3× warning threshold, so a
#: clean run prints these and no `[WARN]` line at all.
REPORT_SECTIONS = ("Size", "Access", "Footprint", "Fields", "Targets")


# ---------------------------------------------------------------------------
# Checking
# ---------------------------------------------------------------------------


class Checker:
    """Collects every deviation so one run reports all failures, not the first."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def _row(self, name: str, got: object, ok: bool) -> None:
        print(f"  {name:<38} {got!s:<34} {'ok' if ok else 'FAILED'}")

    def exact(self, name: str, got: object, want: object) -> None:
        """Assert a count, string, dtype or key list — no tolerance applies."""
        ok = got == want
        if not ok:
            self.failures.append(f"{name}: got {got!r}, want {want!r}")
        self._row(name, got, ok)

    def close(self, name: str, got: float, want: float) -> None:
        """Assert a float golden within :data:`REL_TOL`."""
        ok = math.isclose(got, want, rel_tol=REL_TOL, abs_tol=0.0)
        if not ok:
            self.failures.append(f"{name}: got {got!r}, want {want!r} (rel_tol={REL_TOL})")
        self._row(name, got, ok)

    def truth(self, name: str, holds: bool, message: str) -> None:
        """Assert a boolean contract (a path was taken, a name is re-exported, ...)."""
        if not holds:
            self.failures.append(f"{name}: {message}")
        self._row(name, holds, holds)


def profile_samples(directory: Path) -> DatasetResult:
    """Run the whole user-facing scenario inside *directory*.

    Args:
        directory: Scratch directory for the cache file; caller owns its
            lifetime. Nothing about the path enters a golden.

    Returns:
        The `DatasetResult` for the five literal samples, read back through a
        real `CachedDataset` — i.e. after a genuine pack / save / load round
        trip, not from the in-memory dicts.
    """
    sink = directory / "dataset-profiler-salvage.pt"
    PackedCache(sink).save(SAMPLES)
    dataset = CachedDataset(sink)
    return DatasetProfiler(n_samples=N_SAMPLES).run(dataset)


def check_reexport(checker: Checker, result: DatasetResult) -> None:
    """The documented package-level import path resolves to the real objects.

    `molix.profiler.__init__` re-exports both names; the unit suite imports
    from `molix.profiler.dataset` directly, so nothing but this file would
    notice the package-level path breaking.

    Args:
        checker: Failure collector.
        result: A result produced by the package-level profiler class.
    """
    print("Package re-export (from molix.profiler import DatasetProfiler, DatasetResult)")
    checker.truth(
        "reexport.profiler_is_submodule_class",
        DatasetProfiler is dataset_module.DatasetProfiler,
        "molix.profiler.DatasetProfiler is not molix.profiler.dataset.DatasetProfiler",
    )
    checker.truth(
        "reexport.result_is_submodule_class",
        DatasetResult is dataset_module.DatasetResult,
        "molix.profiler.DatasetResult is not molix.profiler.dataset.DatasetResult",
    )
    checker.truth(
        "reexport.run_returns_that_result",
        isinstance(result, DatasetResult),
        f"run() returned {type(result).__name__}, not the re-exported DatasetResult",
    )
    exported = list(profiler_package.__all__)
    checker.truth(
        "reexport.both_names_in_all",
        {"DatasetProfiler", "DatasetResult"} <= set(exported),
        f"molix.profiler.__all__ is missing "
        f"{sorted({'DatasetProfiler', 'DatasetResult'} - set(exported))}",
    )
    checker.truth(
        "reexport.all_is_alphabetised",
        exported == sorted(exported),
        f"molix.profiler.__all__ is out of order: {exported}",
    )


def check_sizes(checker: Checker, result: DatasetResult) -> None:
    """Size statistics, and that they came from the packed pointers.

    Args:
        checker: Failure collector.
        result: The profiled result.
    """
    print("\nSizes (exact packed-pointer fast path)")
    checker.exact("size.n_total", result.n_total, N_TOTAL)
    checker.exact("size.n_sampled", result.n_sampled, N_SAMPLES)
    checker.truth(
        "size.counts_exact",
        result.counts_exact,
        "size stats fell back to the sampled path — atom_ptr / edge_ptr were "
        "not read, so the numbers below cover only the sampled records",
    )
    checker.close("size.atom_stats.mean", result.atom_stats.mean, ATOM_MEAN)
    checker.close("size.atom_stats.std", result.atom_stats.std, ATOM_STD)
    checker.truth(
        "size.edge_stats_present",
        result.edge_stats is not None,
        "edge_stats is None — the cache lost its edge_ptr",
    )
    if result.edge_stats is not None:
        checker.close("size.edge_stats.mean", result.edge_stats.mean, EDGE_MEAN)
    checker.exact("size.max_atoms", result.max_atoms, MAX_ATOMS)
    checker.exact("size.max_edges", result.max_edges, MAX_EDGES)
    checker.close("size.avg_num_neighbors", result.avg_num_neighbors, AVG_NUM_NEIGHBORS)


def check_footprint(checker: Checker, result: DatasetResult) -> None:
    """Per-record byte footprint and the full-materialisation extrapolation.

    Args:
        checker: Failure collector.
        result: The profiled result.
    """
    print("\nFootprint (leaf tensors, analytic — the dtype tripwire)")
    checker.close("footprint.sample_bytes.mean", result.sample_bytes.mean, SAMPLE_BYTES_MEAN)
    checker.close("footprint.est_total_mb", result.est_total_mb, EST_TOTAL_MB)


def check_fields(checker: Checker, result: DatasetResult) -> None:
    """Field layout, read off the packed schema rather than inferred.

    Args:
        checker: Failure collector.
        result: The profiled result.
    """
    print("\nFields (packed payload['schema'], sorted by key)")
    checker.truth(
        "fields.fields_exact",
        result.fields_exact,
        "field layout was inferred from the sampled records — the packed "
        "schema was not reachable, so axes and trailing shapes are guesses",
    )
    observed = sorted(
        (f.key, f.axis, str(f.dtype).removeprefix("torch."), tuple(f.extra_shape))
        for f in result.fields
    )
    checker.exact("fields.keys", tuple(row[0] for row in observed), tuple(f[0] for f in FIELDS))
    for want in FIELDS:
        got = next((row for row in observed if row[0] == want[0]), None)
        checker.exact(f"fields.{want[0]}", got, want)


def check_targets(checker: Checker, result: DatasetResult) -> None:
    """Label statistics for the single `targets.U0` column.

    Args:
        checker: Failure collector.
        result: The profiled result.
    """
    print("\nTargets (targets.U0)")
    checker.exact("targets.keys", tuple(t.key for t in result.targets), (TARGET_KEY,))
    target = next((t for t in result.targets if t.key == TARGET_KEY), None)
    if target is None:
        checker.truth("targets.U0_present", False, f"no TargetStat for {TARGET_KEY!r}")
        return
    checker.close("targets.U0.mean", target.stat.mean, TARGET_MEAN)
    checker.close("targets.U0.min", target.min, TARGET_MIN)
    checker.close("targets.U0.max", target.max, TARGET_MAX)
    checker.exact("targets.U0.n_nonfinite", target.n_nonfinite, TARGET_NONFINITE)


def check_report(checker: Checker, result: DatasetResult) -> None:
    """`print_report` emits every section, and this clean run warns about nothing.

    The report text is captured rather than printed: it carries wall-clock
    latencies, and this file's own output has to stay byte-identical between
    runs. Only the section labels and the absence of `[WARN]` are asserted —
    both deterministic.

    Args:
        checker: Failure collector.
        result: The profiled result.
    """
    print("\nReport (captured, not printed — it carries timings)")
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        result.print_report()
    text = buffer.getvalue()
    missing = [section for section in REPORT_SECTIONS if f"  {section}" not in text]
    checker.truth(
        "report.sections_present",
        not missing,
        f"print_report omitted section(s) {missing}",
    )
    checker.truth(
        "report.rule_present",
        "─" * 72 in text,
        "print_report lost its 72-char section rule",
    )
    checker.exact("report.warnings", tuple(result.warnings), ())
    checker.truth(
        "report.no_warn_line",
        "[WARN]" not in text,
        "print_report emitted a [WARN] line on a cache that is exact in every "
        "section and has no non-finite label",
    )


def main() -> int:
    """Profile the five literal samples and compare every non-timing field."""
    checker = Checker()
    with tempfile.TemporaryDirectory(prefix="molnex-dataset-profiler-") as directory:
        result = profile_samples(Path(directory))

    check_reexport(checker, result)
    check_sizes(checker, result)
    check_footprint(checker, result)
    check_fields(checker, result)
    check_targets(checker, result)
    check_report(checker, result)

    if checker.failures:
        print("\nFAILED — DatasetProfiler no longer matches the analytic goldens:")
        for failure in checker.failures:
            print(f"  {failure}")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
