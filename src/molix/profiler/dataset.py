"""Dataset characterisation profiler.

Answers *"what is in my data, and what does one sample cost?"* for any object
yielding **flat sample dicts** (the raw-sample tier of the two-tier data
contract): :class:`~molix.data.dataset.CachedDataset` /
:class:`~molix.data.dataset.MmapDataset` /
:class:`~molix.data.dataset.SubsetDataset`, a
:class:`~molix.profiler.mock.MockSource`, or a plain ``Sequence[dict]``.
Complements :class:`~molix.profiler.dataloader.DataLoaderProfiler`, which
times batch *throughput* rather than the data itself.

:meth:`DatasetProfiler.run` merges two paths:

1. **Exact fast path** — when the object exposes the packed-pointer
   properties (``atom_counts`` / ``edge_counts`` / ``avg_num_neighbors`` /
   ``max_atoms`` / ``max_edges``), size statistics are read straight off the
   ``atom_ptr`` / ``edge_ptr`` cumsums: **every** record, zero unpacking.
   The result then carries ``counts_exact=True``.
2. **Sampled slow path** — only for what genuinely needs per-sample reads:
   cold / steady-state ``__getitem__`` latency, per-sample byte footprint,
   and target-value statistics. ``n_samples`` records are visited on a
   ``stride``. Objects without packed pointers get their size statistics
   here too, and report ``counts_exact=False``.

Field layout comes from the packed ``payload["schema"]`` when available
(``fields_exact=True``), else it is inferred from the sampled records.

Diagnostics — non-finite targets, atom-count skew, a missing ``edge_ptr``,
any inexact section — are emitted as ``[WARN]`` lines and **never** raise;
only degenerate *inputs* raise :class:`ValueError`. No unit conversion is
performed and no units are guessed: positions, energies and every target
are printed exactly as the dataset stores them.

Example::

    from molix.profiler import DatasetProfiler

    result = DatasetProfiler(n_samples=500).run(train_dataset)
    result.print_report()
    result.avg_num_neighbors  # exact, straight off the packed pointers
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

from molix.profiler._utils import Timer, TimingStat, ValueStat, _fmt_table, sample_counts

#: Atom-count skew — ``p95 / p50`` over the records — above which the report
#: warns that padded batches will waste compute. Documented as the "3×"
#: threshold in ``docs/molix/user-guide/profiling.md``.
_ATOM_SKEW_WARN_RATIO = 3.0

# ---------------------------------------------------------------------------
# Field / target descriptions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldSpec:
    """Packed layout of one sample key, as shown in the report.

    This is the **report view** of one entry of a packed cache's
    ``payload["schema"]`` — not to be confused with
    :class:`molix.data.collate.TargetSchema`, which is a collate-time
    routing rule. ``FieldSpec`` describes storage, ``TargetSchema``
    describes destination.

    Attributes:
        key: Dotted key path into the flat sample, e.g. ``"targets.U0"``.
        axis: Packing axis — ``"atom"``, ``"edge"``, ``"graph"`` or
            ``"scalar"``, matching :func:`molix.data.cache._infer_schema_across`.
        dtype: ``torch.dtype`` of the tensor, or the Python type name for a
            non-tensor scalar.
        extra_shape: Trailing shape after the packing axis — ``(3,)`` for
            ``pos`` ``(N, 3)``, ``()`` for ``Z`` ``(N,)``. ``"graph"`` fields
            keep their full per-sample shape; ``"scalar"`` fields use ``()``.
    """

    key: str
    axis: str
    dtype: torch.dtype | str
    extra_shape: tuple[int, ...]

    @classmethod
    def from_packed(cls, key: str, spec: tuple) -> "FieldSpec":
        """Build the report view of one packed ``payload["schema"]`` entry.

        Args:
            key: Dotted key the entry describes.
            spec: Schema entry — ``(axis, dtype, extra_shape)`` for packed
                tensors, or the two-element ``("scalar", type_name)`` form.

        Returns:
            The corresponding :class:`FieldSpec`.
        """
        return cls(
            key=key,
            axis=spec[0],
            dtype=spec[1],
            extra_shape=tuple(spec[2]) if len(spec) > 2 else (),
        )

    @classmethod
    def from_sample(cls, key: str, value: object, n_atoms: int, n_edges: int) -> "FieldSpec":
        """Infer the layout of one leaf from a single sampled record.

        Best-effort fallback for datasets with no packed view; classification
        mirrors :func:`molix.data.cache._infer_schema_across` (leading
        dimension tracks atoms, else edges, else the field is per-graph) but
        sees one record instead of all of them — the caller reports
        ``fields_exact=False``.

        Args:
            key: Dotted key of the leaf.
            value: The leaf value, normally a tensor.
            n_atoms: Atom count of the record this leaf came from.
            n_edges: Edge count of the record this leaf came from.

        Returns:
            The inferred :class:`FieldSpec`.
        """
        if not isinstance(value, torch.Tensor):
            return cls(key=key, axis="scalar", dtype=type(value).__name__, extra_shape=())
        if value.ndim and int(value.shape[0]) == n_atoms:
            return cls(key=key, axis="atom", dtype=value.dtype, extra_shape=tuple(value.shape[1:]))
        if value.ndim and int(value.shape[0]) == n_edges:
            return cls(key=key, axis="edge", dtype=value.dtype, extra_shape=tuple(value.shape[1:]))
        return cls(key=key, axis="graph", dtype=value.dtype, extra_shape=tuple(value.shape))


@dataclass(frozen=True)
class TargetStat:
    """Numeric distribution of one label column across the sampled records.

    Unrelated to :class:`molix.data.collate.TargetSchema`: this is a
    *measurement* of the label values, whereas ``TargetSchema`` is the rule
    that routes labels into batch namespaces at collate time.

    Attributes:
        key: Dotted key of the label, e.g. ``"targets.U0"``.
        stat: Mean / std / p50 / p95 over the **finite** sampled values.
        min: Smallest finite value seen (``nan`` if none was).
        max: Largest finite value seen (``nan`` if none was).
        n_nonfinite: How many sampled values were ``nan`` or ``inf``.
    """

    key: str
    stat: ValueStat
    min: float
    max: float
    n_nonfinite: int

    @classmethod
    def from_values(cls, key: str, values: list[float]) -> "TargetStat":
        """Summarise one label column, tolerating non-finite entries.

        Args:
            key: Dotted key of the label column.
            values: Sampled scalar values, possibly containing ``nan`` /
                ``inf`` — those are counted, then excluded from the moments
                so a single bad row cannot poison the whole column.

        Returns:
            The corresponding :class:`TargetStat`.
        """
        finite = [v for v in values if math.isfinite(v)]
        empty = ValueStat(mean=math.nan, std=math.nan, p50=math.nan, p95=math.nan)
        return cls(
            key=key,
            stat=ValueStat.from_list(finite) if finite else empty,
            min=min(finite) if finite else math.nan,
            max=max(finite) if finite else math.nan,
            n_nonfinite=len(values) - len(finite),
        )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class DatasetResult:
    """Profiling results for a dataset.

    Attributes:
        n_total: Records in the dataset (``len(data)``).
        n_sampled: Records actually read by the slow path.
        counts_exact: Size statistics came from the packed pointers (all
            records) rather than the sampled subset.
        fields_exact: Field layout came from the packed ``payload["schema"]``
            rather than sampled inference.
        atom_stats: Atoms per record.
        edge_stats: Edges per record, or ``None`` when the dataset carries no
            edges (see :attr:`warnings`).
        max_atoms: Largest single-record atom count.
        max_edges: Largest single-record edge count (``0`` without edges).
        avg_num_neighbors: ``total_edges / total_atoms``, the MACE/Allegro
            normalisation constant. Exact iff :attr:`counts_exact`.
        access_ms: Steady-state ``__getitem__`` latency.
        cold_access_ms: Latency of the very first access (page-in / mmap
            fault included).
        sample_bytes: Leaf-tensor footprint of one record, in bytes.
        est_total_mb: ``sample_bytes.mean * n_total`` in MB — what a full
            in-RAM materialisation would cost.
        fields: Per-key packed layout.
        targets: Per-label value statistics.
        task_states: ``{task_name: state}`` restored from the cache, i.e.
            which fitted pipeline tasks (e.g. ``AtomicDress``) baked it.
        warnings: Diagnostic lines rendered as ``[WARN]`` by
            :meth:`print_report`.
        data_description: Human-readable description of the input.
    """

    n_total: int
    n_sampled: int
    counts_exact: bool
    fields_exact: bool
    atom_stats: ValueStat
    edge_stats: ValueStat | None
    max_atoms: int
    max_edges: int
    avg_num_neighbors: float
    access_ms: TimingStat
    cold_access_ms: float
    sample_bytes: ValueStat
    est_total_mb: float
    fields: list[FieldSpec] = field(default_factory=list)
    targets: list[TargetStat] = field(default_factory=list)
    task_states: dict[str, object] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    data_description: str = ""

    def print_report(self) -> None:
        """Print a sectioned dataset characterisation report to stdout."""
        print(f"\nDataset Profile  (n_total={self.n_total:,}, n_sampled={self.n_sampled:,})")
        print(f"Data: {self.data_description}")
        print("─" * 72)
        self._print_size()
        self._print_access()
        self._print_footprint()
        self._print_fields()
        self._print_targets()
        self._print_task_states()
        print("─" * 72)
        self._print_warnings()

    def _print_size(self) -> None:
        """Print the ``Size`` section: atom / edge counts per record."""
        print("  Size")
        size_rows = [
            {
                "Quantity": "atoms / sample",
                "mean": f"{self.atom_stats.mean:.2f}",
                "std": f"{self.atom_stats.std:.2f}",
                "p50": f"{self.atom_stats.p50:.0f}",
                "p95": f"{self.atom_stats.p95:.0f}",
                "max": str(self.max_atoms),
            }
        ]
        if self.edge_stats is not None:
            size_rows.append(
                {
                    "Quantity": "edges / sample",
                    "mean": f"{self.edge_stats.mean:.2f}",
                    "std": f"{self.edge_stats.std:.2f}",
                    "p50": f"{self.edge_stats.p50:.0f}",
                    "p95": f"{self.edge_stats.p95:.0f}",
                    "max": str(self.max_edges),
                }
            )
        print(_fmt_table(size_rows, ["Quantity", "mean", "std", "p50", "p95", "max"], col_width=8))
        print(
            f"    avg_num_neighbors (E/N): {self.avg_num_neighbors:.4f}"
            f"   (exact={self.counts_exact})"
        )
        print()

    def _print_access(self) -> None:
        """Print the ``Access`` section: steady-state and cold ``__getitem__`` latency."""
        print("  Access")
        s = self.access_ms
        access_rows = [
            {
                "Metric": "__getitem__",
                "mean(ms)": f"{s.mean_ms:.4f}",
                "std(ms)": f"{s.std_ms:.4f}",
                "p50(ms)": f"{s.p50_ms:.4f}",
                "p95(ms)": f"{s.p95_ms:.4f}",
                "cold(ms)": f"{self.cold_access_ms:.4f}",
            }
        ]
        cols = ["Metric", "mean(ms)", "std(ms)", "p50(ms)", "p95(ms)", "cold(ms)"]
        print(_fmt_table(access_rows, cols, col_width=8))
        print()

    def _print_footprint(self) -> None:
        """Print the ``Footprint`` section: per-record bytes and the full-set estimate."""
        print("  Footprint")
        print(
            f"    {self.sample_bytes.mean / 1e6:.6f} MB / sample"
            f"   (~{self.est_total_mb:,.1f} MB for all {self.n_total:,} records)"
        )
        print()

    def _print_fields(self) -> None:
        """Print the ``Fields`` section: packed layout of every sample key."""
        print(f"  Fields  (exact={self.fields_exact})")
        field_rows = [
            {
                "Key": f.key,
                "axis": f.axis,
                "dtype": str(f.dtype).removeprefix("torch."),
                "extra_shape": str(tuple(f.extra_shape)),
            }
            for f in self.fields
        ]
        print(_fmt_table(field_rows, ["Key", "axis", "dtype", "extra_shape"], col_width=10))
        print()

    def _print_targets(self) -> None:
        """Print the ``Targets`` section: value distribution of every label column."""
        print("  Targets")
        if self.targets:
            target_rows = [
                {
                    "Target": t.key,
                    "mean": f"{t.stat.mean:.4g}",
                    "std": f"{t.stat.std:.4g}",
                    "min": f"{t.min:.4g}",
                    "max": f"{t.max:.4g}",
                    "nonfinite": str(t.n_nonfinite),
                }
                for t in self.targets
            ]
            cols = ["Target", "mean", "std", "min", "max", "nonfinite"]
            print(_fmt_table(target_rows, cols, col_width=8))
        else:
            print("    (none)")
        print()

    def _print_task_states(self) -> None:
        """Print the fitted ``DatasetTask`` names, when the cache restored any."""
        if self.task_states:
            print(f"  Fitted task states: {', '.join(sorted(self.task_states))}")
            print()

    def _print_warnings(self) -> None:
        """Print the diagnostics collected during the run as ``[WARN]`` lines."""
        for message in self.warnings:
            print(f"  [WARN] {message}")
        print()


# ---------------------------------------------------------------------------
# Sample helpers
# ---------------------------------------------------------------------------


def _flatten_leaves(sample: Mapping[str, object], prefix: str = "") -> dict[str, object]:
    """Flatten a nested sample dict to dotted keys, read-only and never raising.

    Deliberately diverges from :func:`molix.data.cache._flatten`, the
    packing-time flattener: that one is a **validator** and raises
    :class:`ValueError` when a sample key collides with a reserved
    packed-cache key (``schema``, ``atom_ptr``, …). This walker performs
    **no** reserved-key validation and raises nothing, because the profiler
    must survive any malformed sample — diagnostics are ``[WARN]`` lines,
    never exceptions.

    Args:
        sample: Sample dict, possibly with nested sub-dicts (``targets``).
        prefix: Dotted prefix accumulated during recursion.

    Returns:
        ``{dotted_key: leaf}``; leaf values are returned by reference, not
        copied.
    """
    leaves: dict[str, object] = {}
    for name, value in sample.items():
        key = f"{prefix}{name}"
        if isinstance(value, Mapping):
            leaves.update(_flatten_leaves(value, prefix=f"{key}."))
        else:
            leaves[key] = value
    return leaves


def _sample_bytes(leaves: Mapping[str, object]) -> int:
    """Total leaf-tensor footprint of one sample, in bytes.

    Args:
        leaves: Flattened sample, as returned by :func:`_flatten_leaves`.

    Returns:
        ``Σ numel() * element_size()`` over tensor leaves; non-tensor leaves
        contribute nothing (their Python overhead is not the dataset's cost).
    """
    return sum(
        value.numel() * value.element_size()
        for value in leaves.values()
        if isinstance(value, torch.Tensor)
    )


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SampledPass:
    """What one pass over the sampled records measured.

    Everything here comes from the slow path — ``len(access_ms)`` records were
    actually read. The size counts are only *used* when the packed pointers of
    the exact fast path are unavailable.

    Attributes:
        cold_access_ms: Latency of the very first access, page-in included.
        access_ms: Steady-state per-record ``__getitem__`` latency, in ms.
        byte_counts: Leaf-tensor footprint of each visited record, in bytes.
        atom_counts: Atoms per visited record.
        edge_counts: Edges per visited record.
        fields: ``{dotted_key: FieldSpec}`` inferred from the first record that
            carried each key.
        target_values: ``{dotted_key: values}`` for every scalar ``targets.*``
            leaf seen.
    """

    cold_access_ms: float
    access_ms: list[float]
    byte_counts: list[int]
    atom_counts: list[int]
    edge_counts: list[int]
    fields: dict[str, FieldSpec]
    target_values: dict[str, list[float]]


@dataclass(frozen=True)
class _SizeStats:
    """Record-size statistics, from the packed pointers or the sampled records.

    Attributes:
        counts_exact: Statistics cover every record (packed pointers) rather
            than the sampled subset.
        atom_stats: Atoms per record.
        edge_stats: Edges per record, or ``None`` when there are no edges.
        max_atoms: Largest single-record atom count.
        max_edges: Largest single-record edge count (``0`` without edges).
        avg_num_neighbors: ``total_edges / total_atoms``.
    """

    counts_exact: bool
    atom_stats: ValueStat
    edge_stats: ValueStat | None
    max_atoms: int
    max_edges: int
    avg_num_neighbors: float


class DatasetProfiler:
    """Profile a dataset's sizes, access cost, field layout and labels.

    Args:
        n_samples: Upper bound on records read by the sampled slow path.
        stride: Step between inspected indices — ``stride > 1`` spreads the
            sample across an ordered dataset instead of reading a prefix.
        n_warmup: Accesses discarded before the timed ones, so page-in and
            allocator warm-up do not land in :attr:`DatasetResult.access_ms`
            (they show up in ``cold_access_ms`` instead).

    Raises:
        ValueError: ``n_samples`` or ``stride`` is not positive.

    Example::

        profiler = DatasetProfiler(n_samples=500, stride=4)
        profiler.run(cached_dataset).print_report()
        profiler.run([sample_a, sample_b]).print_report()   # plain Sequence
    """

    def __init__(self, n_samples: int = 200, stride: int = 1, n_warmup: int = 3) -> None:
        if n_samples <= 0:
            raise ValueError(
                f"n_samples must be > 0, got {n_samples}. Pass the number of records to "
                "inspect (default 200); the sampled path needs at least one read."
            )
        if stride <= 0:
            raise ValueError(
                f"stride must be > 0, got {stride}. Pass stride=1 (default) to walk "
                "consecutive records, or a larger step to spread the sample out."
            )
        self.n_samples = n_samples
        self.stride = stride
        self.n_warmup = max(0, n_warmup)

    def run(self, data: object) -> DatasetResult:
        """Profile *data* and return the grouped result.

        Args:
            data: Any object with ``__len__`` and ``__getitem__`` returning
                flat sample dicts. Packed-cache-backed datasets additionally
                get exact, whole-dataset size statistics and an exact field
                layout for free.

        Returns:
            :class:`DatasetResult`. Everything diagnostic — non-finite
            labels, size skew, missing ``edge_ptr``, inexact sections — lands
            in :attr:`DatasetResult.warnings`, not in an exception.

        Raises:
            ValueError: *data* is empty, or exposes no ``__len__`` /
                ``__getitem__``.
        """
        n_total = self._require_indexable(data)
        indices = list(range(0, n_total, self.stride))[: self.n_samples]

        sampled = self._sampled_pass(data, indices)
        sizes, size_warnings = self._size_stats(
            data, sampled.atom_counts, sampled.edge_counts, len(indices)
        )
        fields, fields_exact, field_warnings = self._field_layout(data, sampled.fields)
        targets, target_warnings = self._target_diagnostics(sampled.target_values, sizes.atom_stats)

        bytes_stat = ValueStat.from_list(sampled.byte_counts)
        return DatasetResult(
            n_total=n_total,
            n_sampled=len(indices),
            counts_exact=sizes.counts_exact,
            fields_exact=fields_exact,
            atom_stats=sizes.atom_stats,
            edge_stats=sizes.edge_stats,
            max_atoms=sizes.max_atoms,
            max_edges=sizes.max_edges,
            avg_num_neighbors=sizes.avg_num_neighbors,
            access_ms=TimingStat.from_list(sampled.access_ms),
            cold_access_ms=sampled.cold_access_ms,
            sample_bytes=bytes_stat,
            est_total_mb=bytes_stat.mean * n_total / 1e6,
            fields=fields,
            targets=targets,
            task_states=dict(data.stats()) if hasattr(data, "stats") else {},
            warnings=[*size_warnings, *field_warnings, *target_warnings],
            data_description=getattr(data, "describe", lambda: type(data).__name__)(),
        )

    # -- Stages -------------------------------------------------------------------

    def _sampled_pass(self, data: object, indices: list[int]) -> _SampledPass:
        """Read every index in *indices* once, measuring the sampled slow path.

        Args:
            data: The dataset being profiled.
            indices: Record indices to visit, in order. ``indices[0]`` is read
                once beforehand for the cold measurement, then
                :attr:`n_warmup` further reads are discarded, so page-in does
                not land in the steady-state latency.

        Returns:
            The measurements of the pass — latency, footprint, sampled sizes,
            inferred field layout and label values.
        """
        with Timer() as timer:
            data[indices[0]]
        cold_access_ms = timer.elapsed * 1000.0

        for i in range(self.n_warmup):
            data[indices[i % len(indices)]]

        access_ms: list[float] = []
        byte_counts: list[int] = []
        sampled_atoms: list[int] = []
        sampled_edges: list[int] = []
        sampled_fields: dict[str, FieldSpec] = {}
        target_values: dict[str, list[float]] = {}
        for idx in indices:
            with Timer() as timer:
                sample = data[idx]
            access_ms.append(timer.elapsed * 1000.0)

            n_atoms, n_edges = sample_counts(sample)
            sampled_atoms.append(n_atoms)
            sampled_edges.append(n_edges)

            leaves = _flatten_leaves(sample)
            byte_counts.append(_sample_bytes(leaves))
            for key, value in leaves.items():
                if key not in sampled_fields:
                    sampled_fields[key] = FieldSpec.from_sample(key, value, n_atoms, n_edges)
                if not key.startswith("targets."):
                    continue
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    target_values.setdefault(key, []).append(float(value.item()))
                elif isinstance(value, (int, float)) and not isinstance(value, bool):
                    target_values.setdefault(key, []).append(float(value))

        return _SampledPass(
            cold_access_ms=cold_access_ms,
            access_ms=access_ms,
            byte_counts=byte_counts,
            atom_counts=sampled_atoms,
            edge_counts=sampled_edges,
            fields=sampled_fields,
            target_values=target_values,
        )

    def _size_stats(
        self,
        data: object,
        sampled_atoms: list[int],
        sampled_edges: list[int],
        n_sampled: int,
    ) -> tuple[_SizeStats, list[str]]:
        """Summarise record sizes, preferring the exact packed-pointer fast path.

        Args:
            data: The dataset being profiled.
            sampled_atoms: Atoms per sampled record, used only as the fallback.
            sampled_edges: Edges per sampled record, used only as the fallback.
            n_sampled: How many records the sampled pass read, for the message.

        Returns:
            ``(stats, warnings)``. ``warnings`` carries any missing-pointer
            reason plus, on the fallback, the ``counts_exact=False`` notice.
        """
        warnings: list[str] = []
        atom_counts, atom_warning = self._packed_counts(data, "atom_counts")
        edge_counts, edge_warning = self._packed_counts(data, "edge_counts")
        warnings.extend(w for w in (atom_warning, edge_warning) if w is not None)
        counts_exact = atom_counts is not None

        if atom_counts is not None:
            atom_stats = ValueStat.from_list(atom_counts.tolist())
            edge_stats = None if edge_counts is None else ValueStat.from_list(edge_counts.tolist())
            max_atoms = int(getattr(data, "max_atoms", 0))
            max_edges = int(getattr(data, "max_edges", 0)) if edge_counts is not None else 0
            avg_num_neighbors = float(getattr(data, "avg_num_neighbors", 0.0))
        else:
            atom_stats = ValueStat.from_list(sampled_atoms)
            edge_stats = ValueStat.from_list(sampled_edges) if any(sampled_edges) else None
            max_atoms = max(sampled_atoms, default=0)
            max_edges = max(sampled_edges, default=0)
            total_atoms = sum(sampled_atoms)
            avg_num_neighbors = sum(sampled_edges) / total_atoms if total_atoms else 0.0
            warnings.append(
                f"size stats estimated from {n_sampled} sampled records (counts_exact=False)"
                " — a packed-cache-backed dataset would give exact whole-dataset counts."
            )

        stats = _SizeStats(
            counts_exact=counts_exact,
            atom_stats=atom_stats,
            edge_stats=edge_stats,
            max_atoms=max_atoms,
            max_edges=max_edges,
            avg_num_neighbors=avg_num_neighbors,
        )
        return stats, warnings

    def _field_layout(
        self, data: object, sampled_fields: Mapping[str, FieldSpec]
    ) -> tuple[list[FieldSpec], bool, list[str]]:
        """Describe the packed layout of every key, exactly where possible.

        Args:
            data: The dataset being profiled.
            sampled_fields: Layout inferred during the sampled pass, used only
                when *data* exposes no packed ``payload["schema"]``.

        Returns:
            ``(fields, fields_exact, warnings)``. ``warnings`` carries the
            inference notice when the packed schema was unreachable.
        """
        schema = self._packed_schema(data)
        if schema is not None:
            fields = [FieldSpec.from_packed(key, spec) for key, spec in schema.items()]
            return fields, True, []

        fields = [sampled_fields[key] for key in sorted(sampled_fields)]
        warning = (
            "field layout inferred from the sampled records (fields_exact=False)"
            " — trailing shapes and axes may differ on unsampled records."
        )
        return fields, False, [warning]

    def _target_diagnostics(
        self, target_values: Mapping[str, list[float]], atom_stats: ValueStat
    ) -> tuple[list[TargetStat], list[str]]:
        """Summarise the label columns and flag distribution problems.

        Args:
            target_values: ``{dotted_key: values}`` collected by the sampled pass.
            atom_stats: Record-size statistics, checked for padding-wasting skew.

        Returns:
            ``(targets, warnings)``. ``warnings`` covers non-finite label
            values and an atom-count skew above :data:`_ATOM_SKEW_WARN_RATIO`.
        """
        targets = [
            TargetStat.from_values(key, values) for key, values in sorted(target_values.items())
        ]
        warnings = [
            f"target {t.key!r} has {t.n_nonfinite} non-finite value(s) in the sample"
            for t in targets
            if t.n_nonfinite
        ]
        if atom_stats.p50 > 0:
            skew = atom_stats.p95 / atom_stats.p50
            if skew > _ATOM_SKEW_WARN_RATIO:
                warnings.append(
                    f"atom-count skew p95/p50 = {skew:.1f}x"
                    " — padded batches will waste compute; consider a token-budget sampler."
                )
        return targets, warnings

    # -- Helpers ----------------------------------------------------------------

    def _require_indexable(self, data: object) -> int:
        """Return ``len(data)``, rejecting inputs the sampled path cannot read.

        Args:
            data: Candidate dataset.

        Returns:
            Number of records.

        Raises:
            ValueError: *data* is not a non-empty indexable sequence.
        """
        remedy = (
            "Point it at a prepared cache (PipelineSpec.run(...) → CachedDataset) "
            "or a non-empty Sequence[dict]."
        )
        if not hasattr(data, "__len__") or not hasattr(data, "__getitem__"):
            raise ValueError(
                f"{type(data).__name__} exposes no __len__ / __getitem__, so no sample "
                f"can be read. {remedy}"
            )
        n_total = len(data)
        if n_total == 0:
            raise ValueError(f"Cannot profile an empty dataset ({type(data).__name__}). {remedy}")
        return n_total

    def _packed_counts(self, data: object, attr: str) -> tuple[torch.Tensor | None, str | None]:
        """Read an exact per-record count vector off a packed-pointer property.

        Args:
            data: The dataset being profiled.
            attr: Property name — ``"atom_counts"`` or ``"edge_counts"``.

        Returns:
            ``(counts, warning)``. ``counts`` is ``None`` when the dataset is
            not cache-backed (silent — the sampled path covers it) or when the
            cache was packed without that pointer (a legitimate case, e.g. no
            :class:`~molix.data.tasks.NeighborList` in the pipeline), in which
            case ``warning`` carries the reason for the report.
        """
        try:
            counts = getattr(data, attr)
        except AttributeError:
            return None, None
        except ValueError as exc:  # cache built without the matching pointer
            return None, f"exact {attr} unavailable — {exc}"
        if not isinstance(counts, torch.Tensor):
            return None, None
        return counts, None

    def _packed_schema(self, data: object) -> Mapping[str, tuple] | None:
        """Return the packed ``payload["schema"]`` mapping, or ``None``.

        Args:
            data: The dataset being profiled.

        Returns:
            The exact per-key ``(axis, dtype, extra_shape)`` mapping inferred
            when the cache was packed, or ``None`` when *data* has no packed
            view (``SubsetDataset`` over a non-cache dataset raises
            :class:`AttributeError` here, which is the documented "no fast
            path" signal).
        """
        try:
            return data.packed_view().payload["schema"]
        except (AttributeError, KeyError, TypeError):
            return None
