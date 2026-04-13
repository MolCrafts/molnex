"""DataLoader throughput profiler.

Measures how much time a training loop spends blocked waiting for the next
batch — the DataLoader stall time.  Accepts either a
:class:`~molix.profiler.mock.MockSource` or a real
:class:`~molix.data.source.DataSource` / :class:`torch.utils.data.Dataset`.

The stall time is measured via the **inter-batch gap technique**: the
wall-clock elapsed from ``next(iter)`` returning batch *i* to the point
where ``next(iter)`` is called for batch *i+1*.  This captures worker
scheduling, collation, and pin_memory overhead.

Example::

    from molix.profiler.dataloader import DataLoaderProfiler
    from molix.profiler.mock import MockSource

    source = MockSource(n_samples=2000, n_atoms=(5, 20))
    profiler = DataLoaderProfiler(batch_size=32, num_workers=4)
    result = profiler.run(source, n_batches=100)
    result.print_report()
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from torch.utils.data import DataLoader, Dataset

from molix.data.collate import DEFAULT_TARGET_SCHEMA, TargetSchema, collate_molecules
from molix.data.dataset import CachedDataset
from molix.data.pipeline import PipelineSpec
from molix.profiler._utils import TimingStat, ValueStat, _fmt_table


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class DataLoaderResult:
    """Profiling results for a DataLoader.

    Attributes:
        load_time: Wall-clock time per batch (ms).
        throughput_graphs_per_sec: Graphs produced per second.
        throughput_atoms_per_sec: Atoms produced per second.
        batch_atom_stats: Atom count distribution per batch.
        batch_graph_stats: Graph count distribution per batch.
        n_batches: Number of measured batches.
        num_workers: DataLoader ``num_workers`` setting.
        batch_size: DataLoader ``batch_size`` setting.
        pin_memory: DataLoader ``pin_memory`` setting.
        data_description: Human-readable description of the input.
    """

    load_time: TimingStat
    throughput_graphs_per_sec: float
    throughput_atoms_per_sec: float
    batch_atom_stats: ValueStat
    batch_graph_stats: ValueStat
    n_batches: int
    num_workers: int
    batch_size: int
    pin_memory: bool
    data_description: str

    def print_report(self) -> None:
        """Print a human-readable throughput report to stdout."""
        s = self.load_time
        print(
            f"\nDataLoader Profile  "
            f"(n={self.n_batches} batches, num_workers={self.num_workers})"
        )
        print(f"Data: {self.data_description}")
        print("─" * 72)

        rows = [
            {
                "Metric": "Load time / batch",
                "mean": f"{s.mean_ms:.3f} ms",
                "std": f"{s.std_ms:.3f}",
                "p50": f"{s.p50_ms:.3f}",
                "p95": f"{s.p95_ms:.3f}",
                "min": f"{s.min_ms:.3f}",
                "max": f"{s.max_ms:.3f}",
            }
        ]
        print(_fmt_table(rows, ["Metric", "mean", "std", "p50", "p95", "min", "max"], col_width=8))
        print()
        print(f"  Throughput:  {self.throughput_graphs_per_sec:>12,.0f} graphs/s")
        print(f"               {self.throughput_atoms_per_sec:>12,.0f} atoms/s")
        print()
        print(
            f"  Batch size (graphs): "
            f"mean={self.batch_graph_stats.mean:.1f}  "
            f"p95={self.batch_graph_stats.p95:.1f}"
        )
        print(
            f"  Batch size (atoms):  "
            f"mean={self.batch_atom_stats.mean:.1f}  "
            f"std={self.batch_atom_stats.std:.1f}  "
            f"p50={self.batch_atom_stats.p50:.1f}  "
            f"p95={self.batch_atom_stats.p95:.1f}"
        )
        print()
        print(
            f"  Config: batch_size={self.batch_size}  "
            f"num_workers={self.num_workers}  "
            f"pin_memory={self.pin_memory}"
        )
        print("─" * 72)

        # Diagnostic
        if s.mean_ms > 0 and s.p95_ms > s.mean_ms * 3:
            print(
                f"  [WARN] p95/mean ratio = {s.p95_ms / s.mean_ms:.1f}x"
                " — worker stall or collation spikes detected."
                f" Try num_workers > {self.num_workers} or persistent_workers=True."
            )
        print()


# ---------------------------------------------------------------------------
# Batch stats extraction
# ---------------------------------------------------------------------------


def _extract_batch_counts(batch: object) -> tuple[int, int]:
    """Extract (n_atoms, n_graphs) from a GraphBatch."""
    try:
        n_atoms = int(batch["atoms"]["Z"].shape[0])  # type: ignore[index]
        n_graphs = int(batch["graphs"]["num_atoms"].shape[0])  # type: ignore[index]
        return n_atoms, n_graphs
    except (KeyError, AttributeError, TypeError):
        return 0, 0


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------


class DataLoaderProfiler:
    """Profile DataLoader batch-production throughput.

    Measures the time spent waiting for each batch using the inter-batch gap
    technique.  Builds a :class:`~torch.utils.data.DataLoader` internally
    from the provided source/dataset.

    Args:
        batch_size: Batch size for the DataLoader.
        num_workers: Number of DataLoader worker processes.
        pin_memory: Whether to pin memory for GPU transfer.
        persistent_workers: Keep workers alive between epochs.
        target_schema: How targets are collated into batches.
        pipeline: Optional :class:`~molix.data.pipeline.PipelineSpec` whose
            ``batch_tasks`` are applied inside ``collate_fn``.

    Example::

        profiler = DataLoaderProfiler(batch_size=32, num_workers=4)

        # Mock data
        result = profiler.run(MockSource(n_samples=2000, n_atoms=(5, 20)))
        result.print_report()

        # Real CachedDataset
        result = profiler.run(cached_dataset, n_batches=200)
        result.print_report()
    """

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        target_schema: TargetSchema = DEFAULT_TARGET_SCHEMA,
        pipeline: PipelineSpec | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.target_schema = target_schema
        self.pipeline = pipeline

    def run(
        self,
        data: object,
        n_batches: int = 50,
        n_warmup: int = 5,
    ) -> DataLoaderResult:
        """Run the DataLoader throughput measurement.

        Args:
            data: One of:

                - :class:`~molix.profiler.mock.MockSource` or any
                  :class:`~molix.data.source.DataSource` — samples are prepared
                  and wrapped in a :class:`~molix.data.dataset.CachedDataset`
                  automatically.
                - A :class:`torch.utils.data.Dataset` — used directly.
                - An existing :class:`torch.utils.data.DataLoader` — used as-is,
                  no new DataLoader is constructed.

            n_batches: Number of batches to measure (after warmup).
            n_warmup: Number of initial batches to discard.

        Returns:
            :class:`DataLoaderResult` with timing and throughput statistics.
        """
        desc = getattr(data, "describe", lambda: type(data).__name__)()

        # Resolve data → DataLoader
        if isinstance(data, DataLoader):
            dl = data
        elif isinstance(data, Dataset):
            dl = self._make_dataloader(data)
        else:
            # Assume DataSource protocol: has __len__ and __getitem__
            dataset = self._source_to_dataset(data)
            dl = self._make_dataloader(dataset)

        # --- Measure ---
        load_times_ms: list[float] = []
        atom_counts: list[int] = []
        graph_counts: list[int] = []
        total = n_warmup + n_batches

        t0 = time.perf_counter()
        for i, batch in enumerate(dl):
            load_ms = (time.perf_counter() - t0) * 1000
            if i >= n_warmup:
                load_times_ms.append(load_ms)
                n_a, n_g = _extract_batch_counts(batch)
                atom_counts.append(n_a)
                graph_counts.append(n_g)
            if i + 1 >= total:
                break
            t0 = time.perf_counter()

        if not load_times_ms:
            raise RuntimeError(
                f"Not enough batches in DataLoader: needed {total}, got fewer."
            )

        timing = TimingStat.from_list(load_times_ms)
        mean_load_s = timing.mean_ms / 1000.0
        mean_atoms = sum(atom_counts) / len(atom_counts) if atom_counts else 0.0
        mean_graphs = sum(graph_counts) / len(graph_counts) if graph_counts else 0.0

        return DataLoaderResult(
            load_time=timing,
            throughput_graphs_per_sec=mean_graphs / mean_load_s if mean_load_s > 0 else 0.0,
            throughput_atoms_per_sec=mean_atoms / mean_load_s if mean_load_s > 0 else 0.0,
            batch_atom_stats=ValueStat.from_list(atom_counts) if atom_counts else ValueStat(0, 0, 0, 0),
            batch_graph_stats=ValueStat.from_list(graph_counts) if graph_counts else ValueStat(0, 0, 0, 0),
            n_batches=len(load_times_ms),
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            data_description=desc,
        )

    # -- Helpers ----------------------------------------------------------------

    def _source_to_dataset(self, source: object) -> Dataset:
        """Convert a DataSource to a CachedDataset of plain sample dicts."""
        n = len(source)  # type: ignore[arg-type]
        samples = [source[i] for i in range(n)]  # type: ignore[index]
        return CachedDataset(samples)

    def _make_dataloader(self, dataset: Dataset) -> DataLoader:
        schema = self.target_schema
        pipeline = self.pipeline

        def collate_fn(batch_samples: list[dict]) -> object:
            from molix.data.pipeline import _call_task as _ct
            batch = collate_molecules(batch_samples, schema)
            if pipeline is not None:
                for entry in pipeline.batch_tasks:
                    batch = _ct(entry.task, batch)
            return batch

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=collate_fn,
        )
