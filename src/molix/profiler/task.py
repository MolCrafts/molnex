"""Pipeline task profiler.

Profiles a single :class:`~molix.data.task.Task` (SampleTask, DatasetTask,
or BatchTask) in complete isolation.  Accepts either a
:class:`~molix.profiler.mock.MockSource` or a real
:class:`~molix.data.source.DataSource` as the sample provider.

For :class:`~molix.data.task.DatasetTask`, a ``fit()`` call is made on the
provided source before timing begins (mirroring what
:func:`~molix.data.execute.run` does).

Example::

    from molix.profiler.task import TaskProfiler
    from molix.profiler.mock import MockSource
    from molix.data.tasks import NeighborList

    source = MockSource(n_samples=500, n_atoms=(5, 20))
    profiler = TaskProfiler(NeighborList(cutoff=5.0))
    result = profiler.run(source, n_samples=100)
    result.print_report()
"""

from __future__ import annotations

from dataclasses import dataclass

from molix.data.execute import call_task
from molix.data.task import DatasetTask
from molix.profiler._utils import Timer, TimingStat, _fmt_table


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class TaskResult:
    """Profiling results for a single pipeline task.

    Attributes:
        task_name: ``type(task).__name__``.
        task_id: ``task.task_id`` if available.
        timing: Wall-clock timing statistics (ms).
        n_samples: Number of samples measured.
        data_description: Human-readable description of the input source.
    """

    task_name: str
    task_id: str
    timing: TimingStat
    n_samples: int
    data_description: str

    def print_report(self) -> None:
        """Print a human-readable timing report to stdout."""
        print(f"\nTask : {self.task_name}  (id={self.task_id})")
        print(f"Data : {self.data_description}")
        print("─" * 72)

        rows = [
            {
                "Metric": "Execute time",
                "mean(ms)": f"{self.timing.mean_ms:.3f}",
                "std(ms)": f"{self.timing.std_ms:.3f}",
                "p50(ms)": f"{self.timing.p50_ms:.3f}",
                "p95(ms)": f"{self.timing.p95_ms:.3f}",
                "min(ms)": f"{self.timing.min_ms:.3f}",
                "max(ms)": f"{self.timing.max_ms:.3f}",
            }
        ]
        cols = ["Metric", "mean(ms)", "std(ms)", "p50(ms)", "p95(ms)", "min(ms)", "max(ms)"]
        print(_fmt_table(rows, cols, col_width=10))
        print(f"\n  Samples measured: {self.n_samples}")
        print("─" * 72)
        print()


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------


class TaskProfiler:
    """Profile a single pipeline :class:`~molix.data.task.Task`.

    Measures the wall-clock time of ``task.execute(sample)`` in isolation.
    For :class:`~molix.data.task.DatasetTask`, ``fit()`` is called once on
    the source before timing starts.

    Args:
        task: The task to profile.

    Example::

        profiler = TaskProfiler(NeighborList(cutoff=5.0))

        # Mock source
        result = profiler.run(MockSource(n_samples=500, n_atoms=(5, 20)), n_samples=100)
        result.print_report()

        # Real source
        result = profiler.run(qm9_source, n_samples=200)
        result.print_report()
    """

    def __init__(self, task: object) -> None:
        self.task = task

    def run(
        self,
        source: object,
        n_samples: int = 100,
        n_warmup: int = 10,
    ) -> TaskResult:
        """Run the task profiler on ``source``.

        Args:
            source: A :class:`~molix.data.source.DataSource` (or
                :class:`~molix.profiler.mock.MockSource`) providing raw samples.
            n_samples: Number of samples to time (after warmup).
            n_warmup: Number of samples to discard at the start.

        Returns:
            :class:`TaskResult` with timing statistics.
        """
        n_source = len(source)  # type: ignore[arg-type]

        # For DatasetTask: fit on the full source first
        if isinstance(self.task, DatasetTask):
            all_samples = [source[i] for i in range(n_source)]  # type: ignore[index]
            self.task.fit(all_samples)

        total = n_warmup + n_samples
        times_ms: list[float] = []

        for i in range(total):
            idx = i % n_source
            sample = source[idx]  # type: ignore[index]
            with Timer() as t:
                call_task(self.task, sample)
            if i >= n_warmup:
                times_ms.append(t.elapsed * 1000)

        task_name = type(self.task).__name__
        task_id = getattr(self.task, "task_id", task_name)
        desc = getattr(source, "describe", lambda: type(source).__name__)()

        return TaskResult(
            task_name=task_name,
            task_id=str(task_id),
            timing=TimingStat.from_list(times_ms),
            n_samples=n_samples,
            data_description=desc,
        )
