"""Single-transform profiler.

Times one transform in complete isolation. Dispatch is **structural**, so any
object works: a :class:`~molix.data.NeighborList`, a
:class:`~molix.data.task.BatchTask`, or a plain callable. Samples come from
anything with ``__len__`` / ``__getitem__`` — a
:class:`~molix.profiler.mock.MockSource` or a real dataset source.

A task that carries a callable ``fit`` is fitted once over the whole source
before timing begins, so a two-phase transform is measured in its fitted state.

Example::

    from molix.profiler import task
    from molix.profiler.mock import MockSource
    from molix.neighbor import NeighborList

    source = MockSource(n_samples=500, n_atoms=(5, 20))
    profiler = task.Profiler(NeighborList(cutoff=5.0))
    result = profiler.run(source, n_samples=100)
    result.print_report()
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from molix.profiler._utils import Timer, TimingStat, _fmt_table

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class Result:
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


class Profiler:
    """Profile a single sample transform.

    Measures the wall-clock time of ``task.execute(sample)`` in isolation,
    falling back to ``task(sample)`` when the task carries no ``execute``.
    A task carrying a callable ``fit`` is fitted once on the source before
    timing starts.

    Args:
        task: The transform to profile — anything with an ``execute`` method
            or anything callable.

    Example::

        profiler = task.Profiler(NeighborList(cutoff=5.0))

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
    ) -> Result:
        """Run the task profiler on ``source``.

        Args:
            source: A sample provider supporting ``__len__`` / ``__getitem__``
                (e.g. :class:`~molix.profiler.mock.MockSource` or a dataset).
            n_samples: Number of samples to time (after warmup).
            n_warmup: Number of samples to discard at the start.

        Returns:
            :class:`Result` with timing statistics.

        Raises:
            TypeError: *task* has neither an ``execute`` method nor a
                ``__call__``, so there is nothing to time.
        """
        n_source = len(source)

        # Two-phase transform: fit over the whole source before timing, so the
        # measured ``execute`` is the fitted one.
        fit = getattr(self.task, "fit", None)
        if callable(fit):
            all_samples = [source[i] for i in range(n_source)]
            fit(all_samples)

        task_name = type(self.task).__name__
        task_id = getattr(self.task, "task_id", task_name)
        # Dispatch structurally: ``execute`` when the task carries one, else the
        # task's own ``__call__``. The profiler times the transform, so it must
        # not route through a DAG wrapper it would otherwise be timing too.
        run_once: Callable[[object], object]
        execute = getattr(self.task, "execute", None)
        call = getattr(self.task, "__call__", None)
        if callable(execute):
            run_once = execute
        elif callable(call):
            run_once = call
        else:
            raise TypeError(
                f"Task {task_name} has neither an execute() nor a __call__: {type(self.task)}"
            )

        total = n_warmup + n_samples
        times_ms: list[float] = []

        for i in range(total):
            idx = i % n_source
            sample = source[idx]
            with Timer() as t:
                run_once(sample)
            if i >= n_warmup:
                times_ms.append(t.elapsed * 1000)
        desc = getattr(source, "describe", lambda: type(source).__name__)()

        return Result(
            task_name=task_name,
            task_id=str(task_id),
            timing=TimingStat.from_list(times_ms),
            n_samples=n_samples,
            data_description=desc,
        )
