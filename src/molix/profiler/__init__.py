"""Development profiler suite for MolNex.

Standalone, OOP profiling tools for identifying performance bottlenecks
**before and outside** of training.  No Trainer, no hooks.

Three profilers, each targeting a single component:

- :class:`TaskProfiler` — wall-clock timing of a single pipeline task
  (:class:`~molix.data.task.SampleTask`, :class:`~molix.data.task.DatasetTask`,
  or :class:`~molix.data.task.BatchTask`).

- :class:`ModuleProfiler` — forward/backward timing and GPU memory for any
  :class:`torch.nn.Module`.

- :class:`DataLoaderProfiler` — DataLoader stall-time measurement.

Two data generators for use when no real dataset is available:

- :class:`MockBatch` — callable that produces a
  :class:`~molix.data.types.GraphBatch` with configurable (optionally random)
  atom / edge / graph counts.

- :class:`MockSource` — :class:`~molix.data.source.DataSource` returning
  synthetic ``{"Z": ..., "pos": ...}`` sample dicts.

Typical workflow::

    from molix.profiler import TaskProfiler, MockSource
    from molix.data.tasks import NeighborList

    # Profile a single task with synthetic data
    profiler = TaskProfiler(NeighborList(cutoff=5.0))
    result = profiler.run(MockSource(n_samples=500, n_atoms=(5, 20)), n_samples=100)
    result.print_report()

    # Profile a model with a mock batch factory (GraphBatch modules)
    from molix.profiler import ModuleProfiler, MockBatch

    profiler = ModuleProfiler(encoder, loss_fn=my_loss, device="cuda:0")
    result = profiler.run(MockBatch(n_atoms=(32, 96), n_edges=(100, 600), n_graphs=4))
    result.print_report()

    # Profile a sub-module with arbitrary inputs (run_fn)
    dist = torch.rand(512)
    result = ModuleProfiler(rbf).run_fn(
        forward_fn=lambda: rbf(dist),
        backward_fn=lambda out: out.sum(),
        label="bond_dist E=512",
    )
    result.print_report()

    # Profile DataLoader throughput
    from molix.profiler import DataLoaderProfiler

    profiler = DataLoaderProfiler(batch_size=32, num_workers=4)
    result = profiler.run(MockSource(n_samples=2000, n_atoms=(5, 20)))
    result.print_report()
"""

from molix.profiler.dataloader import DataLoaderProfiler, DataLoaderResult
from molix.profiler.mock import MockBatch, MockSource
from molix.profiler.module import ModuleProfiler, ModuleResult
from molix.profiler.task import TaskProfiler, TaskResult

__all__ = [
    # Profilers
    "TaskProfiler",
    "ModuleProfiler",
    "DataLoaderProfiler",
    # Results
    "TaskResult",
    "ModuleResult",
    "DataLoaderResult",
    # Data generators
    "MockBatch",
    "MockSource",
]
