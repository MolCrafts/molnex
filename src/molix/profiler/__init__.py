"""Development profiler suite for MolNex.

Standalone, OOP profiling tools for identifying performance bottlenecks
**before and outside** of training.  No Trainer, no hooks.

Five profilers, each targeting a single component:

- :class:`TaskProfiler` — wall-clock timing of a single transform
  (:class:`~molix.data.NeighborList`, a :class:`~molix.data.task.BatchTask`,
  or any callable), dispatched structurally.

- :class:`ProfileSweep` / :class:`ProfileSpec` / :class:`Profiler` /
  :class:`Report` — reproducible model and end-to-end workload measurement.

- :class:`DataLoaderProfiler` — DataLoader stall-time measurement.

- :class:`DatasetProfiler` — dataset characterisation: exact packed-pointer
  size distributions, per-sample access latency and byte footprint, field
  layout and per-target value statistics.

- :class:`TrainerProfiler` — Trainer-loop per-step framework overhead.
  Drives a real Trainer over a near-zero-compute
  :class:`~molix.profiler.mock.MockModel` so a ``cProfile`` window
  attributes wall time to loop machinery (Step dispatch, hooks,
  batch moves, TrainState writes), not model FLOPs.

Two data generators for use when no real dataset is available:

- :class:`MockBatch` — callable that produces a
  ``TensorDict`` with configurable (optionally random)
  atom / edge / graph counts.

- :class:`MockSource` — indexable sample source returning synthetic
  ``{"Z": ..., "pos": ...}`` sample dicts.

Typical workflow::

    from molix.profiler import TaskProfiler, MockSource
    from molix.neighbor import NeighborList

    # Profile a single task with synthetic data
    profiler = TaskProfiler(NeighborList(cutoff=5.0))
    result = profiler.run(MockSource(n_samples=500, n_atoms=(5, 20)), n_samples=100)
    result.print_report()

    # Profile classes with an explicit measurement protocol
    from molix.profiler import Profiler, ProfilerConfig, ProfileSpec

    spec = ProfileSpec("my-model", model_class=MyModel, dataloader_class=DataLoader)
    config = ProfilerConfig(
        device="cpu", warmup=10, steps=100, transfer_batch=False,
        memory="none", activities=("cpu",), hotspot_steps=5, hotspot_limit=20,
    )
    report = Profiler().profile(spec, config)
    print(report.model_dump_json(indent=2))

    # Profile DataLoader throughput
    from molix.profiler import DataLoaderProfiler

    profiler = DataLoaderProfiler(batch_size=32, num_workers=4)
    result = profiler.run(MockSource(n_samples=2000, n_atoms=(5, 20)))
    result.print_report()
"""

from molix.profiler.dataloader import DataLoaderProfiler, DataLoaderResult
from molix.profiler.dataset import DatasetProfiler, DatasetResult
from molix.profiler.mock import MockBatch, MockModel, MockSource, mock_node_feature_loss
from molix.profiler.module import (
    Hotspot,
    Measurement,
    Memory,
    Profiler,
    ProfilerConfig,
    ProfileSpec,
    ProfileSweep,
    Report,
    ReportSpec,
    Timing,
)
from molix.profiler.nsight import KernelCost, NsightCompute, NsightSystems
from molix.profiler.operator import OperatorProfiler, OperatorProtocol, PreparedOperator
from molix.profiler.ptxas import KernelResources, parse_ptxas
from molix.profiler.task import TaskProfiler, TaskResult
from molix.profiler.trainer import TrainerProfiler, TrainerResult

# Profilers, their result types and the mock data generators share one
# alphabetically sorted list (notes.md 2026-08-09: `__all__` stays alphabetized).
__all__ = [
    "DataLoaderProfiler",
    "DataLoaderResult",
    "DatasetProfiler",
    "DatasetResult",
    "Hotspot",
    "KernelCost",
    "KernelResources",
    "Measurement",
    "Memory",
    "MockBatch",
    "MockModel",
    "MockSource",
    "NsightCompute",
    "NsightSystems",
    "OperatorProfiler",
    "OperatorProtocol",
    "PreparedOperator",
    "ProfileSpec",
    "ProfileSweep",
    "Profiler",
    "ProfilerConfig",
    "Report",
    "ReportSpec",
    "TaskProfiler",
    "TaskResult",
    "Timing",
    "TrainerProfiler",
    "TrainerResult",
    "mock_node_feature_loss",
    "parse_ptxas",
]
