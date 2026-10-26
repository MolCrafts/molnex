"""Development profiler suite for MolNex.

Standalone, OOP profiling tools for finding performance bottlenecks before and
outside of training: no Trainer, no hooks. Each module owns one ``Profiler``
(and its ``Result`` where it has one); this package decides which modules are
loaded and exposes them by name — a profiler is reached through its module,
never through a flat re-export:

- :mod:`.task` — wall-clock timing of a single transform
  (:class:`~molix.data.NeighborList`, a :class:`~molix.data.task.BatchTask`,
  or any callable), dispatched structurally.
- :mod:`.module` — reproducible model and end-to-end workload measurement
  (``ProfileSweep`` / ``ProfileSpec`` / ``Profiler`` / ``Report``).
- :mod:`.dataloader` — DataLoader stall-time measurement.
- :mod:`.dataset` — dataset characterisation: exact packed-pointer size
  distributions, per-sample access latency and byte footprint, field layout
  and per-target value statistics.
- :mod:`.trainer` — Trainer-loop per-step framework overhead, driving a real
  Trainer over a near-zero-compute :class:`~molix.profiler.mock.MockModel`.
- :mod:`.operator` — a compiled CUDA operator or energy/force model: explicit
  preparation, uninstrumented CUDA-event timing, Torch tracing, a bounded
  Nsight capture window, and the comparison of two measurements.
- :mod:`.nsight` — Nsight Systems / Nsight Compute wrappers and the kernel
  summary of a capture; :mod:`.ptxas` — registers and spills from a build log.
- :mod:`.mock` — synthetic batches, sample sources and a near-zero model.

Typical workflow::

    from molix.profiler import task, mock
    from molix.neighbor import NeighborList

    profiler = task.Profiler(NeighborList(cutoff=5.0))
    result = profiler.run(mock.MockSource(n_samples=500, n_atoms=(5, 20)), n_samples=100)
    result.print_report()

    from molix.profiler import module

    spec = module.ProfileSpec("my-model", model_class=MyModel, dataloader_class=DataLoader)
    config = module.ProfilerConfig(
        device="cpu", warmup=10, steps=100, transfer_batch=False,
        memory="none", activities=("cpu",), hotspot_steps=5, hotspot_limit=20,
    )
    report = module.Profiler().profile(spec, config)
    print(report.model_dump_json(indent=2))

    from molix.profiler import dataloader

    profiler = dataloader.Profiler(batch_size=32, num_workers=4)
    result = profiler.run(mock.MockSource(n_samples=2000, n_atoms=(5, 20)))
    result.print_report()
"""

from molix.profiler import dataloader, dataset, mock, module, nsight, operator, ptxas, task, trainer

# The modules this package loads, alphabetized (notes.md 2026-08-09).
__all__ = [
    "dataloader",
    "dataset",
    "mock",
    "module",
    "nsight",
    "operator",
    "ptxas",
    "task",
    "trainer",
]
