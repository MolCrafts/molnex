"""nn.Module forward/backward throughput profiler.

Profiles any :class:`torch.nn.Module` in complete isolation — no Trainer,
no DataModule, no hook machinery.  Accepts either a
:class:`~molix.profiler.mock.MockBatch` factory or a pre-built list / DataLoader
of batches as input data.

Uses ``torch.cuda.Event`` for sub-millisecond accurate GPU timing and avoids
the ``torch.cuda.synchronize()`` overhead of CPU-side timers.

Example::

    from molix.profiler.module import ModuleProfiler
    from molix.profiler.mock import MockBatch

    factory = MockBatch(n_atoms=64, n_edges=512, n_graphs=8, device="cuda:0")

    profiler = ModuleProfiler(model, loss_fn=my_loss, device="cuda:0")
    result = profiler.run(factory, n_steps=200)
    result.print_report()
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn

from molix.profiler._utils import TimingStat, ValueStat, _fmt_table, reset_peak_memory


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class ModuleResult:
    """Profiling results for an nn.Module.

    Attributes:
        module_name: ``type(module).__name__``.
        forward_ms: Forward-pass timing statistics.
        backward_ms: Backward-pass timing statistics (all zeros if ``loss_fn`` is None).
        peak_memory_mb: Peak CUDA memory per step during the forward pass.
        throughput_atoms_per_sec: Atoms processed per second (forward only).
        throughput_graphs_per_sec: Graphs processed per second (forward only).
        n_params: Total trainable parameter count.
        device: Device used during profiling.
        n_steps: Number of measured steps.
        data_description: Human-readable description of the input data.
    """

    module_name: str
    forward_ms: TimingStat
    backward_ms: TimingStat | None
    peak_memory_mb: ValueStat
    throughput_atoms_per_sec: float
    throughput_graphs_per_sec: float
    n_params: int
    device: str
    n_steps: int
    data_description: str

    def print_report(self) -> None:
        """Print a human-readable performance report to stdout."""
        print(f"\nModule: {self.module_name}  |  device={self.device}  |  n_steps={self.n_steps}")
        print(f"Data  : {self.data_description}")
        print("─" * 72)

        rows = [
            {
                "Pass": "Forward",
                "mean(ms)": f"{self.forward_ms.mean_ms:.3f}",
                "std(ms)": f"{self.forward_ms.std_ms:.3f}",
                "p50(ms)": f"{self.forward_ms.p50_ms:.3f}",
                "p95(ms)": f"{self.forward_ms.p95_ms:.3f}",
            }
        ]
        if self.backward_ms is not None:
            rows.append(
                {
                    "Pass": "Backward",
                    "mean(ms)": f"{self.backward_ms.mean_ms:.3f}",
                    "std(ms)": f"{self.backward_ms.std_ms:.3f}",
                    "p50(ms)": f"{self.backward_ms.p50_ms:.3f}",
                    "p95(ms)": f"{self.backward_ms.p95_ms:.3f}",
                }
            )
            total = self.forward_ms.mean_ms + self.backward_ms.mean_ms
            rows.append({"Pass": "Total/step", "mean(ms)": f"{total:.3f}", "std(ms)": "", "p50(ms)": "", "p95(ms)": ""})

        print(_fmt_table(rows, ["Pass", "mean(ms)", "std(ms)", "p50(ms)", "p95(ms)"], col_width=10))
        print()

        if self.peak_memory_mb.mean > 0:
            print(
                f"  Peak CUDA memory (fwd):  "
                f"mean={self.peak_memory_mb.mean:.1f} MB  "
                f"p95={self.peak_memory_mb.p95:.1f} MB"
            )
            print()

        print(f"  Throughput:  {self.throughput_atoms_per_sec:>12,.0f} atoms/s")
        print(f"               {self.throughput_graphs_per_sec:>12,.0f} graphs/s")
        print(f"  Parameters:  {self.n_params:>12,}")
        print("─" * 72)
        print()


# ---------------------------------------------------------------------------
# GPU timer helper
# ---------------------------------------------------------------------------


class _CUDATimer:
    """Pair of CUDA events for accurate GPU-side timing."""

    def __init__(self) -> None:
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)

    def start(self) -> None:
        self._start.record()

    def stop(self) -> None:
        self._end.record()

    def elapsed_ms(self) -> float:
        """Synchronize and return elapsed milliseconds."""
        torch.cuda.synchronize()
        return self._start.elapsed_time(self._end)


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------


def _move_to_device(batch: object, device: torch.device) -> object:
    """Move a batch (TensorDict, plain dict, or tensor) to ``device``."""
    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    return batch


def _extract_counts(batch: object) -> tuple[int, int]:
    """Extract (n_atoms, n_graphs) from a GraphBatch; returns (0, 0) on failure."""
    try:
        n_atoms = int(batch["atoms"]["Z"].shape[0])  # type: ignore[index]
        n_graphs = int(batch["graphs"]["num_atoms"].shape[0])  # type: ignore[index]
        return n_atoms, n_graphs
    except (KeyError, AttributeError, TypeError):
        return 0, 0


def _make_batch_iter(
    data: object,
    n_total: int,
) -> Iterable:
    """Return an iterable of batches suitable for the profiling loop.

    Accepts:
    - A callable (``MockBatch`` factory) — called once per step.
    - A list / sequence — cycled through.
    - A DataLoader — iterated and restarted as needed.
    """
    if callable(data) and not isinstance(data, (list, tuple)):
        # Factory: call on each step
        return (data() for _ in range(n_total))
    if isinstance(data, (list, tuple)):
        # Cycle through the list
        return (data[i % len(data)] for i in range(n_total))
    # Assume DataLoader or other iterable — wrap with cycling
    def _cycle(iterable: Iterable, n: int):  # type: ignore[return]
        buf: list = []
        it = iter(iterable)
        count = 0
        while count < n:
            try:
                item = next(it)
                buf.append(item)
                yield item
                count += 1
            except StopIteration:
                if not buf:
                    return
                it = iter(buf)
    return _cycle(data, n_total)


class ModuleProfiler:
    """Profile the forward (and optionally backward) pass of any ``nn.Module``.

    No Trainer or DataModule is required.  The module is timed in isolation
    using either synthetic data from a :class:`~molix.profiler.mock.MockBatch`
    factory or real batches.

    Args:
        module: The module to profile.
        loss_fn: Loss function ``(output, batch) -> scalar Tensor``.
            If *None*, only the forward pass is timed.
        device: Device to run on.  The module is moved to this device
            at the start of :meth:`run`.

    Example::

        from molix.profiler.module import ModuleProfiler
        from molix.profiler.mock import MockBatch

        profiler = ModuleProfiler(encoder, loss_fn=my_loss, device="cuda:0")

        # Synthetic data
        result = profiler.run(MockBatch(n_atoms=(32, 96), n_edges=(100, 600), n_graphs=4))
        result.print_report()

        # Real batches (list)
        result = profiler.run(prebuilt_batch_list, n_steps=50)
        result.print_report()
    """

    def __init__(
        self,
        module: nn.Module,
        loss_fn: Callable | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.module = module
        self.loss_fn = loss_fn
        self.device = torch.device(device)

    def run_fn(
        self,
        forward_fn: Callable[[], object],
        backward_fn: Callable[[object], torch.Tensor] | None = None,
        n_steps: int = 100,
        n_warmup: int = 10,
        label: str = "",
    ) -> ModuleResult:
        """Profile using explicit forward/backward callables.

        Use this when the module does not accept a :class:`~molix.data.types.GraphBatch`
        — i.e. for sub-modules that take raw tensors or other inputs.

        The forward and backward callables are timed separately so you still
        get the forward/backward breakdown.

        Args:
            forward_fn: Zero-argument callable that runs the forward pass and
                returns the output.  All inputs should be captured in the closure.
            backward_fn: Takes the forward output and returns a scalar ``Tensor``
                on which ``.backward()`` is called.  If *None*, only the forward
                pass is timed.
            n_steps: Number of steps to measure.
            n_warmup: Steps to discard before timing.
            label: Optional description shown in the report.

        Returns:
            :class:`ModuleResult` with forward/backward statistics.

        Example::

            # BesselRBF takes a raw 1D distance tensor
            dist = torch.rand(256)
            result = ModuleProfiler(rbf).run_fn(
                forward_fn=lambda: rbf(dist),
                backward_fn=lambda out: out.sum(),
                n_steps=200,
            )
            result.print_report()

            # MessageAggregation takes 4 separate tensors
            result = ModuleProfiler(agg).run_fn(
                forward_fn=lambda: agg(messages, edge_index, cutoff, n_nodes),
                backward_fn=lambda out: out.sum(),
            )
        """
        use_cuda = self.device.type == "cuda"
        module = self.module.to(self.device)
        original_training = module.training
        module.train()

        fwd_times_ms: list[float] = []
        bwd_times_ms: list[float] = []
        peak_mem_mb: list[float] = []
        total = n_warmup + n_steps

        for step in range(total):
            # --- Forward ---
            if use_cuda:
                fwd_t = _CUDATimer()
                reset_peak_memory()
                fwd_t.start()
                output = forward_fn()
                fwd_t.stop()
                fwd_ms = fwd_t.elapsed_ms()
                mem_mb = torch.cuda.max_memory_allocated() / 1e6
            else:
                reset_peak_memory()
                t0 = time.perf_counter()
                output = forward_fn()
                fwd_ms = (time.perf_counter() - t0) * 1000
                mem_mb = 0.0

            # --- Backward (optional) ---
            bwd_ms = 0.0
            if backward_fn is not None:
                loss = backward_fn(output)
                if use_cuda:
                    bwd_t = _CUDATimer()
                    bwd_t.start()
                    loss.backward()
                    bwd_t.stop()
                    bwd_ms = bwd_t.elapsed_ms()
                else:
                    t1 = time.perf_counter()
                    loss.backward()
                    bwd_ms = (time.perf_counter() - t1) * 1000
                module.zero_grad(set_to_none=True)

            if step >= n_warmup:
                fwd_times_ms.append(fwd_ms)
                bwd_times_ms.append(bwd_ms)
                peak_mem_mb.append(mem_mb)

        if not original_training:
            module.eval()

        fwd_stat = TimingStat.from_list(fwd_times_ms)
        mem_stat = ValueStat.from_list(peak_mem_mb)
        bwd_stat = TimingStat.from_list(bwd_times_ms) if backward_fn is not None else None
        n_params = sum(p.numel() for p in module.parameters())
        desc = label or f"forward_fn={forward_fn!r}"

        return ModuleResult(
            module_name=type(module).__name__,
            forward_ms=fwd_stat,
            backward_ms=bwd_stat,
            peak_memory_mb=mem_stat,
            throughput_atoms_per_sec=0.0,
            throughput_graphs_per_sec=0.0,
            n_params=n_params,
            device=str(self.device),
            n_steps=n_steps,
            data_description=desc,
        )

    def run(
        self,
        data: object,
        n_steps: int = 100,
        n_warmup: int = 10,
    ) -> ModuleResult:
        """Run the profiler.

        Args:
            data: Input data source.  One of:

                - :class:`~molix.profiler.mock.MockBatch` (called once per step)
                - ``list`` of pre-built batches (cycled through)
                - A ``DataLoader`` or other iterable (cycled through)

            n_steps: Number of steps to measure.
            n_warmup: Number of steps to discard before timing starts.

        Returns:
            :class:`ModuleResult` with forward/backward statistics.
        """
        use_cuda = self.device.type == "cuda"
        module = self.module.to(self.device)
        original_training = module.training
        module.train()  # train mode: keep dropout / BN behaviour consistent

        fwd_times_ms: list[float] = []
        bwd_times_ms: list[float] = []
        peak_mem_mb: list[float] = []
        atom_counts: list[int] = []
        graph_counts: list[int] = []

        total = n_warmup + n_steps
        batch_iter = iter(_make_batch_iter(data, total))

        for step in range(total):
            batch = next(batch_iter)
            batch = _move_to_device(batch, self.device)

            # --- Forward ---
            if use_cuda:
                fwd_t = _CUDATimer()
                reset_peak_memory()
                fwd_t.start()
                output = module(batch)
                fwd_t.stop()
                fwd_ms = fwd_t.elapsed_ms()
                mem_mb = torch.cuda.max_memory_allocated() / 1e6
            else:
                reset_peak_memory()
                t0 = time.perf_counter()
                output = module(batch)
                fwd_ms = (time.perf_counter() - t0) * 1000
                mem_mb = 0.0

            # --- Backward (optional) ---
            bwd_ms = 0.0
            if self.loss_fn is not None:
                loss = self.loss_fn(output, batch)
                if use_cuda:
                    bwd_t = _CUDATimer()
                    bwd_t.start()
                    loss.backward()
                    bwd_t.stop()
                    bwd_ms = bwd_t.elapsed_ms()
                else:
                    t1 = time.perf_counter()
                    loss.backward()
                    bwd_ms = (time.perf_counter() - t1) * 1000
                module.zero_grad(set_to_none=True)

            if step >= n_warmup:
                fwd_times_ms.append(fwd_ms)
                bwd_times_ms.append(bwd_ms)
                peak_mem_mb.append(mem_mb)
                n_a, n_g = _extract_counts(batch)
                atom_counts.append(n_a)
                graph_counts.append(n_g)

        if not original_training:
            module.eval()

        fwd_stat = TimingStat.from_list(fwd_times_ms)
        mem_stat = ValueStat.from_list(peak_mem_mb)
        bwd_stat = TimingStat.from_list(bwd_times_ms) if self.loss_fn is not None else None

        mean_atoms = sum(atom_counts) / len(atom_counts) if atom_counts else 0.0
        mean_graphs = sum(graph_counts) / len(graph_counts) if graph_counts else 0.0
        mean_fwd_s = fwd_stat.mean_ms / 1000.0

        n_params = sum(p.numel() for p in module.parameters())

        # Build a human-readable data description
        desc = getattr(data, "describe", lambda: type(data).__name__)()

        return ModuleResult(
            module_name=type(module).__name__,
            forward_ms=fwd_stat,
            backward_ms=bwd_stat,
            peak_memory_mb=mem_stat,
            throughput_atoms_per_sec=mean_atoms / mean_fwd_s if mean_fwd_s > 0 else 0.0,
            throughput_graphs_per_sec=mean_graphs / mean_fwd_s if mean_fwd_s > 0 else 0.0,
            n_params=n_params,
            device=str(self.device),
            n_steps=n_steps,
            data_description=desc,
        )
