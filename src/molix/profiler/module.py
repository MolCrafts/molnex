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
        optimizer_ms: Optimizer-step timing (``None`` if no optimizer was given).
        peak_memory_mb: Peak CUDA memory per step during the forward pass.
        throughput_atoms_per_sec: Atoms processed per second (forward only).
        throughput_graphs_per_sec: Graphs processed per second (forward only).
        n_params: Total trainable parameter count.
        device: Device used during profiling.
        n_steps: Number of measured steps.
        data_description: Human-readable description of the input data.
        wall_ms_per_step: End-to-end CPU wall-clock per full step, measured over
            a back-to-back block with **no intra-step synchronization** (``None``
            on CPU). This is the number that reflects real training throughput.
        gpu_active_ms_per_step: Real GPU compute per step = sum of per-kernel
            device self-times from a ``torch.profiler`` window. (Note: a single
            CUDA-event pair spanning the block would *not* give this — it
            measures the GPU-timeline span, which already includes the launch
            gaps, so it equals the wall time. Summing kernel self-times is the
            only way to isolate true GPU-active time on one stream.)
        cpu_dispatch_ms_per_step: Sum of per-op host self-time per step — the
            CPU cost of building and dispatching the kernels.
        launch_bound_pct: ``(wall - gpu_active) / wall * 100`` — fraction of the
            step the GPU sits idle waiting on CPU kernel dispatch. High (>50%)
            ⇒ launch-bound (batch too small / too many tiny kernels); low ⇒
            compute-bound. ``None`` on CPU.
        op_table: Pre-rendered top-op breakdown from ``torch.profiler``.
        op_calls_per_step: Mean number of profiler op calls per step (proxy for
            launch pressure — hundreds of tiny ops ⇒ launch-bound).
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
    optimizer_ms: TimingStat | None = None
    wall_ms_per_step: float | None = None
    gpu_active_ms_per_step: float | None = None
    cpu_dispatch_ms_per_step: float | None = None
    launch_bound_pct: float | None = None
    op_table: str | None = None
    op_calls_per_step: float | None = None
    submodule_table: str | None = None

    def print_report(self) -> None:
        """Print a human-readable performance report to stdout."""
        print(f"\nModule: {self.module_name}  |  device={self.device}  |  n_steps={self.n_steps}")
        print(f"Data  : {self.data_description}")
        print("─" * 72)

        def _row(name: str, s: TimingStat) -> dict:
            return {
                "Pass": name,
                "mean(ms)": f"{s.mean_ms:.3f}",
                "std(ms)": f"{s.std_ms:.3f}",
                "p50(ms)": f"{s.p50_ms:.3f}",
                "p95(ms)": f"{s.p95_ms:.3f}",
            }

        rows = [_row("Forward", self.forward_ms)]
        total = self.forward_ms.mean_ms
        if self.backward_ms is not None:
            rows.append(_row("Backward", self.backward_ms))
            total += self.backward_ms.mean_ms
        if self.optimizer_ms is not None:
            rows.append(_row("Optimizer", self.optimizer_ms))
            total += self.optimizer_ms.mean_ms
        if len(rows) > 1:
            rows.append({"Pass": "Σ components", "mean(ms)": f"{total:.3f}"})

        print(_fmt_table(rows, ["Pass", "mean(ms)", "std(ms)", "p50(ms)", "p95(ms)"], col_width=10))
        print()

        if self.wall_ms_per_step is not None:
            sps = 1000.0 / self.wall_ms_per_step if self.wall_ms_per_step > 0 else 0.0
            print(
                f"  Full step (no intra-step sync):  "
                f"wall={self.wall_ms_per_step:.3f} ms/step  ({sps:,.1f} step/s)"
            )
            if self.gpu_active_ms_per_step is not None:
                print(
                    f"    GPU-active={self.gpu_active_ms_per_step:.3f} ms  "
                    f"CPU-dispatch={self.cpu_dispatch_ms_per_step:.3f} ms  "
                    f"launch-bound={self.launch_bound_pct:.1f}%"
                )
                print(f"    → {self._verdict()}")
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
        if self.op_calls_per_step is not None:
            print(f"  Op calls/step:{self.op_calls_per_step:>12,.0f}")
        print("─" * 72)
        if self.submodule_table is not None:
            print("Forward time by submodule (per step):")
            print(self.submodule_table)
            print("─" * 72)
        if self.op_table is not None:
            print(self.op_table)
        print()

    def _verdict(self) -> str:
        """One-word classification of the launch-bound percentage."""
        p = self.launch_bound_pct or 0.0
        if p >= 60:
            return "LAUNCH-BOUND — too many tiny kernels / batch too small"
        if p >= 30:
            return "partially launch-bound"
        return "compute-bound"


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
    """Extract (n_atoms, n_graphs) from a TensorDict batch; returns (0, 0) on failure."""
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
        optimizer: Optional optimizer. When given, the full train step
            (forward → loss → backward → ``optimizer.step()`` →
            ``zero_grad``) is profiled, matching what the real
            :class:`~molix.core.trainer.Trainer` runs. The optimizer step is
            timed as its own column and is included in the launch-bound
            measurement. The optimizer must already wrap ``module.parameters()``.

    Example::

        from molix.profiler.module import ModuleProfiler
        from molix.profiler.mock import MockBatch

        opt = torch.optim.Adam(model.parameters(), lr=1e-4, fused=True)
        profiler = ModuleProfiler(model, loss_fn=my_loss, device="cuda:0", optimizer=opt)

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
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        self.module = module
        self.loss_fn = loss_fn
        self.device = torch.device(device)
        self.optimizer = optimizer

    def run_fn(
        self,
        forward_fn: Callable[[], object],
        backward_fn: Callable[[object], torch.Tensor] | None = None,
        n_steps: int = 100,
        n_warmup: int = 10,
        label: str = "",
    ) -> ModuleResult:
        """Profile using explicit forward/backward callables.

        Use this when the module does not accept a ``TensorDict`` batch
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
        op_rows: int = 12,
        submodules: bool = False,
    ) -> ModuleResult:
        """Run the profiler.

        Three measurements are taken (the last two only on CUDA):

        1. **Component breakdown** — forward / backward / optimizer timed
           separately with :class:`torch.cuda.Event`. Each component
           synchronizes, so these numbers are *upper bounds* (they serialize
           CPU and GPU) — read them for relative cost, not absolute throughput.
        2. **Full-step block** — ``n_steps`` run back-to-back with **no
           intra-step sync**, then one synchronize. Yields the true
           ``wall_ms_per_step``.
        3. **Op breakdown** — a short :class:`torch.profiler` window. Summing
           per-kernel device self-time gives ``gpu_active_ms_per_step``; the
           wall/active gap is ``launch_bound_pct``. The top ops by self-CUDA
           time are rendered into ``op_table``.

        Args:
            data: Input data source.  One of:

                - :class:`~molix.profiler.mock.MockBatch` (called once per step)
                - ``list`` of pre-built batches (cycled through)
                - A ``DataLoader`` or other iterable (cycled through)

            n_steps: Number of steps to measure.
            n_warmup: Number of steps to discard before timing starts.
            op_rows: Number of top ops to show in the breakdown table.
            submodules: Also attribute forward time to the module's top-level
                named children (architectural breakdown — e.g. embedding vs
                interaction vs readout — complementing the aten-level op
                table). Measured in a separate hooked forward-only pass so it
                does not perturb the wall / op windows.

        Returns:
            :class:`ModuleResult` with component, full-step and op statistics.
        """
        use_cuda = self.device.type == "cuda"
        module = self.module.to(self.device)
        original_training = module.training
        module.train()  # train mode: keep dropout / BN behaviour consistent
        opt = self.optimizer

        fwd_times_ms: list[float] = []
        bwd_times_ms: list[float] = []
        opt_times_ms: list[float] = []
        peak_mem_mb: list[float] = []
        atom_counts: list[int] = []
        graph_counts: list[int] = []

        total = n_warmup + n_steps
        batch_iter = iter(_make_batch_iter(data, total))

        for step in range(total):
            batch = next(batch_iter)
            batch = _move_to_device(batch, self.device)

            if opt is not None:
                opt.zero_grad(set_to_none=True)
            else:
                module.zero_grad(set_to_none=True)

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
            loss = None
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

            # --- Optimizer step (optional) ---
            opt_ms = 0.0
            if opt is not None and loss is not None:
                if use_cuda:
                    opt_t = _CUDATimer()
                    opt_t.start()
                    opt.step()
                    opt_t.stop()
                    opt_ms = opt_t.elapsed_ms()
                else:
                    t2 = time.perf_counter()
                    opt.step()
                    opt_ms = (time.perf_counter() - t2) * 1000

            if step >= n_warmup:
                fwd_times_ms.append(fwd_ms)
                bwd_times_ms.append(bwd_ms)
                opt_times_ms.append(opt_ms)
                peak_mem_mb.append(mem_mb)
                n_a, n_g = _extract_counts(batch)
                atom_counts.append(n_a)
                graph_counts.append(n_g)

        # --- Full-step block (no intra-step sync) + op breakdown ---
        wall_ms = gpu_active = cpu_dispatch = lb_pct = None
        op_table = None
        op_calls = None
        if use_cuda:
            wall_ms = self._measure_block(module, data, n_steps, n_warmup)
            gpu_active, cpu_dispatch, op_calls, op_table = self._op_breakdown(module, data, op_rows)
            lb_pct = (wall_ms - gpu_active) / wall_ms * 100.0 if wall_ms > 0 else 0.0

        submodule_table = None
        if submodules:
            submodule_table = self._submodule_breakdown(module, data, n_steps, n_warmup, use_cuda)

        if not original_training:
            module.eval()

        fwd_stat = TimingStat.from_list(fwd_times_ms)
        mem_stat = ValueStat.from_list(peak_mem_mb)
        bwd_stat = TimingStat.from_list(bwd_times_ms) if self.loss_fn is not None else None
        opt_stat = TimingStat.from_list(opt_times_ms) if opt is not None else None

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
            optimizer_ms=opt_stat,
            peak_memory_mb=mem_stat,
            throughput_atoms_per_sec=mean_atoms / mean_fwd_s if mean_fwd_s > 0 else 0.0,
            throughput_graphs_per_sec=mean_graphs / mean_fwd_s if mean_fwd_s > 0 else 0.0,
            n_params=n_params,
            device=str(self.device),
            n_steps=n_steps,
            data_description=desc,
            wall_ms_per_step=wall_ms,
            gpu_active_ms_per_step=gpu_active,
            cpu_dispatch_ms_per_step=cpu_dispatch,
            launch_bound_pct=lb_pct,
            op_table=op_table,
            op_calls_per_step=op_calls,
            submodule_table=submodule_table,
        )

    def _submodule_breakdown(
        self,
        module: nn.Module,
        data: object,
        n_steps: int,
        n_warmup: int,
        use_cuda: bool,
    ) -> str | None:
        """Attribute forward time to top-level named children via hooks.

        A separate forward-only pass (no backward/optimizer) so the hook
        overhead never lands in the wall / op windows. Each child is
        bracket-timed with CUDA events (GPU) or ``perf_counter`` (CPU);
        children not invoked in ``forward`` are dropped. Returns a rendered
        table (child, mean ms/step, % of summed children), or ``None`` when
        the module has no named children.
        """
        # Expand container children (ModuleList / Sequential / ModuleDict) into
        # their entries: the container itself is never *called* (you index into
        # it), so a hook on it would never fire and the bulk of an interaction
        # stack would go unattributed.
        children: list[tuple[str, nn.Module]] = []
        for name, child in module.named_children():
            if isinstance(child, (nn.ModuleList, nn.Sequential, nn.ModuleDict)):
                children.extend((f"{name}.{sub}", m) for sub, m in child.named_children())
            else:
                children.append((name, child))
        if not children:
            return None

        pending: dict[str, object] = {}
        elapsed: dict[str, list[float]] = {name: [] for name, _ in children}
        handles = []

        def _pre(name):
            def hook(_mod, _inp):
                if use_cuda:
                    ev = torch.cuda.Event(enable_timing=True)
                    ev.record()
                    pending[name] = ev
                else:
                    pending[name] = time.perf_counter()

            return hook

        def _post(name):
            def hook(_mod, _inp, _out):
                start = pending.pop(name, None)
                if start is None:
                    return
                if use_cuda:
                    end = torch.cuda.Event(enable_timing=True)
                    end.record()
                    elapsed[name].append(("cuda", start, end))
                else:
                    elapsed[name].append(time.perf_counter() - start)

            return hook

        for name, child in children:
            handles.append(child.register_forward_pre_hook(_pre(name)))
            handles.append(child.register_forward_hook(_post(name)))

        try:
            batch_iter = iter(_make_batch_iter(data, n_warmup + n_steps))
            with torch.no_grad():
                for step in range(n_warmup + n_steps):
                    batch = _move_to_device(next(batch_iter), self.device)
                    module(batch)
                    if use_cuda and step == n_warmup - 1:
                        torch.cuda.synchronize()
                        for v in elapsed.values():
                            v.clear()  # drop warmup samples
            if use_cuda:
                torch.cuda.synchronize()
        finally:
            for h in handles:
                h.remove()

        # Resolve CUDA event pairs to milliseconds.
        means: dict[str, float] = {}
        for name, samples in elapsed.items():
            if not samples:
                continue
            if use_cuda:
                ms = [s.elapsed_time(e) for _, s, e in samples]
            else:
                ms = [v * 1000.0 for v in samples]
            means[name] = sum(ms) / len(ms)
        if not means:
            return None

        captured = sum(means.values())
        denom = captured or 1.0
        rows = [
            {
                "submodule": name[:36],
                "fwd_ms": f"{ms:.4f}",
                "%captured": f"{ms / denom * 100:.1f}",
            }
            for name, ms in sorted(means.items(), key=lambda kv: kv[1], reverse=True)
        ]
        # Σ row: compare to the report's Forward mean to read coverage —
        # the gap is inline/functional ops not inside a named child.
        rows.append({"submodule": "Σ captured", "fwd_ms": f"{captured:.4f}", "%captured": "100.0"})
        return _fmt_table(rows, ["submodule", "fwd_ms", "%captured"], col_width=10)

    def _full_step(self, module: nn.Module, batch: object) -> None:
        """One untimed train step: zero_grad → fwd → loss → backward → opt.step."""
        opt = self.optimizer
        if opt is not None:
            opt.zero_grad(set_to_none=True)
        else:
            module.zero_grad(set_to_none=True)
        output = module(batch)
        if self.loss_fn is not None:
            loss = self.loss_fn(output, batch)
            loss.backward()
            if opt is not None:
                opt.step()

    def _measure_block(self, module: nn.Module, data: object, n_steps: int, n_warmup: int) -> float:
        """Wall-clock ms per step over ``n_steps`` run back-to-back with **no
        intra-step sync** (one synchronize at the end). This is real throughput.
        """
        batch_iter = iter(_make_batch_iter(data, n_warmup + n_steps))
        for _ in range(n_warmup):
            self._full_step(module, _move_to_device(next(batch_iter), self.device))
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_steps):
            self._full_step(module, _move_to_device(next(batch_iter), self.device))
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0 / n_steps

    def _op_breakdown(
        self, module: nn.Module, data: object, op_rows: int, n_steps: int = 20
    ) -> tuple[float, float, float, str]:
        """Run a short ``torch.profiler`` window.

        Returns ``(gpu_active_ms_per_step, cpu_dispatch_ms_per_step,
        op_calls_per_step, op_table)``. GPU-active time is the **sum of
        per-kernel device self-times** — on a single stream kernels run
        serially, so this sum is the true GPU-busy time, distinct from the
        GPU-timeline span a CUDA-event pair would report.
        """
        from torch.profiler import ProfilerActivity, profile

        batch_iter = iter(_make_batch_iter(data, n_steps + 3))
        for _ in range(3):  # warm up before the profiler window
            self._full_step(module, _move_to_device(next(batch_iter), self.device))
        torch.cuda.synchronize()

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(n_steps):
                self._full_step(module, _move_to_device(next(batch_iter), self.device))
            torch.cuda.synchronize()

        ka = prof.key_averages()
        gpu_active = sum(e.self_device_time_total for e in ka) / 1e3 / n_steps
        cpu_dispatch = sum(e.self_cpu_time_total for e in ka) / 1e3 / n_steps
        op_calls = sum(e.count for e in ka) / n_steps

        ranked = sorted(ka, key=lambda e: e.self_device_time_total, reverse=True)[:op_rows]
        rows = [
            {
                "op": e.key[:34],
                "cuda_ms/step": f"{e.self_device_time_total / 1e3 / n_steps:.3f}",
                "cpu_ms/step": f"{e.self_cpu_time_total / 1e3 / n_steps:.3f}",
                "calls/step": f"{e.count / n_steps:.0f}",
            }
            for e in ranked
        ]
        table = "  Top ops by self-CUDA time (per step):\n" + _fmt_table(
            rows, ["op", "cuda_ms/step", "cpu_ms/step", "calls/step"], col_width=10
        )
        return gpu_active, cpu_dispatch, op_calls, table
