"""Explicit preparation, uninstrumented timing and bounded operator capture.

Profiling is a host-side development operation. The callable itself is compiled
with fullgraph=True; NVTX and CUDA profiler controls stay outside that graph.
Callers own inputs, correctness checks and derivative definitions.
"""

from __future__ import annotations

import hashlib
import math
import platform
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from molix.compile import Compiler
from molix.profiler.module import Timing


@dataclass(frozen=True)
class OperatorProtocol:
    """Steady-state measurement budget, excluding compilation and profiling."""

    warmup: int = 30
    groups: int = 5
    min_calls: int = 100
    min_group_ms: float = 200.0
    max_calls: int = 1_000_000

    def __post_init__(self):
        if self.warmup < 0 or self.groups < 1 or self.min_calls < 1:
            raise ValueError("warmup must be nonnegative; groups and min_calls positive")
        if not math.isfinite(self.min_group_ms) or self.min_group_ms <= 0:
            raise ValueError("min_group_ms must be finite and positive")
        if self.max_calls < self.min_calls:
            raise ValueError("max_calls must cover min_calls")


class PreparedOperator:
    """A warmed callable or captured graph with stable, caller-owned inputs."""

    def __init__(self, function, args, graph, output, preparation):
        self.function = function
        self.args = args
        self.graph = graph
        self.output = output
        self.preparation = preparation

    def __call__(self):
        if self.graph is None:
            self.output = self.function(*self.args)
        else:
            self.graph.replay()
        return self.output


class OperatorProfiler:
    """Measure one already-defined forward, adjoint or force-training workload.

    Compilation/capture, latency measurement, Torch tracing, and external-tool
    capture are separate methods. Never run latency measurement inside Nsight.
    Outputs of a captured graph alias its private storage: clone before reuse.
    """

    def __init__(self, device: str, protocol: OperatorProtocol):
        self.device = torch.device(device)
        self.protocol = protocol
        if self.device.type != "cuda":
            raise ValueError("OperatorProfiler requires an explicit CUDA device")

    @staticmethod
    def fingerprint(inputs: Mapping[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
        """Hash exact tensor values outside all measured windows."""
        return {
            name: {
                "shape": list(t.shape),
                "stride": list(t.stride()),
                "dtype": str(t.dtype),
                "sha256": hashlib.sha256(
                    t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
            }
            for name, t in inputs.items()
        }

    def prepare(
        self, function: Callable, args: tuple = (), *, compile: bool, cuda_graph: bool
    ) -> PreparedOperator:
        """Compile, warm up and optionally capture; retain transient memory peaks."""
        with torch.cuda.device(self.device):
            fn = Compiler(fullgraph=True, dynamic=False)(function) if compile else function
            torch.cuda.synchronize(self.device)
            start = time.perf_counter()
            output = fn(*args)
            torch.cuda.synchronize(self.device)
            first_call_s = time.perf_counter() - start
            baseline = torch.cuda.memory_allocated(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            for _ in range(self.protocol.warmup):
                output = fn(*args)
            torch.cuda.synchronize(self.device)
            graph = None
            if cuda_graph:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = fn(*args)
                graph.replay()
                torch.cuda.synchronize(self.device)
            preparation = {
                "compile": compile,
                "cuda_graph": cuda_graph,
                "first_call_s": first_call_s,
                "allocated_before_warmup_bytes": baseline,
                "warmup_capture_peak_bytes": torch.cuda.max_memory_allocated(self.device),
                "warmup_capture_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
            }
            return PreparedOperator(fn, args, graph, output, preparation)

    def benchmark(self, prepared: PreparedOperator) -> dict[str, Any]:
        """CUDA-event groups with explicit minimum duration and call count."""
        samples, counts = [], []
        with torch.cuda.device(self.device):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            calls = self.protocol.min_calls
            for _ in range(self.protocol.groups):
                while True:
                    start.record()
                    for _ in range(calls):
                        prepared()
                    end.record()
                    end.synchronize()
                    elapsed = start.elapsed_time(end)
                    if not math.isfinite(elapsed) or elapsed <= 0:
                        raise RuntimeError(f"invalid CUDA event duration: {elapsed}")
                    if elapsed >= self.protocol.min_group_ms:
                        break
                    calls = math.ceil(calls * self.protocol.min_group_ms * 1.05 / elapsed)
                    if calls > self.protocol.max_calls:
                        raise RuntimeError("measurement exceeds max_calls; revise explicit budget")
                samples.append(elapsed / calls)
                counts.append(calls)
        return {
            "environment": {
                "host": platform.node(),
                "machine": platform.machine(),
                "device": str(self.device),
                "gpu": torch.cuda.get_device_name(self.device),
                "compute_capability": list(torch.cuda.get_device_capability(self.device)),
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
                "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            },
            "protocol": asdict(self.protocol),
            "preparation": prepared.preparation,
            "latency": Timing.summarize(samples).model_dump(),
            "calls_per_group": counts,
            "timing_instrumentation": "CUDA events only",
        }

    @staticmethod
    def compare(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
        """Speed-up of ``candidate`` over ``reference`` from two ``benchmark`` records.

        Both must come from the same GPU allocation and exact inputs: the
        reference kernel's own latency moved by 20 % between nodes on one
        day's runs (cuEquivariance's forward, 1.05–1.25 ms), more than many of
        the differences being measured. ``separated`` is whether the two sets
        of group means do not overlap at all — a coarse, protocol-level
        distinguishability check, not a confidence interval.
        """
        ref = Timing.model_validate(reference["latency"])
        cand = Timing.model_validate(candidate["latency"])
        if ref.mean_ms is None or cand.mean_ms is None or cand.mean_ms <= 0:
            raise ValueError("both records need a positive mean latency")
        return {
            "reference_ms": ref.mean_ms,
            "candidate_ms": cand.mean_ms,
            "speedup": ref.mean_ms / cand.mean_ms,
            "separated": bool(cand.max_ms < ref.min_ms or ref.max_ms < cand.min_ms),
            "same_environment": reference.get("environment") == candidate.get("environment"),
        }

    def trace(self, prepared: PreparedOperator, destination: Path, *, iterations: int = 3):
        """Write a Chrome trace and return op/device-event attribution separately."""
        if iterations < 1:
            raise ValueError("iterations must be positive")
        destination = Path(destination)
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with torch.cuda.device(self.device):
            torch.cuda.synchronize(self.device)
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
            ) as profiler:
                for _ in range(iterations):
                    with torch.profiler.record_function("molnex.op"):
                        prepared()
                torch.cuda.synchronize(self.device)
        profiler.export_chrome_trace(str(destination))
        return {
            "iterations": iterations,
            "operators": [
                {
                    "name": e.key,
                    "calls": e.count,
                    "self_cpu_us": e.self_cpu_time_total,
                    "self_device_us": e.self_device_time_total,
                }
                for e in profiler.key_averages()
            ],
            "device_activities": [
                {"name": e.name, "duration_us": e.time_range.elapsed_us()}
                for e in profiler.events()
                if e.device_type == torch.autograd.DeviceType.CUDA
            ],
        }

    def capture(self, prepared: PreparedOperator, *, iterations: int = 1):
        """Mark a bounded Nsight window after compile/warmup, across autograd threads.

        cudaProfilerStart/Stop gates both tools; NVTX labels the host range.
        NCU can filter kernel names without relying on thread-local NVTX ranges.
        """
        if iterations < 1:
            raise ValueError("iterations must be positive")
        with torch.cuda.device(self.device):
            torch.cuda.synchronize(self.device)
            torch.cuda.profiler.start()
            try:
                with torch.cuda.nvtx.range("molnex.op"):
                    for _ in range(iterations):
                        prepared()
                    torch.cuda.synchronize(self.device)
            finally:
                torch.cuda.profiler.stop()
