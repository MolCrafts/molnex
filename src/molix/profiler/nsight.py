"""Bounded NVIDIA CLI invocations, raw artifacts and machine-readable status.

No shell interpolation, implicit installation, administrator changes, or GPU
clock changes. Profiling failures remain failures, including missing reports.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


class NsightTool:
    """An explicitly selected executable, launched in an owned process group."""

    def __init__(self, executable: str):
        self.executable = executable

    def inspect(self) -> dict:
        """Discover the installed tool without requiring a GPU allocation."""
        resolved = shutil.which(self.executable)
        if resolved is None:
            return {"executable": self.executable, "available": False, "version": None}
        result = subprocess.run(
            [resolved, "--version"], capture_output=True, text=True, timeout=15, check=False
        )
        return {
            "executable": resolved,
            "available": result.returncode == 0,
            "version": (result.stdout + result.stderr).strip(),
        }

    def command(self, target: list[str], output: Path) -> list[str]:
        raise NotImplementedError

    @property
    def report_name(self) -> str:
        raise NotImplementedError

    def run(self, target: list[str], output: Path, *, timeout: float = 600) -> dict:
        """Capture one subprocess in a new directory and retain failure provenance."""
        if not target or timeout <= 0 or not math.isfinite(timeout):
            raise ValueError("target command and finite positive timeout are required")
        output = Path(output).resolve()
        output.mkdir(parents=True, exist_ok=False)
        command = self.command(target, output)
        record = {
            "command": command,
            "cwd": str(Path.cwd()),
            "timeout_s": timeout,
            "status": "starting",
            "tool": None,
        }
        destination = output / "capture.json"
        destination.write_text(json.dumps(record, indent=2) + "\n")
        start = time.monotonic()
        try:
            record["tool"] = self.inspect()
            if not record["tool"]["available"]:
                raise FileNotFoundError(f"profiler executable unavailable: {self.executable}")
            with (
                (output / "stdout.log").open("w") as stdout,
                (output / "stderr.log").open("w") as stderr,
            ):
                process = subprocess.Popen(
                    command, stdout=stdout, stderr=stderr, start_new_session=True
                )
                try:
                    code = process.wait(timeout=timeout)
                except BaseException:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass  # The child can finish between interruption and cleanup.
                    record["returncode"] = process.wait()
                    raise
            record["returncode"] = code
            if code:
                raise RuntimeError(f"profiler exited {code}; see {output}")
            report = output / self.report_name
            if not report.is_file() or report.stat().st_size == 0:
                raise RuntimeError(f"profiler produced no report: {report}")
            record["status"] = "complete"
            record["report"] = str(report)
        except BaseException as error:
            record["status"] = "failed"
            record["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            record["elapsed_s"] = time.monotonic() - start
            destination.write_text(json.dumps(record, indent=2) + "\n")
        return record


@dataclass(frozen=True)
class KernelCost:
    """One kernel's share of a captured window, per iteration of the workload."""

    name: str  #: kernel name without template arguments and parameter list
    instances: int  #: launches in the window
    total_ms: float  #: over the whole window
    per_iteration_ms: float  #: ``total_ms / iterations``
    mean_us: float  #: per launch


_API_ROW = re.compile(r"^cu(da)?[A-Z]")  # cudaDeviceSynchronize, cudaGraphLaunch, cuProfilerStart


def strip_template(name: str) -> str:
    """``void ns::kernel<float, 1>(const float*, …)`` → ``ns::kernel``."""
    name = name.strip().strip('"')
    if name.startswith("void "):
        name = name[5:]
    depth, out = 0, []
    for ch in name:
        if ch in "<(":
            depth += 1
        elif ch in ">)":
            depth -= 1
        elif depth == 0:
            out.append(ch)
    return "".join(out).strip()


class NsightSystems(NsightTool):
    """CUDA/NVTX timeline gated by the workload's CUDA profiler API window."""

    def __init__(self, executable: str = "nsys"):
        super().__init__(executable)

    @property
    def report_name(self) -> str:
        return "timeline.nsys-rep"

    def command(self, target: list[str], output: Path) -> list[str]:
        return [
            self.executable,
            "profile",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cpuctxsw=none",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--cuda-graph-trace=node",
            "--output",
            str(output / "timeline"),
            *target,
        ]

    def stats(self, report: Path, destination: Path) -> dict[str, list[dict[str, str]]]:
        """Export kernel/API summaries; the original timeline remains authoritative."""
        destination = Path(destination)
        with (
            destination.open("x") as stream,
            destination.with_suffix(".stderr.log").open("x") as errors,
        ):
            subprocess.run(
                [
                    self.executable,
                    "stats",
                    "--report",
                    "cuda_gpu_kern_sum,cuda_api_sum",
                    "--format",
                    "csv",
                    str(report),
                ],
                stdout=stream,
                stderr=errors,
                text=True,
                timeout=120,
                check=True,
            )
        return self.parse_stats(destination.read_text())

    @staticmethod
    def parse_stats(text: str) -> dict[str, list[dict[str, str]]]:
        """Separate NVIDIA's concatenated kernel/API CSV tables and diagnostics."""
        tables = {"kernels": [], "cuda_api": []}
        header, table = None, None
        for row in csv.reader(io.StringIO(text)):
            if {"Name", "Total Time (ns)", "Instances"}.issubset(row):
                header, table = row, "kernels"
            elif {"Name", "Total Time (ns)", "Num Calls"}.issubset(row):
                header, table = row, "cuda_api"
            elif header is not None and len(row) == len(header):
                tables[table].append(dict(zip(header, row, strict=True)))
            else:
                header, table = None, None
        if not any(tables.values()):
            raise ValueError("Nsight Systems statistics contain no CUDA events")
        return tables

    @staticmethod
    def kernel_summary(
        tables: dict[str, list[dict[str, str]]], *, iterations: int = 1, top: int | None = None
    ) -> list[KernelCost]:
        """Kernels of a captured window by total time, per iteration of the workload.

        Template arguments are stripped so the instantiations of one kernel
        (dtypes, orders, channel widths) stay distinguishable by count but
        readable; CUDA API rows nsys lists among the kernels
        (``cudaDeviceSynchronize``, ``cudaGraphLaunch``) are dropped. The
        workload's iteration count is the caller's: the capture window is
        theirs (``OperatorProfiler.capture(iterations=…)``).
        """
        if iterations < 1:
            raise ValueError("iterations must be positive")
        costs = []
        for row in tables.get("kernels", []):
            name = strip_template(row["Name"])
            if _API_ROW.match(name):
                continue
            total_ms = float(row["Total Time (ns)"]) / 1e6
            instances = int(row["Instances"])
            costs.append(
                KernelCost(
                    name=name,
                    instances=instances,
                    total_ms=total_ms,
                    per_iteration_ms=total_ms / iterations,
                    mean_us=float(row["Avg (ns)"]) / 1e3,
                )
            )
        costs.sort(key=lambda k: k.total_ms, reverse=True)
        return costs if top is None else costs[:top]

    @staticmethod
    def format_summary(costs: list[KernelCost]) -> str:
        """A table with the per-iteration sum on its last line."""
        rows = [
            f"{k.per_iteration_ms:9.3f} ms/iter {k.instances:6d}x {k.mean_us:9.1f} us  {k.name}"
            for k in costs
        ]
        rows.append(f"{sum(k.per_iteration_ms for k in costs):9.3f} ms/iter  kernel sum")
        return "\n".join(rows)


class NsightCompute(NsightTool):
    """Selected-kernel counters with bounded launches, no clock manipulation."""

    def __init__(
        self,
        executable: str = "ncu",
        *,
        kernel: str,
        launches: int = 4,
        sections: tuple[str, ...] = (
            "SpeedOfLight",
            "LaunchStats",
            "Occupancy",
            "MemoryWorkloadAnalysis",
            "WarpStateStats",
        ),
    ):
        super().__init__(executable)
        if not kernel or launches < 1 or not sections:
            raise ValueError("kernel filter, positive launches and sections are required")
        self.kernel = kernel
        self.launches = launches
        self.sections = sections

    @property
    def report_name(self) -> str:
        return "kernels.ncu-rep"

    def command(self, target: list[str], output: Path) -> list[str]:
        section_args = [value for section in self.sections for value in ("--section", section)]
        return [
            self.executable,
            "--target-processes",
            "application-only",
            "--profile-from-start",
            "off",
            "--kernel-name-base",
            "demangled",
            "--kernel-name",
            f"regex:{self.kernel}",
            "--launch-count",
            str(self.launches),
            "--clock-control",
            "none",
            "--cache-control",
            "none",
            *section_args,
            "--export",
            str(output / "kernels"),
            *target,
        ]

    def metrics(self, report: Path, destination: Path) -> list[dict[str, str]]:
        """Export and parse raw per-kernel counters, retaining NVIDIA metric units."""
        result = subprocess.run(
            [
                self.executable,
                "--import",
                str(report),
                "--csv",
                "--page",
                "raw",
                "--print-units",
                "base",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        destination = Path(destination)
        with destination.open("x") as stream:
            stream.write(result.stdout)
        with destination.with_suffix(".stderr.log").open("x") as stream:
            stream.write(result.stderr)
        result.check_returncode()
        return self.parse_metrics(result.stdout)

    @staticmethod
    def parse_metrics(text: str) -> list[dict[str, str]]:
        """Locate NVIDIA's CSV header after diagnostics; reject empty captures."""
        rows = list(csv.reader(io.StringIO(text)))
        for index, row in enumerate(rows):
            if {"ID", "Kernel Name", "Metric Name", "Metric Unit", "Metric Value"}.issubset(row):
                metrics = [
                    dict(zip(row, values, strict=True))
                    for values in rows[index + 1 :]
                    if len(values) == len(row) and values[0].isdigit()
                ]
                if metrics:
                    return metrics
                break
            if {"ID", "Kernel Name", "CC"}.issubset(row):
                # NCU's raw page is wide: one kernel per row, preceded by a
                # second header row carrying the unit for each metric column.
                if index + 1 >= len(rows):
                    break
                units = rows[index + 1]
                if len(units) != len(row) or units[0] != "":
                    raise ValueError("Nsight Compute raw CSV is missing its unit row")
                first_metric = row.index("CC") + 1
                metrics = []
                for values in rows[index + 2 :]:
                    if len(values) != len(row) or not values[0].isdigit():
                        continue
                    identity = dict(zip(row[:first_metric], values[:first_metric], strict=True))
                    metrics.extend(
                        {
                            **identity,
                            "Metric Name": row[column],
                            "Metric Unit": units[column],
                            "Metric Value": values[column],
                        }
                        for column in range(first_metric, len(row))
                    )
                if metrics:
                    return metrics
                break
        raise ValueError("Nsight Compute CSV contains no kernel metrics")
