"""Stateless profiling of explicitly configured PyTorch model classes."""

from __future__ import annotations

import itertools
import statistics
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from molix.core.steps import batch_to
from molix.profiler._utils import batch_counts


class Timing(BaseModel):
    """Raw latency samples and summary statistics in milliseconds."""

    model_config = ConfigDict(frozen=True)

    samples_ms: tuple[float, ...] = ()
    mean_ms: float | None = None
    std_ms: float | None = None
    p50_ms: float | None = None
    p95_ms: float | None = None
    min_ms: float | None = None
    max_ms: float | None = None

    @classmethod
    def summarize(cls, samples: Sequence[float]) -> "Timing":
        """Summarize an explicit sequence of millisecond observations."""
        values = tuple(float(value) for value in samples)
        if not values:
            return cls()
        ordered = sorted(values)

        def percentile(q: float) -> float:
            position = (len(ordered) - 1) * q
            low = int(position)
            high = min(low + 1, len(ordered) - 1)
            return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

        return cls(
            samples_ms=values,
            mean_ms=statistics.fmean(values),
            std_ms=statistics.pstdev(values) if len(values) > 1 else 0.0,
            p50_ms=percentile(0.50),
            p95_ms=percentile(0.95),
            min_ms=min(values),
            max_ms=max(values),
        )


class Memory(BaseModel):
    """CUDA allocator measurements for the model-only latency window."""

    model_config = ConfigDict(frozen=True)

    allocated_baseline_mb: float | None = None
    allocated_peak_mb: float | None = None
    allocated_delta_mb: float | None = None
    reserved_baseline_mb: float | None = None
    reserved_peak_mb: float | None = None


class Hotspot(BaseModel):
    """One operator aggregated by ``torch.profiler``."""

    model_config = ConfigDict(frozen=True)

    name: str
    calls: int
    self_cpu_ms: float | None = None
    self_device_ms: float | None = None


class Measurement(BaseModel):
    """Measurements for one declared boundary."""

    model_config = ConfigDict(frozen=True)

    latency: Timing = Field(default_factory=Timing)
    throughput_steps_per_sec: float | None = None
    throughput_atoms_per_sec: float | None = None
    throughput_graphs_per_sec: float | None = None
    error: str | None = None


class ProfilerConfig(BaseModel):
    """Complete, explicit measurement configuration.

    No field is derived from the host or model. ``device`` controls batch
    transfer and synchronization. The caller is responsible for constructing
    the model class on that same device.
    """

    model_config = ConfigDict(frozen=True)

    device: str
    warmup: int
    steps: int
    transfer_batch: bool
    memory: Literal["none", "cuda"]
    activities: tuple[Literal["cpu", "cuda"], ...]
    hotspot_steps: int
    hotspot_limit: int

    @model_validator(mode="after")
    def validate_protocol(self) -> "ProfilerConfig":
        if self.warmup < 0:
            raise ValueError("warmup must be non-negative")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.hotspot_steps < 0:
            raise ValueError("hotspot_steps must be non-negative")
        if self.hotspot_limit <= 0:
            raise ValueError("hotspot_limit must be positive")
        if self.memory == "cuda" and torch.device(self.device).type != "cuda":
            raise ValueError("memory='cuda' requires a CUDA device")
        if self.hotspot_steps and not self.activities:
            raise ValueError("hotspot_steps requires at least one profiler activity")
        return self


class ReportSpec(BaseModel):
    """Serializable identity and labels supplied by the caller."""

    model_config = ConfigDict(frozen=True)

    name: str
    model_class: str
    dataloader_class: str
    parameters: dict[str, JsonValue]


class Report(BaseModel):
    """Immutable Pydantic result for one concrete profile point."""

    model_config = ConfigDict(frozen=True)

    spec: ReportSpec
    config: ProfilerConfig
    model: Measurement
    end_to_end: Measurement
    memory: Memory
    hotspots: tuple[Hotspot, ...] = ()
    errors: tuple[str, ...] = ()

    def flat(self) -> dict[str, JsonValue]:
        """Return one row for tabular analysis."""
        return {
            **self.spec.parameters,
            "name": self.spec.name,
            "model.mean_ms": self.model.latency.mean_ms,
            "model.p95_ms": self.model.latency.p95_ms,
            "model.steps_per_sec": self.model.throughput_steps_per_sec,
            "model.atoms_per_sec": self.model.throughput_atoms_per_sec,
            "model.graphs_per_sec": self.model.throughput_graphs_per_sec,
            "end_to_end.mean_ms": self.end_to_end.latency.mean_ms,
            "end_to_end.p95_ms": self.end_to_end.latency.p95_ms,
            "end_to_end.steps_per_sec": self.end_to_end.throughput_steps_per_sec,
            "memory.allocated_delta_mb": self.memory.allocated_delta_mb,
            "model.error": self.model.error,
            "end_to_end.error": self.end_to_end.error,
        }


@dataclass(frozen=True)
class ProfileSpec:
    """One concrete pair of classes and their constructor arguments."""

    name: str
    model_class: type[torch.nn.Module]
    dataloader_class: type[Any]
    model_args: tuple[Any, ...] = ()
    model_kwargs: Mapping[str, Any] = field(default_factory=dict)
    dataloader_args: tuple[Any, ...] = ()
    dataloader_kwargs: Mapping[str, Any] = field(default_factory=dict)
    parameters: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.model_class, type) or not issubclass(
            self.model_class, torch.nn.Module
        ):
            raise TypeError("model_class must be an nn.Module class")
        if not isinstance(self.dataloader_class, type):
            raise TypeError("dataloader_class must be a class")
        object.__setattr__(self, "model_args", tuple(self.model_args))
        object.__setattr__(self, "dataloader_args", tuple(self.dataloader_args))
        object.__setattr__(self, "model_kwargs", MappingProxyType(dict(self.model_kwargs)))
        object.__setattr__(
            self, "dataloader_kwargs", MappingProxyType(dict(self.dataloader_kwargs))
        )
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))

    @contextmanager
    def build(self) -> Iterator[tuple[torch.nn.Module, Any]]:
        """Directly instantiate the configured model and dataloader classes."""
        model = self.model_class(*self.model_args, **self.model_kwargs)
        dataloader = self.dataloader_class(*self.dataloader_args, **self.dataloader_kwargs)
        try:
            yield model, dataloader
        finally:
            del model
            del dataloader

    @property
    def report_spec(self) -> ReportSpec:
        """Return only caller-supplied, serializable report metadata."""
        return ReportSpec(
            name=self.name,
            model_class=f"{self.model_class.__module__}.{self.model_class.__qualname__}",
            dataloader_class=(
                f"{self.dataloader_class.__module__}.{self.dataloader_class.__qualname__}"
            ),
            parameters=dict(self.parameters),
        )


class ProfileSweep:
    """Pure Cartesian expansion of constructor keyword arguments."""

    def __init__(
        self,
        name: str,
        model_class: type[torch.nn.Module],
        dataloader_class: type[Any],
        *,
        model_args: Sequence[Any] = (),
        model_kwargs: Mapping[str, Any] | None = None,
        dataloader_args: Sequence[Any] = (),
        dataloader_kwargs: Mapping[str, Any] | None = None,
        parameters: Mapping[str, JsonValue] | None = None,
        _model_grid: Mapping[str, Sequence[Any]] | None = None,
        _data_grid: Mapping[str, Sequence[Any]] | None = None,
    ) -> None:
        if not isinstance(model_class, type) or not issubclass(model_class, torch.nn.Module):
            raise TypeError("model_class must be an nn.Module class")
        if not isinstance(dataloader_class, type):
            raise TypeError("dataloader_class must be a class")
        self.name = name
        self.model_class = model_class
        self.dataloader_class = dataloader_class
        self.model_args = tuple(model_args)
        self.model_kwargs = dict(model_kwargs or {})
        self.dataloader_args = tuple(dataloader_args)
        self.dataloader_kwargs = dict(dataloader_kwargs or {})
        self.parameters = dict(parameters or {})
        self._model_grid = self._normalize_grid(_model_grid)
        self._data_grid = self._normalize_grid(_data_grid)

    @staticmethod
    def _normalize_grid(
        grid: Mapping[str, Sequence[Any]] | None,
    ) -> dict[str, tuple[Any, ...]]:
        normalized: dict[str, tuple[Any, ...]] = {}
        for key, values in (grid or {}).items():
            if isinstance(values, str) or not isinstance(values, Sequence):
                raise TypeError(f"grid values for {key!r} must be a sequence")
            normalized[key] = tuple(values)
        return normalized

    def grid(
        self,
        *,
        model: Mapping[str, Sequence[Any]] | None = None,
        data: Mapping[str, Sequence[Any]] | None = None,
    ) -> "ProfileSweep":
        """Return a new sweep; scan values override fixed kwargs."""
        return ProfileSweep(
            self.name,
            self.model_class,
            self.dataloader_class,
            model_args=self.model_args,
            model_kwargs=self.model_kwargs,
            dataloader_args=self.dataloader_args,
            dataloader_kwargs=self.dataloader_kwargs,
            parameters=self.parameters,
            _model_grid=self._model_grid | self._normalize_grid(model),
            _data_grid=self._data_grid | self._normalize_grid(data),
        )

    @staticmethod
    def _points(grid: Mapping[str, Sequence[Any]]) -> Iterator[dict[str, Any]]:
        keys = tuple(grid)
        for values in itertools.product(*(grid[key] for key in keys)):
            yield dict(zip(keys, values, strict=True))

    def __iter__(self) -> Iterator[ProfileSpec]:
        for model_point, data_point in itertools.product(
            self._points(self._model_grid), self._points(self._data_grid)
        ):
            point_parameters: dict[str, JsonValue] = dict(self.parameters)
            point_parameters.update({f"model.{key}": value for key, value in model_point.items()})
            point_parameters.update({f"data.{key}": value for key, value in data_point.items()})
            yield ProfileSpec(
                name=self.name,
                model_class=self.model_class,
                dataloader_class=self.dataloader_class,
                model_args=self.model_args,
                model_kwargs=self.model_kwargs | model_point,
                dataloader_args=self.dataloader_args,
                dataloader_kwargs=self.dataloader_kwargs | data_point,
                parameters=point_parameters,
            )


class _BatchStream:
    """Cycle a re-iterable dataloader without caching its batches."""

    def __init__(self, dataloader: Any) -> None:
        if iter(dataloader) is dataloader:
            raise TypeError("dataloader must be re-iterable")
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def next(self) -> Any:
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            try:
                return next(self.iterator)
            except StopIteration as exc:
                raise RuntimeError("dataloader yielded no batches") from exc


class Profiler:
    """Stateless executor; every call receives its complete configuration."""

    def profile(self, spec: ProfileSpec, config: ProfilerConfig) -> Report:
        """Measure one spec without changing global RNG or restoring object state."""
        errors: list[str] = []
        model_latency, memory = self._model_latency(spec, config)
        errors.extend(self._errors(model_latency))

        model = model_latency
        if model.error is None:
            throughput = self._model_throughput(spec, config)
            errors.extend(self._errors(throughput))
            model = model.model_copy(
                update={
                    "throughput_steps_per_sec": throughput.throughput_steps_per_sec,
                    "throughput_atoms_per_sec": throughput.throughput_atoms_per_sec,
                    "throughput_graphs_per_sec": throughput.throughput_graphs_per_sec,
                    "error": throughput.error,
                }
            )

        end_to_end = self._end_to_end(spec, config)
        errors.extend(self._errors(end_to_end))
        hotspots, hotspot_error = self._hotspots(spec, config)
        if hotspot_error is not None:
            errors.append(hotspot_error)

        return Report(
            spec=spec.report_spec,
            config=config,
            model=model,
            end_to_end=end_to_end,
            memory=memory,
            hotspots=hotspots,
            errors=tuple(errors),
        )

    @staticmethod
    def _errors(measurement: Measurement) -> tuple[str, ...]:
        return (measurement.error,) if measurement.error is not None else ()

    def _model_latency(
        self, spec: ProfileSpec, config: ProfilerConfig
    ) -> tuple[Measurement, Memory]:
        try:
            with spec.build() as (model, dataloader):
                self._validate_model(model, config)
                batch = self._prepare(_BatchStream(dataloader).next(), config)
                for _ in range(config.warmup):
                    output = model(batch)
                    del output
                self._synchronize(config)
                baseline = self._reset_memory(config)
                samples = [self._timed_call(model, batch, config) for _ in range(config.steps)]
                memory = self._finish_memory(baseline, config)
                atoms, graphs = batch_counts(batch)
                timing = Timing.summarize(samples)
                mean_seconds = (timing.mean_ms or 0.0) / 1000
                return (
                    Measurement(
                        latency=timing,
                        throughput_atoms_per_sec=atoms / mean_seconds if atoms else None,
                        throughput_graphs_per_sec=graphs / mean_seconds if graphs else None,
                    ),
                    memory,
                )
        except Exception as exc:
            return Measurement(error=self._message(exc)), Memory()

    def _model_throughput(self, spec: ProfileSpec, config: ProfilerConfig) -> Measurement:
        try:
            with spec.build() as (model, dataloader):
                self._validate_model(model, config)
                batch = self._prepare(_BatchStream(dataloader).next(), config)
                for _ in range(config.warmup):
                    output = model(batch)
                    del output
                self._synchronize(config)
                start = time.perf_counter()
                for _ in range(config.steps):
                    output = model(batch)
                    del output
                self._synchronize(config)
                elapsed = time.perf_counter() - start
                atoms, graphs = batch_counts(batch)
                steps_per_second = config.steps / elapsed
                return Measurement(
                    throughput_steps_per_sec=steps_per_second,
                    throughput_atoms_per_sec=atoms * steps_per_second if atoms else None,
                    throughput_graphs_per_sec=graphs * steps_per_second if graphs else None,
                )
        except Exception as exc:
            return Measurement(error=self._message(exc))

    def _end_to_end(self, spec: ProfileSpec, config: ProfilerConfig) -> Measurement:
        latency = self._end_to_end_latency(spec, config)
        if latency.error is not None:
            return latency
        throughput = self._end_to_end_throughput(spec, config)
        return latency.model_copy(
            update={
                "throughput_steps_per_sec": throughput.throughput_steps_per_sec,
                "throughput_atoms_per_sec": throughput.throughput_atoms_per_sec,
                "throughput_graphs_per_sec": throughput.throughput_graphs_per_sec,
                "error": throughput.error,
            }
        )

    def _end_to_end_latency(self, spec: ProfileSpec, config: ProfilerConfig) -> Measurement:
        try:
            with spec.build() as (model, dataloader):
                self._validate_model(model, config)
                stream = _BatchStream(dataloader)
                for _ in range(config.warmup):
                    output = model(self._prepare(stream.next(), config))
                    del output
                self._synchronize(config)
                samples: list[float] = []
                for _ in range(config.steps):
                    self._synchronize(config)
                    start_step = time.perf_counter()
                    batch = self._prepare(stream.next(), config)
                    output = model(batch)
                    del output
                    self._synchronize(config)
                    samples.append((time.perf_counter() - start_step) * 1000)
                return Measurement(latency=Timing.summarize(samples))
        except Exception as exc:
            return Measurement(error=self._message(exc))

    def _end_to_end_throughput(self, spec: ProfileSpec, config: ProfilerConfig) -> Measurement:
        try:
            with spec.build() as (model, dataloader):
                self._validate_model(model, config)
                stream = _BatchStream(dataloader)
                for _ in range(config.warmup):
                    output = model(self._prepare(stream.next(), config))
                    del output
                self._synchronize(config)
                atoms = 0
                graphs = 0
                start = time.perf_counter()
                for _ in range(config.steps):
                    batch = self._prepare(stream.next(), config)
                    output = model(batch)
                    del output
                    batch_atoms, batch_graphs = batch_counts(batch)
                    atoms += batch_atoms
                    graphs += batch_graphs
                self._synchronize(config)
                elapsed = time.perf_counter() - start
                return Measurement(
                    throughput_steps_per_sec=config.steps / elapsed,
                    throughput_atoms_per_sec=atoms / elapsed if atoms else None,
                    throughput_graphs_per_sec=graphs / elapsed if graphs else None,
                )
        except Exception as exc:
            return Measurement(error=self._message(exc))

    def _hotspots(
        self, spec: ProfileSpec, config: ProfilerConfig
    ) -> tuple[tuple[Hotspot, ...], str | None]:
        if config.hotspot_steps == 0:
            return (), None
        try:
            activities = {
                "cpu": torch.profiler.ProfilerActivity.CPU,
                "cuda": torch.profiler.ProfilerActivity.CUDA,
            }
            with spec.build() as (model, dataloader):
                self._validate_model(model, config)
                batch = self._prepare(_BatchStream(dataloader).next(), config)
                for _ in range(config.warmup):
                    output = model(batch)
                    del output
                with torch.profiler.profile(
                    activities=[activities[name] for name in config.activities],
                    record_shapes=True,
                ) as profile:
                    for _ in range(config.hotspot_steps):
                        output = model(batch)
                        del output
                self._synchronize(config)
                key = (
                    "self_device_time_total"
                    if "cuda" in config.activities
                    else "self_cpu_time_total"
                )
                events = sorted(
                    profile.key_averages(),
                    key=lambda event: float(getattr(event, key, 0.0)),
                    reverse=True,
                )[: config.hotspot_limit]
                return (
                    tuple(
                        Hotspot(
                            name=event.key,
                            calls=int(event.count),
                            self_cpu_ms=self._milliseconds(event, "self_cpu_time_total"),
                            self_device_ms=self._milliseconds(event, "self_device_time_total"),
                        )
                        for event in events
                    ),
                    None,
                )
        except Exception as exc:
            return (), self._message(exc)

    @staticmethod
    def _prepare(batch: Any, config: ProfilerConfig) -> Any:
        if not config.transfer_batch:
            return batch
        return batch_to(batch, device=config.device)

    @staticmethod
    def _validate_model(model: torch.nn.Module, config: ProfilerConfig) -> None:
        expected = torch.device(config.device)
        devices = {value.device for value in itertools.chain(model.parameters(), model.buffers())}
        if devices and devices != {expected}:
            found = ", ".join(sorted(str(device) for device in devices))
            raise ValueError(f"model tensors are on [{found}], expected {expected}")

    @staticmethod
    def _synchronize(config: ProfilerConfig) -> None:
        device = torch.device(config.device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def _timed_call(self, model: torch.nn.Module, batch: Any, config: ProfilerConfig) -> float:
        self._synchronize(config)
        start = time.perf_counter()
        output = model(batch)
        del output
        self._synchronize(config)
        return (time.perf_counter() - start) * 1000

    def _reset_memory(self, config: ProfilerConfig) -> Memory:
        if config.memory == "none":
            return Memory()
        device = torch.device(config.device)
        self._synchronize(config)
        allocated = torch.cuda.memory_allocated(device) / 1e6
        reserved = torch.cuda.memory_reserved(device) / 1e6
        torch.cuda.reset_peak_memory_stats(device)
        return Memory(allocated_baseline_mb=allocated, reserved_baseline_mb=reserved)

    def _finish_memory(self, baseline: Memory, config: ProfilerConfig) -> Memory:
        if config.memory == "none":
            return baseline
        device = torch.device(config.device)
        peak = torch.cuda.max_memory_allocated(device) / 1e6
        return Memory(
            allocated_baseline_mb=baseline.allocated_baseline_mb,
            allocated_peak_mb=peak,
            allocated_delta_mb=peak - (baseline.allocated_baseline_mb or 0.0),
            reserved_baseline_mb=baseline.reserved_baseline_mb,
            reserved_peak_mb=torch.cuda.max_memory_reserved(device) / 1e6,
        )

    @staticmethod
    def _milliseconds(event: Any, attribute: str) -> float | None:
        value = getattr(event, attribute, None)
        return float(value) / 1000 if value is not None else None

    @staticmethod
    def _message(exc: Exception) -> str:
        return f"{type(exc).__name__}: {str(exc).splitlines()[0][:300]}"


__all__ = [
    "Hotspot",
    "Measurement",
    "Memory",
    "Profiler",
    "ProfilerConfig",
    "ProfileSpec",
    "ProfileSweep",
    "Report",
    "ReportSpec",
    "Timing",
]
