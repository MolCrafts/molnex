"""Tests for the stateless sweep/spec/profiler/report API."""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn

from molix.profiler import Profiler, ProfilerConfig, ProfileSpec, ProfileSweep


class _Counter(nn.Module):
    def __init__(self, width: int = 1) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return batch.sum() * self.weight.sum()


class _Loader:
    def __init__(self, batch_size: int = 1) -> None:
        self.batch_size = batch_size

    def __iter__(self):
        yield torch.ones(self.batch_size, 1)
        yield torch.full((self.batch_size, 1), 2.0)


class _OneShotLoader:
    def __init__(self) -> None:
        self.done = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.done:
            raise StopIteration
        self.done = True
        return torch.ones(1, 1)


class _FailingModel(nn.Module):
    def forward(self, batch):
        raise RuntimeError("kernel failed")


def test_sweep_builds_classes_lazily_and_scan_values_override_kwargs():
    sweep = ProfileSweep(
        "counter",
        _Counter,
        _Loader,
        model_kwargs={"width": 1},
        dataloader_kwargs={"batch_size": 1},
        parameters={"precision": "fp32"},
    ).grid(model={"width": [2, 3]}, data={"batch_size": [4]})

    specs = list(sweep)

    assert len(specs) == 2
    assert specs[0].model_kwargs["width"] == 2
    assert specs[1].model_kwargs["width"] == 3
    assert specs[0].dataloader_kwargs["batch_size"] == 4
    assert specs[0].report_spec.parameters == {
        "precision": "fp32",
        "model.width": 2,
        "data.batch_size": 4,
    }


def test_spec_directly_constructs_the_declared_classes():
    spec = ProfileSpec("counter", _Counter, _Loader, model_args=(3,))

    with spec.build() as (model, dataloader):
        assert type(model) is _Counter
        assert type(dataloader) is _Loader
        assert model.weight.numel() == 3


def test_spec_rejects_factory_functions():
    with pytest.raises(TypeError, match="nn.Module class"):
        ProfileSpec("invalid", lambda: _Counter(), _Loader)
    with pytest.raises(TypeError, match="dataloader_class"):
        ProfileSpec("invalid", _Counter, lambda: _Loader())


def test_profiler_returns_an_immutable_pydantic_report():
    config = ProfilerConfig(
        device="cpu",
        warmup=1,
        steps=3,
        transfer_batch=False,
        memory="none",
        activities=("cpu",),
        hotspot_steps=2,
        hotspot_limit=10,
    )
    report = Profiler().profile(
        ProfileSpec("counter", _Counter, _Loader, parameters={"system": "test"}), config
    )

    assert report.model.error is None
    assert report.end_to_end.error is None
    assert len(report.model.latency.samples_ms) == 3
    assert len(report.end_to_end.latency.samples_ms) == 3
    assert report.model.throughput_steps_per_sec is not None
    assert report.end_to_end.throughput_steps_per_sec is not None
    assert report.hotspots
    payload = json.loads(report.model_dump_json())
    assert payload["config"]["device"] == "cpu"
    assert payload["spec"]["parameters"]["system"] == "test"
    assert "environment" not in payload
    with pytest.raises(Exception):
        report.errors = ("changed",)


def test_profiler_does_not_change_the_global_rng_state():
    config = ProfilerConfig(
        device="cpu",
        warmup=0,
        steps=1,
        transfer_batch=False,
        memory="none",
        activities=(),
        hotspot_steps=0,
        hotspot_limit=1,
    )
    before = torch.random.get_rng_state().clone()

    Profiler().profile(ProfileSpec("counter", _Counter, _Loader), config)

    assert torch.equal(torch.random.get_rng_state(), before)


def test_one_shot_loader_is_reported_without_hiding_the_exception_type():
    config = ProfilerConfig(
        device="cpu",
        warmup=0,
        steps=1,
        transfer_batch=False,
        memory="none",
        activities=(),
        hotspot_steps=0,
        hotspot_limit=1,
    )

    report = Profiler().profile(ProfileSpec("one-shot", _Counter, _OneShotLoader), config)

    assert "TypeError" in (report.model.error or "")
    assert "re-iterable" in (report.model.error or "")


def test_model_device_must_match_explicit_config():
    config = ProfilerConfig(
        device="meta",
        warmup=0,
        steps=1,
        transfer_batch=False,
        memory="none",
        activities=(),
        hotspot_steps=0,
        hotspot_limit=1,
    )

    report = Profiler().profile(ProfileSpec("wrong-device", _Counter, _Loader), config)

    assert "expected meta" in (report.model.error or "")


def test_failed_model_is_returned_as_one_error_without_stage_objects():
    config = ProfilerConfig(
        device="cpu",
        warmup=0,
        steps=1,
        transfer_batch=False,
        memory="none",
        activities=(),
        hotspot_steps=0,
        hotspot_limit=1,
    )

    report = Profiler().profile(ProfileSpec("failure", _FailingModel, _Loader), config)

    assert "kernel failed" in (report.model.error or "")
    assert not hasattr(report, "model_stage")


def test_grid_requires_sequence_values():
    sweep = ProfileSweep("counter", _Counter, _Loader)
    with pytest.raises(TypeError, match="sequence"):
        sweep.grid(model={"width": 2})
