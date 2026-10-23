"""Host-side protocol and exact input identity checks."""

import pytest
import torch

from molix.profiler.operator import OperatorProfiler, OperatorProtocol


class TestOperatorProtocol:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"warmup": -1},
            {"groups": 0},
            {"min_calls": 0},
            {"min_group_ms": float("nan")},
            {"min_group_ms": 0},
            {"max_calls": 99},
        ],
    )
    def test_invalid_budget(self, kwargs):
        with pytest.raises(ValueError):
            OperatorProtocol(**kwargs)


class TestOperatorProfiler:
    def test_fingerprint_tracks_mutation_and_layout(self):
        x = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4).T
        before = OperatorProfiler.fingerprint({"x": x})["x"]
        contiguous = OperatorProfiler.fingerprint({"x": x.contiguous()})["x"]
        assert before["sha256"] == contiguous["sha256"]
        assert before["stride"] != contiguous["stride"]
        x[0, 0] = 99
        after = OperatorProfiler.fingerprint({"x": x})["x"]
        assert before["sha256"] != after["sha256"]

    def test_cpu_is_rejected(self):
        with pytest.raises(ValueError, match="CUDA"):
            OperatorProfiler("cpu", OperatorProtocol())

    def test_compare_reports_speedup_and_separation(self):
        def record(samples, host):
            from molix.profiler.module import Timing

            return {
                "latency": Timing.summarize(samples).model_dump(),
                "environment": {"host": host},
            }

        reference = record([2.0, 2.1, 2.2], "n1")
        candidate = record([1.0, 1.05, 1.1], "n1")
        out = OperatorProfiler.compare(reference, candidate)
        assert out["speedup"] == pytest.approx(2.1 / 1.05)
        assert out["separated"] and out["same_environment"]
        overlapping = OperatorProfiler.compare(reference, record([1.9, 2.0, 2.3], "n2"))
        assert not overlapping["separated"] and not overlapping["same_environment"]
        with pytest.raises(ValueError):
            OperatorProfiler.compare(reference, record([0.0, 0.0], "n1"))
