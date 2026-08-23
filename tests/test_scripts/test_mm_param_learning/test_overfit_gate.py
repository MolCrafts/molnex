"""Unit tests for the Class-I overfit gate."""

from __future__ import annotations

import torch
from scripts.mm_param_learning.overfit_gate import _force_ok, _model_with_device_retry


class _StubFit:
    def __init__(self, results):
        self._results = list(results)
        self.calls = 0

    def model(self):
        self.calls += 1
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _busy_error() -> torch.AcceleratorError:
    return torch.AcceleratorError("CUDA error: CUDA-capable device(s) is/are busy or unavailable")


def test_retries_transient_device_busy(monkeypatch):
    """A busy-device error at model move is retried until it clears."""
    fit = _StubFit([_busy_error(), _busy_error(), "model"])
    monkeypatch.setattr("scripts.mm_param_learning.overfit_gate.time.sleep", lambda _: None)
    assert _model_with_device_retry(fit) == "model"
    assert fit.calls == 3


def test_other_accelerator_error_is_not_retried():
    """A non-busy accelerator error must propagate immediately."""
    fit = _StubFit([torch.AcceleratorError("CUDA error: out of memory")])
    try:
        _model_with_device_retry(fit, delay_s=0.0)
    except torch.AcceleratorError as exc:
        assert "out of memory" in str(exc)
    else:
        raise AssertionError("expected AcceleratorError to propagate")
    assert fit.calls == 1


def test_force_ok_requires_drop_and_scale():
    """Force overfit needs a Huber drop and predicted |F| at least half of ref."""
    assert _force_ok(final=9.0, initial=14.0, pred_med=8.0, ref_med=14.0)
    assert not _force_ok(final=14.0, initial=14.0, pred_med=8.0, ref_med=14.0)
    assert not _force_ok(final=9.0, initial=14.0, pred_med=2.0, ref_med=14.0)
