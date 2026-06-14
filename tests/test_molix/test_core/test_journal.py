"""Tests for ``molix.hooks.JournalHook`` (post io-hooks-refactor).

The hook now drives :class:`molix.io.JournalWriter` directly via a
kwargs ``append`` API (no ``MetricRecord`` dataclass, no
``MetricStore`` Protocol). This test suite mirrors the legacy
``Journal`` regression suite and pins the same invariants under
the new contract.

Acceptance traces:
    ac-004 — kwargs API used by JournalHook.
    ac-011 — train/val behavioural parity preserved.
"""

from __future__ import annotations

from typing import Any

import pytest

from molix.core.state import TrainState


class _CapturingWriter:
    """Test double — captures every kwargs append call without disk IO."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.flushed = False
        self.closed = False

    def append(
        self,
        *,
        type: str,
        key: str,
        step: int | None,
        wall_time_ns: int,
        value: Any,
        tags: dict[str, Any] | None = None,
    ) -> None:
        self.records.append(
            {
                "type": type,
                "key": key,
                "step": step,
                "wall_time_ns": wall_time_ns,
                "value": value,
                "tags": tags,
            }
        )

    def flush(self) -> None:
        self.flushed = True

    def close(self) -> None:
        self.closed = True


def test_mirrors_four_namespaces() -> None:
    """JournalHook mirrors all four TrainState namespaces as scalar records."""
    from molix.hooks import JournalHook

    state = TrainState()
    state["global_step"] = 100
    state["train"]["loss"] = 0.5
    state["performance"]["step_per_second"] = 12.3
    state["gpu"]["peak_gib"] = 1.4
    state["eval"]["MAE"] = 0.07

    writer = _CapturingWriter()
    hook = JournalHook(every_n_steps=1, store=writer)

    hook.on_train_batch_end(trainer=None, state=state, batch=None, outputs=None)
    hook.on_eval_step_complete(trainer=None, state=state)

    by_key = {r["key"]: r for r in writer.records if r["type"] == "scalar"}
    assert set(by_key) == {
        "train/loss",
        "performance/step_per_second",
        "gpu/peak_gib",
        "eval/MAE",
    }
    assert by_key["train/loss"]["value"] == pytest.approx(0.5)
    assert by_key["performance/step_per_second"]["value"] == pytest.approx(12.3)
    assert by_key["gpu/peak_gib"]["value"] == pytest.approx(1.4)
    assert by_key["eval/MAE"]["value"] == pytest.approx(0.07)

    other_types = [r for r in writer.records if r["type"] != "scalar"]
    assert other_types == [], f"Mirror pass should emit only scalar records; got {other_types!r}"


def test_hparams_emits_single_json_record() -> None:
    """hparams emitted as a single type=json record at step=0."""
    from molix.hooks import JournalHook

    state = TrainState()
    writer = _CapturingWriter()
    hparams = {"lr": 1e-3, "model": "mace"}
    hook = JournalHook(
        every_n_steps=10,
        store=writer,
        log_hparams=True,
        hparams=hparams,
    )

    hook.on_train_start(trainer=None, state=state)

    assert len(writer.records) == 1, (
        f"Expected exactly one record at on_train_start with hparams; "
        f"got {len(writer.records)}: {writer.records!r}"
    )
    rec = writer.records[0]
    assert rec["type"] == "json"
    assert rec["key"] == "hparams"
    assert rec["step"] == 0
    assert rec["value"] == hparams


def test_no_hparams_when_disabled() -> None:
    """log_hparams=False (default) emits zero records on on_train_start."""
    from molix.hooks import JournalHook

    state = TrainState()
    writer = _CapturingWriter()
    hook = JournalHook(every_n_steps=10, store=writer)
    hook.on_train_start(trainer=None, state=state)
    assert writer.records == []


def test_every_n_steps_gating() -> None:
    """on_train_batch_end skips when global_step % every_n_steps != 0."""
    from molix.hooks import JournalHook

    state = TrainState()
    state["global_step"] = 7
    state["train"]["loss"] = 0.5

    writer = _CapturingWriter()
    hook = JournalHook(every_n_steps=10, store=writer)
    hook.on_train_batch_end(trainer=None, state=state, batch=None, outputs=None)
    assert writer.records == []

    state["global_step"] = 10
    hook.on_train_batch_end(trainer=None, state=state, batch=None, outputs=None)
    assert len(writer.records) == 1
    assert writer.records[0]["key"] == "train/loss"


def test_close_propagates_to_store() -> None:
    """on_train_end calls store.close()."""
    from molix.hooks import JournalHook

    state = TrainState()
    writer = _CapturingWriter()
    hook = JournalHook(every_n_steps=1, store=writer)
    hook.on_train_end(trainer=None, state=state)
    assert writer.closed is True


def test_non_scalar_state_value_skipped() -> None:
    """Vector / non-scalar state values are silently skipped."""
    import torch

    from molix.hooks import JournalHook

    state = TrainState()
    state["global_step"] = 1
    state["train"]["loss"] = 0.5
    state["train"]["embedding"] = torch.zeros(8)

    writer = _CapturingWriter()
    hook = JournalHook(every_n_steps=1, store=writer)
    hook.on_train_batch_end(trainer=None, state=state, batch=None, outputs=None)

    keys = {r["key"] for r in writer.records}
    assert keys == {"train/loss"}, f"Non-scalar values should be skipped; got keys {keys!r}"


def test_no_interference_with_existing_hooks() -> None:
    """JournalHook does not perturb other hooks' state mutations."""
    from molix.hooks import JournalHook

    state = TrainState()
    state["global_step"] = 1
    state["train"]["loss"] = 0.5
    state["eval"]["MAE"] = 0.07

    writer = _CapturingWriter()
    journal = JournalHook(every_n_steps=1, store=writer)

    journal.on_train_batch_end(trainer=None, state=state, batch=None, outputs=None)
    journal.on_eval_step_complete(trainer=None, state=state)

    assert state["train"]["loss"] == 0.5
    assert state["eval"]["MAE"] == 0.07
    assert state["global_step"] == 1
    assert state["train"] == {"loss": 0.5}
    assert state["eval"] == {"MAE": 0.07}


# ---------------------------------------------------------------------------
# Async writer (pinned-buffer + cuda.Event + background drain) — these run on
# CPU by driving the drain loop with fake events, so they need no GPU.
# ---------------------------------------------------------------------------


class _FakeEvent:
    """Stand-in for a torch.cuda.Event: reports not-ready for the first
    ``ready_after`` ``query()`` calls, then ready forever."""

    def __init__(self, ready_after: int = 0) -> None:
        self._calls = 0
        self._ready_after = ready_after

    def query(self) -> bool:
        ready = self._calls >= self._ready_after
        self._calls += 1
        return ready


def _scalar_factory(key: str, step: int, value: float):
    return lambda: {
        "type": "scalar",
        "key": key,
        "step": step,
        "wall_time_ns": 0,
        "value": value,
    }


def test_async_writer_writes_ready_records() -> None:
    """An event=None record is appended verbatim by the background thread."""
    from molix.hooks.journal import _AsyncJournalWriter

    writer = _CapturingWriter()
    aw = _AsyncJournalWriter(writer, poll_interval_s=1e-4)
    aw.submit(None, _scalar_factory("train/loss", 1, 0.5))
    aw.shutdown()

    assert len(writer.records) == 1
    assert writer.records[0]["key"] == "train/loss"
    assert writer.records[0]["value"] == 0.5


def test_async_writer_preserves_per_key_step_order() -> None:
    """A not-ready record is written BEFORE a later ready record for the same
    key — the drain loop never advances past a pending event."""
    from molix.hooks.journal import _AsyncJournalWriter

    writer = _CapturingWriter()
    aw = _AsyncJournalWriter(writer, poll_interval_s=1e-4)
    # step 10's event is not ready for its first 3 queries; step 20 is ready.
    aw.submit(_FakeEvent(ready_after=3), _scalar_factory("train/loss", 10, 1.0))
    aw.submit(None, _scalar_factory("train/loss", 20, 2.0))
    aw.shutdown()

    steps = [r["step"] for r in writer.records]
    assert steps == [10, 20], f"expected in-order [10, 20], got {steps}"


def test_async_writer_drains_all_before_shutdown_returns() -> None:
    """shutdown() blocks until every queued record (incl. delayed ones) lands."""
    from molix.hooks.journal import _AsyncJournalWriter

    writer = _CapturingWriter()
    aw = _AsyncJournalWriter(writer, poll_interval_s=1e-4)
    for i in range(5):
        aw.submit(_FakeEvent(ready_after=2), _scalar_factory("train/loss", i, float(i)))
    aw.shutdown()

    assert [r["step"] for r in writer.records] == [0, 1, 2, 3, 4]


def test_async_writer_shutdown_without_submit_is_noop() -> None:
    """shutdown() is safe when the thread was never started."""
    from molix.hooks.journal import _AsyncJournalWriter

    writer = _CapturingWriter()
    aw = _AsyncJournalWriter(writer, poll_interval_s=1e-4)
    aw.shutdown()  # must not hang or raise
    assert writer.records == []


def test_async_falls_back_to_sync_without_cuda(monkeypatch) -> None:
    """async_writes=True with no CUDA behaves exactly like the sync path."""
    import torch

    from molix.hooks import JournalHook

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    state = TrainState()
    state["global_step"] = 1
    state["train"]["loss"] = 0.5

    writer = _CapturingWriter()
    hook = JournalHook(every_n_steps=1, store=writer, async_writes=True)
    assert hook._async is None  # no background thread spun up

    hook.on_train_batch_end(trainer=None, state=state, batch=None, outputs=None)
    keys = {r["key"] for r in writer.records}
    assert keys == {"train/loss"}
    assert writer.records[0]["value"] == pytest.approx(0.5)


def test_async_on_train_end_drains_then_closes() -> None:
    """on_train_end joins the drain thread before closing the writer, so all
    enqueued records are present and no append races close()."""
    from molix.hooks import JournalHook
    from molix.hooks.journal import _AsyncJournalWriter

    writer = _CapturingWriter()
    hook = JournalHook(every_n_steps=1, store=writer)
    # Force-activate async regardless of CUDA availability for the test.
    hook._async = _AsyncJournalWriter(writer, poll_interval_s=1e-4)
    hook._async.submit(_FakeEvent(ready_after=2), _scalar_factory("train/loss", 1, 0.5))

    hook.on_train_end(trainer=None, state=TrainState())

    assert writer.closed is True
    assert any(r["key"] == "train/loss" for r in writer.records), "record dropped before close"
