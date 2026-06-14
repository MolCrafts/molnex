"""Regression tests for the MetricsHook / TrainState namespace contract.

Guards three invariants: (1) ``TrainState`` rejects slash-prefix
top-level writes; (2) ``MetricsHook`` keeps independent metric
instances per phase; (3) a train→eval→train cycle leaves ``train/*``
reflecting only the latest train batch.
"""

from __future__ import annotations

import pytest
import torch

from molix.core.metrics import MAE, RMSE
from molix.core.state import TrainState, display, resolve
from molix.hooks import MetricsHook

# ---------------------------------------------------------------------------
# Invariant 1 — TrainState rejects flat slash-prefix writes
# ---------------------------------------------------------------------------


def test_trainstate_rejects_slash_topwrite():
    """Writing ``state["train/loss"] = x`` raises — must nest explicitly."""
    import pytest

    state = TrainState()
    with pytest.raises(ValueError, match="namespace"):
        state["train/loss"] = 0.5


def test_trainstate_rejects_non_dict_namespace():
    """Writing a non-dict to a namespace sub-dict key is rejected."""
    import pytest

    state = TrainState()
    with pytest.raises(ValueError, match="sub-namespace"):
        state["train"] = 0.5


def test_resolve_walks_tuple_paths():
    """``resolve(state, ("train", "loss"))`` returns ``state["train"]["loss"]``."""
    state = TrainState()
    state["train"]["loss"] = 0.5
    state["eval"]["MAE"] = 0.1
    state["gpu"]["peak_gib"] = 2.3
    assert resolve(state, ("train", "loss")) == 0.5
    assert resolve(state, ("eval", "MAE")) == 0.1
    assert resolve(state, ("gpu", "peak_gib")) == 2.3
    assert resolve(state, ("missing", "key"), default=-1.0) == -1.0
    # Top-level string keys still work.
    assert resolve(state, "epoch") == 0


def test_display_renders_tuple_paths():
    """Display strings join tuples with ``/`` for user-facing rendering."""
    assert display(("train", "loss")) == "train/loss"
    assert display("epoch") == "epoch"


def test_trainstate_namespaces_prewired():
    """The four namespace sub-dicts exist on a fresh TrainState."""
    state = TrainState()
    for ns in ("train", "eval", "performance", "gpu"):
        assert isinstance(state[ns], dict), f"{ns} namespace missing or wrong type"


# ---------------------------------------------------------------------------
# Invariant 2 — MetricsHook isolates train and val metric instances
# ---------------------------------------------------------------------------


def test_metrics_hook_train_and_val_are_separate_instances():
    """Train and val metric accumulators must be distinct objects."""
    mae = MAE()
    hook = MetricsHook(metrics=[mae], pred_key="predictions", target_key="targets")
    # The user-supplied list is deep-copied; no shared reference leaks through.
    assert hook.train_metrics[0] is not hook.val_metrics[0]
    assert hook.train_metrics[0] is not mae
    assert hook.val_metrics[0] is not mae


def test_metrics_hook_val_update_does_not_touch_train_buffer():
    """Calling val pathway must leave train accumulator unchanged."""
    hook = MetricsHook(metrics=[MAE()], pred_key="predictions", target_key="targets")
    state = TrainState()

    # Run one eval batch: accumulates into val_metrics only.
    preds = torch.zeros(4)
    targets = torch.ones(4)
    batch = {"targets": targets}
    outputs = {"predictions": preds}
    hook.on_eval_batch_end(trainer=None, state=state, batch=batch, outputs=outputs)

    # Train buffer must be empty — val path must not have written into it.
    train_mae = hook.train_metrics[0]
    assert not getattr(train_mae, "preds", []), "val path leaked into train_metrics buffer"


# ---------------------------------------------------------------------------
# Invariant 3 — full train→eval→train cycle keeps train state uncorrupted
# ---------------------------------------------------------------------------


def test_train_eval_train_cycle_train_mae_reflects_last_batch_only():
    """After train→eval→train, train/MAE must be the last train batch's MAE.

    This is the direct regression test for the observed bug where
    ``train/MAE`` reported cumulative-plus-eval-polluted values after
    each eval phase.
    """
    hook = MetricsHook(
        metrics=[MAE(), RMSE()],
        pred_key="predictions",
        target_key="targets",
    )
    state = TrainState()

    def train_batch(err: float):
        preds = torch.full((4,), fill_value=err, dtype=torch.float32)
        targets = torch.zeros(4)
        hook.on_train_batch_end(
            trainer=None,
            state=state,
            batch={"targets": targets},
            outputs={"predictions": preds},
        )

    def eval_batch(err: float):
        preds = torch.full((4,), fill_value=err, dtype=torch.float32)
        targets = torch.zeros(4)
        hook.on_eval_batch_end(
            trainer=None,
            state=state,
            batch={"targets": targets},
            outputs={"predictions": preds},
        )

    hook.on_epoch_start(trainer=None, state=state)

    # Two train batches with MAE=0.1 then MAE=0.2 (per batch, not cumulative).
    # Train metrics are stored as 0-d device tensors (no hot-path sync), so
    # materialise with float() before comparing.
    train_batch(0.1)
    assert float(state["train"]["MAE"]) == pytest.approx(0.1)
    train_batch(0.2)
    assert float(state["train"]["MAE"]) == pytest.approx(0.2)

    # Eval phase with an outlier batch (MAE=50) and a normal one (MAE=0.3).
    hook.on_eval_phase_start(trainer=None, state=state)
    eval_batch(50.0)
    eval_batch(0.3)
    hook.on_eval_step_complete(trainer=None, state=state)
    # Val MAE averages over the two eval batches: (50 + 0.3) / 2 = 25.15
    assert abs(state["eval"]["MAE"] - 25.15) < 1e-4

    # Next train batch: MAE must be exactly this batch's value, NOT
    # contaminated by the outlier eval batch.
    train_batch(0.25)
    assert abs(float(state["train"]["MAE"]) - 0.25) < 1e-6, (
        f"train/MAE polluted after eval: got {state['train']['MAE']}, expected 0.25"
    )


def test_val_metrics_reset_between_eval_phases():
    """Repeated eval phases must not accumulate across phases."""
    hook = MetricsHook(metrics=[MAE()], pred_key="predictions", target_key="targets")
    state = TrainState()

    def eval_batch(err: float):
        preds = torch.full((4,), fill_value=err, dtype=torch.float32)
        targets = torch.zeros(4)
        hook.on_eval_batch_end(
            trainer=None,
            state=state,
            batch={"targets": targets},
            outputs={"predictions": preds},
        )

    # First eval phase — MAE=10.
    hook.on_eval_phase_start(trainer=None, state=state)
    eval_batch(10.0)
    hook.on_eval_step_complete(trainer=None, state=state)
    assert abs(state["eval"]["MAE"] - 10.0) < 1e-6

    # Second eval phase — MAE=0.1. The start-of-phase reset must clear the
    # first phase's accumulation so it is not influenced by it.
    hook.on_eval_phase_start(trainer=None, state=state)
    eval_batch(0.1)
    hook.on_eval_step_complete(trainer=None, state=state)
    assert abs(state["eval"]["MAE"] - 0.1) < 1e-6, (
        f"val/MAE from second phase contaminated by first: {state['eval']['MAE']}"
    )
