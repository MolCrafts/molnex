"""Regression tests for :class:`TrainState` read ergonomics.

``TrainState.__getitem__`` / ``.get`` / ``__contains__`` walk nested
namespaces for both slash-string and tuple paths, so callers don't
need a helper just to read ``state["eval"]["MAE"]``. Writes stay
nested-only — that's what enforces namespace ownership.

These tests guard against regressions to the old "tuple paths
internally, dotted strings only for display" rule, which forced
verbose ``resolve(state, ("eval", "MAE"))`` reads at every callsite.
"""

from __future__ import annotations

import pytest

from molix.core.state import TrainState, resolve


@pytest.fixture
def state() -> TrainState:
    s = TrainState()
    s["train"]["loss"] = 0.5
    s["train"]["MAE"] = 17.3
    s["eval"]["MAE"] = 14.6
    s["gpu"]["util_pct"] = 18.0
    s["best_metric"] = 0.42
    return s


class TestTrainStateReads:
    def test_nested_read(self, state):
        assert state["train"]["loss"] == 0.5

    def test_slash_string_read(self, state):
        assert state["train/loss"] == 0.5
        assert state["eval/MAE"] == 14.6
        assert state["gpu/util_pct"] == 18.0

    def test_tuple_path_read(self, state):
        assert state[("train", "loss")] == 0.5
        assert state[("eval", "MAE")] == 14.6

    def test_get_slash_default(self, state):
        assert state.get("eval/MAE") == 14.6
        assert state.get("eval/missing") is None
        assert state.get("eval/missing", -1.0) == -1.0

    def test_get_tuple_default(self, state):
        assert state.get(("eval", "MAE")) == 14.6
        assert state.get(("eval", "missing"), -1.0) == -1.0

    def test_get_top_level(self, state):
        assert state.get("epoch") == 0
        assert state.get("best_metric") == 0.42
        assert state.get("missing_top_level", "x") == "x"

    def test_contains_slash(self, state):
        assert "eval/MAE" in state
        assert "eval/missing" not in state
        assert "epoch" in state

    def test_contains_tuple(self, state):
        assert ("eval", "MAE") in state
        assert ("eval", "missing") not in state

    def test_getitem_keyerror(self, state):
        with pytest.raises(KeyError):
            _ = state["eval/missing"]
        with pytest.raises(KeyError):
            _ = state[("eval", "missing")]

    def test_walk_through_non_mapping_returns_default(self, state):
        # state["epoch"] is an int — walking past it gracefully returns default
        assert state.get("epoch/foo", "default") == "default"
        assert state.get(("epoch", "foo"), "default") == "default"


class TestTrainStateWrites:
    """Writes still go through nested dict access only."""

    def test_slash_write_rejected(self):
        s = TrainState()
        with pytest.raises(ValueError, match="slash-prefix"):
            s["train/loss"] = 0.5

    def test_tuple_write_rejected(self):
        s = TrainState()
        with pytest.raises(ValueError, match="tuple-path"):
            s[("train", "loss")] = 0.5

    def test_namespace_replacement_rejected(self):
        s = TrainState()
        with pytest.raises(ValueError, match="must be a dict"):
            s["train"] = 0.5  # would clobber the sub-dict

    def test_nested_write_works(self):
        s = TrainState()
        s["train"]["loss"] = 0.5
        s["eval"]["MAE"] = 1.0
        assert s["train/loss"] == 0.5
        assert s["eval/MAE"] == 1.0


class TestResolveAcceptsSlashStrings:
    """``resolve`` is the read helper for arbitrary mappings (e.g. dict
    snapshots). It must accept the same shapes ``TrainState`` does."""

    def test_resolve_tuple(self):
        m = {"a": {"b": 7}}
        assert resolve(m, ("a", "b")) == 7

    def test_resolve_slash_string(self):
        m = {"a": {"b": 7}}
        assert resolve(m, "a/b") == 7

    def test_resolve_flat_string(self):
        m = {"epoch": 5}
        assert resolve(m, "epoch") == 5

    def test_resolve_default_on_missing(self):
        m = {"a": {"b": 7}}
        assert resolve(m, "a/missing", default=-1) == -1
        assert resolve(m, ("missing", "key"), default=-1) == -1
