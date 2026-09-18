"""Tests for the eager step prologue :func:`molix.core.steps.base._mark_leaves`."""

from __future__ import annotations

import torch
import torch.nn as nn

from molix.core.steps.base import _mark_leaves


class _PhysicsModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 1)
        self.physics_calls = 0

    def physics(self, batch: object) -> None:
        self.physics_calls += 1

    def forward(self, batch: dict[str, dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {"energy": self.linear(batch["atoms"]["pos"]).squeeze(-1)}


class TestMarkLeaves:
    def test_calls_physics_when_the_model_exposes_it(self) -> None:
        model = _PhysicsModel()
        _mark_leaves(model, {"atoms": {"pos": torch.zeros(2, 3)}})
        assert model.physics_calls == 1

    def test_unwraps_a_compile_wrapper_before_physics(self) -> None:
        inner = _PhysicsModel()

        class _Wrapped:
            _orig_mod = inner

        _mark_leaves(_Wrapped(), {"atoms": {"pos": torch.zeros(2, 3)}})
        assert inner.physics_calls == 1

    def test_a_model_without_physics_is_a_no_op(self) -> None:
        _mark_leaves(nn.Linear(3, 1), {})
