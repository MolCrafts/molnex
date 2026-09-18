"""``WeightedLoss`` — combining terms of both kinds this package produces."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from molix.core.losses import WeightedLoss


class _StatefulTerm(nn.Module):
    """A term that carries a buffer, so registration is observable."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self.register_buffer("scale", torch.tensor(value))

    def forward(self, pred, target) -> torch.Tensor:
        return self.scale * pred["a"]


class TestWeightedLoss:
    def test_accepts_a_plain_closure(self):
        # ``energy_loss`` / ``force_loss`` return closures by design — molnex
        # adds no loss class of its own — so refusing them would make the two
        # halves of this package unusable together.
        loss = WeightedLoss([(2.0, lambda pred, target: pred["a"])])
        assert float(loss({"a": torch.tensor(3.0)}, None)) == pytest.approx(6.0)

    def test_registers_a_module_term_so_state_follows(self):
        loss = WeightedLoss([(1.0, _StatefulTerm(4.0))])
        assert "losses.0.scale" in loss.state_dict()

    def test_sums_both_kinds_under_their_weights(self):
        loss = WeightedLoss(
            [
                (1.0, _StatefulTerm(4.0)),
                (0.5, lambda pred, target: pred["a"]),
            ]
        )
        # 1.0 * (4 * 3) + 0.5 * 3
        assert float(loss({"a": torch.tensor(3.0)}, None)) == pytest.approx(13.5)

    def test_a_closure_does_not_shift_a_module_state_dict_key(self):
        # Call order holds both kinds; only the modules are numbered in the
        # ModuleList, so adding a closure term cannot silently rename a
        # checkpointed buffer.
        loss = WeightedLoss(
            [
                (1.0, lambda pred, target: pred["a"]),
                (1.0, _StatefulTerm(4.0)),
            ]
        )
        assert "losses.0.scale" in loss.state_dict()

    def test_a_failing_term_propagates(self):
        # Silently skipping a term means training a different objective than
        # the one that was asked for, with no signal.
        def boom(pred, target):
            raise KeyError("forces")

        loss = WeightedLoss([(1.0, lambda pred, target: pred["a"]), (1.0, boom)])
        with pytest.raises(KeyError):
            loss({"a": torch.tensor(1.0)}, None)
