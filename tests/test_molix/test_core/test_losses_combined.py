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


class TestNamedTerms:
    """A combined objective has to be watchable in parts, not only as a sum."""

    @staticmethod
    def _pair():
        return [
            (1.0, torch.nn.MSELoss()),
            (2.0, lambda pred, target: (pred - target).abs().mean()),
        ]

    def test_names_default_to_the_term_and_its_position(self) -> None:
        """A module names itself; a closure has no name to take."""
        assert WeightedLoss(self._pair()).names == ("mse", "term1")

    def test_given_names_are_kept_and_must_match_the_term_count(self) -> None:
        loss = WeightedLoss(self._pair(), names=("energy", "force"))
        assert loss.names == ("energy", "force")
        with pytest.raises(ValueError, match="one name per term"):
            WeightedLoss(self._pair(), names=("only-one",))

    def test_terms_are_unweighted_and_the_forward_is_their_weighted_sum(self) -> None:
        """The parts are for reporting; the sum is what gets differentiated.

        Reporting the *weighted* part would hide a weight of zero — a term that is
        declared and then silently not optimised.
        """
        loss = WeightedLoss(self._pair(), names=("sq", "abs"))
        pred, target = torch.zeros(3), torch.ones(3)
        parts = {name: float(value) for name, value in loss.terms(pred, target).items()}
        assert parts == pytest.approx({"sq": 1.0, "abs": 1.0})
        assert float(loss(pred, target)) == pytest.approx(1.0 * 1.0 + 2.0 * 1.0)
