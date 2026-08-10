"""Tests for TypingRecoveryMetrics."""

from __future__ import annotations

import torch

from molrep.chem.typing_metrics import TypingRecoveryMetrics


class TestTypingRecoveryMetrics:
    def test_perfect_accuracy(self):
        m = TypingRecoveryMetrics(num_types=3)
        pred = torch.tensor([0, 1, 2, 1])
        true = torch.tensor([0, 1, 2, 1])
        m.update(pred, true, Z=torch.tensor([6, 6, 1, 8]), molecule_ids=["a", "a", "b", "b"])
        r = m.compute()
        assert r.overall_accuracy == 1.0
        assert r.n_atoms == 4
        assert r.molecule_error_counts == {}
        assert r.per_element_accuracy[6] == 1.0

    def test_errors_and_confusion(self):
        m = TypingRecoveryMetrics(num_types=2, rare_max_count=1)
        m.update(
            torch.tensor([0, 1, 0]),
            torch.tensor([0, 0, 1]),
            molecule_ids=["m0", "m0", "m1"],
        )
        r = m.compute()
        assert abs(r.overall_accuracy - 1 / 3) < 1e-6
        assert r.confusion[0, 0] == 1
        assert r.confusion[0, 1] == 1
        assert r.confusion[1, 0] == 1
        assert r.molecule_error_counts["m0"] == 1
        assert r.molecule_error_counts["m1"] == 1

    def test_reset(self):
        m = TypingRecoveryMetrics(num_types=2)
        m.update(torch.tensor([0]), torch.tensor([0]))
        m.reset()
        assert m.compute().n_atoms == 0
