"""Tests for :class:`molix.data.tasks.ConstantLabel`."""

from __future__ import annotations

import pytest
import torch

from molix.data.task import Runnable, SampleTask
from molix.data.tasks import ConstantLabel


class TestConstantLabel:
    def test_is_sample_task(self):
        t = ConstantLabel(key="total_charge", value=0.0)
        assert isinstance(t, SampleTask)
        assert isinstance(t, Runnable)

    def test_writes_target(self):
        t = ConstantLabel(key="total_charge", value=0.0)
        out = t({"Z": torch.tensor([1, 6]), "targets": {"U0": torch.tensor([1.5])}})
        assert "total_charge" in out["targets"]
        torch.testing.assert_close(out["targets"]["total_charge"], torch.tensor([0.0]))

    def test_writes_anion_target(self):
        """Charged-system semantics: value can be any float."""
        t = ConstantLabel(key="total_charge", value=-1.0)
        out = t({"Z": torch.tensor([7, 1, 1, 1]), "targets": {}})
        torch.testing.assert_close(out["targets"]["total_charge"], torch.tensor([-1.0]))

    def test_does_not_clobber_other_targets(self):
        t = ConstantLabel(key="total_charge", value=0.0)
        sample = {
            "Z": torch.tensor([1]),
            "pos": torch.zeros(1, 3),
            "targets": {"U0": torch.tensor([1.5]), "mu": torch.tensor([0.7])},
        }
        out = t(sample)
        assert out["targets"]["U0"].item() == pytest.approx(1.5)
        assert out["targets"]["mu"].item() == pytest.approx(0.7)
        assert out["targets"]["total_charge"].item() == 0.0

    def test_creates_targets_dict_if_missing(self):
        t = ConstantLabel(key="total_charge", value=0.0)
        out = t({"Z": torch.tensor([1])})
        assert "targets" in out
        assert out["targets"]["total_charge"].item() == 0.0

    def test_does_not_mutate_input(self):
        t = ConstantLabel(key="total_charge", value=0.0)
        sample = {"Z": torch.tensor([1]), "targets": {"U0": torch.tensor([1.5])}}
        out = t(sample)
        assert "total_charge" not in sample["targets"]
        assert "total_charge" in out["targets"]

    def test_rejects_empty_key(self):
        with pytest.raises(ValueError, match="key"):
            ConstantLabel(key="", value=0.0)

    def test_task_id_is_deterministic(self):
        a = ConstantLabel(key="total_charge", value=0.0)
        b = ConstantLabel(key="total_charge", value=0.0)
        c = ConstantLabel(key="total_charge", value=-1.0)
        d = ConstantLabel(key="spin", value=0.0)
        assert a.task_id == b.task_id
        assert a.task_id != c.task_id  # different value → different cache key
        assert a.task_id != d.task_id  # different key → different cache key

    def test_dtype_is_default_float(self):
        t = ConstantLabel(key="total_charge", value=0.0)
        out = t({"Z": torch.tensor([1])})
        assert out["targets"]["total_charge"].dtype == torch.get_default_dtype()
