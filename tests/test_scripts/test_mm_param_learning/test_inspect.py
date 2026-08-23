"""Scale / NaN / clip inspection for checklist §2."""

from __future__ import annotations

import numpy as np
import torch
from molpy import Block, Frame
from scripts.mm_param_learning.class_i_fit import ClassIFit, label_sample
from scripts.mm_param_learning.scale_audit import inspect_scales


def _frame(name: str, *, energy: float = -76.0, shift: float = 0.0) -> Frame:
    atoms = Block()
    atoms["element"] = np.array(["O", "H", "H"], dtype="U2")
    atoms["number"] = np.array([8, 1, 1], dtype=np.int64)
    atoms["x"] = np.array([0.0, 0.96, -0.24], dtype=np.float64) + shift
    atoms["y"] = np.array([0.0, 0.0, 0.93], dtype=np.float64)
    atoms["z"] = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    atoms["fx"] = np.array([0.1, -0.05, -0.05], dtype=np.float64)
    atoms["fy"] = np.zeros(3, dtype=np.float64)
    atoms["fz"] = np.zeros(3, dtype=np.float64)
    bonds = Block()
    bonds["atomi"] = np.array([0, 0], dtype=np.int64)
    bonds["atomj"] = np.array([1, 2], dtype=np.int64)
    frame = Frame()
    frame["atoms"] = atoms
    frame["bonds"] = bonds
    frame.meta["energy"] = energy
    frame.meta["molecule_id"] = name
    return frame


class TestInspectScales:
    def test_two_named_waters_are_singletons(self) -> None:
        torch.manual_seed(0)
        records = [(_frame("a"),), (_frame("b", shift=0.2),)]
        fit = ClassIFit(
            records,
            records,
            hidden=16,
            n_blocks=1,
            n_torsion_terms=2,
            device="cpu",
            epochs=1,
            energy_scope="full",
        )
        samples = [label_sample(rec[0], group=float(i)) for i, rec in enumerate(records)]
        report = inspect_scales(fit.model(), samples)
        assert report["finite_ok"] == 1.0, report
        assert report["n_groups"] == 2.0, report
        assert report["n_singletons"] == 2.0, report
        assert report["energy_term"] > 0.0, report
        assert report["collapse_ok"] == 1.0, report
        assert report["energy_scale_ok"] == 1.0, report
        assert report["groups_ok"] == 1.0, report

    def test_empty_raises(self) -> None:
        import pytest

        torch.manual_seed(0)
        records = [(_frame("a"),)]
        fit = ClassIFit(records, records, hidden=16, n_blocks=1, device="cpu")
        with pytest.raises(ValueError, match="at least one"):
            inspect_scales(fit.model(), [])
