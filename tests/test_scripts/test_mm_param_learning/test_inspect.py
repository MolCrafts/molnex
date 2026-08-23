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


def _c4_close_contact(name: str = "c4") -> Frame:
    """Four-carbon chain whose 1–4 pair sits at ~1.0 Å.

    With the retired LJ init (sigma ≈ 2.5 Å) the ``(sigma/r)**12`` wall on
    that close contact is ~1e5 kcal/mol and the dressed energy MSE exceeds
    ``ENERGY_TERM_WARN``. The zero-energy LJ init must keep it near zero.
    """
    atoms = Block()
    atoms["element"] = np.array(["C", "C", "C", "C"], dtype="U2")
    atoms["number"] = np.array([6, 6, 6, 6], dtype=np.int64)
    atoms["x"] = np.array([0.0, 1.5, 1.5, 0.8], dtype=np.float64)
    atoms["y"] = np.array([0.0, 0.0, 1.0, 0.6], dtype=np.float64)
    atoms["z"] = np.zeros(4, dtype=np.float64)
    atoms["fx"] = np.array([0.1, -0.1, 0.1, -0.1], dtype=np.float64)
    atoms["fy"] = np.zeros(4, dtype=np.float64)
    atoms["fz"] = np.zeros(4, dtype=np.float64)
    bonds = Block()
    bonds["atomi"] = np.array([0, 1, 2], dtype=np.int64)
    bonds["atomj"] = np.array([1, 2, 3], dtype=np.int64)
    frame = Frame()
    frame["atoms"] = atoms
    frame["bonds"] = bonds
    frame.meta["energy"] = -1500.0
    frame.meta["molecule_id"] = name
    return frame


class TestInspectEnergyScale:
    def test_close_contact_lj_does_not_break_energy_scale(self) -> None:
        torch.manual_seed(0)
        records = [(_c4_close_contact(),)]
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
        samples = [label_sample(records[0][0], group=0.0)]
        report = inspect_scales(fit.model(), samples)
        assert report["finite_ok"] == 1.0, report
        assert report["energy_scale_ok"] == 1.0, report
        # The force ratio is not meaningful here: packing pads a single-molecule
        # batch with zero reference forces, so ``ref_force_med`` is ~0 no matter
        # how healthy the model is. Guard the explosion instead.
        assert report["clip_would_fire"] == 0.0, report


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
