"""Packed vs per-molecule Class-I parity."""

from __future__ import annotations

import numpy as np
import torch
from molpy import Block, Frame
from scripts.mm_param_learning.align import (
    ATOL_ENERGY,
    RTOL_ENERGY,
    _abs_max,
    _max_abs,
    _within,
    align_energy_shift,
    align_finite,
    align_invariants,
    align_leak,
    align_packed,
    align_suite,
    bucket_of,
)
from scripts.mm_param_learning.class_i_fit import ClassIFit, label_sample


class TestMaxAbsEmpty:
    def test_empty_tensors_return_zero(self) -> None:
        empty = torch.zeros(0)
        assert _max_abs(empty, empty) == 0.0
        assert _abs_max(empty) == 0.0


def _frame(name: str, *, shift: float = 0.0) -> Frame:
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
    frame.meta["energy"] = -76.0
    frame.meta["molecule_id"] = name
    return frame


class TestAlignPacked:
    def test_two_waters_agree(self) -> None:
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
        report = align_packed(fit.model(), samples)
        assert report["energy_ok"] == 1.0, report
        assert report["force_ok"] == 1.0, report
        assert report["loss_ok"] == 1.0, report
        assert report["grad_ok"] == 1.0, report
        assert report["update_ok"] == 1.0, report

    def test_empty_raises(self) -> None:
        import pytest

        torch.manual_seed(0)
        records = [(_frame("a"),)]
        fit = ClassIFit(records, records, hidden=16, n_blocks=1, device="cpu")
        with pytest.raises(ValueError, match="at least one"):
            align_packed(fit.model(), [])


def _two_waters():
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
    return fit.model(), samples


class TestAlignInvariants:
    def test_water_rigid_and_charge(self) -> None:
        model, samples = _two_waters()
        report = align_invariants(model, samples[0])
        assert report["trans_ok"] == 1.0, report
        assert report["rot_ok"] == 1.0, report
        assert report["perm_ok"] == 1.0, report
        assert report["charge_ok"] == 1.0, report


class TestAlignLeak:
    def test_second_molecule_does_not_change_first(self) -> None:
        model, samples = _two_waters()
        report = align_leak(model, samples)
        assert report["leak_ok"] == 1.0, report


class TestAlignEnergyShift:
    def test_constant_energy_shift_changes_loss(self) -> None:
        model, samples = _two_waters()
        report = align_energy_shift(model, samples)
        assert report["energy_shift_ok"] == 1.0, report
        # 50 kcal/mol label shift is in the Huber linear region; a real
        # energy term must move by more than float noise.
        assert report["energy_shift_abs"] > 1.0, report


class TestAlignFinite:
    def test_two_waters_finite(self) -> None:
        model, samples = _two_waters()
        report = align_finite(model, samples)
        assert report["finite_ok"] == 1.0, report
        assert report["groups_ok"] == 1.0, report
        assert report["n_nonzero_groups"] >= 1.0


class TestAlignSuite:
    def test_two_waters_ok(self) -> None:
        model, samples = _two_waters()
        report = align_suite(model, samples)
        assert report["ok"] == 1.0, report


class TestDressedEnergyTol:
    def test_ulp_on_megakcal_passes(self) -> None:
        assert _within(0.0625, 2.0e6, atol=ATOL_ENERGY, rtol=RTOL_ENERGY)
        assert _within(0.5, 4.0e6, atol=ATOL_ENERGY, rtol=RTOL_ENERGY)

    def test_kcal_error_on_megakcal_fails(self) -> None:
        assert not _within(100.0, 2.0e6, atol=ATOL_ENERGY, rtol=RTOL_ENERGY)


class TestBucketOf:
    def test_declared_edges(self) -> None:
        assert bucket_of(1) == "small"
        assert bucket_of(8) == "small"
        assert bucket_of(9) == "medium"
        assert bucket_of(16) == "medium"
        assert bucket_of(17) == "large"
        assert bucket_of(32) == "large"
        assert bucket_of(0) is None
        assert bucket_of(33) is None
