"""Tests for GTOHartreeEnergy (isolated molecule, l=0)."""

from __future__ import annotations

import math

import pytest
import torch

from molix import config
from molpot.potentials.elec.prefactors import eV_A
from molpot.potentials.gto.hartree import GTOHartreeEnergy, gto_self_overlap


@pytest.fixture(autouse=True)
def fp64():
    previous = config["ftype"]
    config.set_precision("fp64")
    yield
    config.set_precision("fp64" if previous == torch.float64 else "fp32")


class TestGTOHartreeEnergy:
    """l=0 pair Hartree: k_e q_i q_j erf(r/(2σ))/r."""

    def test_two_unit_charges_one_angstrom_apart(self):
        """σ=1.5 Å, r=1 Å, q=1,1 → eV_A * erf(1/3)."""
        energy = GTOHartreeEnergy(sigma=1.5, l_max=0)
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
        q = torch.ones(2, 1, dtype=torch.float64)
        batch = torch.zeros(2, dtype=torch.long)
        out = energy(q, pos, batch)
        expect = eV_A * math.erf(1.0 / 3.0)
        assert out.shape == (1,)
        assert out.item() == pytest.approx(expect, rel=1e-10)

    def test_self_pairs_are_excluded(self):
        """A single charge has no pair energy (self-interaction is a separate term)."""
        energy = GTOHartreeEnergy(sigma=1.5, l_max=0, include_self_interaction=False)
        pos = torch.zeros(1, 3, dtype=torch.float64)
        q = torch.ones(1, 1, dtype=torch.float64)
        batch = torch.zeros(1, dtype=torch.long)
        assert energy(q, pos, batch).item() == pytest.approx(0.0)

    def test_unit_monopole_self_energy(self):
        """Dump σ=1.5 Å: ½ q² eV_A/(σ√π) for a lone unit charge."""
        energy = GTOHartreeEnergy(sigma=1.5, l_max=0, include_self_interaction=True)
        pos = torch.zeros(1, 3, dtype=torch.float64)
        q = torch.ones(1, 1, dtype=torch.float64)
        batch = torch.zeros(1, dtype=torch.long)
        expect = 0.5 * eV_A / (1.5 * math.sqrt(math.pi))
        assert energy(q, pos, batch).item() == pytest.approx(expect, rel=1e-8)

    def test_self_overlap_polar_dump_sigma(self):
        """Polar-1-M dump overlap_constants at σ=1.5 Å, l_max=1."""
        omega = gto_self_overlap(1.5, 1)
        assert omega.shape == (4,)
        assert omega[0].item() == pytest.approx(eV_A / (1.5 * math.sqrt(math.pi)), rel=1e-10)
        assert omega[1].item() == pytest.approx(0.401191604, rel=1e-6)
        assert torch.allclose(omega[1:], omega[1:2].expand(3))

    def test_zero_dipole_l1_matches_l0(self):
        """Finite-difference copies of a pure monopole must not add extra pairs."""
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
        batch = torch.zeros(2, dtype=torch.long)
        q0 = torch.ones(2, 1, dtype=torch.float64)
        q1 = torch.zeros(2, 4, dtype=torch.float64)
        q1[:, 0] = 1.0
        e0 = GTOHartreeEnergy(sigma=1.5, l_max=0)(q0, pos, batch)
        e1 = GTOHartreeEnergy(sigma=1.5, l_max=1)(q1, pos, batch)
        assert torch.allclose(e0, e1, rtol=1e-10, atol=1e-12)

    def test_l1_pair_count_skips_same_atom_copies(self):
        """2 atoms × 4 copies → 32 directed pairs, not 8×7."""
        from molpot.potentials.gto.hartree import _complete_directed_pairs

        orig = torch.arange(2).repeat_interleave(4)
        batch = torch.zeros(8, dtype=torch.long)
        src, dst = _complete_directed_pairs(batch, orig)
        assert src.numel() == 32
        assert int((orig[src] == orig[dst]).sum()) == 0

    def test_graphs_do_not_mix(self):
        """Two isolated pairs in different graphs keep their own energies."""
        energy = GTOHartreeEnergy(sigma=1.5, l_max=0)
        pos = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [11.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
        )
        q = torch.ones(4, 1, dtype=torch.float64)
        batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        out = energy(q, pos, batch)
        expect = eV_A * math.erf(1.0 / 3.0)
        assert out.shape == (2,)
        assert torch.allclose(out, torch.full((2,), expect, dtype=torch.float64), rtol=1e-10)
