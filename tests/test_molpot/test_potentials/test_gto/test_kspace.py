"""Tests for official Polar GTO k-space Hartree (SI V.6 direct sum)."""

from __future__ import annotations

import math

import pytest
import torch

from molix import config
from molpot.potentials.elec.prefactors import eV_A
from molpot.potentials.gto.hartree import GTOHartreeEnergy
from molpot.potentials.gto.kspace import hartree_energy, kspace_cutoff


@pytest.fixture(autouse=True)
def fp64():
    previous = config["ftype"]
    config.set_precision("fp64")
    yield
    config.set_precision("fp64" if previous == torch.float64 else "fp32")


class TestKspaceCutoff:
    def test_polar_dump_defaults(self):
        """σ=1.5 Å, l_max=1, κ=1.0 → 2.25/σ · 2^{0.3}."""
        expect = 2.25 / 1.5 * (2.0**0.3)
        assert kspace_cutoff(1.5, 1, 1.0) == pytest.approx(expect, rel=1e-12)


class TestGTOKspaceHartree:
    def test_two_charges_energy_is_finite(self):
        energy = GTOHartreeEnergy(sigma=1.5, l_max=0)
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
        q = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
        batch = torch.zeros(2, dtype=torch.long)
        cell = torch.eye(3, dtype=torch.float64) * 12.0
        out = energy(q, pos, batch, cell=cell)
        assert out.shape == (1,)
        assert torch.isfinite(out).all()

    def test_forces_obey_newtons_third_law(self):
        """Opposite charges in a cube: F_1 = −F_0, attraction along the bond."""
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=torch.float64)
        q = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
        batch = torch.zeros(2, dtype=torch.long)
        cell = torch.eye(3, dtype=torch.float64) * 14.0
        leaf = pos.detach().requires_grad_(True)
        energy = GTOHartreeEnergy(sigma=1.5, l_max=0)(q, leaf, batch, cell=cell).sum()
        (d_energy,) = torch.autograd.grad(energy, leaf)
        assert torch.allclose(d_energy.sum(dim=0), torch.zeros(3, dtype=torch.float64), atol=1e-10)
        # Attraction: moving atom 0 toward atom 1 lowers E (∂E/∂x_0 < 0).
        assert d_energy[0, 0].item() < 0.0
        assert d_energy[1, 0].item() > 0.0

    def test_origin_translation_leaves_energy(self):
        energy = GTOHartreeEnergy(sigma=1.5, l_max=0)
        pos = torch.tensor([[1.0, 2.0, 3.0], [2.4, 2.0, 3.0]], dtype=torch.float64)
        q = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
        batch = torch.zeros(2, dtype=torch.long)
        cell = torch.eye(3, dtype=torch.float64) * 14.0
        e0 = energy(q, pos, batch, cell=cell)
        e1 = energy(q, pos + 1.7, batch, cell=cell)
        assert torch.allclose(e0, e1, atol=1e-9, rtol=0.0)

    def test_zero_dipole_matches_monopole(self):
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.3, -0.2]], dtype=torch.float64)
        cell = torch.eye(3, dtype=torch.float64) * 16.0
        q0 = torch.tensor([0.4, -0.4], dtype=torch.float64)
        k_cut = kspace_cutoff(1.5, 1, 1.0)
        e0 = hartree_energy(pos, q0, cell, sigma=1.5, k_cutoff=k_cut)
        e1 = hartree_energy(
            pos, q0, cell, sigma=1.5, k_cutoff=k_cut, dipoles=torch.zeros(2, 3, dtype=torch.float64)
        )
        assert torch.allclose(e0, e1, atol=1e-10, rtol=0.0)

    def test_half_sphere_matches_full_sphere(self):
        """Conjugate-symmetry factor 2 reproduces the full k ≠ 0 sum."""
        pos = torch.tensor([[0.1, 0.2, 0.0], [1.0, 0.2, 0.4]], dtype=torch.float64)
        q = torch.tensor([0.7, -0.7], dtype=torch.float64)
        cell = torch.eye(3, dtype=torch.float64) * 11.0
        sigma = 1.5
        k_cut = kspace_cutoff(sigma, 0, 1.0)
        half = hartree_energy(pos, q, cell, sigma=sigma, k_cutoff=k_cut)
        g = (2.0 * math.pi) * torch.linalg.inv(cell).T
        nk = max(1, int(11.0 * k_cut / (2.0 * math.pi)) + 1)
        n = torch.arange(-nk, nk + 1)
        nvec = torch.stack(torch.meshgrid(n, n, n, indexing="ij"), dim=-1).reshape(-1, 3)
        kvec = nvec.to(cell.dtype) @ g
        k_sq = (kvec * kvec).sum(dim=1)
        keep = (k_sq > 0.0) & (k_sq < k_cut * k_cut)
        kvec, k_sq = kvec[keep], k_sq[keep]
        kr = pos @ kvec.T
        s_real = q @ torch.cos(kr)
        s_imag = q @ torch.sin(kr)
        s_sq = s_real * s_real + s_imag * s_imag
        kfac = torch.exp(-(sigma * sigma) * k_sq) / k_sq
        full = eV_A * (2.0 * math.pi) / torch.det(cell).abs() * (kfac * s_sq).sum()
        assert half.item() == pytest.approx(full.item(), rel=1e-10)

    def test_cell_gradient_is_nonzero(self):
        """Hartree depends on the cell, so strain autograd yields a virial."""
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
        q = torch.tensor([1.0, -1.0], dtype=torch.float64)
        cell = (torch.eye(3, dtype=torch.float64) * 13.0).requires_grad_(True)
        energy = hartree_energy(pos, q, cell, sigma=1.5, k_cutoff=kspace_cutoff(1.5, 0, 1.0))
        (grad,) = torch.autograd.grad(energy, cell)
        assert torch.isfinite(grad).all()
        assert grad.abs().sum().item() > 0.0

    def test_batched_graphs_are_rejected(self):
        energy = GTOHartreeEnergy(sigma=1.5, l_max=0)
        pos = torch.zeros(4, 3, dtype=torch.float64)
        q = torch.ones(4, 1, dtype=torch.float64)
        batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        cell = torch.eye(3, dtype=torch.float64) * 12.0
        with pytest.raises(ValueError, match="single-system"):
            energy(q, pos, batch, cell=cell)
