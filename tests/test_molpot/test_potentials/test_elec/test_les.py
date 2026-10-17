"""Unit tests for :mod:`molpot.potentials.elec.les`."""

from __future__ import annotations

import math

import pytest
import torch

from molpot.potentials.elec.les import LatentEwaldEnergy

#: LES's ``norm_factor``; ``c / 2π`` is the Coulomb constant in eV·Å/e².
NORM = 90.0474
SIGMA = 1.0


def _pair_energy(r: float, qi: float, qj: float) -> float:
    return qi * qj * math.erf(r / (math.sqrt(2.0) * SIGMA)) / r * NORM / (2.0 * math.pi)


class TestLatentEwaldEnergy:
    def test_isolated_is_the_gaussian_pair_sum_per_graph(self):
        """Two graphs in one batch: each is its own erf pair sum, no cross terms."""
        pos = torch.tensor(
            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 2.0, 0.0], [5.0, 5.0, 5.0], [5.0, 5.0, 7.5]],
            dtype=torch.float64,
        )
        q = torch.tensor([0.4, -0.3, -0.1, 0.25, -0.25], dtype=torch.float64)
        batch = torch.tensor([0, 0, 0, 1, 1])
        energy = LatentEwaldEnergy(sigma=SIGMA, norm_factor=NORM)(pos, q, batch, 2)
        e0 = (
            _pair_energy(1.5, 0.4, -0.3)
            + _pair_energy(2.0, 0.4, -0.1)
            + _pair_energy(2.5, -0.3, -0.1)
        )
        e1 = _pair_energy(2.5, 0.25, -0.25)
        torch.testing.assert_close(energy, torch.tensor([e0, e1], dtype=torch.float64))

    def test_periodic_approaches_isolated_in_a_large_box(self):
        """A neutral pair in a 20 Å box: periodic minus isolated is the image dipole term.

        Tin-foil Ewald of a dipole ``p`` in a cube of volume ``V`` differs from
        the isolated pair by ``O(2π p² k_e / 3V)`` (≈ 0.015 eV here); a tight
        reciprocal cutoff (``Δ = 1`` Å) keeps the k-sum itself converged.
        """
        pos = torch.tensor([[9.0, 10.0, 10.0], [11.0, 10.0, 10.0]], dtype=torch.float64)
        q = torch.tensor([1.0, -1.0], dtype=torch.float64)
        batch = torch.zeros(2, dtype=torch.long)
        cell = 20.0 * torch.eye(3, dtype=torch.float64)
        module = LatentEwaldEnergy(sigma=SIGMA, dl=1.0, norm_factor=NORM)
        module.bind_cell(cell)
        periodic = module(pos, q, batch, 1, cell=cell.unsqueeze(0))
        isolated = module(pos, q, batch, 1)
        bound = 2.0 * math.pi * 4.0 * NORM / (2.0 * math.pi) / (3.0 * 20.0**3)
        assert abs(float(periodic - isolated)) < 2.0 * bound

    def test_periodic_energy_is_invariant_to_a_lattice_translation(self):
        """Moving one atom by a lattice vector changes nothing."""
        torch.manual_seed(0)
        cell = torch.tensor(
            [[9.0, 0.0, 0.0], [1.0, 8.5, 0.0], [0.5, -0.4, 10.0]], dtype=torch.float64
        )
        pos = torch.rand(6, 3, dtype=torch.float64) @ cell
        q = torch.randn(6, dtype=torch.float64)
        batch = torch.zeros(6, dtype=torch.long)
        module = LatentEwaldEnergy(sigma=SIGMA, norm_factor=NORM)
        module.bind_cell(cell)
        moved = pos.clone()
        moved[2] += cell[1] - cell[0]
        torch.testing.assert_close(
            module(pos, q, batch, 1, cell=cell), module(moved, q, batch, 1, cell=cell)
        )

    @pytest.mark.slow
    def test_periodic_forward_compiles_to_one_graph(self):
        """After ``bind_cell`` the periodic energy traces with zero graph breaks."""
        cell = 12.0 * torch.eye(3, dtype=torch.float64)
        pos = torch.rand(8, 3, dtype=torch.float64) * 12.0
        q = torch.randn(8, dtype=torch.float64)
        batch = torch.zeros(8, dtype=torch.long)
        module = LatentEwaldEnergy()
        module.bind_cell(cell)
        explained = torch._dynamo.explain(module)(pos, q, batch, 1, cell=cell)
        assert explained.graph_break_count == 0
