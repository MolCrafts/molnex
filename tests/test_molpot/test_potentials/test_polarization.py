"""Tests for the Polarization module."""

from __future__ import annotations

import pytest
import torch

from molpot.potentials.polarization import Polarization


@pytest.fixture
def water_like():
    """A single 3-atom molecule resembling water."""
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [-0.24, 0.93, 0.0],
        ],
        dtype=torch.float64,
    )
    charge = torch.tensor([-0.8, 0.4, 0.4], dtype=torch.float64)
    alpha = torch.tensor([1.0, 0.5, 0.5], dtype=torch.float64)
    batch = torch.zeros(3, dtype=torch.long)
    # Fully-connected edges (exclude self-loops)
    edge_index = torch.tensor(
        [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]], dtype=torch.long
    )
    return pos, charge, alpha, batch, edge_index


@pytest.fixture
def two_mol_batch():
    """Two 2-atom molecules."""
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [11.5, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    charge = torch.tensor([-0.5, 0.5, -0.5, 0.5], dtype=torch.float64)
    alpha = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    edge_index = torch.tensor(
        [[0, 1], [1, 0], [2, 3], [3, 2]], dtype=torch.long
    )
    return pos, charge, alpha, batch, edge_index


class TestPolarization:
    def test_output_shape_single_mol(self, water_like):
        pos, charge, alpha, batch, edge_index = water_like
        pol = Polarization().double()
        energy = pol(
            pos=pos, charge=charge, alpha=alpha,
            batch=batch, edge_index=edge_index, num_graphs=1,
        )
        assert energy.shape == (1,)

    def test_output_shape_batched(self, two_mol_batch):
        pos, charge, alpha, batch, edge_index = two_mol_batch
        pol = Polarization().double()
        energy = pol(
            pos=pos, charge=charge, alpha=alpha,
            batch=batch, edge_index=edge_index, num_graphs=2,
        )
        assert energy.shape == (2,)

    def test_energy_is_negative(self, water_like):
        """Polarization energy should be stabilizing (negative)."""
        pos, charge, alpha, batch, edge_index = water_like
        pol = Polarization().double()
        energy = pol(
            pos=pos, charge=charge, alpha=alpha,
            batch=batch, edge_index=edge_index,
        )
        assert energy.item() < 0

    def test_zero_charges_give_zero_energy(self, water_like):
        pos, _, alpha, batch, edge_index = water_like
        charge = torch.zeros(3, dtype=torch.float64)
        pol = Polarization().double()
        energy = pol(
            pos=pos, charge=charge, alpha=alpha,
            batch=batch, edge_index=edge_index,
        )
        assert abs(energy.item()) < 1e-10

    def test_larger_alpha_gives_larger_magnitude(self, water_like):
        pos, charge, _, batch, edge_index = water_like
        pol = Polarization().double()
        e_small = pol(
            pos=pos, charge=charge,
            alpha=torch.tensor([0.5, 0.25, 0.25], dtype=torch.float64),
            batch=batch, edge_index=edge_index,
        )
        e_large = pol(
            pos=pos, charge=charge,
            alpha=torch.tensor([5.0, 2.5, 2.5], dtype=torch.float64),
            batch=batch, edge_index=edge_index,
        )
        # Larger polarizability → more negative energy
        assert e_large.item() < e_small.item()

    def test_identical_molecules_same_energy(self, two_mol_batch):
        pos, charge, alpha, batch, edge_index = two_mol_batch
        pol = Polarization().double()
        energy = pol(
            pos=pos, charge=charge, alpha=alpha,
            batch=batch, edge_index=edge_index, num_graphs=2,
        )
        assert torch.allclose(energy[0], energy[1], atol=1e-8)

    def test_grad_wrt_positions(self, water_like):
        pos, charge, alpha, batch, edge_index = water_like
        pos = pos.clone().requires_grad_(True)
        pol = Polarization().double()
        energy = pol(
            pos=pos, charge=charge, alpha=alpha,
            batch=batch, edge_index=edge_index,
        )
        energy.sum().backward()
        assert pos.grad is not None
        assert pos.grad.shape == (3, 3)

    def test_num_graphs_inferred(self, two_mol_batch):
        pos, charge, alpha, batch, edge_index = two_mol_batch
        pol = Polarization().double()
        energy = pol(
            pos=pos, charge=charge, alpha=alpha,
            batch=batch, edge_index=edge_index,
        )
        assert energy.shape == (2,)
