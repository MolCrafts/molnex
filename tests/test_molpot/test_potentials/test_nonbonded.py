"""Tests for nonbonded pairwise potentials: RepulsionExp6, DispersionC6, ChargeTransfer."""

from __future__ import annotations

import pytest
import torch

from molpot.potentials.nonbonded import (
    RepulsionExp6,
    DispersionC6,
    ChargeTransfer,
    repulsion_mixing,
    dispersion_mixing,
    ct_mixing,
)
from molpot.potentials.mixing import geometric_arithmetic_mixing

from tests.utils import assert_compile_compatible


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pair_data():
    """Minimal 2-molecule batch with 6 edges."""
    edge_index = torch.tensor(
        [[0, 1], [1, 0], [1, 2], [2, 1], [3, 4], [4, 3]], dtype=torch.long
    )
    distance = torch.tensor([1.4, 1.4, 1.5, 1.5, 2.0, 2.0])
    edge_batch = torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.long)
    return distance, edge_index, edge_batch


@pytest.fixture
def atom_params():
    """Per-atom parameters for 5 atoms."""
    return {
        "eps_rep": torch.tensor([0.1, 0.2, 0.15, 0.12, 0.18]),
        "lam_rep": torch.tensor([10.0, 12.0, 11.0, 10.5, 11.5]),
        "r_star": torch.tensor([1.5, 1.6, 1.55, 1.52, 1.58]),
        "c6": torch.tensor([5.0, 6.0, 5.5, 5.2, 5.8]),
        "eps_ct": torch.tensor([0.01, 0.02, 0.015, 0.012, 0.018]),
        "lam_ct": torch.tensor([3.0, 3.5, 3.2, 3.1, 3.3]),
    }


# ---------------------------------------------------------------------------
# Mixing tests
# ---------------------------------------------------------------------------

class TestMixing:
    def test_geometric_arithmetic_mixing_shapes(self, atom_params, pair_data):
        _, edge_index, _ = pair_data
        result = geometric_arithmetic_mixing(
            atom_params, edge_index,
            geometric_keys=["eps_rep"],
            arithmetic_keys=["r_star"],
        )
        assert result["eps_rep_ij"].shape == (6,)
        assert result["r_star_ij"].shape == (6,)

    def test_geometric_mean_is_symmetric(self, atom_params, pair_data):
        _, edge_index, _ = pair_data
        result = geometric_arithmetic_mixing(
            atom_params, edge_index,
            geometric_keys=["eps_rep"],
            arithmetic_keys=[],
        )
        # Edge 0→1 and 1→0 should have the same mixed value
        assert torch.allclose(result["eps_rep_ij"][0], result["eps_rep_ij"][1])

    def test_arithmetic_mean_is_symmetric(self, atom_params, pair_data):
        _, edge_index, _ = pair_data
        result = geometric_arithmetic_mixing(
            atom_params, edge_index,
            geometric_keys=[],
            arithmetic_keys=["r_star"],
        )
        assert torch.allclose(result["r_star_ij"][0], result["r_star_ij"][1])

    def test_repulsion_mixing(self, atom_params, pair_data):
        _, edge_index, _ = pair_data
        result = repulsion_mixing(atom_params, edge_index)
        assert "eps_rep_ij" in result
        assert "lam_rep_ij" in result
        assert "r_star_ij" in result

    def test_dispersion_mixing(self, atom_params, pair_data):
        _, edge_index, _ = pair_data
        result = dispersion_mixing(atom_params, edge_index)
        assert "c6_ij" in result
        assert "r_star_ij" in result

    def test_ct_mixing(self, atom_params, pair_data):
        _, edge_index, _ = pair_data
        result = ct_mixing(atom_params, edge_index)
        assert "eps_ct_ij" in result
        assert "lam_ct_ij" in result
        assert "r_star_ij" in result


# ---------------------------------------------------------------------------
# RepulsionExp6 tests
# ---------------------------------------------------------------------------

class TestRepulsionExp6:
    def test_output_shape_batched(self, pair_data):
        distance, _, edge_batch = pair_data
        pot = RepulsionExp6()
        E = 6
        energy = pot(
            distance=distance,
            eps_rep_ij=torch.full((E,), 0.1),
            lam_rep_ij=torch.full((E,), 10.0),
            r_star_ij=torch.full((E,), 1.5),
            edge_batch=edge_batch,
            num_graphs=2,
        )
        assert energy.shape == (2,)

    def test_output_scalar_unbatched(self):
        pot = RepulsionExp6(bidirectional=False)
        energy = pot(
            distance=torch.tensor([2.0]),
            eps_rep_ij=torch.tensor([0.1]),
            lam_rep_ij=torch.tensor([10.0]),
            r_star_ij=torch.tensor([1.5]),
        )
        assert energy.shape == ()

    def test_energy_positive(self):
        """Repulsion energy should be positive at short range."""
        pot = RepulsionExp6(bidirectional=False)
        energy = pot(
            distance=torch.tensor([1.0]),
            eps_rep_ij=torch.tensor([0.5]),
            lam_rep_ij=torch.tensor([12.0]),
            r_star_ij=torch.tensor([1.5]),
        )
        assert energy.item() > 0

    def test_energy_decreases_with_distance(self):
        pot = RepulsionExp6(bidirectional=False)
        distances = torch.tensor([1.0, 2.0, 3.0])
        params = {
            "eps_rep_ij": torch.full((3,), 0.5),
            "lam_rep_ij": torch.full((3,), 12.0),
            "r_star_ij": torch.full((3,), 1.5),
        }
        energies = []
        for i in range(3):
            e = pot(distance=distances[i:i+1], **{k: v[i:i+1] for k, v in params.items()})
            energies.append(e.item())
        assert energies[0] > energies[1] > energies[2]

    def test_energy_scale(self, pair_data):
        distance, _, edge_batch = pair_data
        E = 6
        kwargs = dict(
            distance=distance,
            eps_rep_ij=torch.full((E,), 0.1),
            lam_rep_ij=torch.full((E,), 10.0),
            r_star_ij=torch.full((E,), 1.5),
            edge_batch=edge_batch,
            num_graphs=2,
        )
        e1 = RepulsionExp6(energy_scale=1.0)(**kwargs)
        e2 = RepulsionExp6(energy_scale=2.0)(**kwargs)
        assert torch.allclose(e2, 2.0 * e1, atol=1e-6)

    def test_bidirectional_halves_energy(self):
        kwargs = dict(
            distance=torch.tensor([1.5, 1.5]),
            eps_rep_ij=torch.tensor([0.1, 0.1]),
            lam_rep_ij=torch.tensor([10.0, 10.0]),
            r_star_ij=torch.tensor([1.5, 1.5]),
        )
        e_uni = RepulsionExp6(bidirectional=False)(**kwargs)
        e_bi = RepulsionExp6(bidirectional=True)(**kwargs)
        assert torch.allclose(e_bi, 0.5 * e_uni, atol=1e-6)

    @pytest.mark.xfail(reason="batched path uses index_add_ scatter; may break graph", strict=False)
    def test_compile(self):
        pot = RepulsionExp6(bidirectional=False)
        assert_compile_compatible(
            pot,
            strict=False,
            distance=torch.tensor([2.0]),
            eps_rep_ij=torch.tensor([0.1]),
            lam_rep_ij=torch.tensor([10.0]),
            r_star_ij=torch.tensor([1.5]),
        )


# ---------------------------------------------------------------------------
# DispersionC6 tests
# ---------------------------------------------------------------------------

class TestDispersionC6:
    def test_output_shape_batched(self, pair_data):
        distance, _, edge_batch = pair_data
        pot = DispersionC6()
        E = 6
        energy = pot(
            distance=distance,
            c6_ij=torch.full((E,), 5.0),
            r_star_ij=torch.full((E,), 1.5),
            edge_batch=edge_batch,
            num_graphs=2,
        )
        assert energy.shape == (2,)

    def test_energy_negative(self):
        """Dispersion energy should be negative (attractive)."""
        pot = DispersionC6(bidirectional=False)
        energy = pot(
            distance=torch.tensor([2.0]),
            c6_ij=torch.tensor([5.0]),
            r_star_ij=torch.tensor([1.5]),
        )
        assert energy.item() < 0

    def test_energy_approaches_zero_at_large_distance(self):
        pot = DispersionC6(bidirectional=False)
        e_close = pot(
            distance=torch.tensor([2.0]),
            c6_ij=torch.tensor([5.0]),
            r_star_ij=torch.tensor([1.5]),
        )
        e_far = pot(
            distance=torch.tensor([100.0]),
            c6_ij=torch.tensor([5.0]),
            r_star_ij=torch.tensor([1.5]),
        )
        assert abs(e_far.item()) < abs(e_close.item())

    def test_energy_scale(self):
        kwargs = dict(
            distance=torch.tensor([2.0]),
            c6_ij=torch.tensor([5.0]),
            r_star_ij=torch.tensor([1.5]),
        )
        e1 = DispersionC6(bidirectional=False, energy_scale=1.0)(**kwargs)
        e3 = DispersionC6(bidirectional=False, energy_scale=3.0)(**kwargs)
        assert torch.allclose(e3, 3.0 * e1, atol=1e-6)

    @pytest.mark.xfail(reason="batched path uses index_add_ scatter; may break graph", strict=False)
    def test_compile(self):
        pot = DispersionC6(bidirectional=False)
        assert_compile_compatible(
            pot,
            strict=False,
            distance=torch.tensor([2.0]),
            c6_ij=torch.tensor([5.0]),
            r_star_ij=torch.tensor([1.5]),
        )


# ---------------------------------------------------------------------------
# ChargeTransfer tests
# ---------------------------------------------------------------------------

class TestChargeTransfer:
    def test_output_shape_batched(self, pair_data):
        distance, _, edge_batch = pair_data
        pot = ChargeTransfer()
        E = 6
        energy = pot(
            distance=distance,
            eps_ct_ij=torch.full((E,), 0.01),
            lam_ct_ij=torch.full((E,), 3.0),
            r_star_ij=torch.full((E,), 1.5),
            edge_batch=edge_batch,
            num_graphs=2,
        )
        assert energy.shape == (2,)

    def test_energy_positive(self):
        """CT energy should be positive (repulsive envelope)."""
        pot = ChargeTransfer(bidirectional=False)
        energy = pot(
            distance=torch.tensor([2.0]),
            eps_ct_ij=torch.tensor([0.01]),
            lam_ct_ij=torch.tensor([3.0]),
            r_star_ij=torch.tensor([1.5]),
        )
        assert energy.item() > 0

    def test_energy_decreases_at_long_range(self):
        """Past the peak, CT energy decreases with distance."""
        pot = ChargeTransfer(bidirectional=False)
        params = {
            "eps_ct_ij": torch.tensor([0.01]),
            "lam_ct_ij": torch.tensor([0.5]),
            "r_star_ij": torch.tensor([1.0]),
        }
        e_close = pot(distance=torch.tensor([4.0]), **params)
        e_far = pot(distance=torch.tensor([8.0]), **params)
        assert e_close.item() > e_far.item()

    def test_bidirectional(self):
        kwargs = dict(
            distance=torch.tensor([2.0, 2.0]),
            eps_ct_ij=torch.tensor([0.01, 0.01]),
            lam_ct_ij=torch.tensor([3.0, 3.0]),
            r_star_ij=torch.tensor([1.5, 1.5]),
        )
        e_uni = ChargeTransfer(bidirectional=False)(**kwargs)
        e_bi = ChargeTransfer(bidirectional=True)(**kwargs)
        assert torch.allclose(e_bi, 0.5 * e_uni, atol=1e-6)

    @pytest.mark.xfail(reason="batched path uses index_add_ scatter; may break graph", strict=False)
    def test_compile(self):
        pot = ChargeTransfer(bidirectional=False)
        assert_compile_compatible(
            pot,
            strict=False,
            distance=torch.tensor([2.0]),
            eps_ct_ij=torch.tensor([0.01]),
            lam_ct_ij=torch.tensor([3.0]),
            r_star_ij=torch.tensor([1.5]),
        )


# ---------------------------------------------------------------------------
# Gradient tests
# ---------------------------------------------------------------------------

class TestGradients:
    def test_repulsion_grad_wrt_distance(self):
        pot = RepulsionExp6(bidirectional=False)
        d = torch.tensor([2.0], requires_grad=True)
        energy = pot(
            distance=d,
            eps_rep_ij=torch.tensor([0.1]),
            lam_rep_ij=torch.tensor([10.0]),
            r_star_ij=torch.tensor([1.5]),
        )
        energy.backward()
        assert d.grad is not None
        assert d.grad.shape == (1,)

    def test_dispersion_grad_wrt_distance(self):
        pot = DispersionC6(bidirectional=False)
        d = torch.tensor([2.0], requires_grad=True)
        energy = pot(
            distance=d,
            c6_ij=torch.tensor([5.0]),
            r_star_ij=torch.tensor([1.5]),
        )
        energy.backward()
        assert d.grad is not None

    def test_ct_grad_wrt_distance(self):
        pot = ChargeTransfer(bidirectional=False)
        d = torch.tensor([2.0], requires_grad=True)
        energy = pot(
            distance=d,
            eps_ct_ij=torch.tensor([0.01]),
            lam_ct_ij=torch.tensor([3.0]),
            r_star_ij=torch.tensor([1.5]),
        )
        energy.backward()
        assert d.grad is not None
