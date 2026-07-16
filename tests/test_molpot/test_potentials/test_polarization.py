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
    edge_index = torch.tensor([[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]], dtype=torch.long)
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
    edge_index = torch.tensor([[0, 1], [1, 0], [2, 3], [3, 2]], dtype=torch.long)
    return pos, charge, alpha, batch, edge_index


class TestPolarization:
    def test_output_shape_single_mol(self, water_like):
        pos, charge, alpha, batch, edge_index = water_like
        pol = Polarization().double()
        energy = pol(
            pos=pos,
            charge=charge,
            alpha=alpha,
            batch=batch,
            edge_index=edge_index,
            num_graphs=1,
        )
        assert energy.shape == (1,)

    def test_output_shape_batched(self, two_mol_batch):
        pos, charge, alpha, batch, edge_index = two_mol_batch
        pol = Polarization().double()
        energy = pol(
            pos=pos,
            charge=charge,
            alpha=alpha,
            batch=batch,
            edge_index=edge_index,
            num_graphs=2,
        )
        assert energy.shape == (2,)

    def test_energy_is_negative(self, water_like):
        """Polarization energy should be stabilizing (negative)."""
        pos, charge, alpha, batch, edge_index = water_like
        pol = Polarization().double()
        energy = pol(
            pos=pos,
            charge=charge,
            alpha=alpha,
            batch=batch,
            edge_index=edge_index,
        )
        assert energy.item() < 0

    def test_zero_charges_give_zero_energy(self, water_like):
        pos, _, alpha, batch, edge_index = water_like
        charge = torch.zeros(3, dtype=torch.float64)
        pol = Polarization().double()
        energy = pol(
            pos=pos,
            charge=charge,
            alpha=alpha,
            batch=batch,
            edge_index=edge_index,
        )
        assert abs(energy.item()) < 1e-10

    def test_larger_alpha_gives_larger_magnitude(self, water_like):
        pos, charge, _, batch, edge_index = water_like
        pol = Polarization().double()
        e_small = pol(
            pos=pos,
            charge=charge,
            alpha=torch.tensor([0.5, 0.25, 0.25], dtype=torch.float64),
            batch=batch,
            edge_index=edge_index,
        )
        e_large = pol(
            pos=pos,
            charge=charge,
            alpha=torch.tensor([5.0, 2.5, 2.5], dtype=torch.float64),
            batch=batch,
            edge_index=edge_index,
        )
        # Larger polarizability → more negative energy
        assert e_large.item() < e_small.item()

    def test_identical_molecules_same_energy(self, two_mol_batch):
        pos, charge, alpha, batch, edge_index = two_mol_batch
        pol = Polarization().double()
        energy = pol(
            pos=pos,
            charge=charge,
            alpha=alpha,
            batch=batch,
            edge_index=edge_index,
            num_graphs=2,
        )
        assert torch.allclose(energy[0], energy[1], atol=1e-8)

    def test_grad_wrt_positions(self, water_like):
        pos, charge, alpha, batch, edge_index = water_like
        pos = pos.clone().requires_grad_(True)
        pol = Polarization().double()
        energy = pol(
            pos=pos,
            charge=charge,
            alpha=alpha,
            batch=batch,
            edge_index=edge_index,
        )
        energy.sum().backward()
        assert pos.grad is not None
        assert pos.grad.shape == (3, 3)

    def test_num_graphs_inferred(self, two_mol_batch):
        pos, charge, alpha, batch, edge_index = two_mol_batch
        pol = Polarization().double()
        energy = pol(
            pos=pos,
            charge=charge,
            alpha=alpha,
            batch=batch,
            edge_index=edge_index,
        )
        assert energy.shape == (2,)


class TestPolarizationQuantitative:
    """Numerical-oracle tests pinning energy magnitude (not just sign).

    These probes guarantee dimensional consistency with the rest of molpot
    (which uses the eV-Å-e unit system with Coulomb prefactor
    ``k_e = 1/(4πε₀) ≈ 14.3996 eV·Å·e⁻²``).
    """

    K_E = 14.3996  # 1/(4πε₀) in eV·Å·e⁻²

    def test_isolated_pair_matches_analytic_induced_dipole(self):
        """Single polarizable atom + fixed point charge at long range.

        In the decoupled limit (one atom polarizable with α₀=1, the other
        effectively rigid with α₁≈0; T·μ₁ ≈ 0 because μ₁ ≈ 0), the CG
        solve reduces to ``μ₀ = α₀ · E_perm,₀`` and the energy collapses
        to ``U = -½ α₀ (k_e · q / r²)²``.
        """
        r = 5.0
        alpha_val = 1.0
        q_val = 1.0

        pos = torch.tensor([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=torch.float64)
        charge = torch.tensor([0.0, q_val], dtype=torch.float64)
        # Atom 1 made effectively rigid (α≈0) so its induced dipole is negligible.
        alpha = torch.tensor([alpha_val, 1e-12], dtype=torch.float64)
        batch = torch.zeros(2, dtype=torch.long)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

        pol = Polarization().double()
        energy = pol(
            pos=pos,
            charge=charge,
            alpha=alpha,
            batch=batch,
            edge_index=edge_index,
            num_graphs=1,
        )

        e_field_mag = self.K_E * q_val / r**2
        expected = -0.5 * alpha_val * e_field_mag**2

        rel_err = abs(energy.item() - expected) / abs(expected)
        assert rel_err < 0.01, (
            f"impl {energy.item():.6f} vs analytic {expected:.6f}, rel_err {rel_err:.4e}"
        )

    def test_unit_consistency_with_ewald_multipole(self):
        """``Polarization`` and ``EwaldMultipoleEnergy`` α-mode must agree on energy.

        Both implement induced-dipole polarization (SC-CG vs one-shot
        non-self-consistent). At long range (r ≫ σ) and weak coupling
        (T·μ ≈ 0), the SC iteration converges in one step and the two
        outputs must agree to within numerical noise — confirming both
        are in the same eV unit system.
        """
        from molpot.potentials import EwaldMultipoleEnergy

        r = 5.0
        pos = torch.tensor([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=torch.float64)
        charge = torch.tensor([0.0, 1.0], dtype=torch.float64)
        alpha = torch.tensor([1.0, 0.0], dtype=torch.float64)
        batch = torch.zeros(2, dtype=torch.long)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

        pol = Polarization().double()
        e_pol = pol(
            pos=pos,
            charge=charge,
            alpha=torch.tensor([1.0, 1e-12], dtype=torch.float64),
            batch=batch,
            edge_index=edge_index,
            num_graphs=1,
        )

        # σ small enough that erf(r/(σ√2)) ≈ 1 at r=5; kernel field == bare Coulomb.
        ewald = EwaldMultipoleEnergy(sigma=0.5)
        out = ewald.forward(q=charge.double(), pos=pos.double(), alpha=alpha.double())
        e_ewald = out["pot"]

        rel_err = abs(e_pol.item() - e_ewald.item()) / abs(e_ewald.item())
        assert rel_err < 0.05, (
            f"polarization {e_pol.item():.6f} vs ewald α-mode {e_ewald.item():.6f}, "
            f"rel_err {rel_err:.4e}"
        )

    def test_dimer_energy_scales_inverse_r4(self):
        """``U_pol ∝ 1/r⁴`` (since E ∝ 1/r² and U ∝ E²).

        Sanity check that the corrected 1/r² Coulomb scaling produces the
        right power law. With the buggy 1/r scaling, U would scale as
        1/r² instead — this test would catch a regression of the original
        bug.
        """
        ratios = []
        for r in [4.0, 6.0, 8.0]:
            pos = torch.tensor([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=torch.float64)
            charge = torch.tensor([0.0, 1.0], dtype=torch.float64)
            alpha = torch.tensor([1.0, 1e-12], dtype=torch.float64)
            batch = torch.zeros(2, dtype=torch.long)
            edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
            pol = Polarization().double()
            energy = pol(
                pos=pos,
                charge=charge,
                alpha=alpha,
                batch=batch,
                edge_index=edge_index,
                num_graphs=1,
            )
            ratios.append((r, abs(energy.item()) * r**4))

        # |U| · r⁴ should be approximately constant (to within 1%) if scaling is 1/r⁴.
        baseline = ratios[0][1]
        for r, scaled in ratios:
            rel_err = abs(scaled - baseline) / baseline
            assert rel_err < 0.01, (
                f"r={r}: |U|·r⁴ = {scaled:.6f}, baseline {baseline:.6f}, rel_err {rel_err:.4e}"
            )
