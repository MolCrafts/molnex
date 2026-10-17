"""Tests for molpot.potentials.repulsion module."""

import math

import pytest
import torch

from molpot.potentials.repulsion import UniversalZBL, ZBLRepulsion

#: Floating dtype every module under test is built at.
DTYPE = torch.float32


@pytest.fixture
def zbl():
    """ZBL term at MACE's foundation-model settings (p = 5)."""
    return ZBLRepulsion(exponent=5, dtype=DTYPE).double()


class TestZBLRepulsion:
    """Test the ZBL screened-nuclear-repulsion pair term."""

    def test_output_is_per_atom(self, zbl):
        """One energy per atom, not per edge."""
        Z = torch.tensor([8, 1, 1])
        pair_index = torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]]).t()  # (E, 2)
        r = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
        assert zbl(r, Z, pair_index).shape == (3,)

    def test_repulsion_is_positive(self, zbl):
        """Nuclear repulsion never lowers the energy."""
        Z = torch.tensor([8, 8])
        pair_index = torch.tensor([[0, 1], [1, 0]]).t()  # (E, 2)
        r = torch.tensor([0.8, 0.8], dtype=torch.float64)
        assert bool((zbl(r, Z, pair_index) > 0).all())

    def test_decays_with_distance(self, zbl):
        """Closer nuclei repel harder."""
        Z = torch.tensor([8, 8])
        pair_index = torch.tensor([[0, 1], [1, 0]]).t()  # (E, 2)
        close = zbl(torch.tensor([0.6, 0.6], dtype=torch.float64), Z, pair_index)
        far = zbl(torch.tensor([1.1, 1.1], dtype=torch.float64), Z, pair_index)
        assert float(close[0]) > float(far[0])

    def test_vanishes_beyond_summed_covalent_radii(self, zbl):
        """The envelope cuts the pair off at ``R_i + R_j``, not at the model cutoff."""
        Z = torch.tensor([1, 1])  # H-H: covalent radii sum to 0.62 A
        pair_index = torch.tensor([[0, 1], [1, 0]]).t()  # (E, 2)
        beyond = zbl(torch.tensor([0.7, 0.7], dtype=torch.float64), Z, pair_index)
        assert float(beyond.abs().max()) == 0.0

    def test_heavier_nuclei_repel_more(self, zbl):
        """The ``Z_i Z_j`` prefactor makes heavy pairs stiffer at equal distance."""
        pair_index = torch.tensor([[0, 1], [1, 0]]).t()  # (E, 2)
        r = torch.tensor([0.5, 0.5], dtype=torch.float64)
        light = zbl(r, torch.tensor([1, 1]), pair_index)
        heavy = zbl(r, torch.tensor([8, 8]), pair_index)
        assert float(heavy[0]) > float(light[0])

    def test_pair_energy_is_split_between_both_atoms(self, zbl):
        """Each direction carries half, so a bidirectional list sums to the pair."""
        Z = torch.tensor([8, 1])
        both = torch.tensor([[0, 1], [1, 0]])  # (E, 2): 0->1 and 1->0
        one = torch.tensor([[0, 1]])  # (E, 2): the single direction 0->1
        r2 = torch.tensor([0.7, 0.7], dtype=torch.float64)
        r1 = torch.tensor([0.7], dtype=torch.float64)
        assert float(zbl(r2, Z, both).sum()) == pytest.approx(
            2.0 * float(zbl(r1, Z, one).sum()), rel=1e-12
        )

    def test_is_differentiable_wrt_distance(self, zbl):
        """Forces come from ``-dE/dr``, so the term must carry gradient."""
        Z = torch.tensor([8, 8])
        pair_index = torch.tensor([[0, 1], [1, 0]]).t()  # (E, 2)
        r = torch.tensor([0.8, 0.8], dtype=torch.float64, requires_grad=True)
        zbl(r, Z, pair_index).sum().backward()
        assert r.grad is not None
        # Repulsive: energy decreases as the pair separates.
        assert bool((r.grad < 0).all())

    def test_frozen_parameters_by_default(self, zbl):
        """Screening constants are buffers unless the caller asks for training."""
        assert not any(p.requires_grad for p in zbl.parameters())
        assert any(n == "a_exp" for n, _ in zbl.named_buffers())

    def test_trainable_promotes_screening_to_parameters(self):
        """``trainable=True`` exposes the screening length for fine-tuning."""
        trainable = ZBLRepulsion(trainable=True, dtype=DTYPE)
        names = {n for n, _ in trainable.named_parameters()}
        assert {"a_exp", "a_prefactor"} <= names


class TestUniversalZBL:
    def test_hand_value_for_an_h2_pair(self):
        """½ k_e Z²/r φ(x) u(r) per direction, summed onto the source; LAMMPS constants."""
        r = torch.tensor([0.5, 0.5], dtype=torch.float64)
        out = UniversalZBL(r_max=4.0, dtype=DTYPE)(
            r, torch.tensor([1, 1]), torch.tensor([[0, 1], [1, 0]])
        )
        x = 2.0 * 0.5 / 0.46850
        phi = sum(
            c * math.exp(d * x)
            for c, d in zip(
                (0.02817, 0.28022, 0.50986, 0.18175), (-0.20162, -0.40290, -0.94229, -3.19980)
            )
        )
        y = 0.5 / 4.0
        envelope = 1 - 28 * y**6 + 48 * y**7 - 21 * y**8
        expected = 0.5 * 14.399645 / 0.5 * phi * envelope
        torch.testing.assert_close(out, torch.tensor([expected, expected], dtype=torch.float64))

    def test_vanishes_at_the_cutoff(self):
        out = UniversalZBL(r_max=2.0, dtype=DTYPE)(
            torch.tensor([2.0], dtype=torch.float64), torch.tensor([6, 8]), torch.tensor([[0, 1]])
        )
        torch.testing.assert_close(out, torch.zeros(2, dtype=torch.float64))
