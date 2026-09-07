"""LJ 12-6 kernel — mirrors ``src/molpot/potentials/vdw/lj126.py``.

``lj126_pair_energy_split`` is the S3 kernel: it returns the repulsive wall
``4ε(σ/r)¹²`` and the attractive tail ``−4ε(σ/r)⁶`` separately so the assembled
potential can label them ``local`` and ``disp``. Splitting must not change the
number it sums to, which is what makes the S-decomposition a grouping of one
energy path rather than a second one.
"""

from __future__ import annotations

import pytest
import torch

from molpot.potentials.vdw.lj126 import lj126_pair_energy, lj126_pair_energy_split

DTYPE = torch.float64
EPSILON = 0.2
SIGMA = 3.0
SEED = 20260907


class TestLJ126PairEnergySplit:
    def test_halves_sum_to_the_unsplit_kernel(self) -> None:
        """Exact identity on ``r ∈ [0.95σ, 3σ]``, fp64."""
        generator = torch.Generator().manual_seed(SEED)
        distance = SIGMA * (0.95 + 2.05 * torch.rand(64, generator=generator, dtype=DTYPE))
        epsilon = torch.full_like(distance, EPSILON)
        sigma = torch.full_like(distance, SIGMA)
        repulsion, dispersion = lj126_pair_energy_split(distance, epsilon, sigma)
        assert torch.allclose(
            repulsion + dispersion,
            lj126_pair_energy(distance, epsilon, sigma),
            atol=1e-14,
            rtol=0.0,
        )

    def test_at_the_minimum_the_halves_are_plus_epsilon_and_minus_two_epsilon(self) -> None:
        """``r = 2^{1/6}σ`` → ``(σ/r)⁶ = ½`` → ``+ε`` and ``−2ε``."""
        distance = torch.tensor([2.0 ** (1.0 / 6.0) * SIGMA], dtype=DTYPE)
        repulsion, dispersion = lj126_pair_energy_split(
            distance,
            torch.tensor([EPSILON], dtype=DTYPE),
            torch.tensor([SIGMA], dtype=DTYPE),
        )
        assert float(repulsion) == pytest.approx(EPSILON, abs=1e-12)
        assert float(dispersion) == pytest.approx(-2.0 * EPSILON, abs=1e-12)
