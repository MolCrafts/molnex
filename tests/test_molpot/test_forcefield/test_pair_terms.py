"""Pair terms owned by the Potentials assembly — mirrors ``pair_terms.py``.

``LJCutPotential`` is the seam where S3's split becomes observable: it returns
the repulsive and attractive halves of ``4ε[(σ/r)¹² − (σ/r)⁶]`` separately, so
``Potentials`` can label one ``local`` and the other ``disp`` without a second
energy path. Analytic-force values are pinned one layer up
(``test_analytic_forces.py``); what is pinned here is the return *shape* of
both entry points and the split at the LJ minimum.
"""

from __future__ import annotations

import pytest
import torch

from molpot.forcefield.pair_terms import LJCutPotential

DTYPE = torch.float64
EPSILON = 0.2
SIGMA = 3.0
#: ``r`` at the LJ minimum, where ``4ε(σ/r)¹² = +ε`` and ``−4ε(σ/r)⁶ = −2ε``.
R_MIN = 2.0 ** (1.0 / 6.0) * SIGMA


def _one_pair_kwargs(*, r: float = R_MIN) -> dict[str, torch.Tensor]:
    """A single half-list pair at separation ``r`` along x̂, unscaled."""
    return {
        "edge_index": torch.tensor([[0, 1]], dtype=torch.long),
        "distances": torch.tensor([r], dtype=DTYPE),
        "epsilon": torch.full((2,), EPSILON, dtype=DTYPE),
        "sigma": torch.full((2,), SIGMA, dtype=DTYPE),
        "scales": torch.ones(1, dtype=DTYPE),
    }


class TestLJCutPotential:
    def test_energy_returns_the_repulsion_dispersion_pair(self) -> None:
        """At ``r = 2^{1/6}σ`` the halves are ``+ε`` and ``−2ε`` exactly."""
        repulsion, dispersion = LJCutPotential(cutoff=10.0).energy(**_one_pair_kwargs())
        assert float(repulsion) == pytest.approx(EPSILON, abs=1e-12)
        assert float(dispersion) == pytest.approx(-2.0 * EPSILON, abs=1e-12)

    def test_energy_forces_returns_the_same_pair_alongside_the_forces(self) -> None:
        """``((rep, disp), forces)`` — the two entry points may not disagree."""
        potential = LJCutPotential(cutoff=10.0)
        positions = torch.tensor([[0.0, 0.0, 0.0], [0.9 * R_MIN, 0.0, 0.0]], dtype=DTYPE)
        kwargs = _one_pair_kwargs(r=0.9 * R_MIN)
        (repulsion, dispersion), forces = potential.energy_forces(positions=positions, **kwargs)
        expected_repulsion, expected_dispersion = potential.energy(**kwargs)
        assert float(repulsion) == pytest.approx(float(expected_repulsion), abs=1e-10)
        assert float(dispersion) == pytest.approx(float(expected_dispersion), abs=1e-10)
        assert forces.shape == (2, 3)
