"""ImproperHarmonic — E = 1/2 k (chi - chi0)^2.

Hard-coded geometry goldens. improper_index is molrs layout
[center, i, j, k] (center at row 0); chi is dihedral(i, center, j, k).
"""

import math

import pytest
import torch

from molpot.potentials import ImproperHarmonic


def _planar_pos() -> torch.Tensor:
    """chi = 0 (planar)."""
    return torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],  # center
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=torch.float64,
    )


def _pi_over_6_pos() -> torch.Tensor:
    """chi = pi/6: k rotated out of the plane.

    i=(0,1,0), center=(0,0,0), j=(1,0,0), k=(1, cos(pi/6), sin(pi/6))
    → atan2 path yields phi = pi/6 on (i,center,j,k).
    """
    return torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],  # center
            [1.0, 0.0, 0.0],
            [1.0, math.cos(math.pi / 6.0), math.sin(math.pi / 6.0)],
        ],
        dtype=torch.float64,
    )


def _improper_index() -> torch.Tensor:
    """molrs [center, i, j, k] with center = atom 1."""
    return torch.tensor([[1], [0], [2], [3]], dtype=torch.long)


class TestImproperHarmonic:
    def test_planar_equilibrium_gives_zero_energy(self):
        # chi = chi0 = 0 → E = 0
        pot = ImproperHarmonic(
            k=torch.tensor([2.0], dtype=torch.float64),
            chi0=torch.tensor([0.0], dtype=torch.float64),
        )
        e = pot(
            pos=_planar_pos(),
            improper_index=_improper_index(),
            improper_types=torch.tensor([0], dtype=torch.long),
        )
        assert math.isclose(float(e), 0.0, rel_tol=1e-10, abs_tol=1e-10)

    def test_pi_over_6_matches_half_k_delta_squared(self):
        # k=2, chi=pi/6, chi0=0 → E = 0.5 * 2 * (pi/6)^2 = (pi/6)^2
        pot = ImproperHarmonic(
            k=torch.tensor([2.0], dtype=torch.float64),
            chi0=torch.tensor([0.0], dtype=torch.float64),
        )
        e = pot(
            pos=_pi_over_6_pos(),
            improper_index=_improper_index(),
            improper_types=torch.tensor([0], dtype=torch.long),
        )
        expected = (math.pi / 6.0) ** 2
        # float64 geometry + atan2 leaves ~1e-9 residual on (pi/6)**2
        assert math.isclose(float(e), expected, rel_tol=1e-8, abs_tol=1e-8)

    def test_empty_improper_index_returns_zero(self):
        pot = ImproperHarmonic(
            k=torch.tensor([2.0], dtype=torch.float64),
            chi0=torch.tensor([0.0], dtype=torch.float64),
        )
        e = pot(
            pos=_planar_pos(),
            improper_index=torch.zeros(4, 0, dtype=torch.long),
            improper_types=torch.zeros(0, dtype=torch.long),
        )
        assert float(e) == 0.0

    def test_rejects_row_major_n_by_4_shape(self):
        pot = ImproperHarmonic(
            k=torch.tensor([2.0], dtype=torch.float64),
            chi0=torch.tensor([0.0], dtype=torch.float64),
        )
        wrong = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        with pytest.raises(ValueError, match=r"\[4"):
            pot(
                pos=_planar_pos(),
                improper_index=wrong,
                improper_types=torch.tensor([0], dtype=torch.long),
            )

    def test_forces_are_finite_off_equilibrium(self):
        pot = ImproperHarmonic(
            k=torch.tensor([2.0], dtype=torch.float64),
            chi0=torch.tensor([0.0], dtype=torch.float64),
        )
        pos = _pi_over_6_pos().clone().requires_grad_(True)
        forces = pot.calc_forces(
            pos=pos.detach(),
            improper_index=_improper_index(),
            improper_types=torch.tensor([0], dtype=torch.long),
            as_numpy=False,
        )
        assert torch.isfinite(forces).all()
