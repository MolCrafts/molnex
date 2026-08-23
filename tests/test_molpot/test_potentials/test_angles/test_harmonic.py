"""AngleHarmonic energy goldens and linear-angle force-matching stability."""

import math

import torch

from molpot.potentials.angles import AngleHarmonic


def _right_angle_pos() -> torch.Tensor:
    return torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )


def _linear_pos() -> torch.Tensor:
    """Exactly 180° at the central atom (O-C-O-like)."""
    return torch.tensor(
        [[-1.16, 0.0, 0.0], [0.0, 0.0, 0.0], [1.16, 0.0, 0.0]],
        dtype=torch.float32,
    )


def _angle_index() -> torch.Tensor:
    return torch.tensor([[0], [1], [2]], dtype=torch.long)


class TestAngleHarmonic:
    def test_right_angle_matches_analytic(self):
        pot = AngleHarmonic(
            k=torch.tensor([2.0], dtype=torch.float64),
            theta0=torch.tensor([math.pi / 3], dtype=torch.float64),
        )
        energy = pot(
            pos=_right_angle_pos(),
            angle_index=_angle_index(),
            angle_types=torch.tensor([0], dtype=torch.long),
        )
        assert math.isclose(float(energy), (math.pi / 6) ** 2, rel_tol=1e-10, abs_tol=1e-10)

    def test_linear_angle_force_matching_stays_finite(self):
        k = 100.0
        theta0 = 2.094395
        pot = AngleHarmonic(
            k=torch.tensor([k], dtype=torch.float32),
            theta0=torch.tensor([theta0], dtype=torch.float32),
        )
        pos = _linear_pos().clone().requires_grad_(True)
        energy = pot(
            pos=pos,
            angle_index=_angle_index(),
            angle_types=torch.tensor([0], dtype=torch.long),
        )
        expected = 0.5 * k * (math.pi - theta0) ** 2
        assert torch.isfinite(energy)
        assert math.isclose(float(energy.detach()), expected, rel_tol=1e-4, abs_tol=1e-3)
        (forces,) = torch.autograd.grad(energy, pos, create_graph=True)
        assert torch.isfinite(forces).all()
        forces.square().mean().backward()
        assert pos.grad is not None
        assert torch.isfinite(pos.grad).all()
