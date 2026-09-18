"""C^infty angle / dihedral trigonometry goldens."""

import math

import torch

from molpot.potentials.geom import bond_cos_sin, dihedral_cos_sin, fourier_cos


class TestBondCosSin:
    def test_right_angle(self):
        u = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        v = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
        c, s = bond_cos_sin(u, v)
        assert math.isclose(float(c), 0.0, abs_tol=1e-12)
        assert math.isclose(float(s), 1.0, abs_tol=1e-12)

    def test_linear_is_cos_minus_one(self):
        u = torch.tensor([[-1.16, 0.0, 0.0]], dtype=torch.float64)
        v = torch.tensor([[1.16, 0.0, 0.0]], dtype=torch.float64)
        c, s = bond_cos_sin(u, v)
        assert math.isclose(float(c), -1.0, abs_tol=1e-8)
        assert float(s) >= 0.0
        assert float(s) < 1e-5


class TestDihedralFourier:
    def test_cis_trans_perp_and_colinear(self):
        cis = torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            dtype=torch.float64,
        )
        trans = cis.clone()
        trans[3, 1] = -1.0
        perp = cis.clone()
        perp[3] = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64)
        colin = torch.tensor(
            [[-2.2, 0.0, 0.0], [-1.2, 0.0, 0.0], [0.0, 0.0, 0.0], [1.06, 0.0, 0.0]],
            dtype=torch.float64,
        )
        n = torch.tensor([1], dtype=torch.long)
        phase = torch.zeros(1, 1, dtype=torch.float64)
        got = []
        for pos in (cis, trans, perp, colin):
            b1 = pos[1] - pos[0]
            b2 = pos[2] - pos[1]
            b3 = pos[3] - pos[2]
            c, s = dihedral_cos_sin(b1.unsqueeze(0), b2.unsqueeze(0), b3.unsqueeze(0))
            got.append(float(fourier_cos(c, s, n, phase, n_max=1)))
        assert math.isclose(got[0], 1.0, abs_tol=1e-12)  # cis cos 0
        assert math.isclose(got[1], -1.0, abs_tol=1e-12)  # trans cos π
        assert math.isclose(got[2], 0.0, abs_tol=1e-12)  # perp cos π/2
        assert math.isclose(got[3], 0.0, abs_tol=1e-12)  # colinear, LAMMPS rginv=0
