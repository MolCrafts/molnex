"""Dipolar calculator against the analytic three-dipole chain.

Three parallel dipoles ``(1, 1, 0)`` at ``y = 0, 2, 4`` in a 10 Å cube. The
direct dipole-dipole energy is ``-0.265625``; the short-range part of the
range-separated potential tends to it as ``smearing → ∞`` and to zero as
``smearing → 0``. The espressomd DipolarP3M parity lives in
``regressions/elec-reference-parity.py``.
"""

import pytest
import torch

from molpot.potentials.elec import CalculatorDipole, PotentialDipole
from tests.test_molpot.test_elec.conftest import DTYPES

DIRECT_ENERGY = -0.265625


def parallel_dipoles(dtype):
    dipoles = torch.tensor([[1.0, 1.0, 0.0]] * 3, dtype=dtype)
    cell = 10.0 * torch.eye(3, dtype=dtype)
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 4.0, 0.0]], dtype=dtype)
    neighbor_indices = torch.tensor([[0, 1], [1, 2], [0, 2]], dtype=torch.int64)
    neighbor_vectors = torch.tensor(
        [[0.0, 2.0, 0.0], [0.0, 2.0, 0.0], [0.0, 4.0, 0.0]], dtype=dtype
    )
    return dipoles, cell, positions, neighbor_indices, neighbor_vectors


@pytest.mark.parametrize("dtype", DTYPES)
def test_direct_energy(dtype):
    calculator = CalculatorDipole(potential=PotentialDipole(), full_neighbor_list=False)
    calculator.to(dtype=dtype)
    system = parallel_dipoles(dtype)
    energy = (calculator(*system) * system[0]).sum()
    torch.testing.assert_close(energy, torch.tensor(DIRECT_ENERGY, dtype=dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    ("smearing", "expected"),
    [(1e10, DIRECT_ENERGY), (1e-10, 0.0)],
    ids=["smearing-to-inf-is-direct", "smearing-to-zero-vanishes"],
)
def test_short_range_limits(dtype, smearing, expected):
    calculator = CalculatorDipole(
        potential=PotentialDipole(smearing=smearing),
        full_neighbor_list=False,
        lr_wavelength=1.0,
    )
    calculator.to(dtype=dtype)
    dipoles, _, _, neighbor_indices, neighbor_vectors = parallel_dipoles(dtype)
    pot = calculator._compute_rspace(
        dipoles=dipoles, neighbor_indices=neighbor_indices, neighbor_vectors=neighbor_vectors
    )
    torch.testing.assert_close((pot * dipoles).sum(), torch.tensor(expected, dtype=dtype))
