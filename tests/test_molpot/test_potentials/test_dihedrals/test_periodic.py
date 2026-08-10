"""ProperTorsionPeriodic — Class-I multi-term cosine proper torsion.

Energy: E = sum_n (k_n / s) * [1 + cos(n * phi - gamma_n)]

Hard-coded goldens (no external oracle). Geometry fixtures yield known
dihedral angles under the standard atan2(n1,n2) convention used by
DihedralHarmonic (i-j-k-l, b1=j-i, b2=k-j, b3=l-k).
"""

import math

import pytest
import torch

from molpot.potentials import ProperTorsionPeriodic

# ---------------------------------------------------------------------------
# Geometry fixtures — known dihedral angles
# ---------------------------------------------------------------------------


def _cis_pos() -> torch.Tensor:
    """Four atoms in a plane with proper torsion phi = 0 (cis).

    i=(0,1,0), j=(0,0,0), k=(1,0,0), l=(1,1,0)
    → n1 = n2 = (0,0,1) → phi = 0.
    """
    return torch.tensor(
        [
            [0.0, 1.0, 0.0],  # i
            [0.0, 0.0, 0.0],  # j
            [1.0, 0.0, 0.0],  # k
            [1.0, 1.0, 0.0],  # l
        ],
        dtype=torch.float64,
    )


def _trans_pos() -> torch.Tensor:
    """phi = pi (trans): l flipped below the jk axis."""
    return torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, -1.0, 0.0],
        ],
        dtype=torch.float64,
    )


def _perp_pos() -> torch.Tensor:
    """phi = pi/2: l out of plane along +z."""
    return torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )


def _proper_index() -> torch.Tensor:
    """COO-style [4, 1] proper torsion over atoms 0-1-2-3."""
    return torch.tensor([[0], [1], [2], [3]], dtype=torch.long)


def _single_term_potential(*, k: float = 1.0, n: int = 1, gamma: float = 0.0, s: float = 1.0):
    """Build a one-type, one-term ProperTorsionPeriodic.

    Formula-correct goldens use k=1 so cis (phi=0) yields E=2.0:
        E = (k/s)[1 + cos(n*phi - gamma)] = 1*[1+1] = 2.0
    """
    return ProperTorsionPeriodic(
        k=torch.tensor([[k]], dtype=torch.float64),
        periodicity=torch.tensor([n], dtype=torch.long),
        phase=torch.tensor([[gamma]], dtype=torch.float64),
        idivf=torch.tensor([s], dtype=torch.float64),
    )


class TestProperTorsionPeriodic:
    def test_cis_phi0_k1_gives_energy_2(self):
        # E = (1/1)[1 + cos(0)] = 2.0  (hard-coded golden; Class-I identity)
        pot = _single_term_potential(k=1.0, n=1, gamma=0.0, s=1.0)
        e = pot(
            pos=_cis_pos(),
            proper_index=_proper_index(),
            proper_types=torch.tensor([0], dtype=torch.long),
        )
        assert math.isclose(float(e), 2.0, rel_tol=1e-10, abs_tol=1e-10)

    def test_trans_phi_pi_gives_energy_0(self):
        # E = (1/1)[1 + cos(pi)] = 0.0
        pot = _single_term_potential(k=1.0, n=1, gamma=0.0, s=1.0)
        e = pot(
            pos=_trans_pos(),
            proper_index=_proper_index(),
            proper_types=torch.tensor([0], dtype=torch.long),
        )
        assert math.isclose(float(e), 0.0, rel_tol=1e-10, abs_tol=1e-10)

    def test_perp_phi_half_pi_gives_energy_1(self):
        # E = (1/1)[1 + cos(pi/2)] = 1.0
        pot = _single_term_potential(k=1.0, n=1, gamma=0.0, s=1.0)
        e = pot(
            pos=_perp_pos(),
            proper_index=_proper_index(),
            proper_types=torch.tensor([0], dtype=torch.long),
        )
        assert math.isclose(float(e), 1.0, rel_tol=1e-10, abs_tol=1e-10)

    def test_empty_proper_index_returns_zero(self):
        pot = _single_term_potential()
        e = pot(
            pos=_cis_pos(),
            proper_index=torch.zeros(4, 0, dtype=torch.long),
            proper_types=torch.zeros(0, dtype=torch.long),
        )
        assert float(e) == 0.0

    def test_rejects_row_major_n_by_4_shape(self):
        pot = _single_term_potential()
        wrong = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)  # [1, 4] == [N, 4]
        with pytest.raises(ValueError, match=r"\[4"):
            pot(
                pos=_cis_pos(),
                proper_index=wrong,
                proper_types=torch.tensor([0], dtype=torch.long),
            )

    def test_rejects_edge_index_shape(self):
        pot = _single_term_potential()
        edge_like = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)  # [E, 2]
        with pytest.raises(ValueError, match=r"\[4"):
            pot(
                pos=_cis_pos(),
                proper_index=edge_like,
                proper_types=torch.tensor([0, 0], dtype=torch.long),
            )

    def test_rejects_bond_like_2_by_n_shape(self):
        pot = _single_term_potential()
        bond_like = torch.tensor([[0], [1]], dtype=torch.long)  # [2, 1] COO bond shape
        with pytest.raises(ValueError, match=r"\[4"):
            pot(
                pos=_cis_pos(),
                proper_index=bond_like,
                proper_types=torch.tensor([0], dtype=torch.long),
            )

    def test_multi_term_sums_component_energies(self):
        # Two terms at phi=0, gamma=0:
        #   term1: k=1, n=1 → (1/1)[1+cos(0)] = 2.0
        #   term2: k=0.5, n=2 → (0.5/1)[1+cos(0)] = 1.0
        #   total = 3.0
        pot = ProperTorsionPeriodic(
            k=torch.tensor([[1.0, 0.5]], dtype=torch.float64),
            periodicity=torch.tensor([1, 2], dtype=torch.long),
            phase=torch.tensor([[0.0, 0.0]], dtype=torch.float64),
            idivf=torch.tensor([1.0], dtype=torch.float64),
        )
        e = pot(
            pos=_cis_pos(),
            proper_index=_proper_index(),
            proper_types=torch.tensor([0], dtype=torch.long),
        )
        assert math.isclose(float(e), 3.0, rel_tol=1e-10, abs_tol=1e-10)
