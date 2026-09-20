"""Seminario force constants: analytic goldens and the invariances they must hold.

No dataset access — the fast tier stays a unit tier. The physical validation
against THEMol's QM Hessians (per-pair medians against literature Class-I
values) is recorded in the module docstring of
:mod:`molpot.forcefield.seminario`, not re-run here.
"""

from __future__ import annotations

import math

import pytest
import torch

from molpot.forcefield.seminario import SeminarioForceConstants

#: An exactly-representable spring constant, kcal mol⁻¹ Å⁻².
K_EXACT: float = 500.0
#: Analytic goldens are exact to a few ULP of the eigen-decomposition.
EXACT_ATOL: float = 1e-9


def _diatomic_hessian(k: float) -> torch.Tensor:
    """Hessian of ``E = 1/2 k (r - r0)^2`` for two atoms along x.

    The only curvature is along the bond, so the mixed block is ``-k`` on ``xx``
    and zero elsewhere — the textbook case in which Seminario must return ``k``.
    """
    hessian = torch.zeros(6, 6, dtype=torch.float64)
    for a, b, sign in ((0, 0, 1.0), (1, 1, 1.0), (0, 1, -1.0), (1, 0, -1.0)):
        hessian[3 * a, 3 * b] = sign * k
    return hessian


class TestSeminarioBondConstants:
    """One bond, where the answer is analytic."""

    def test_diatomic_recovers_its_own_force_constant(self) -> None:
        """``E = 1/2 k (r - r0)^2`` in, ``k`` out — the convention pinned."""
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float64)
        bond_index = torch.tensor([[0], [1]])
        k = SeminarioForceConstants().bond_constants(pos, _diatomic_hessian(K_EXACT), bond_index)
        assert k.shape == (1,)
        torch.testing.assert_close(
            k, torch.tensor([K_EXACT], dtype=torch.float64), atol=EXACT_ATOL, rtol=0.0
        )

    def test_rotating_the_molecule_does_not_change_the_constant(self) -> None:
        """A force constant is a scalar; the method must not depend on the frame.

        The Hessian rotates as ``R H R^T`` block-wise, so a rotated molecule with
        its rotated Hessian has to give the same answer. This is the property
        that catches a transposed block or a direction taken in the wrong frame.
        """
        angle = 0.7
        rot = torch.tensor(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float64)
        hessian = _diatomic_hessian(K_EXACT)
        big = torch.block_diag(rot, rot)
        seminario = SeminarioForceConstants()
        bond_index = torch.tensor([[0], [1]])
        plain = seminario.bond_constants(pos, hessian, bond_index)
        turned = seminario.bond_constants(pos @ rot.T, big @ hessian @ big.T, bond_index)
        torch.testing.assert_close(plain, turned, atol=EXACT_ATOL, rtol=0.0)

    def test_swapping_the_bond_ends_does_not_change_the_constant(self) -> None:
        """``k_AB == k_BA``: a bond has no direction."""
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float64)
        hessian = _diatomic_hessian(K_EXACT)
        seminario = SeminarioForceConstants()
        forward = seminario.bond_constants(pos, hessian, torch.tensor([[0], [1]]))
        reverse = seminario.bond_constants(pos, hessian, torch.tensor([[1], [0]]))
        torch.testing.assert_close(forward, reverse, atol=EXACT_ATOL, rtol=0.0)

    def test_empty_bond_list_gives_an_empty_result(self) -> None:
        """Dtype and device follow ``pos``, as the sibling Wilson builder does."""
        pos = torch.zeros(2, 3, dtype=torch.float64)
        k = SeminarioForceConstants().bond_constants(
            pos, _diatomic_hessian(K_EXACT), torch.zeros(2, 0, dtype=torch.long)
        )
        assert k.shape == (0,)
        assert k.dtype == pos.dtype

    def test_a_wrong_shaped_bond_index_is_refused(self) -> None:
        """``(n, 2)`` is the pair-list layout, not the COO one this takes."""
        pos = torch.zeros(2, 3, dtype=torch.float64)
        with pytest.raises(ValueError, match=r"\(2, n_bonds\)"):
            SeminarioForceConstants().bond_constants(
                pos, _diatomic_hessian(K_EXACT), torch.zeros(4, 2, dtype=torch.long)
            )


class TestSeminarioAngleConstants:
    """Bends, where the two arms act as springs in series."""

    @staticmethod
    def _trimer() -> tuple[torch.Tensor, torch.Tensor]:
        """A right-angle trimer about atom 1 with a diagonal arm curvature."""
        pos = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
        hessian = torch.zeros(9, 9, dtype=torch.float64)
        # Curvature perpendicular to each arm, equal on both, so the series sum
        # is exactly half of one arm's contribution.
        for a, b in ((0, 1), (1, 0)):
            hessian[3 * a + 1, 3 * b + 1] = -K_EXACT
        for a, b in ((2, 1), (1, 2)):
            hessian[3 * a + 0, 3 * b + 0] = -K_EXACT
        return pos, hessian

    def test_equal_arms_put_two_springs_in_series(self) -> None:
        """``1/k = 1/k_a + 1/k_c`` — equal unit arms give ``k_a / 2``.

        Both arms are unit length and carry ``K_EXACT`` perpendicular to
        themselves, so each term is ``K_EXACT`` and the series result is half.
        An implementation that *added* the arms instead would give ``2 K`` and
        pass no test that only checked the order of magnitude.
        """
        pos, hessian = self._trimer()
        angle_index = torch.tensor([[0], [1], [2]])
        k = SeminarioForceConstants().angle_constants(pos, hessian, angle_index)
        torch.testing.assert_close(
            k, torch.tensor([K_EXACT / 2.0], dtype=torch.float64), atol=EXACT_ATOL, rtol=0.0
        )

    def test_swapping_the_outer_atoms_does_not_change_the_constant(self) -> None:
        """``k_ABC == k_CBA``: the apex is what matters, not the arm order."""
        pos, hessian = self._trimer()
        seminario = SeminarioForceConstants()
        forward = seminario.angle_constants(pos, hessian, torch.tensor([[0], [1], [2]]))
        reverse = seminario.angle_constants(pos, hessian, torch.tensor([[2], [1], [0]]))
        torch.testing.assert_close(forward, reverse, atol=EXACT_ATOL, rtol=0.0)

    def test_empty_angle_list_gives_an_empty_result(self) -> None:
        pos, hessian = self._trimer()
        k = SeminarioForceConstants().angle_constants(
            pos, hessian, torch.zeros(3, 0, dtype=torch.long)
        )
        assert k.shape == (0,)
        assert k.dtype == pos.dtype

    def test_a_wrong_shaped_angle_index_is_refused(self) -> None:
        pos, hessian = self._trimer()
        with pytest.raises(ValueError, match=r"\(3, n_angles\)"):
            SeminarioForceConstants().angle_constants(
                pos, hessian, torch.zeros(2, 5, dtype=torch.long)
            )
