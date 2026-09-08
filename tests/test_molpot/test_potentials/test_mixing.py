"""Shared mixing utilities — mirrors ``src/molpot/potentials/mixing.py``.

``geometric_arithmetic_mixing`` combines per-atom parameters into per-pair ones
by name: the geometric mean ``sqrt(p_i p_j)`` for energy-like parameters, the
arithmetic mean ``½(p_i + p_j)`` for distance-like ones. It is the generic
sibling of :func:`molpot.potentials.vdw.lj126.lorentz_berthelot`, which mixes
the Lennard-Jones tables specifically.

**The additive epsilon under the root is the thing these tests pin.** Writing
``sqrt(p_i p_j + 1e-12)`` to keep the derivative finite at zero does not
compute a geometric mean at all — it computes a *biased* one, and the bias is
absolute, so it is invisible on large parameters and ruinous on small ones. The
same guard was removed from ``lorentz_berthelot`` on 2026-09-07 and is recorded
as ledger finding **F10** here (``.claude/notes/composable-potentials.md``); the
zero case it was protecting is handled instead by masking the zero product,
which gives the exact mean *and* a finite backward pass. Measured against the
current implementation, in fp64:

===============  ==============================  ==================
``p_i = p_j``    ``sqrt(p_i p_j + 1e-12)``       error
===============  ==============================  ==================
``0.2``          ``0.2000000000025``             ``+2.5e-12``
``1e-6``         ``1.414213562373095e-06``       ``+41.42 %``
``0`` (product)  ``1e-06``                       not zero at all
===============  ==============================  ==================

A well depth 41 % too deep is not a rounding artefact of a learnable parameter
set, it is a different force field; and the ``2.5e-12`` on a physical ``ε`` is
still above the ``1e-12`` these goldens hold the mean to, i.e. visible at the
Lennard-Jones minimum.
"""

from __future__ import annotations

import torch

from molpot.potentials.mixing import geometric_arithmetic_mixing

DTYPE = torch.float64

#: The single pair ``(i, j)`` every case below mixes over.
PAIR = torch.tensor([[0, 1]], dtype=torch.long)

#: Distance-like table carried through every call, so the arithmetic branch is
#: exercised beside the geometric one — ``½(3.5 + 2.5) = 3.0`` exactly.
SIGMA = (3.5, 2.5)
SIGMA_ARITHMETIC = 3.0

#: A physical Lennard-Jones well depth in kcal/mol, and a learnable one that
#: has not yet grown. The geometric mean of a value with itself *is* that
#: value, which is what makes both cases exact goldens needing no oracle.
PHYSICAL_EPSILON = 0.2
TINY_EPSILON = 1e-6


def _mixed(
    epsilon: tuple[float, float],
    *,
    requires_grad: bool = False,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Mix one pair, geometric in ``epsilon`` and arithmetic in ``sigma``.

    Args:
        epsilon: The two per-atom well depths in kcal/mol.
        requires_grad: Make ``epsilon`` a differentiable leaf.

    Returns:
        ``(result, epsilon_leaf)`` — the ``{"epsilon_ij", "sigma_ij"}`` mapping
        and the tensor the epsilons were read from, so a caller can
        differentiate the first with respect to the second.
    """
    leaf = torch.tensor(epsilon, dtype=DTYPE, requires_grad=requires_grad)
    result = geometric_arithmetic_mixing(
        {"epsilon": leaf, "sigma": torch.tensor(SIGMA, dtype=DTYPE)},
        PAIR,
        ["epsilon"],
        ["sigma"],
    )
    return result, leaf


class TestGeometricArithmeticMixing:
    def test_the_geometric_mean_of_a_physical_epsilon_is_exact(self) -> None:
        """``sqrt(0.2 · 0.2)`` is ``0.2``, to the last fp64 bit.

        Nothing is added under the root, so mixing a table with itself returns
        it unchanged. The current ``+1e-12`` returns ``0.2000000000025``.
        """
        result, _ = _mixed((PHYSICAL_EPSILON, PHYSICAL_EPSILON))
        torch.testing.assert_close(
            result["epsilon_ij"],
            torch.tensor([PHYSICAL_EPSILON], dtype=DTYPE),
            rtol=0.0,
            atol=1e-15,
        )

    def test_a_learnable_epsilon_far_below_the_guard_is_not_inflated(self) -> None:
        """``sqrt(1e-6 · 1e-6)`` is ``1e-6``, not ``sqrt(2) · 1e-6``.

        An absolute ``1e-12`` under the root dominates a product of ``1e-12``,
        so the guard returns ``1.414213562373095e-06`` — 41 % too deep. A
        parameterizer head whose output starts small would train against a
        well depth it never predicted.
        """
        result, _ = _mixed((TINY_EPSILON, TINY_EPSILON))
        torch.testing.assert_close(
            result["epsilon_ij"],
            torch.tensor([TINY_EPSILON], dtype=DTYPE),
            rtol=0.0,
            atol=1e-15,
        )

    def test_a_zero_product_mixes_to_zero_with_a_finite_gradient(self) -> None:
        """The zero the guard existed for: exactly ``0``, and still differentiable.

        ``d sqrt(x)/dx`` diverges at ``x = 0``, which is what the additive term
        was avoiding — at the cost of returning ``1e-06`` for a pair that has
        no interaction at all. Masking the zero product gives both: the exact
        value and a finite backward pass. (A *negative* product is a broken
        parameterization and is not swallowed here.)
        """
        result, epsilon = _mixed((0.0, 0.03), requires_grad=True)
        assert float(result["epsilon_ij"].detach()) == 0.0
        (gradient,) = torch.autograd.grad(result["epsilon_ij"].sum(), epsilon)
        assert bool(torch.isfinite(gradient).all())

    def test_arithmetic_keys_are_the_plain_mean(self) -> None:
        """``½(3.5 + 2.5) = 3.0`` — the distance branch has no guard to remove."""
        result, _ = _mixed((PHYSICAL_EPSILON, PHYSICAL_EPSILON))
        torch.testing.assert_close(
            result["sigma_ij"],
            torch.tensor([SIGMA_ARITHMETIC], dtype=DTYPE),
            rtol=0.0,
            atol=1e-15,
        )
