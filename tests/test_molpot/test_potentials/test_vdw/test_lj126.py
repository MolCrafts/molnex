"""LJ 12-6 kernel — mirrors ``src/molpot/potentials/vdw/lj126.py``.

``lj126_pair_energy_split`` is the S3 kernel: it returns the repulsive wall
``4ε(σ/r)¹²`` and the attractive tail ``−4ε(σ/r)⁶`` separately so the assembled
potential can label them ``local`` and ``disp``. Splitting must not change the
number it sums to, which is what makes the S-decomposition a grouping of one
energy path rather than a second one.

The module also owns the **combining rules** that turn the per-atom ``ε`` /
``σ`` tables into the per-pair values that kernel takes: ``lorentz_berthelot``
(``σ`` arithmetic, ``ε`` geometric — AMBER / CHARMM) and ``geometric_mixing``
(both geometric — OPLS-AA), with :class:`MixingRule` naming the two as the
values a ``pair:lj/cut`` style declares and the model manifest records. The
rule is model identity: on a cross pair the two rules are different force
fields, and every molnex model mixed Lorentz–Berthelot with nothing saying so
until 2026-09-08 (ledger finding F24).
"""

from __future__ import annotations

import pytest
import torch

from molpot.potentials.vdw.lj126 import (
    MixingRule,
    geometric_mixing,
    lj126_pair_energy,
    lj126_pair_energy_split,
    lorentz_berthelot,
)

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


#: Per-atom Lennard-Jones tables of the two-type mixing goldens below —
#: ``ε`` in kcal/mol, ``σ`` in Å. Alkane-shaped (an OPLS-AA ``CT`` carbon and
#: an ``HC`` hydrogen), and deliberately *unequal*, because the two combining
#: rules coincide on a same-type pair and only a cross pair tells them apart.
EPS_I, EPS_J = 0.066, 0.03
SIG_I, SIG_J = 3.5, 2.5

#: **Hard-coded goldens** — the geometric mean of each table over the one cross
#: pair, ``sqrt(0.066 · 0.03)`` and ``sqrt(3.5 · 2.5)`` = ``sqrt(8.75)``,
#: evaluated as fp64 literals::
#:
#:     python -c "import math; print(repr(math.sqrt(0.066*0.03)), repr(math.sqrt(3.5*2.5)))"
#:     0.044497190922573975 2.958039891549808
#:
#: Pure arithmetic on two exactly-stated products — no oracle, no third-party
#: implementation. ``EPS_GEOMETRIC`` is also what Lorentz–Berthelot gives for
#: ``ε`` (both rules take the geometric mean there); ``SIG_GEOMETRIC`` is what
#: separates them, since Lorentz–Berthelot's arithmetic mean of the same two
#: ``σ`` is exactly ``3.0``.
EPS_GEOMETRIC = 0.044497190922573975
SIG_GEOMETRIC = 2.958039891549808

#: The single cross pair ``(i, j)`` the mixing rules are read on.
CROSS_PAIR = torch.tensor([[0, 1]], dtype=torch.long)


def _two_type_params(
    *,
    epsilon: tuple[float, float] = (EPS_I, EPS_J),
    sigma: tuple[float, float] = (SIG_I, SIG_J),
    requires_grad: bool = False,
) -> dict[str, torch.Tensor]:
    """The ``{"epsilon", "sigma"}`` per-atom tables a mixing rule takes."""
    return {
        "epsilon": torch.tensor(epsilon, dtype=DTYPE, requires_grad=requires_grad),
        "sigma": torch.tensor(sigma, dtype=DTYPE, requires_grad=requires_grad),
    }


class TestGeometricMixing:
    """``geometric_mixing`` — the OPLS-AA combining rule (Jorgensen 1996).

    ``ε_ij = sqrt(ε_i ε_j)`` **and** ``σ_ij = sqrt(σ_i σ_j)``: OPLS-AA takes
    the geometric mean of *both* tables, where Lorentz–Berthelot takes the
    arithmetic mean of ``σ``. On a cross pair the two are different force
    fields, which is why the rule is model identity rather than a kernel
    detail.

    Reference:
        Jorgensen, Maxwell, Tirado-Rives. "Development and Testing of the OPLS
        All-Atom Force Field on Conformational Energetics and Properties of
        Organic Liquids" J. Am. Chem. Soc. 118, 11225 (1996).
        https://doi.org/10.1021/ja9621760
    """

    def test_both_tables_are_the_root_of_their_product(self) -> None:
        """Hard-coded fp64 goldens, provenance beside the constants."""
        mixed = geometric_mixing(_two_type_params(), CROSS_PAIR)
        assert float(mixed["epsilon_ij"]) == pytest.approx(EPS_GEOMETRIC, abs=1e-12)
        assert float(mixed["sigma_ij"]) == pytest.approx(SIG_GEOMETRIC, abs=1e-12)

    def test_a_zero_product_mixes_to_zero_with_a_finite_gradient(self) -> None:
        """Both roots at once: ``0`` out, and a backward pass that stays finite.

        One call, because ``ε`` and ``σ`` go through the *same* masked-sqrt code
        path here — asserting it twice would pin one behaviour in two places.
        Geometric mixing is where a zero ``σ`` can matter at all: the
        Lorentz–Berthelot ``σ`` is an arithmetic mean with no root to protect,
        while both rules take the root of the ``ε`` product.

        The mask is what replaces the additive ``1e-12`` under the root that
        ``lorentz_berthelot`` carried until 2026-09-07: that guard biased every
        well depth (a learnable ``ε = 1e-6`` came out 41 % too deep), while the
        unguarded root has an infinite derivative at zero and poisons a
        parameterizer's backward pass with ``inf`` / ``NaN``. Masking gives the
        exact mean *and* a finite gradient. A zero table is not hypothetical: it
        is what an untrained parameterizer head emits before it has learned
        anything.
        """
        params = _two_type_params(epsilon=(0.0, EPS_J), sigma=(0.0, SIG_J), requires_grad=True)
        mixed = geometric_mixing(params, CROSS_PAIR)
        assert float(mixed["epsilon_ij"].detach()) == 0.0
        assert float(mixed["sigma_ij"].detach()) == 0.0
        gradients = torch.autograd.grad(
            mixed["epsilon_ij"].sum() + mixed["sigma_ij"].sum(),
            [params["epsilon"], params["sigma"]],
        )
        assert all(bool(torch.isfinite(g).all()) for g in gradients)


class TestMixingRule:
    """``MixingRule`` — the declared combining rule, as a value.

    The enum is the vocabulary a ``pair:lj/cut`` style option is validated
    against and the string the model manifest records, so its ``mix`` is a
    *dispatch* onto the two free functions and never a third implementation of
    either mean.
    """

    def test_geometric_dispatches_to_the_free_function(self) -> None:
        """``MixingRule("geometric").mix`` is ``geometric_mixing``, value for value."""
        params = _two_type_params()
        mixed = MixingRule("geometric").mix(params, CROSS_PAIR)
        expected = geometric_mixing(params, CROSS_PAIR)
        assert set(mixed) == {"epsilon_ij", "sigma_ij"}
        for key, value in expected.items():
            assert torch.equal(mixed[key], value), key

    def test_lorentz_berthelot_dispatches_to_the_free_function(self) -> None:
        """The other member is the untouched ``σ`` arithmetic mean — here ``3.0``."""
        params = _two_type_params()
        mixed = MixingRule("lorentz-berthelot").mix(params, CROSS_PAIR)
        expected = lorentz_berthelot(params, CROSS_PAIR)
        assert set(mixed) == {"epsilon_ij", "sigma_ij"}
        for key, value in expected.items():
            assert torch.equal(mixed[key], value), key
