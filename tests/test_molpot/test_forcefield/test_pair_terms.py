"""Pair terms owned by the Potentials assembly — mirrors ``pair_terms.py``.

``LJCutPotential`` is the seam where S3's split becomes observable: it returns
the repulsive and attractive halves of ``4ε[(σ/r)¹² − (σ/r)⁶]`` separately, so
``Potentials`` can label one ``local`` and the other ``disp`` without a second
energy path. Analytic-force values are pinned one layer up
(``test_analytic_forces.py``); what is pinned here is the return *shape* of
both entry points, the split at the LJ minimum, and — from
``composable-a0-02`` — the quintic switching function and the ``edge_mask``.

The switch is ``S = 1 − x³(10 − 15x + 6x²)`` on
``x = clamp((r − r_s)/(r_c − r_s), 0, 1)`` (OpenMM ``useSwitchingFunction``;
algebraically the LAMMPS ``lj/mdf`` taper ``(1 − x)³(1 + 3x + 6x²)``). It has
``S(r_s) = 1``, ``S(r_c) = 0`` and ``S'(r_s) = S'(r_c) = 0``, so energy *and*
force go to zero smoothly at the cutoff — the S8 continuity gate. The hard
truncation it replaces put a step in the force at ``r_c``, which an MD
integrator turns into a slow energy drift.

``edge_mask`` is the other half of the same change: a fixed-capacity neighbour
list carries dead rows, and they are multiplied out rather than sliced out, so
the shape of every intermediate is known at trace time (zero graph breaks).
"""

from __future__ import annotations

import pytest
import torch

from molpot.forcefield.pair_terms import CoulombOpenPotential, LJCutPotential

DTYPE = torch.float64
EPSILON = 0.2
SIGMA = 3.0
#: ``r`` at the LJ minimum, where ``4ε(σ/r)¹² = +ε`` and ``−4ε(σ/r)⁶ = −2ε``.
R_MIN = 2.0 ** (1.0 / 6.0) * SIGMA
#: Real-space cutoff ``r_c`` (Å) and switch start ``r_s = 0.9 r_c`` of the
#: switched fixture — the window width the acceptance criterion names.
R_CUT = 10.0
R_SWITCH = 0.9 * R_CUT
#: The two probes that straddle ``r_c`` in the continuity gate (Å).
DELTA = 1e-6


def _one_pair_kwargs(*, r: float = R_MIN) -> dict[str, torch.Tensor]:
    """A single half-list pair at separation ``r`` along x̂, unscaled."""
    return {
        "edge_index": torch.tensor([[0, 1]], dtype=torch.long),
        "distances": torch.tensor([r], dtype=DTYPE),
        "epsilon": torch.full((2,), EPSILON, dtype=DTYPE),
        "sigma": torch.full((2,), SIGMA, dtype=DTYPE),
        "scales": torch.ones(1, dtype=DTYPE),
    }


def _from_positions(positions: torch.Tensor, edge_index: torch.Tensor) -> dict[str, object]:
    """The two geometry arguments :meth:`LJCutPotential.energy_forces` takes.

    That method no longer sees the positions: it takes the separation vectors,
    so direction and distance always come out of the *same* vector (under a
    cell that vector is the minimum image, which the raw difference is not),
    and an explicit ``n_atoms`` for the force buffer it scatters into. These
    fixtures are all open-boundary, so ``pos[j] - pos[i]`` is that vector and
    the values below are unchanged.
    """
    return {
        "pair_vectors": positions[edge_index[:, 1]] - positions[edge_index[:, 0]],
        "n_atoms": int(positions.shape[0]),
    }


def _sum(halves: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """``repulsion + dispersion`` — the full LJ energy, autograd graph intact."""
    repulsion, dispersion = halves
    return repulsion + dispersion


def _total(halves: tuple[torch.Tensor, torch.Tensor]) -> float:
    """:func:`_sum` as a Python float."""
    return float(_sum(halves))


def _dedr(potential: LJCutPotential, r: float) -> float:
    """``dE/dr`` of one pair at ``r`` (kcal mol⁻¹ Å⁻¹), by autograd on the distance."""
    distances = torch.tensor([r], dtype=DTYPE, requires_grad=True)
    kwargs = _one_pair_kwargs(r=r)
    kwargs["distances"] = distances
    (gradient,) = torch.autograd.grad(_sum(potential.energy(**kwargs)), distances)
    return float(gradient)


def _two_atom_energy_force(potential: LJCutPotential, r: float) -> tuple[float, float]:
    """``(E, F_x)`` of a two-atom pair at separation ``r`` along x̂."""
    positions = torch.tensor([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=DTYPE)
    kwargs = _one_pair_kwargs(r=r)
    halves, forces = potential.energy_forces(
        **_from_positions(positions, kwargs["edge_index"]), **kwargs
    )
    return _total(halves), float(forces[0, 0])


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
        (repulsion, dispersion), forces = potential.energy_forces(
            **_from_positions(positions, kwargs["edge_index"]), **kwargs
        )
        expected_repulsion, expected_dispersion = potential.energy(**kwargs)
        assert float(repulsion) == pytest.approx(float(expected_repulsion), abs=1e-10)
        assert float(dispersion) == pytest.approx(float(expected_dispersion), abs=1e-10)
        assert forces.shape == (2, 3)

    # --- quintic switch (composable-a0-02, S8 gate 2) ---

    def test_switch_is_one_at_the_window_start(self) -> None:
        """``S(r_s) = 1`` — inside the window the plain LJ energy is untouched."""
        kwargs = _one_pair_kwargs(r=R_SWITCH)
        switched = _total(LJCutPotential(cutoff=R_CUT, switch_start=R_SWITCH).energy(**kwargs))
        plain = _total(LJCutPotential(cutoff=R_CUT).energy(**kwargs))
        assert plain != 0.0
        assert switched == pytest.approx(plain, rel=1e-12, abs=1e-14)

    def test_switch_is_half_at_the_window_midpoint(self) -> None:
        """``S(x = 0.5) = 1 − 0.125·(10 − 7.5 + 1.5) = 0.5`` exactly."""
        r_mid = 0.5 * (R_SWITCH + R_CUT)
        kwargs = _one_pair_kwargs(r=r_mid)
        switched = _total(LJCutPotential(cutoff=R_CUT, switch_start=R_SWITCH).energy(**kwargs))
        plain = _total(LJCutPotential(cutoff=R_CUT).energy(**kwargs))
        assert switched / plain == pytest.approx(0.5, rel=1e-12)

    def test_switch_derivative_vanishes_at_both_window_ends(self) -> None:
        """``S'(r_s) = S'(r_c) = 0``, read off ``dE/dr`` by autograd.

        At ``r_s`` the only surviving term of ``d(S·E)/dr`` is ``S·E'``, so the
        switched derivative equals the plain one; at ``r_c`` both ``S`` and
        ``S'`` vanish, so the derivative does too. A non-zero ``S'`` at either
        end is exactly the force discontinuity the switch exists to remove.
        """
        potential = LJCutPotential(cutoff=R_CUT, switch_start=R_SWITCH)
        plain = LJCutPotential(cutoff=R_CUT)
        assert _dedr(potential, R_SWITCH) == pytest.approx(_dedr(plain, R_SWITCH), abs=1e-12)
        assert abs(_dedr(potential, R_CUT - DELTA)) < 1e-8

    def test_energy_and_force_are_continuous_across_the_cutoff(self) -> None:
        """``|ΔE| < 1e-10`` kcal/mol and ``|ΔF| < 1e-8`` kcal mol⁻¹ Å⁻¹ at ``r_c``.

        Beyond ``r_c`` the pair contributes nothing, so continuity means the
        switched value has already reached zero on the inside — the S8 gate.
        """
        potential = LJCutPotential(cutoff=R_CUT, switch_start=R_SWITCH)
        e_inside, f_inside = _two_atom_energy_force(potential, R_CUT - DELTA)
        e_outside, f_outside = _two_atom_energy_force(potential, R_CUT + DELTA)
        assert e_outside == 0.0
        assert abs(e_inside - e_outside) < 1e-10
        assert abs(f_inside - f_outside) < 1e-8

    def test_analytic_forces_match_autograd_inside_the_switching_window(self) -> None:
        """``dS/dr`` is in the closed form, not only in the energy.

        Dropping the ``E·dS/dr`` term leaves an energy that tapers and a force
        that does not — a non-conservative model that silently heats an MD run.
        """
        potential = LJCutPotential(cutoff=R_CUT, switch_start=R_SWITCH)
        r = 0.5 * (R_SWITCH + R_CUT)
        positions = torch.tensor([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=DTYPE)
        window = _one_pair_kwargs(r=r)
        _, analytic = potential.energy_forces(
            **_from_positions(positions, window["edge_index"]), **window
        )

        leaf = positions.clone().requires_grad_(True)
        distances = (leaf[1] - leaf[0]).norm().reshape(1)
        kwargs = _one_pair_kwargs(r=r)
        kwargs["distances"] = distances
        energy = _sum(potential.energy(**kwargs))
        (autograd,) = torch.autograd.grad(energy, leaf)
        assert torch.allclose(analytic, -autograd, atol=1e-8, rtol=0.0)

    # --- fixed-capacity lists (composable-a0-02) ---

    def test_masked_pairs_contribute_exactly_zero(self) -> None:
        """A dead row changes neither energy nor force — no slicing needed."""
        potential = LJCutPotential(cutoff=R_CUT)
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [R_MIN, 0.0, 0.0], [0.0, R_MIN, 0.0]], dtype=DTYPE
        )
        padded = {
            "edge_index": torch.tensor([[0, 1], [0, 2]], dtype=torch.long),
            "distances": torch.tensor([R_MIN, R_MIN], dtype=DTYPE),
            "epsilon": torch.full((3,), EPSILON, dtype=DTYPE),
            "sigma": torch.full((3,), SIGMA, dtype=DTYPE),
            "scales": torch.ones(2, dtype=DTYPE),
            "edge_mask": torch.tensor([True, False]),
        }
        masked = _total(potential.energy(**padded))
        assert masked == pytest.approx(_total(potential.energy(**_one_pair_kwargs())), abs=1e-14)

        (_, _), forces = potential.energy_forces(
            **_from_positions(positions, padded["edge_index"]), **padded
        )
        assert torch.equal(forces[2], torch.zeros(3, dtype=DTYPE))

    @pytest.mark.slow
    def test_energy_forces_has_no_graph_breaks_on_a_masked_padded_list(self) -> None:
        """Multiplicative mask, no ``edge_index[within]`` — one dynamo graph."""
        from molix.compile import Compiler

        potential = LJCutPotential(cutoff=R_CUT, switch_start=R_SWITCH)
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [R_MIN, 0.0, 0.0], [0.0, R_MIN, 0.0]], dtype=DTYPE
        )
        edge_index = torch.tensor([[0, 1], [0, 2], [0, 0], [0, 0]], dtype=torch.long)
        explanation = Compiler.explain(
            potential.energy_forces,
            **_from_positions(positions, edge_index),
            edge_index=edge_index,
            distances=torch.tensor([R_MIN, R_MIN, 1.0, 1.0], dtype=DTYPE),
            epsilon=torch.full((3,), EPSILON, dtype=DTYPE),
            sigma=torch.full((3,), SIGMA, dtype=DTYPE),
            scales=torch.ones(4, dtype=DTYPE),
            edge_mask=torch.tensor([True, True, False, False]),
        )
        assert (explanation.graph_count, explanation.graph_break_count) == (1, 0)


class TestCoulombOpenPotential:
    def test_masked_pairs_contribute_exactly_zero(self) -> None:
        """The same fixed-shape contract as LJ — Coulomb masks, it does not slice."""
        potential = CoulombOpenPotential(cutoff=R_CUT)
        charges = torch.tensor([1.0, -1.0, -1.0], dtype=DTYPE)
        live_only = potential.energy(
            edge_index=torch.tensor([[0, 1]], dtype=torch.long),
            distances=torch.tensor([2.0], dtype=DTYPE),
            charges=charges,
            scales=torch.ones(1, dtype=DTYPE),
        )
        padded = potential.energy(
            edge_index=torch.tensor([[0, 1], [0, 2]], dtype=torch.long),
            distances=torch.tensor([2.0, 2.0], dtype=DTYPE),
            charges=charges,
            scales=torch.ones(2, dtype=DTYPE),
            edge_mask=torch.tensor([True, False]),
        )
        assert float(padded) == pytest.approx(float(live_only), abs=1e-14)
