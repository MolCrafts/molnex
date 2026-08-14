"""Unit tests for the Cartesian Hessian producer in molpot.derivation.hessian.

Mirrors ``src/molpot/derivation/hessian.py``. Each test targets exactly one
method of :class:`~molpot.derivation.hessian.HessianDerivation` on a system
whose ``H = ∂²E/∂pos²`` is known in closed form — the quadratic toy
``E = s Σ‖r‖²`` (conftest) and the harmonic dimer
``E = ½k(r − r₀)²`` (:class:`molpot.potentials.BondHarmonic`). No reference
implementation, no finite differences, no third-party oracle: every expected
matrix below is hard-coded from the Wilson closed form

    ∂²E/∂d∂d = k u uᵀ + k (r − r₀)/r · (I − u uᵀ),   H₁₁ = H₂₂ = −H₁₂

    Wilson, E. B. *J. Chem. Phys.* **9**, 76 (1941).
    https://doi.org/10.1063/1.1750829

Layout: ``(3N, 3N)`` with flat Cartesian index ``i = 3a + c`` (``x0,y0,z0,x1,…``).
Units: energy per Å² — eV/Å² for the toy, kcal·mol⁻¹·Å⁻² for a kcal/mol core.
All assertions run in float64; ``atol=1e-12`` for the exactly-representable toy
goldens and ``atol=1e-10`` for the dimer goldens (one ``norm`` + one division
away from exact), always with ``rtol=0``.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from molpot.derivation.hessian import HessianDerivation
from molpot.potentials import BondHarmonic
from tests.test_molpot.test_derivation.conftest import KernelToyPotential, kernel_toy_batch

#: Harmonic dimer force constant (energy/Å²) and equilibrium length (Å).
DIMER_K: float = 2.0
DIMER_R0: float = 1.5

#: ``H = k BᵀB`` with the Wilson stretch row ``B = [[-1,0,0,1,0,0]]`` — the
#: equilibrium (``r = r₀``) dimer, where the transverse term vanishes. Chains
#: the ``k = 2.0`` golden of ``regressions/universal-class-i-03-losses.py``.
EQUILIBRIUM_HESSIAN: list[list[float]] = [
    [2.0, 0.0, 0.0, -2.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [-2.0, 0.0, 0.0, 2.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
]

#: Stretched dimer at ``r = 2.0``, ``r₀ = 1.5``, ``k = 2.0``: longitudinal
#: ``k = 2.0`` along x, transverse ``k(r − r₀)/r = 2.0 · 0.5 / 2.0 = 0.5`` on
#: y and z. Dropping the ``(I − uuᵀ)`` term zeroes the ±0.5 entries.
STRETCHED_HESSIAN: list[list[float]] = [
    [2.0, 0.0, 0.0, -2.0, 0.0, 0.0],
    [0.0, 0.5, 0.0, 0.0, -0.5, 0.0],
    [0.0, 0.0, 0.5, 0.0, 0.0, -0.5],
    [-2.0, 0.0, 0.0, 2.0, 0.0, 0.0],
    [0.0, -0.5, 0.0, 0.0, 0.5, 0.0],
    [0.0, 0.0, -0.5, 0.0, 0.0, 0.5],
]

#: Wilson stretch row ``B = ∂r/∂x = (−u, u)`` for the x-aligned dimer.
WILSON_B: list[list[float]] = [[-1.0, 0.0, 0.0, 1.0, 0.0, 0.0]]

EXACT_ATOL: float = 1e-12
DIMER_ATOL: float = 1e-10


def _dimer_energy_fn(
    k: float = DIMER_K, r0: float = DIMER_R0
) -> Callable[[torch.Tensor], torch.Tensor]:
    """``E(pos) = ½k(r − r₀)²`` for a single-bond dimer, via the public potential."""
    bond = BondHarmonic(
        k=torch.tensor([k], dtype=torch.float64),
        r0=torch.tensor([r0], dtype=torch.float64),
    )
    bond_index = torch.tensor([[0], [1]], dtype=torch.long)  # COO [2, num_bonds]
    bond_types = torch.zeros(1, dtype=torch.long)

    def energy_fn(pos: torch.Tensor) -> torch.Tensor:
        return bond(pos=pos, bond_index=bond_index, bond_types=bond_types)

    return energy_fn


def _dimer_pos(separation: float) -> torch.Tensor:
    """Two atoms on the x axis at ``(0,0,0)`` and ``(separation,0,0)`` Å."""
    return torch.tensor([[0.0, 0.0, 0.0], [separation, 0.0, 0.0]], dtype=torch.float64)


def _centroid_energy(pos: torch.Tensor) -> torch.Tensor:
    """Translation-invariant energy ``Σ‖r − r̄‖²`` — identically 0 for one atom."""
    return (pos - pos.mean(dim=0, keepdim=True)).pow(2).sum()


class TestHessianDerivation:
    """``HessianDerivation`` — dense ``forward`` and per-graph ``graph_blocks``."""

    def test_forward_quadratic_toy(self) -> None:
        """``E = 1.0 · Σ‖r‖²`` ⇒ ``H = 2·I₃ₙ`` (hard-coded analytic golden)."""
        toy = KernelToyPotential(scale=1.0)
        pos = kernel_toy_batch()["atoms", "pos"].detach()

        hessian = HessianDerivation().forward(toy.total_energy, pos)

        assert hessian.shape == (12, 12)
        torch.testing.assert_close(
            hessian,
            2.0 * torch.eye(12, dtype=torch.float64),
            atol=EXACT_ATOL,
            rtol=0.0,
        )

    def test_forward_equilibrium_dimer(self) -> None:
        """``r = r₀`` dimer ⇒ rank-1 ``H = k BᵀB``; transverse term vanishes."""
        hessian = HessianDerivation().forward(_dimer_energy_fn(), _dimer_pos(DIMER_R0))

        assert hessian.shape == (6, 6)
        torch.testing.assert_close(
            hessian,
            torch.tensor(EQUILIBRIUM_HESSIAN, dtype=torch.float64),
            atol=DIMER_ATOL,
            rtol=0.0,
        )
        wilson_b = torch.tensor(WILSON_B, dtype=torch.float64)
        torch.testing.assert_close(
            hessian,
            DIMER_K * wilson_b.T @ wilson_b,
            atol=DIMER_ATOL,
            rtol=0.0,
        )

    def test_forward_stretched_dimer(self) -> None:
        """``r = 2.0 > r₀`` dimer ⇒ longitudinal 2.0 plus transverse ±0.5."""
        hessian = HessianDerivation().forward(_dimer_energy_fn(), _dimer_pos(2.0))

        assert hessian.shape == (6, 6)
        torch.testing.assert_close(
            hessian,
            torch.tensor(STRETCHED_HESSIAN, dtype=torch.float64),
            atol=DIMER_ATOL,
            rtol=0.0,
        )

    def test_forward_symmetry_and_translation(self) -> None:
        """Schwarz symmetry ``H = Hᵀ`` and zero column sums (translation modes)."""
        hessian = HessianDerivation().forward(_dimer_energy_fn(), _dimer_pos(2.0))

        assert (hessian - hessian.T).abs().max().item() <= DIMER_ATOL
        assert hessian.sum(dim=0).abs().max().item() <= DIMER_ATOL

    def test_forward_single_atom_and_bad_shape(self) -> None:
        """One atom + translation-invariant energy ⇒ ``(3,3)`` zeros; bad shape ⇒ ``ValueError``."""
        deriv = HessianDerivation()

        hessian = deriv.forward(_centroid_energy, torch.zeros(1, 3, dtype=torch.float64))

        assert hessian.shape == (3, 3)
        torch.testing.assert_close(
            hessian,
            torch.zeros(3, 3, dtype=torch.float64),
            atol=EXACT_ATOL,
            rtol=0.0,
        )

        with pytest.raises(ValueError):
            deriv.forward(_centroid_energy, torch.zeros(6, dtype=torch.float64))
        with pytest.raises(ValueError):
            deriv.forward(_centroid_energy, torch.zeros(3, 2, dtype=torch.float64))

    def test_graph_blocks_toy_batch(self) -> None:
        """Two graphs ⇒ two ``(6, 6)`` blocks of ``2s·I₆ = 3·I₆``, one energy forward."""
        toy = KernelToyPotential(scale=1.5)
        batch = kernel_toy_batch(n_atoms=4, n_graphs=2)

        blocks = HessianDerivation().graph_blocks(toy, batch)

        assert len(blocks) == 2
        expected = 3.0 * torch.eye(6, dtype=torch.float64)
        for block in blocks:
            assert block.shape == (6, 6)
            torch.testing.assert_close(block, expected, atol=EXACT_ATOL, rtol=0.0)
        assert toy.n_forward == 1

    def test_graph_blocks_block_diagonality(self) -> None:
        """Dense Hessian of the same batch: cross-graph blocks exactly 0, diagonal blocks match."""
        toy = KernelToyPotential(scale=1.5)
        batch = kernel_toy_batch(n_atoms=4, n_graphs=2)
        pos = batch["atoms", "pos"].detach().clone()

        blocks = HessianDerivation().graph_blocks(toy, batch)
        dense = HessianDerivation().forward(toy.total_energy, pos)

        assert dense.shape == (12, 12)
        assert (dense[:6, 6:] == 0.0).all()
        assert (dense[6:, :6] == 0.0).all()
        torch.testing.assert_close(dense[:6, :6], blocks[0], atol=EXACT_ATOL, rtol=0.0)
        torch.testing.assert_close(dense[6:, 6:], blocks[1], atol=EXACT_ATOL, rtol=0.0)

    def test_graph_blocks_contiguity_guard(self) -> None:
        """Non-contiguous ``atoms.batch`` raises; the happy path writes no hessian key."""
        toy = KernelToyPotential()
        shuffled = kernel_toy_batch(n_atoms=4, n_graphs=2)
        shuffled["atoms", "batch"] = torch.tensor([0, 1, 0, 1], dtype=torch.long)

        with pytest.raises(ValueError, match=r"atoms\.batch"):
            HessianDerivation().graph_blocks(toy, shuffled)

        clean = kernel_toy_batch(n_atoms=4, n_graphs=2)
        HessianDerivation().graph_blocks(toy, clean)
        assert "hessian" not in clean["graphs"].keys()
        assert "hessian" not in clean["atoms"].keys()
        assert "hessian" not in clean.keys()

    def test_graph_blocks_grad_policy(self) -> None:
        """Blocks reach parameters under grad and are detached under ``no_grad``."""
        toy = KernelToyPotential(scale=1.5)
        blocks = HessianDerivation().graph_blocks(toy, kernel_toy_batch(n_atoms=4, n_graphs=2))
        blocks[0].sum().backward()

        assert toy.scale.grad is not None
        assert toy.scale.grad.abs().item() > 0.0

        detached_toy = KernelToyPotential(scale=1.5)
        with torch.no_grad():
            detached = HessianDerivation().graph_blocks(
                detached_toy, kernel_toy_batch(n_atoms=4, n_graphs=2)
            )
        assert detached[0].requires_grad is False
