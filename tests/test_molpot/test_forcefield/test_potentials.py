"""Tests for Potentials force methods (canonical autograd delegation).

Mirrors ``src/molpot/forcefield/potentials.py``.

``Potentials.calc_forces`` / ``calc_energy_forces`` must route through the one
canonical force path, :func:`molpot.derivation.autograd_forces_from_energy`,
instead of hand-rolling ``torch.autograd.grad`` (spec
``universal-class-i-08-adapter`` §"Iron law — debt found in the touched
surface"). Numerics are pinned here against the analytic harmonic bond so the
generalization stays behaviour-preserving.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from molpot.forcefield import ForceField, Potentials

DTYPE = torch.float64

# Analytic two-atom harmonic bond: E = ½k(r−r0)², k = 100 kcal mol⁻¹ Å⁻²,
# r0 = 1.0 Å, r = 1.5 Å along x̂ → E = 12.5 kcal/mol, |F| = k(r−r0) = 50.
BOND_K = 100.0
BOND_R0 = 1.0
BOND_R = 1.5
GOLDEN_ENERGY = 12.5
GOLDEN_FORCE_X = 50.0

SOURCE = Path(__file__).resolve().parents[3] / "src/molpot/forcefield/potentials.py"


def _two_atom_bond_batch(*, r: float = BOND_R) -> TensorDict:
    pos = torch.tensor([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=DTYPE)
    return TensorDict(
        {
            "atoms": TensorDict(
                {
                    "pos": pos,
                    "Z": torch.tensor([1, 1], dtype=torch.long),
                    "batch": torch.zeros(2, dtype=torch.long),
                },
                batch_size=[2],
            ),
            "bonds": TensorDict(
                {
                    "atomi": torch.tensor([0], dtype=torch.long),
                    "atomj": torch.tensor([1], dtype=torch.long),
                },
                batch_size=[1],
            ),
        },
        batch_size=[],
    )


def _two_molecule_bond_batch() -> TensorDict:
    """Two copies of the two-atom molecule packed into one batch (B = 2)."""
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [BOND_R, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [BOND_R, 0.0, 0.0],
        ],
        dtype=DTYPE,
    )
    return TensorDict(
        {
            "atoms": TensorDict(
                {
                    "pos": pos,
                    "Z": torch.tensor([1, 1, 1, 1], dtype=torch.long),
                    "batch": torch.tensor([0, 0, 1, 1], dtype=torch.long),
                },
                batch_size=[4],
            ),
            "bonds": TensorDict(
                {
                    "atomi": torch.tensor([0, 2], dtype=torch.long),
                    "atomj": torch.tensor([1, 3], dtype=torch.long),
                },
                batch_size=[2],
            ),
        },
        batch_size=[],
    )


def _bond_potentials(k: torch.Tensor | None = None, *, n_bonds: int = 1) -> Potentials:
    """Harmonic-bond-only :class:`Potentials` with optional differentiable ``k``."""
    force_constant = torch.full((n_bonds,), BOND_K, dtype=DTYPE) if k is None else k
    return (
        ForceField()
        .def_bondstyle("harmonic")
        .with_params(
            bonds={
                "k": force_constant,
                "r0": torch.full((n_bonds,), BOND_R0, dtype=DTYPE),
            }
        )
        .to_potentials()
    )


def _method_source(class_name: str, method_name: str) -> str:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == method_name:
                return ast.unparse(item)
    raise AssertionError(f"{class_name}.{method_name} not found in {SOURCE}")


class TestPotentials:
    def test_force_methods_delegate(self) -> None:
        """No third force path: both methods call the canonical autograd helper."""
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        imported = {
            f"{node.module}.{alias.name}"
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
            for alias in node.names
        }
        assert any(
            name.startswith("molpot.derivation") and name.endswith(".autograd_forces_from_energy")
            for name in imported
        ), "potentials.py must import autograd_forces_from_energy from molpot.derivation"

        for method in ("calc_forces", "calc_energy_forces"):
            source = _method_source("Potentials", method)
            assert "autograd_forces_from_energy" in source, (
                f"Potentials.{method} must delegate to autograd_forces_from_energy"
            )
            assert "torch.autograd.grad" not in source, (
                f"Potentials.{method} must not hand-roll torch.autograd.grad"
            )

    def test_calc_forces_analytic_golden(self) -> None:
        """F = −∂E/∂x = ∓k(r−r0) x̂ for the stretched two-atom harmonic bond."""
        forces = _bond_potentials().calc_forces(_two_atom_bond_batch())
        expected = torch.tensor(
            [[GOLDEN_FORCE_X, 0.0, 0.0], [-GOLDEN_FORCE_X, 0.0, 0.0]],
            dtype=DTYPE,
        )
        assert forces.shape == (2, 3)
        assert torch.allclose(forces.detach(), expected, atol=1e-10, rtol=0.0)

    def test_calc_energy_forces_matches_separate_calls(self) -> None:
        pots = _bond_potentials()
        batch = _two_atom_bond_batch()
        energy, forces = pots.calc_energy_forces(batch)
        assert float(energy.detach()) == pytest.approx(GOLDEN_ENERGY, abs=1e-10)
        assert torch.allclose(forces.detach(), pots.calc_forces(batch).detach(), atol=1e-10)

    def test_calc_forces_double_backward_reaches_parameters(self) -> None:
        """``create_graph`` keeps ∂²E/∂x∂θ alive so a force loss reaches θ."""
        parameter = nn.Parameter(torch.tensor([BOND_K], dtype=DTYPE))
        pots = _bond_potentials(parameter * 1.0)
        forces = pots.calc_forces(_two_atom_bond_batch())
        torch.mean(forces**2).backward()
        assert parameter.grad is not None
        # d/dk mean(F²) with F = ±k(r−r0): mean over 6 components = k²·0.5²·2/6
        assert float(parameter.grad) == pytest.approx(2.0 * BOND_K * 0.25 * 2.0 / 6.0, abs=1e-10)

    def test_calc_forces_on_multi_graph_batch(self) -> None:
        """Per-graph energies ``(B,)`` reduce to one scalar before differentiating."""
        forces = _bond_potentials(n_bonds=2).calc_forces(_two_molecule_bond_batch())
        expected = torch.tensor(
            [
                [GOLDEN_FORCE_X, 0.0, 0.0],
                [-GOLDEN_FORCE_X, 0.0, 0.0],
                [GOLDEN_FORCE_X, 0.0, 0.0],
                [-GOLDEN_FORCE_X, 0.0, 0.0],
            ],
            dtype=DTYPE,
        )
        assert forces.shape == (4, 3)
        assert torch.allclose(forces.detach(), expected, atol=1e-10, rtol=0.0)
