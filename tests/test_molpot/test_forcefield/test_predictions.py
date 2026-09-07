"""Tests for ForceFieldPredictions (Parameterizer → per-task prediction dict).

Mirrors ``src/molpot/forcefield/predictions.py``.

Fixtures follow ``test_parameterizer.py``: a deterministic stub encoder plus
constant-output heads in fp64, so every expected value below is analytic
(spec ``universal-class-i-08-adapter``, §Domain basis).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from molpot import HessianDerivation
from molpot.charge import (
    AtomicChargeModel,
    DiagonalHardnessKernel,
    ElectronegativityEqualization,
)
from molpot.composition.heads import LJParameterHead
from molpot.derivation import ForceDerivation
from molpot.forcefield import ForceField, ForceFieldPredictions, Parameterizer, Potentials

DTYPE = torch.float64

# Analytic two-atom harmonic bond: E = ½k(r−r0)², k = 100 kcal mol⁻¹ Å⁻²,
# r0 = 1.0 Å, r = 1.5 Å along x̂ (spec §Domain basis).
BOND_K = 100.0
BOND_R0 = 1.0
BOND_R = 1.5
GOLDEN_ENERGY = 12.5  # kcal/mol
GOLDEN_FORCE_X = 50.0  # kcal mol⁻¹ Å⁻¹ on atom 0 (−50 on atom 1)
GOLDEN_H_DIAG = 100.0  # k û_x û_x           kcal mol⁻¹ Å⁻²
GOLDEN_H_OFFDIAG = -100.0  # −k û_x û_x
GOLDEN_H_TRANSVERSE = 33.333333333333336  # k(r−r0)/r = 100·0.5/1.5

# EEM/QEq KKT golden: χ = (0, 1), η = (2, 2), Q = 0 → λ = −0.5, q = (+¼, −¼) e.
GOLDEN_CHI = (0.0, 1.0)
GOLDEN_ETA = (2.0, 2.0)
GOLDEN_Q = (0.25, -0.25)


def _two_atom_bond_batch(*, r: float = BOND_R) -> TensorDict:
    """One molecule: two bonded atoms separated by ``r`` Å along x̂."""
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


class _ConstantBondHead(nn.Module):
    """Learnable-but-constant ``k`` / ``r0`` (so force → parameter grads exist)."""

    def __init__(self, k: float = BOND_K, r0: float = BOND_R0) -> None:
        super().__init__()
        self.k = nn.Parameter(torch.tensor(float(k), dtype=DTYPE))
        self.r0 = nn.Parameter(torch.tensor(float(r0), dtype=DTYPE))

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        n = int(features.shape[0])
        return {"k": self.k.expand(n), "r0": self.r0.expand(n)}


class _ConstantChiEtaHead(nn.Module):
    """Constant χ / η with a call counter (charge-solve laziness probe)."""

    def __init__(
        self,
        chi: Sequence[float] = GOLDEN_CHI,
        eta: Sequence[float] = GOLDEN_ETA,
    ) -> None:
        super().__init__()
        self.register_buffer("chi", torch.tensor(list(chi), dtype=DTYPE))
        self.register_buffer("eta", torch.tensor(list(eta), dtype=DTYPE))
        self.calls = 0

    def forward(self, atom_features: torch.Tensor) -> dict[str, torch.Tensor]:
        del atom_features
        self.calls += 1
        return {"chi": self.chi, "eta": self.eta}


class _FakeEncoder(nn.Module):
    """Deterministic per-namespace features, independent of geometry."""

    def __init__(self, features: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self._features = features

    def forward(self, td: TensorDict) -> TensorDict:
        return td

    def embeddings(self, td: TensorDict) -> dict[str, torch.Tensor]:
        del td
        return dict(self._features)


class _CountingHessianDerivation(nn.Module):
    """:class:`HessianDerivation` wrapper counting adapter invocations."""

    def __init__(self) -> None:
        super().__init__()
        self.inner = HessianDerivation()
        self.calls = 0

    def forward(
        self,
        energy_fn: Callable[[torch.Tensor], torch.Tensor],
        pos: torch.Tensor,
    ) -> torch.Tensor:
        self.calls += 1
        return self.inner(energy_fn, pos)


class _NoQChargeModel(nn.Module):
    """``charge_model`` that writes no ``q`` — the silent-zero trap."""

    def write_forcefield(
        self,
        forcefield: ForceField,
        atom_features: torch.Tensor,
        batch: TensorDict,
    ) -> ForceField:
        del atom_features, batch
        return forcefield

    def forward(self, *_: object, **__: object) -> torch.Tensor:
        raise AssertionError("Parameterizer must use write_forcefield, not forward")


def _features(n_atoms: int = 2, n_bonds: int = 1) -> dict[str, torch.Tensor]:
    return {
        "bonds": torch.zeros(n_bonds, 2, dtype=DTYPE),
        "atoms": torch.zeros(n_atoms, 2, dtype=DTYPE),
    }


def _bond_model(
    *,
    n_atoms: int = 2,
    n_bonds: int = 1,
    bond_head: nn.Module | None = None,
    atom_head: nn.Module | None = None,
    charge_model: nn.Module | None = None,
) -> Parameterizer:
    """Bonded-only Class-I model over the analytic two-atom topology."""
    return Parameterizer(
        _FakeEncoder(_features(n_atoms, n_bonds)),
        template=ForceField().def_bondstyle("harmonic"),
        bond_head=_ConstantBondHead() if bond_head is None else bond_head,
        atom_head=atom_head,
        charge_model=charge_model,
    )


def _charge_model(head: nn.Module | None = None) -> AtomicChargeModel:
    """EEM with a diagonal hardness kernel and no jitter (exact goldens)."""
    return AtomicChargeModel(
        _ConstantChiEtaHead() if head is None else head,
        ElectronegativityEqualization(DiagonalHardnessKernel(), jitter=0.0),
    )


class TestForceFieldPredictions:
    def test_supported_keys_are_the_four_task_predictions(self) -> None:
        assert ForceFieldPredictions.SUPPORTED_KEYS == ("energy", "forces", "hessian", "q")

    def test_forcefield_predictions_importable_at_package_roots(self) -> None:
        from molpot import ForceFieldPredictions as TopLevel
        from molpot.forcefield import ForceFieldPredictions as PkgLevel

        assert TopLevel is ForceFieldPredictions
        assert PkgLevel is ForceFieldPredictions

    @pytest.mark.parametrize(
        "keys",
        [("energy",), ("energy", "forces"), ("energy", "hessian"), ("q",)],
    )
    def test_call_returns_exactly_declared_keys(self, keys: tuple[str, ...]) -> None:
        model = _bond_model(charge_model=_charge_model())
        adapter = ForceFieldPredictions(
            model,
            keys=keys,
            hessian_derivation=HessianDerivation(),
        )
        out = adapter(_two_atom_bond_batch())
        assert set(out) == set(keys)

    def test_hessian_derivation_uncalled_unless_declared(self) -> None:
        model = _bond_model()
        derivation = _CountingHessianDerivation()
        batch = _two_atom_bond_batch()

        ForceFieldPredictions(model, keys=("energy", "forces"), hessian_derivation=derivation)(
            batch
        )
        assert derivation.calls == 0

        ForceFieldPredictions(model, keys=("energy", "hessian"), hessian_derivation=derivation)(
            batch
        )
        assert derivation.calls == 1

    def test_q_only_skips_potentials_assembly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``keys=("q",)`` reads topology charges — no energy path is built."""
        assembled: list[str] = []
        original = ForceField.to_potentials

        def counting(self: ForceField, frame: object = None, **kwargs: object) -> Potentials:
            assembled.append("to_potentials")
            return original(self, frame, **kwargs)

        monkeypatch.setattr(ForceField, "to_potentials", counting)
        model = _bond_model(charge_model=_charge_model())
        out = ForceFieldPredictions(model, keys=("q",))(_two_atom_bond_batch())
        assert set(out) == {"q"}
        assert assembled == []

    def test_charge_solve_runs_once_per_call(self) -> None:
        """One model forward per call: ``q`` is read, never re-solved."""
        head = _ConstantChiEtaHead()
        model = _bond_model(charge_model=_charge_model(head))
        ForceFieldPredictions(model, keys=("energy", "q"))(_two_atom_bond_batch())
        assert head.calls == 1

    def test_energy_parity_with_direct_potentials_path(self) -> None:
        model = _bond_model()
        batch = _two_atom_bond_batch()
        expected = model(batch).to_potentials().calc_energy(batch)
        energy = ForceFieldPredictions(model, keys=("energy",)).energy(batch)
        assert torch.allclose(energy, expected, atol=1e-10, rtol=0.0)

    def test_energy_shape_single_and_two_molecule_batch(self) -> None:
        single = ForceFieldPredictions(_bond_model(), keys=("energy",))
        assert single.energy(_two_atom_bond_batch()).shape == ()

        batched = ForceFieldPredictions(_bond_model(n_atoms=4, n_bonds=2), keys=("energy",))
        assert batched.energy(_two_molecule_bond_batch()).shape == (2,)

    def test_force_parity_with_force_derivation(self) -> None:
        model = _bond_model()
        batch = _two_atom_bond_batch()
        pots = model(batch).to_potentials()
        expected = ForceDerivation(method="autograd")(
            lambda p: pots.calc_energy(batch, pos=p),
            batch["atoms", "pos"],
        )
        forces = ForceFieldPredictions(model, keys=("forces",)).forces(batch)
        assert forces.shape == (2, 3)
        assert torch.allclose(forces, expected, atol=1e-10, rtol=0.0)

    def test_energy_forces_matches_separate_calls(self) -> None:
        adapter = ForceFieldPredictions(_bond_model(), keys=("energy", "forces"))
        batch = _two_atom_bond_batch()
        energy, forces = adapter.energy_forces(batch)
        assert torch.allclose(energy, adapter.energy(batch), atol=1e-10, rtol=0.0)
        assert torch.allclose(forces, adapter.forces(batch), atol=1e-10, rtol=0.0)

    def test_force_loss_backward_reaches_bond_head_parameters(self) -> None:
        head = _ConstantBondHead()
        adapter = ForceFieldPredictions(_bond_model(bond_head=head), keys=("forces",))
        forces = adapter.forces(_two_atom_bond_batch())
        torch.mean(forces**2).backward()
        assert head.k.grad is not None
        assert head.r0.grad is not None

    def test_analytic_harmonic_bond_energy_and_force_goldens(self) -> None:
        """E = ½k(r−r0)², F = −∂E/∂x for k=100, r0=1.0 Å, r=1.5 Å along x̂.

        E = ½·100·0.5² = 12.5 kcal/mol; the bond is stretched, so atom 0 is
        pulled toward +x̂ and atom 1 toward −x̂ with |F| = k(r−r0) = 50
        kcal mol⁻¹ Å⁻¹ (spec ``universal-class-i-08-adapter`` §Domain basis).
        """
        adapter = ForceFieldPredictions(_bond_model(), keys=("energy", "forces"))
        energy, forces = adapter.energy_forces(_two_atom_bond_batch())
        expected_forces = torch.tensor(
            [[GOLDEN_FORCE_X, 0.0, 0.0], [-GOLDEN_FORCE_X, 0.0, 0.0]],
            dtype=DTYPE,
        )
        assert float(energy.detach()) == pytest.approx(GOLDEN_ENERGY, abs=1e-10)
        assert torch.allclose(forces.detach(), expected_forces, atol=1e-10, rtol=0.0)

    def test_analytic_harmonic_bond_hessian_goldens(self) -> None:
        """H = k ûûᵀ ⊕ (k(r−r0)/r)(I − ûûᵀ) for the stretched two-atom bond.

        Longitudinal H[0,0] = k = 100, cross term H[0,3] = −k = −100, and the
        transverse curvature H[1,1] = k(r−r0)/r = 100·0.5/1.5 kcal mol⁻¹ Å⁻².
        The Hessian is a symmetric second derivative (Wilson, E. B.
        *J. Chem. Phys.* **9**, 76 (1941), https://doi.org/10.1063/1.1750829),
        checked here rather than symmetrized away.
        """
        adapter = ForceFieldPredictions(
            _bond_model(),
            keys=("hessian",),
            hessian_derivation=HessianDerivation(),
        )
        hessian = adapter.hessian(_two_atom_bond_batch()).detach()
        assert hessian.shape == (6, 6)
        assert float(hessian[0, 0]) == pytest.approx(GOLDEN_H_DIAG, abs=1e-10)
        assert float(hessian[0, 3]) == pytest.approx(GOLDEN_H_OFFDIAG, abs=1e-10)
        assert float(hessian[1, 1]) == pytest.approx(GOLDEN_H_TRANSVERSE, abs=1e-10)
        assert torch.allclose(hessian, hessian.T, atol=1e-10, rtol=0.0)

    def test_eem_charges_analytic_golden(self) -> None:
        """χ = (0, 1), η = (2, 2), Q = 0 → λ = −0.5, q = (+0.25, −0.25) e.

        The KKT system ``[diag(η) 1; 1ᵀ 0][q; λ] = [−χ; Q]`` gives
        ``q_i = −(χ_i + λ)/η_i`` with λ fixed by ``Σq = Q``; the more
        electronegative atom carries the negative charge.
        """
        model = _bond_model(charge_model=_charge_model())
        charges = ForceFieldPredictions(model, keys=("q",)).charges(_two_atom_bond_batch())
        expected = torch.tensor(list(GOLDEN_Q), dtype=DTYPE)
        assert charges.shape == (2,)
        assert torch.allclose(charges.detach(), expected, atol=1e-10, rtol=0.0)

    def test_charges_match_atomic_charge_model(self) -> None:
        charge_model = _charge_model()
        model = _bond_model(charge_model=charge_model)
        batch = _two_atom_bond_batch()
        expected = charge_model.charges(_features()["atoms"], batch)
        charges = ForceFieldPredictions(model, keys=("q",)).charges(batch)
        assert torch.allclose(charges, expected, atol=1e-10, rtol=0.0)

    def test_charges_are_geometry_independent(self) -> None:
        """``dq/dx = 0`` for the diagonal hardness kernel — pins force consistency."""
        adapter = ForceFieldPredictions(_bond_model(charge_model=_charge_model()), keys=("q",))
        first = adapter.charges(_two_atom_bond_batch(r=1.5))
        second = adapter.charges(_two_atom_bond_batch(r=2.7))
        assert torch.equal(first, second)

    def test_bonded_only_composition_evaluates_no_lj(self) -> None:
        """An ``LJParameterHead`` writing ε/σ adds no term to a bonded-only form."""
        model = _bond_model(atom_head=LJParameterHead(feature_dim=2, hidden_dim=4).double())
        batch = _two_atom_bond_batch()
        energy = ForceFieldPredictions(model, keys=("energy",)).energy(batch)
        assert float(energy.detach()) == pytest.approx(GOLDEN_ENERGY, abs=1e-10)

        filled = model(batch)
        atoms = filled.params("atoms")
        assert atoms is not None
        assert {"epsilon", "sigma"} <= set(atoms)
        terms = filled.to_potentials().term_energies(batch)
        assert set(terms) <= {"bonds", "angles", "propers", "impropers"}

    def test_empty_keys_raise(self) -> None:
        with pytest.raises(ValueError, match="keys"):
            ForceFieldPredictions(_bond_model(), keys=())

    def test_duplicate_keys_raise(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            ForceFieldPredictions(_bond_model(), keys=("energy", "energy"))

    def test_unsupported_key_message_lists_supported_keys(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            ForceFieldPredictions(_bond_model(), keys=("nrg",))
        message = str(excinfo.value)
        assert "nrg" in message
        for key in ForceFieldPredictions.SUPPORTED_KEYS:
            assert key in message

    def test_q_without_charge_model_raises(self) -> None:
        with pytest.raises(ValueError, match="charge_model"):
            ForceFieldPredictions(_bond_model(), keys=("q",))

    def test_hessian_without_derivation_raises(self) -> None:
        with pytest.raises(ValueError, match="derivation"):
            ForceFieldPredictions(_bond_model(), keys=("energy", "hessian"))

    def test_missing_q_on_filled_forcefield_raises(self) -> None:
        """No silent zero vector when the charge model wrote nothing."""
        model = _bond_model(
            atom_head=LJParameterHead(feature_dim=2, hidden_dim=4).double(),
            charge_model=_NoQChargeModel(),
        )
        adapter = ForceFieldPredictions(model, keys=("q",))
        with pytest.raises(RuntimeError, match="q"):
            adapter.charges(_two_atom_bond_batch())
