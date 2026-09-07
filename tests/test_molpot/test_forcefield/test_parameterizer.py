"""Tests for ForceField + Parameterizer (schema-gated MM heads)."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from molpot.charge import (
    AtomicChargeModel,
    DiagonalHardnessKernel,
    ElectronegativityEqualization,
)
from molpot.composition.heads import ChargeHead, LJParameterHead
from molpot.composition.mm_heads import (
    AngleParamHead,
    BondParamHead,
    ImproperParamHead,
    ProperTorsionParamHead,
)
from molpot.composition.multihead import MultiHead
from molpot.forcefield import ForceField, OPLSAAForceField, Parameterizer


def _two_atom_bond_batch(
    *,
    r: float = 1.5,
    dtype: torch.dtype = torch.float64,
) -> TensorDict:
    pos = torch.tensor([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=dtype)
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


def _ion_batch(*, total_charge: float | None = None) -> TensorDict:
    """Two atoms in one graph; ``graphs.total_charge`` present only when asked."""
    graphs: dict[str, torch.Tensor] = {"num_atoms": torch.tensor([2], dtype=torch.long)}
    if total_charge is not None:
        graphs["total_charge"] = torch.tensor([total_charge])
    return TensorDict(
        {
            "atoms": TensorDict(
                {
                    "pos": torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]),
                    "Z": torch.tensor([8, 1], dtype=torch.long),
                    "batch": torch.zeros(2, dtype=torch.long),
                },
                batch_size=[2],
            ),
            "graphs": TensorDict(graphs, batch_size=[1]),
        },
        batch_size=[],
    )


class _ConstantBondHead(nn.Module):
    def __init__(self, k: float, r0: float):
        super().__init__()
        self.k = k
        self.r0 = r0

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        n = features.shape[0]
        return {
            "k": torch.full((n,), self.k, dtype=features.dtype, device=features.device),
            "r0": torch.full((n,), self.r0, dtype=features.dtype, device=features.device),
        }


class _MissingK1ProperHead(nn.Module):
    form = "opls"

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        n = features.shape[0]
        z = torch.zeros(n, dtype=features.dtype, device=features.device)
        return {"k2": z, "k3": z, "k4": z}


class _FakeEncoder(nn.Module):
    def __init__(self, features: dict[str, torch.Tensor]):
        super().__init__()
        self._features = features
        self._p = nn.Parameter(torch.zeros(1))

    def forward(self, td: TensorDict) -> TensorDict:
        return td

    def embeddings(self, td: TensorDict):
        class _E:
            def __init__(self, f):
                self._f = f

            def interaction_dict(self):
                return dict(self._f)

        return _E(self._features)


def _charge_model(feature_dim: int = 4) -> AtomicChargeModel:
    from molpot.charge import ChiEtaHead

    return AtomicChargeModel(
        ChiEtaHead(feature_dim=feature_dim, hidden_dim=8),
        ElectronegativityEqualization(DiagonalHardnessKernel()),
    )


def _oplsaa_heads(feature_dim: int = 4) -> dict[str, nn.Module]:
    return {
        "bond_head": BondParamHead(feature_dim=feature_dim, hidden_dim=8),
        "angle_head": AngleParamHead(feature_dim=feature_dim, hidden_dim=8),
        "proper_head": ProperTorsionParamHead(feature_dim=feature_dim, hidden_dim=8, form="opls"),
        "improper_head": ImproperParamHead(feature_dim=feature_dim, hidden_dim=8),
        "atom_head": LJParameterHead(feature_dim=feature_dim, hidden_dim=8),
        "charge_model": _charge_model(feature_dim),
    }


class TestParameterizer:
    def test_forward_writes_continuous_params(self):
        bond_head = BondParamHead(feature_dim=4, hidden_dim=8)
        angle_head = AngleParamHead(feature_dim=4, hidden_dim=8)
        atom_head = MultiHead(
            {
                "lj": LJParameterHead(feature_dim=4, hidden_dim=8),
                "q": ChargeHead(feature_dim=4, hidden_dim=8),
            }
        )
        ff = (
            ForceField(name="abs", units="real")
            .def_bondstyle("harmonic")
            .def_anglestyle("harmonic")
            .def_pairstyle("lj/cut")
        )
        features = {
            "bonds": torch.randn(2, 4),
            "angles": torch.randn(1, 4),
            "atoms": torch.randn(3, 4),
        }
        model = Parameterizer(
            _FakeEncoder(features),
            ff,
            bond_head=bond_head,
            angle_head=angle_head,
            atom_head=atom_head,
        )
        batch = TensorDict(
            {
                "atoms": TensorDict(
                    {
                        "pos": torch.randn(3, 3),
                        "Z": torch.tensor([6, 1, 1]),
                        "batch": torch.zeros(3, dtype=torch.long),
                    },
                    batch_size=[3],
                ),
                "bonds": TensorDict(
                    {
                        "atomi": torch.tensor([0, 0]),
                        "atomj": torch.tensor([1, 2]),
                    },
                    batch_size=[2],
                ),
                "angles": TensorDict(
                    {
                        "atomi": torch.tensor([1]),
                        "atomj": torch.tensor([0]),
                        "atomk": torch.tensor([2]),
                    },
                    batch_size=[1],
                ),
            },
            batch_size=[],
        )
        filled = model(batch)
        assert isinstance(filled, ForceField)
        assert filled.params("bonds") is not None
        assert filled.params("bonds")["k"].shape == (2,)
        assert filled.params("angles") is not None
        assert filled.params("atoms") is not None
        schema = filled.to_potentials().get_schema()
        assert "bond:harmonic" in schema
        assert "angle:harmonic" in schema

    def test_oplsaa_forward_fills_all_namespaces_with_opls_k(self):
        d = 4
        features = {
            "bonds": torch.randn(2, d),
            "angles": torch.randn(1, d),
            "propers": torch.randn(1, d),
            "impropers": torch.randn(1, d),
            "atoms": torch.randn(4, d),
        }
        template = OPLSAAForceField()
        model = Parameterizer(_FakeEncoder(features), template, **_oplsaa_heads(d))
        batch = TensorDict(
            {
                "atoms": TensorDict(
                    {
                        "pos": torch.randn(4, 3),
                        "Z": torch.tensor([6, 6, 6, 6]),
                        "batch": torch.zeros(4, dtype=torch.long),
                    },
                    batch_size=[4],
                ),
                "bonds": TensorDict(
                    {
                        "atomi": torch.tensor([0, 1], dtype=torch.long),
                        "atomj": torch.tensor([1, 2], dtype=torch.long),
                    },
                    batch_size=[2],
                ),
                "angles": TensorDict(
                    {
                        "atomi": torch.tensor([0], dtype=torch.long),
                        "atomj": torch.tensor([1], dtype=torch.long),
                        "atomk": torch.tensor([2], dtype=torch.long),
                    },
                    batch_size=[1],
                ),
                "propers": TensorDict(
                    {
                        "atomi": torch.tensor([0], dtype=torch.long),
                        "atomj": torch.tensor([1], dtype=torch.long),
                        "atomk": torch.tensor([2], dtype=torch.long),
                        "atoml": torch.tensor([3], dtype=torch.long),
                    },
                    batch_size=[1],
                ),
                "impropers": TensorDict(
                    {
                        "atomi": torch.tensor([0], dtype=torch.long),
                        "atomj": torch.tensor([1], dtype=torch.long),
                        "atomk": torch.tensor([2], dtype=torch.long),
                        "atoml": torch.tensor([3], dtype=torch.long),
                    },
                    batch_size=[1],
                ),
                # Neutral molecule, spelled out: the charge model in
                # ``_oplsaa_heads`` makes ``graphs.total_charge`` mandatory.
                "graphs": TensorDict(
                    {
                        "num_atoms": torch.tensor([4], dtype=torch.long),
                        "total_charge": torch.zeros(1),
                    },
                    batch_size=[1],
                ),
            },
            batch_size=[],
        )
        assert template.params("bonds") is None
        filled = model(batch)
        assert filled is not template
        assert template.params("bonds") is None
        for ns in ("bonds", "angles", "propers", "impropers", "atoms"):
            assert filled.params(ns) is not None
        pp = filled.params("propers")
        assert pp is not None
        for key in ("k1", "k2", "k3", "k4"):
            assert key in pp
            assert pp[key].shape == (1,)
        assert "q" in filled.params("atoms")

    def test_forward_does_not_mutate_template(self):
        ff = ForceField().def_bondstyle("harmonic")
        model = Parameterizer(
            _FakeEncoder({"bonds": torch.zeros(1, 1, dtype=torch.float64)}),
            ff,
            bond_head=_ConstantBondHead(k=2.0, r0=1.0),
        )
        batch = _two_atom_bond_batch()
        filled = model(batch)
        assert filled.params("bonds") is not None
        assert ff.params("bonds") is None

    def test_oplsaa_missing_proper_head_raises(self):
        heads = _oplsaa_heads()
        del heads["proper_head"]
        with pytest.raises(ValueError, match="proper_head"):
            Parameterizer(_FakeEncoder({}), OPLSAAForceField(), **heads)

    def test_oplsaa_missing_atom_head_raises(self):
        heads = _oplsaa_heads()
        del heads["atom_head"]
        with pytest.raises(ValueError, match="atom_head"):
            Parameterizer(_FakeEncoder({}), OPLSAAForceField(), **heads)

    def test_oplsaa_missing_charge_model_raises(self):
        heads = _oplsaa_heads()
        del heads["charge_model"]
        with pytest.raises(ValueError, match="charge_model"):
            Parameterizer(_FakeEncoder({}), OPLSAAForceField(), **heads)

    def test_oplsaa_cartesian_proper_form_raises(self):
        heads = _oplsaa_heads()
        heads["proper_head"] = ProperTorsionParamHead(feature_dim=4, hidden_dim=8)
        with pytest.raises(ValueError, match="form"):
            Parameterizer(_FakeEncoder({}), OPLSAAForceField(), **heads)

    def test_oplsaa_proper_missing_k1_raises_on_forward(self):
        heads = _oplsaa_heads()
        heads["proper_head"] = _MissingK1ProperHead()
        features = {
            "bonds": torch.zeros(1, 4),
            "angles": torch.zeros(1, 4),
            "propers": torch.zeros(1, 4),
            "impropers": torch.zeros(1, 4),
            "atoms": torch.zeros(2, 4),
        }
        model = Parameterizer(_FakeEncoder(features), OPLSAAForceField(), **heads)
        batch = TensorDict(
            {
                "atoms": TensorDict(
                    {
                        "pos": torch.randn(2, 3),
                        "Z": torch.tensor([1, 1]),
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
                "angles": TensorDict(
                    {
                        "atomi": torch.tensor([0], dtype=torch.long),
                        "atomj": torch.tensor([0], dtype=torch.long),
                        "atomk": torch.tensor([1], dtype=torch.long),
                    },
                    batch_size=[1],
                ),
                "propers": TensorDict(
                    {
                        "atomi": torch.tensor([0], dtype=torch.long),
                        "atomj": torch.tensor([0], dtype=torch.long),
                        "atomk": torch.tensor([1], dtype=torch.long),
                        "atoml": torch.tensor([1], dtype=torch.long),
                    },
                    batch_size=[1],
                ),
                "impropers": TensorDict(
                    {
                        "atomi": torch.tensor([0], dtype=torch.long),
                        "atomj": torch.tensor([0], dtype=torch.long),
                        "atomk": torch.tensor([1], dtype=torch.long),
                        "atoml": torch.tensor([1], dtype=torch.long),
                    },
                    batch_size=[1],
                ),
            },
            batch_size=[],
        )
        with pytest.raises(ValueError, match="k1"):
            model(batch)

    def test_charge_model_without_atom_features_raises(self):
        charge = _charge_model()
        model = Parameterizer(
            _FakeEncoder({}),
            ForceField().def_pairstyle("coulomb"),
            charge_model=charge,
        )
        batch = TensorDict(
            {
                "atoms": TensorDict(
                    {
                        "pos": torch.randn(2, 3),
                        "Z": torch.tensor([1, 1]),
                        "batch": torch.zeros(2, dtype=torch.long),
                    },
                    batch_size=[2],
                )
            },
            batch_size=[],
        )
        with pytest.raises(ValueError, match="atom features"):
            model(batch)

    def test_charge_model_without_graphs_total_charge_raises(self):
        """Fail-closed (S5): the silent neutral default is unreachable from here.

        ``AtomicChargeModel.total_charge`` defaults to ``0.0``. Reaching that
        default from a Parameterizer means an anion is equilibrated as a
        neutral molecule and every downstream charge is wrong by a constant —
        with nothing in the run record to say so.
        """
        model = Parameterizer(
            _FakeEncoder({"atoms": torch.zeros(2, 4)}),
            ForceField().def_pairstyle("coulomb"),
            charge_model=_charge_model(),
        )
        with pytest.raises(ValueError, match="total_charge"):
            model(_ion_batch())

    def test_graphs_total_charge_reaches_the_charge_model(self):
        """``Q = −1 e`` is forwarded, so the equilibrated charges sum to it."""
        model = Parameterizer(
            _FakeEncoder({"atoms": torch.zeros(2, 4)}),
            ForceField().def_pairstyle("coulomb"),
            charge_model=_charge_model(),
        )
        filled = model(_ion_batch(total_charge=-1.0))
        charges = filled.params("atoms")["q"]
        assert float(charges.sum()) == pytest.approx(-1.0, abs=1e-6)

    def test_polarizability_head_is_refused_until_composable_a1(self):
        """No induced-response style exists yet, so the head has nothing to fill."""
        with pytest.raises(ValueError, match="composable-a1"):
            Parameterizer(
                _FakeEncoder({}),
                ForceField().def_bondstyle("harmonic"),
                bond_head=_ConstantBondHead(k=2.0, r0=1.0),
                polarizability_head=nn.Linear(4, 1),
            )


class TestEnergy:
    def test_bond_harmonic_via_to_potentials(self):
        k, r0, r = 2.0, 1.0, 1.5
        expected = 0.5 * k * (r - r0) ** 2
        ff = ForceField().def_bondstyle("harmonic")
        model = Parameterizer(
            _FakeEncoder({"bonds": torch.zeros(1, 1, dtype=torch.float64)}),
            ff,
            bond_head=_ConstantBondHead(k=k, r0=r0),
        )
        batch = _two_atom_bond_batch(r=r)
        energy = model(batch).to_potentials().calc_energy(batch)
        assert energy.shape == ()
        assert math.isclose(float(energy), expected, rel_tol=1e-5, abs_tol=1e-5)

    def test_potentials_is_embeddable_module(self):
        from molpot.derivation.protocol import ENERGY_KEY, FORCES_KEY
        from molpot.potentials.base import BasePotential

        ff = (
            ForceField()
            .def_bondstyle("harmonic")
            .with_params(bonds={"k": torch.tensor([2.0]), "r0": torch.tensor([1.0])})
        )
        batch = _two_atom_bond_batch(r=1.5)
        pots = ff.to_potentials()
        assert isinstance(pots, BasePotential)
        assert isinstance(pots, nn.Module)
        out = pots(batch)
        assert out is batch
        assert ENERGY_KEY in out.keys(include_nested=True)
        assert FORCES_KEY in out.keys(include_nested=True)
        assert out[FORCES_KEY].shape == (2, 3)

    def test_forward_returns_only_forcefield(self):
        ff = ForceField().def_bondstyle("harmonic")
        model = Parameterizer(
            _FakeEncoder({"bonds": torch.zeros(1, 1, dtype=torch.float64)}),
            ff,
            bond_head=_ConstantBondHead(k=4.0, r0=1.2),
        )
        batch = _two_atom_bond_batch(r=1.4)
        out = model(batch)
        assert isinstance(out, ForceField)
        assert not isinstance(out, dict)


class TestForces:
    def test_calc_forces_on_potentials(self):
        ff = ForceField().def_bondstyle("harmonic")
        model = Parameterizer(
            _FakeEncoder({"bonds": torch.zeros(1, 1, dtype=torch.float64)}),
            ff,
            bond_head=_ConstantBondHead(k=2.0, r0=1.0),
        )
        batch = _two_atom_bond_batch(r=1.5)
        pots = model(batch).to_potentials()
        forces = pots.calc_forces(batch)
        assert forces.shape == (2, 3)
        assert math.isclose(float(forces[0, 0].detach()), 2.0 * 0.5, abs_tol=1e-5)
        assert math.isclose(float(forces[1, 0].detach()), -2.0 * 0.5, abs_tol=1e-5)
