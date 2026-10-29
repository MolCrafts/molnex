"""Rondo: parameters from the covalent graph, energy through the force field's potentials."""

from __future__ import annotations

import molpy as mp
import pytest
import torch
import torch.nn as nn

from molrep.encoders.covalent_graph import CovalentGraphEncoder, CovalentGraphEncoderSpec
from molzoo.rondo import Rondo
from tests.test_molzoo.test_rondo.conftest import (
    DTYPE,
    TINY,
    class_i_forcefield,
    excite,
    with_hessian_pairs,
)


def _model(forcefield, **kwargs) -> Rondo:
    torch.manual_seed(0)
    return Rondo(forcefield, dtype=DTYPE, **{**TINY, **kwargs})


def _flat(module: nn.Module) -> torch.Tensor:
    return torch.cat([parameter.detach().reshape(-1) for parameter in module.parameters()])


def _gap(before: torch.Tensor, after: torch.Tensor) -> tuple[float, float]:
    delta = (after - before).abs()
    scale = before.square().mean().sqrt().clamp_min(1e-12)
    return float(delta.max()), float(delta.square().mean().sqrt() / scale)


class TestRondo:
    def test_swapping_the_encoder_does_not_change_the_head(self, forcefield):
        """Loading another encoder leaves every head tensor where it was."""
        host = _model(forcefield)
        donor = _model(forcefield)
        with torch.no_grad():
            for parameter in donor.graph.parameters():
                parameter.add_(1.0)
            for parameter in donor.heads.parameters():
                parameter.add_(1.0)
        head_before = _flat(host.heads)
        graph_before = _flat(host.graph)
        head_ptrs = [parameter.data_ptr() for parameter in host.heads.parameters()]
        host.load_state_dict(donor.state_dict(), prefixes=("graph.",))
        head_abs, head_rel = _gap(head_before, _flat(host.heads))
        graph_abs, _graph_rel = _gap(graph_before, _flat(host.graph))
        assert head_abs == 0.0
        assert head_rel == 0.0
        assert graph_abs > 0.0
        assert [parameter.data_ptr() for parameter in host.heads.parameters()] == head_ptrs

        graph_after = _flat(host.graph)
        host.load_state_dict(donor.state_dict(), prefixes=("heads.",))
        graph_abs, graph_rel = _gap(graph_after, _flat(host.graph))
        head_abs, _head_rel = _gap(head_before, _flat(host.heads))
        assert graph_abs == 0.0
        assert graph_rel == 0.0
        assert head_abs > 0.0

    def test_a_new_term_trains_only_its_own_parameters(self, butane_batch):
        """The encoder and the old head stay put when a new term is added.

        The new head is the only module trained. Its columns are the classical
        parameter table.
        """
        bond_field = class_i_forcefield(styles=("bond:harmonic",))
        both_field = class_i_forcefield(styles=("bond:harmonic", "angle:harmonic"))
        torch.manual_seed(0)
        trained = Rondo(bond_field, dtype=DTYPE, **TINY)
        optimizer = torch.optim.Adam(
            list(trained.graph.parameters()) + list(trained.heads.parameters()), lr=1e-2
        )
        for _ in range(20):
            optimizer.zero_grad(set_to_none=True)
            trained.energy(butane_batch.clone())["graphs", "energy"].sum().backward()
            optimizer.step()
        added = Rondo(both_field, dtype=DTYPE, **TINY)
        added.load_state_dict(trained.state_dict(), prefixes=("graph.", "heads.bonds."))
        encoder = [parameter.detach().clone() for parameter in added.graph.parameters()]
        bonds = [parameter.detach().clone() for parameter in added.heads["bonds"].parameters()]
        angles = [parameter.detach().clone() for parameter in added.heads["angles"].parameters()]
        optimizer = torch.optim.Adam(added.heads["angles"].parameters(), lr=1e-2)
        for _ in range(20):
            optimizer.zero_grad(set_to_none=True)
            added.energy(butane_batch.clone())["graphs", "energy"].sum().backward()
            optimizer.step()

        def max_abs(module: nn.Module, saved: list[torch.Tensor]) -> float:
            return max(
                float((parameter.detach() - previous).abs().max())
                for parameter, previous in zip(module.parameters(), saved, strict=True)
            )

        encoder_delta = max_abs(added.graph, encoder)
        bond_delta = max_abs(added.heads["bonds"], bonds)
        angle_delta = max_abs(added.heads["angles"], angles)
        assert encoder_delta == 0.0
        assert bond_delta == 0.0
        assert angle_delta > 0.0
        bond_energy = trained.energy(butane_batch.clone())["graphs", "energy"]
        kept = added.energy(butane_batch.clone(), styles=("bond:harmonic",))["graphs", "energy"]
        torch.testing.assert_close(kept, bond_energy)
        table = added.energy(butane_batch.clone())
        features = added.features(butane_batch)
        assert features.atom.shape[-1] == TINY["width"]
        assert features.bond.shape[-1] == TINY["edge_mlp"][1]
        print(
            {
                "fingerprint": tuple(features.atom.shape),
                "encoder_delta": encoder_delta,
                "bond_head_delta": bond_delta,
                "angle_head_delta": angle_delta,
                "bonds.k": table["bonds", "k"].detach().tolist(),
                "bonds.r0": table["bonds", "r0"].detach().tolist(),
                "angles.k": table["angles", "k"].detach().tolist(),
                "angles.theta0": table["angles", "theta0"].detach().tolist(),
            }
        )

    def test_forces_are_the_negative_gradient_of_the_energy(self, forcefield, butane_batch):
        """The force lands beside the label (``atoms.forces_pred``); the energy is the op's."""
        model = _model(forcefield)
        excite(model)
        batch = butane_batch
        out = model(batch.clone())

        pos = batch["atoms", "pos"].clone().requires_grad_(True)
        probe = batch.clone()
        probe["atoms", "pos"] = pos
        energy = model.energy(probe)["graphs", "energy"]
        (grad,) = torch.autograd.grad(energy.sum(), pos)
        torch.testing.assert_close(out["atoms", "forces_pred"], -grad)
        torch.testing.assert_close(out["graphs", "energy"], energy.detach())

    def test_forward_writes_the_tables_as_columns(self, forcefield, butane_batch):
        out = _model(forcefield)(butane_batch.clone())
        assert out["bonds", "k"].shape == (butane_batch["bonds", "bond_index"].shape[1],)
        assert out["angles", "theta0"].shape == (butane_batch["angles"].batch_size[0],)
        assert out["propers", "k"].shape == (butane_batch["propers"].batch_size[0], 4)
        assert out["atoms", "charge"].shape == (14,)

    def test_tables_writes_the_columns_energy_writes_and_no_energy(
        self, forcefield, butane_batch
    ):
        model = _model(forcefield)
        excite(model)
        priced = model.energy(butane_batch.clone())
        before = butane_batch.clone()
        tabled = model.tables(butane_batch.clone())
        # The label, if the batch has one, is left exactly as it was.
        assert sorted(tabled["graphs"].keys()) == sorted(before["graphs"].keys())
        for key in before["graphs"].keys():
            torch.testing.assert_close(tabled["graphs", key], before["graphs", key])
        for key in (("bonds", "k"), ("angles", "theta0"), ("propers", "k"), ("atoms", "charge")):
            torch.testing.assert_close(tabled[key], priced[key])

    def test_charges_meet_the_declared_total_charge(self, forcefield, butane_batch):
        charge = _model(forcefield)(butane_batch.clone())["atoms", "charge"]
        torch.testing.assert_close(
            charge.sum(), torch.zeros((), dtype=charge.dtype), atol=1e-5, rtol=0.0
        )

    def test_energy_of_named_styles_sums_only_those(self, forcefield, butane_batch):
        """The bonded strain alone, the pair force alone, and their sum."""
        model = _model(forcefield)
        excite(model)
        batch = butane_batch
        bonded = ("bond:harmonic", "angle:harmonic")
        rest = tuple(key for key in model.potentials.keys() if key not in bonded)
        full = model.energy(batch.clone())["graphs", "energy"]
        part = model.energy(batch.clone(), styles=bonded)["graphs", "energy"]
        other = model.energy(batch.clone(), styles=rest)["graphs", "energy"]
        torch.testing.assert_close(part + other, full)
        assert not torch.allclose(part, full)

    def test_a_style_selection_outside_the_field_is_refused_by_name(self, forcefield, butane_batch):
        with pytest.raises(ValueError, match="bond:morse"):
            _model(forcefield).energy(butane_batch.clone(), styles=("bond:morse",))

    def test_a_bond_loss_trains_the_bond_head_and_the_encoder(self, forcefield, butane_batch):
        """A style's energy trains that head and the encoder, not the other heads."""
        model = _model(forcefield)
        excite(model)
        batch = butane_batch
        model.energy(batch, styles=("bond:harmonic",))["graphs", "energy"].sum().backward()
        assert all(p.grad is None for p in model.heads["angles"].parameters())
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in model.heads["bonds"].parameters()
        )
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.graph.parameters())

    def test_partial_hessian_is_the_full_hessian_on_the_bonded_pairs(
        self, forcefield, butane_batch
    ):
        model = _model(forcefield)
        excite(model)
        batch = butane_batch
        layout = with_hessian_pairs(batch)
        batch["atoms", "pos"].requires_grad_(True)
        blocks = model.partial_hessian(batch)
        bonded = tuple(key for key in model.potentials if not key.startswith("pair:"))

        def energy(pos: torch.Tensor) -> torch.Tensor:
            probe = batch.clone()
            probe["atoms", "pos"] = pos
            return model.energy(probe, styles=bonded)["graphs", "energy"].sum()

        pos = batch["atoms", "pos"].detach()
        full = torch.autograd.functional.hessian(energy, pos).reshape(pos.numel(), pos.numel())
        torch.testing.assert_close(blocks.detach(), layout.gather(full))

    def test_a_mirror_image_has_the_same_energy(self, forcefield, butane_batch):
        model = _model(forcefield)
        excite(model)
        batch = butane_batch
        flip = torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)
        mirrored = batch.clone()
        mirrored["atoms", "pos"] = batch["atoms", "pos"] * flip
        if "pair_diff" in mirrored["pairs"].keys():
            mirrored["pairs", "pair_diff"] = batch["pairs", "pair_diff"] * flip
        out = model(batch.clone())
        image = model(mirrored)
        torch.testing.assert_close(image["graphs", "energy"], out["graphs", "energy"])
        torch.testing.assert_close(
            image["atoms", "forces_pred"], out["atoms", "forces_pred"] * flip
        )

    def test_energy_is_invariant_under_a_rigid_motion(self, forcefield, butane_batch):
        model = _model(forcefield)
        excite(model)
        batch = butane_batch
        rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
        if torch.linalg.det(rotation) < 0:
            rotation[:, 0] = -rotation[:, 0]
        moved = batch.clone()
        moved["atoms", "pos"] = batch["atoms", "pos"] @ rotation.T + torch.tensor(
            [1.5, -2.0, 0.7], dtype=torch.float64
        )
        if "pair_diff" in moved["pairs"].keys():
            moved["pairs", "pair_diff"] = batch["pairs", "pair_diff"] @ rotation.T
        out = model(batch.clone())
        image = model(moved)
        torch.testing.assert_close(image["graphs", "energy"], out["graphs", "energy"])
        torch.testing.assert_close(
            image["atoms", "forces_pred"], out["atoms", "forces_pred"] @ rotation.T
        )

    def test_propers_through_a_linear_centre_are_silent(self, forcefield, butyne_batch):
        model = _model(forcefield)
        excite(model)
        batch = butyne_batch
        k = model.energy(batch.clone())["propers", "k"]
        assert k[0].abs().sum() > 0  # 5-0-1-2: both central atoms sp3
        torch.testing.assert_close(k[1:], torch.zeros_like(k[1:]))  # through C≡C

    def test_lennard_jones_starts_on_gaff2(self, forcefield, butane_batch):
        atoms = _model(forcefield).energy(butane_batch.clone())["atoms"]
        torch.testing.assert_close(atoms["sigma"], butane_batch["atoms", "gaff2_sigma"])
        torch.testing.assert_close(atoms["epsilon"], butane_batch["atoms", "gaff2_epsilon"])

    def test_a_missing_gaff2_column_is_refused_by_name(self, forcefield, butane_batch):
        batch = butane_batch.clone()
        del batch["atoms", "gaff2_sigma"]
        with pytest.raises(KeyError, match="gaff2_sigma"):
            _model(forcefield).energy(batch)

    def test_an_unsupported_style_is_refused_by_name(self):
        ff = class_i_forcefield()
        ff.def_style("bond", "morse")
        with pytest.raises(ValueError, match="bond:morse"):
            _model(ff)

    def test_untyped_opls_requires_ir_conversion(self):
        # No coefficients exist for the IR to convert; do not guess a series.
        with pytest.raises(ValueError, match="dihedral:opls"):
            _model(class_i_forcefield(dihedral="opls"))

    def test_typed_opls_is_canonicalized_by_molpy(self):
        ff = class_i_forcefield(dihedral="opls")
        atom = ff.def_style("atom", "full").def_type("C", mass=12.0)
        ff.get_style("dihedral", "opls").def_type(
            "torsion", atom, atom, atom, atom, k1=1.2, k2=0.3, k3=-0.2, k4=0.0
        )
        model = _model(ff)
        assert "dihedral:opls" not in model.potentials
        assert "dihedral:periodic" in model.potentials
        canonical = ff.canonical().get_types("dihedral")[0]
        for name, value in canonical.params.items():
            assert model.forcefield.get_types("dihedral")[0].params[name] == value
        assert ff.get_style("dihedral", "opls") is not None

    def test_forward_differentiates_positions(self, forcefield):
        model = _model(forcefield)
        assert model.derivative.wrt == (("atoms", "pos"),)
        assert model.derivative.outputs == (("atoms", "forces_pred"),)

    def test_a_field_with_no_priced_style_is_refused(self):
        ff = mp.ff.forcefield.ForceField(name="atoms-only", units="real")
        ff.def_style("atom", "full")
        with pytest.raises(ValueError, match="no style that prices an energy"):
            _model(ff)


#: Lennard-Jones plus the four valence styles. Coulomb is a later force field.
VALENCE_AND_LJ = (
    "bond:harmonic",
    "angle:harmonic",
    "dihedral:periodic",
    "improper:cvff",
    "pair:lj/cut",
)


class _WithCoulomb(Rondo):
    """The same body on a force field that also declares ``pair coul/cut``."""

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return super().load_state_dict(state_dict, strict=False, assign=assign)


class TestComposition:
    def test_a_field_without_coulomb_prices_only_its_styles(self, forcefield, butane_batch):
        """Styles the field does not declare have no head and are not priced."""
        full = _model(forcefield)
        excite(full)
        part = _model(class_i_forcefield(styles=VALENCE_AND_LJ))
        loaded = part.load_state_dict(full.state_dict(), strict=False)
        assert loaded.missing_keys == []
        assert loaded.unexpected_keys
        assert all(
            key.startswith("heads.charge") or "coul" in key for key in loaded.unexpected_keys
        )
        assert "charge" not in part.heads
        assert "pair:coul/cut" not in part.potentials
        batch = butane_batch
        torch.testing.assert_close(
            part.energy(batch.clone())["graphs", "energy"],
            full.energy(batch.clone(), styles=VALENCE_AND_LJ)["graphs", "energy"],
        )
        assert "charge" not in part.energy(batch.clone())["atoms"].keys()

    def test_a_subclass_loads_the_parent_checkpoint_and_keeps_the_new_head(
        self, forcefield, butane_batch
    ):
        """The new head stays at its initial value; training it leaves the old bond weights."""
        parent = _model(class_i_forcefield(styles=VALENCE_AND_LJ))
        excite(parent)
        batch = butane_batch
        energy_before = parent.energy(batch.clone())["graphs", "energy"].detach()

        extended = _WithCoulomb(forcefield, dtype=DTYPE, **TINY)
        extended.load_state_dict(parent.state_dict())

        for name, value in parent.state_dict().items():
            torch.testing.assert_close(extended.state_dict()[name], value)
        torch.testing.assert_close(
            extended.energy(batch.clone())["graphs", "energy"], energy_before
        )

        with torch.no_grad():
            extended.heads["charge"].param_head.mlp[-1].weight.normal_(0, 0.3)
        coulomb = extended.energy(batch.clone(), styles=("pair:coul/cut",))["graphs", "energy"]
        assert coulomb.abs().sum() > 0
        bond_weight = extended.heads["bonds"].out["k"][1].weight.detach().clone()
        charge_weight = extended.heads["charge"].param_head.mlp[-1].weight.detach().clone()
        optimizer = torch.optim.Adam(extended.parameters(), lr=1e-2)
        extended.energy(batch.clone(), styles=("pair:coul/cut",))[
            "graphs", "energy"
        ].sum().backward()
        optimizer.step()
        torch.testing.assert_close(extended.heads["bonds"].out["k"][1].weight, bond_weight)
        assert not torch.equal(extended.heads["charge"].param_head.mlp[-1].weight, charge_weight)

    def test_the_encoder_predicts_without_a_force_field(self, butane_batch):
        """The graph features are a module of their own; a head on them is not a force term."""
        fields = {
            key: value
            for key, value in TINY.items()
            if key in CovalentGraphEncoderSpec.model_fields
        }
        encoder = CovalentGraphEncoder(CovalentGraphEncoderSpec(**fields), dtype=DTYPE)
        features = encoder(butane_batch)
        assert features.atom.shape == (butane_batch["atoms", "Z"].shape[0], TINY["width"])
        head = nn.Linear(features.atom.shape[-1], 1, dtype=DTYPE)
        head(features.atom).sum().backward()
        assert head.weight.grad is not None
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in encoder.parameters()
        )

    def test_features_train_the_encoder_and_not_the_force_heads(self, butane_batch):
        model = _model(class_i_forcefield(styles=("bond:harmonic",)))
        features = model.features(butane_batch.clone())
        head = nn.Linear(features.atom.shape[-1], 1, dtype=DTYPE)
        head(features.atom).sum().backward()
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in model.graph.parameters()
        )
        assert all(parameter.grad is None for parameter in model.heads["bonds"].parameters())
