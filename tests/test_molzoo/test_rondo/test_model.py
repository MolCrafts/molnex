"""Rondo: parameters from the graph, energy through the force field's potentials, derivatives."""

from __future__ import annotations

import pytest
import torch

from molzoo.rondo import Rondo
from tests.test_molzoo.test_rondo.conftest import (
    DTYPE,
    SPEC,
    class_i_forcefield,
    excite,
    with_hessian_pairs,
)


def _model(forcefield, spec=SPEC, **kwargs) -> Rondo:
    torch.manual_seed(0)
    return Rondo(forcefield, spec, dtype=DTYPE, **kwargs)


class TestRondo:
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
        assert out["angles", "theta0_ref"].shape == (butane_batch["angles"].batch_size[0],)
        assert out["propers", "k"].shape == (butane_batch["propers"].batch_size[0], 2)
        assert out["atoms", "charge"].shape == (14,)

    def test_charges_meet_the_declared_total_charge(self, forcefield, butane_batch):
        charge = _model(forcefield)(butane_batch.clone())["atoms", "charge"]
        torch.testing.assert_close(
            charge.sum(), torch.zeros((), dtype=charge.dtype), atol=1e-5, rtol=0.0
        )

    def test_energy_with_its_own_tables_equals_energy_without(self, forcefield, butane_batch):
        model = _model(forcefield)
        excite(model)
        batch = butane_batch
        tables = model.tables(batch)
        with_tables = model.energy(batch.clone(), tables)["graphs", "energy"]
        alone = model.energy(batch.clone())["graphs", "energy"]
        torch.testing.assert_close(with_tables, alone)

    def test_energy_of_named_styles_sums_only_those(self, forcefield, butane_batch):
        """ByteFF's ``calc_terms``: the bonded strain alone, the pair force alone."""
        model = _model(forcefield)
        excite(model)
        batch = butane_batch
        bonded = ("bond:harmonic", "angle:harmonic")
        rest = tuple(key for key in model.potentials.keys() if key not in bonded)
        tables = model.tables(batch)
        full = model.energy(batch.clone(), tables)["graphs", "energy"]
        part = model.energy(batch.clone(), tables, styles=bonded)["graphs", "energy"]
        other = model.energy(batch.clone(), tables, styles=rest)["graphs", "energy"]
        torch.testing.assert_close(part + other, full)
        assert not torch.allclose(part, full)

    def test_a_style_selection_outside_the_field_is_refused_by_name(self, forcefield, butane_batch):
        with pytest.raises(ValueError, match="bond:morse"):
            _model(forcefield).energy(butane_batch.clone(), styles=("bond:morse",))

    def test_a_detached_table_cuts_its_heads_gradient_and_not_the_encoders(
        self, forcefield, butane_batch
    ):
        """The seam a training step uses: tables in, the ones its loss does not
        name detached. The angle head then gets no gradient while the encoder,
        shared by every table, still does."""
        model = _model(forcefield)
        excite(model)
        batch = butane_batch
        tables = model.tables(batch)
        tables["angles"] = {name: value.detach() for name, value in tables["angles"].items()}
        model.energy(batch, tables)["graphs", "energy"].sum().backward()
        assert all(p.grad is None for p in model.heads["angles"].parameters())
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in model.heads["bonds"].parameters()
        )
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters()
        )

    def test_partial_hessian_is_the_full_hessian_on_the_bonded_pairs(
        self, forcefield, butane_batch
    ):
        model = _model(forcefield)
        excite(model)
        batch = butane_batch
        layout = with_hessian_pairs(batch)
        batch["atoms", "pos"].requires_grad_(True)
        tables = model.tables(batch)
        blocks = model.partial_hessian(batch, tables)

        frozen = {ns: {k: v.detach() for k, v in t.items()} for ns, t in tables.items()}
        bonded = {k: p for k, p in model.potentials.items() if not k.startswith("pair:")}

        def energy(pos: torch.Tensor) -> torch.Tensor:
            probe = batch.clone()
            probe["atoms", "pos"] = pos
            total = pos.new_zeros(())
            for key, potential in bonded.items():
                params = model._style_params(key, frozen)
                total = total + potential(probe, **params).sum()
            return total

        pos = batch["atoms", "pos"].detach()
        full = torch.autograd.functional.hessian(energy, pos).reshape(pos.numel(), pos.numel())
        torch.testing.assert_close(blocks.detach(), layout.gather(full))

    def test_improper_energy_is_symmetric_in_the_outer_legs(self, forcefield, butane_batch):
        """Three cyclic rows per centre: swapping two legs permutes the rows, and
        an even series makes each dihedral the negative of another's."""
        model = _model(forcefield)
        excite(model)
        batch = butane_batch
        swapped = batch.clone()
        # Rows of the first centre: (c,i,j,k), (c,j,k,i), (c,k,i,j) → legs j↔k.
        rows = slice(0, 3)
        j = batch["impropers", "atomk"][rows].clone()
        k = batch["impropers", "atoml"][rows].clone()
        swapped["impropers", "atomk"][rows] = k
        swapped["impropers", "atoml"][rows] = j
        torch.testing.assert_close(
            model.energy(swapped)["graphs", "energy"], model.energy(batch)["graphs", "energy"]
        )

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
        params = model._style_params("dihedral:periodic", model.tables(batch))
        k = params["k"]
        assert k[0].abs().sum() > 0  # 5-0-1-2: both central atoms sp3
        torch.testing.assert_close(k[1:], torch.zeros_like(k[1:]))  # through C≡C

    def test_an_sp_centre_starts_at_a_linear_angle(self, forcefield, butyne_batch):
        theta0 = _model(forcefield).tables(butyne_batch)["angles"]["theta0"]
        # Angles 0-1-2 (centre 1, sp3), 1-2-3 (centre 2, sp), 2-3-4 (centre 3, sp).
        assert theta0[0].item() == pytest.approx(109.47, abs=0.01)
        assert theta0[1].item() == pytest.approx(180.0, abs=0.01)
        assert theta0[2].item() == pytest.approx(180.0, abs=0.01)

    def test_lennard_jones_starts_on_gaff2(self, forcefield, butane_batch):
        tables = _model(forcefield).tables(butane_batch)["atoms"]
        torch.testing.assert_close(tables["sigma"], butane_batch["atoms", "gaff2_sigma"])
        torch.testing.assert_close(tables["epsilon"], butane_batch["atoms", "gaff2_epsilon"])

    def test_a_missing_gaff2_column_is_refused_by_name(self, forcefield, butane_batch):
        batch = butane_batch.clone()
        del batch["atoms", "gaff2_sigma"]
        with pytest.raises(KeyError, match="gaff2_sigma"):
            _model(forcefield).tables(batch)

    def test_an_unsupported_style_is_refused_by_name(self):
        ff = class_i_forcefield()
        ff.def_style("bond", "morse")
        with pytest.raises(ValueError, match="bond:morse"):
            _model(ff)

    def test_opls_needs_the_four_term_series(self):
        ff = class_i_forcefield(dihedral="opls")
        with pytest.raises(ValueError, match="1, 2, 3, 4"):
            _model(ff)
        spec = SPEC.model_copy(
            update={"heads": SPEC.heads.model_copy(update={"proper_periodicity": (1, 2, 3, 4)})}
        )
        assert "dihedral:opls" in _model(ff, spec).potentials

    def test_opls_prices_the_same_series_as_periodic(self, butane_batch):
        """``K_n = 2k_n`` with phases 0/180/0/180 is exactly the OPLS form."""
        spec = SPEC.model_copy(
            update={"heads": SPEC.heads.model_copy(update={"proper_periodicity": (1, 2, 3, 4)})}
        )
        periodic = _model(class_i_forcefield(), spec)
        excite(periodic)
        opls = _model(class_i_forcefield(dihedral="opls"), spec)
        opls.load_state_dict(periodic.state_dict())
        batch = butane_batch
        torch.testing.assert_close(
            opls.energy(batch.clone())["graphs", "energy"],
            periodic.energy(batch.clone())["graphs", "energy"],
        )

    def test_one_state_dict_serves_every_derivative_choice(self, forcefield):
        energy_only = _model(forcefield, wrt=())
        energy_only.load_state_dict(_model(forcefield).state_dict())
        assert energy_only.derivative.wrt == ()
