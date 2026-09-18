"""Rondo's energy/force wrapper: the host-side bind and the traced derivative."""

from __future__ import annotations

import torch

from molix.schema import ENERGY_KEY, FORCES_KEY
from molix.units import units
from molzoo.rondo import Rondo, RondoPotential, RondoSpec


def _model(**over):
    spec = RondoSpec(hidden=8, n_blocks=1, n_torsion_terms=2, eem_max_atoms=32, **over)
    return RondoPotential(Rondo(spec))


class TestRondoPotential:
    def test_predicts_an_energy_per_graph_and_a_force_per_atom(self, butane_batch):
        out = _model()(butane_batch)
        n_atoms = butane_batch["atoms", "Z"].shape[0]
        n_graphs = butane_batch["graphs", "num_atoms"].shape[0]
        assert out["energy"].shape == (n_graphs,)
        assert out["forces"].shape == (n_atoms, 3)
        assert torch.isfinite(out["energy"]).all()
        assert torch.isfinite(out["forces"]).all()

    def test_restores_the_callers_label_columns(self, butane_batch):
        # The derivative op writes in place onto ``graphs.energy`` and
        # ``atoms.forces`` — which is also where a labelled batch carries its
        # DFT targets. If they are not put back, a loss reading the target
        # after the forward compares the prediction with itself: the force
        # term is identically zero, the energy gradient is identically zero,
        # and there is no training signal at all. Silently. (Adaptation A6.)
        butane_batch[ENERGY_KEY] = torch.full_like(butane_batch[ENERGY_KEY], -42.0)
        butane_batch[FORCES_KEY] = torch.full_like(butane_batch[FORCES_KEY], 3.0)
        out = _model()(butane_batch)
        assert torch.equal(
            butane_batch[ENERGY_KEY], torch.full_like(butane_batch[ENERGY_KEY], -42.0)
        )
        assert torch.equal(butane_batch[FORCES_KEY], torch.full_like(butane_batch[FORCES_KEY], 3.0))
        assert not torch.allclose(out["forces"], butane_batch[FORCES_KEY])

    def test_assembles_the_container_once_and_rebinds_per_batch(self, butane_batch):
        # Assembling the container is untraceable molpy work worth 20 graph
        # breaks per batch; the topology bind is per batch by nature.
        model = _model()
        model(butane_batch)
        first = model.physics(butane_batch)
        model(butane_batch.clone())
        assert model.physics(butane_batch) is first

    def test_energy_unit_rescales_the_reported_energy(self, butane_batch):
        # One parameterizer, two wrappers: the container is always kcal/mol,
        # and the unit is a pure float on the way out — not a buffer, and not
        # a second energy path.
        param = Rondo(RondoSpec(hidden=8, n_blocks=1, n_torsion_terms=2, eem_max_atoms=32))
        kcal = RondoPotential(param)(butane_batch.clone())["energy"]
        ev = RondoPotential(param, energy_unit=units.eV)(butane_batch.clone())["energy"]
        scale = float((1.0 * units.kilocalorie_per_mole).to(units.eV).magnitude)
        torch.testing.assert_close(ev, kcal * scale)


class TestRondoPotentialTrainable:
    """Which parameter tables an objective is allowed to move."""

    def _grads(self, butane_batch, **kw):
        param = Rondo(RondoSpec(hidden=8, n_blocks=1, n_torsion_terms=2, eem_max_atoms=32))
        model = RondoPotential(param, **kw)
        model(butane_batch.clone())["energy"].sum().backward()
        return {
            name: any(
                p.grad is not None and bool(p.grad.abs().sum() > 0) for p in child.parameters()
            )
            for name, child in param.named_children()
        }

    def test_everything_trains_by_default(self, butane_batch):
        got = self._grads(butane_batch)
        assert got["proper_head"] and got["bond_head"] and got["encoder"]

    def test_a_named_subset_is_the_only_head_that_moves(self, butane_batch):
        # A torsion profile constrains torsion barriers. Letting it also move
        # Lennard-Jones sigma lets the two compensate — the classical
        # molecular-mechanics degeneracy (spec section 6.1, R2).
        got = self._grads(butane_batch, trainable=("propers",))
        assert got["proper_head"], "the named head must still train"
        assert not got["bond_head"], "an unnamed head must not"
        assert not got["atom_head"], "nor the Lennard-Jones head"

    def test_the_shared_encoder_still_trains_through_the_kept_path(self, butane_batch):
        # Detaching a table is not the same as dropping those parameters from
        # the optimizer: the encoder feeds every head, and it must keep the
        # gradient that arrives through the head that was kept.
        assert self._grads(butane_batch, trainable=("propers",))["encoder"]

    def test_detached_tables_still_contribute_their_value(self, butane_batch):
        # Only the gradient path is cut — the energy is unchanged, which is
        # what makes this a statement about the objective and not about the
        # model.
        param = Rondo(RondoSpec(hidden=8, n_blocks=1, n_torsion_terms=2, eem_max_atoms=32))
        full = RondoPotential(param)(butane_batch.clone())["energy"]
        masked = RondoPotential(param, trainable=("propers",))(butane_batch.clone())["energy"]
        torch.testing.assert_close(full, masked)
