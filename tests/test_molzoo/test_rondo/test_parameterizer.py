"""The Rondo recipe: which heads are mounted, and whose field they fill."""

from __future__ import annotations

import pytest
import torch

from molpot.forcefield import ForceField
from molzoo.rondo import Rondo, RondoForceField, RondoSpec


class TestRondo:
    def test_mounts_a_head_for_every_style_the_template_declares(self):
        # `Parameterizer` raises when a registered style has no head, so a
        # bare construction is the assertion: the recipe is complete.
        model = Rondo(RondoSpec(hidden=8, n_blocks=1, n_torsion_terms=2))
        assert isinstance(model.template, RondoForceField)

    def test_the_field_it_parameterizes_is_the_field_that_prices_the_pairs(self):
        # One object owns both the parameter tables and the nonbonded radius,
        # so a pair list cannot be built at a radius the model does not charge
        # for (ledger finding F21).
        model = Rondo(RondoSpec(hidden=8, n_blocks=1, n_torsion_terms=2, cutoff=9.0))
        assert model.template.neighbor_list().cutoff == pytest.approx(9.0)

    def test_forward_returns_a_filled_field_and_tables_carry_the_charges(self, butane_batch):
        # `forward` is the export path — a molpy ForceField to hand the LAMMPS
        # / OpenMM writers. `tables` is the hot path and is where the energy
        # reads its parameters from; the solved charges live there, as a
        # column of the `atoms` table, not as a batch column.
        model = Rondo(RondoSpec(hidden=8, n_blocks=1, n_torsion_terms=2, eem_max_atoms=32))
        assert isinstance(model(butane_batch), ForceField)
        charge = model.tables(butane_batch)["atoms"]["charge"]
        assert charge.shape == butane_batch["atoms", "Z"].shape
        assert torch.isfinite(charge).all()

    def test_reads_no_coordinate(self, butane_batch):
        # Predicted parameters must be a function of topology alone, or the
        # deployed force is not -dE/dpos but a partial derivative that drops
        # the parameter-response term.
        model = Rondo(RondoSpec(hidden=8, n_blocks=1, n_torsion_terms=2, eem_max_atoms=32))
        moved = butane_batch.clone()
        moved["atoms", "pos"] = moved["atoms", "pos"] + 1.7
        before = model.tables(butane_batch.clone())
        after = model.tables(moved)
        assert set(before) == set(after)
        for namespace, params in before.items():
            for name, tensor in params.items():
                torch.testing.assert_close(tensor, after[namespace][name])
