"""AtomicChargeModel — mirrors ``src/molpot/charge/assignment.py``.

The charge model is the domain object that turns atom features into partial
charges: a ``χ``/``η`` head plus an equilibration solver. It owns two things
nothing else can state — the charges themselves, and, for the S2 manifest,
*how* they were obtained. ``ForceField.manifest`` reports only the
electrostatics the container owns (multipole order and the permanent
exclusion scaling, tested in
``tests/test_molpot/test_forcefield/test_container.py``); the constraint and
the solver come from here, and the caller composes the two records with
``dataclasses.replace``.

Reference:
    ``.claude/notes/composable-potentials.md`` §"S2 manifest" (record table);
    spec ``.claude/specs/composable-a0-04-manifest.md``.
"""

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
from molpot.forcefield import ForceField
from molpot.ir import NonbondedScaling
from molpot.manifest import ElectrostaticsRecord, SolverRecord


class _ConstChiEta(nn.Module):
    """Fixed ``χ``/``η`` for two atoms — the head is not what is under test."""

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        n = x.shape[0]
        return {
            "chi": torch.tensor([2.0, 4.0], dtype=x.dtype)[:n],
            "eta": torch.tensor([1.0, 3.0], dtype=x.dtype)[:n],
        }


def _charge_model() -> AtomicChargeModel:
    """The two-atom EEM model both tests below run."""
    return AtomicChargeModel(
        _ConstChiEta(),
        ElectronegativityEqualization(DiagonalHardnessKernel(), jitter=0.0),
    )


class TestAtomicChargeModel:
    @pytest.mark.slow
    def test_charges_trace_as_one_graph_at_a_declared_capacity(self):
        """The charge path is a hot path: ``features -> q`` must be one graph.

        With ``max_atoms`` declared on the solver and the ``(B,)`` total charge
        the Parameterizer always forwards, nothing on the way from features to
        charges reads a tensor back to the host — ``B`` is the total-charge
        length, the dense layout is the capacity, the empty-molecule pin is a
        mask select. Zero breaks, measured once here at the public entry.
        """
        from molix.compile import Compiler

        model = AtomicChargeModel(
            _ConstChiEta(),
            ElectronegativityEqualization(DiagonalHardnessKernel(), jitter=0.0, max_atoms=4),
        )
        batch = TensorDict(
            {
                "atoms": TensorDict(
                    {
                        "pos": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                        "batch": torch.zeros(2, dtype=torch.long),
                    },
                    batch_size=[2],
                )
            },
            batch_size=[],
        )
        feats = torch.zeros(2, 4)
        total = torch.zeros(1)
        model.charges(feats, batch, total_charge=total)
        explanation = Compiler.explain(model.charges, feats, batch, total_charge=total)
        assert (explanation.graph_count, explanation.graph_break_count) == (1, 0), (
            f"break sites: {explanation.break_sites}"
        )

    def test_write_forcefield_q(self):
        model = _charge_model()
        batch = TensorDict(
            {
                "atoms": TensorDict(
                    {
                        "pos": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                        "batch": torch.zeros(2, dtype=torch.long),
                    },
                    batch_size=[2],
                )
            },
            batch_size=[],
        )
        feats = torch.zeros(2, 4)
        ff = model.write_forcefield(ForceField(name="c"), feats, batch)
        q = ff.params("atoms")["charge"]
        assert math.isclose(float(q.sum()), 0.0, abs_tol=1e-10)

    def test_electrostatics_record_names_the_constraint_and_the_solve_it_ran(self) -> None:
        """S2: the charge model is the only thing that knows how ``charge`` was got.

        The constraint is the per-graph total charge imposed through the KKT
        row of the equilibration system. The solve is *direct* — one bordered
        linear system through ``torch.linalg.solve``, not an iterated fixed
        point — so it reports a single "iteration", a zero tolerance and no
        preconditioner rather than the numbers an iterative solver would
        carry. The permanent scaling is not the model's to invent: it comes
        from the force field the record will be composed with, and is echoed
        back unchanged.
        """
        scaling = NonbondedScaling()
        record = _charge_model().electrostatics_record(permanent_scaling=scaling)
        assert isinstance(record, ElectrostaticsRecord)
        assert record.charge_constraint == "total-charge-kkt"
        assert record.solver == SolverRecord(
            kind="electronegativity-equalization",
            iterations=1,
            tolerance=0.0,
            preconditioner="none",
        )
        assert record.multipole_order == 0
        assert record.polarization_scheme is None
        assert record.permanent_scaling is scaling
