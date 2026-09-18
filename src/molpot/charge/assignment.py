"""Atomic charge assignment: param head + equilibration → q.

The charge model is also the only object that knows *how* a model's charges
were obtained, so it supplies that half of the manifest's electrostatics
(:meth:`AtomicChargeModel.electrostatics_record`, requirement S2).

Reference:
    ``.claude/notes/composable-potentials.md`` §"S2 manifest" (record table).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import torch
import torch.nn as nn
from tensordict import TensorDict

from molpot.charge.problem import ChargeEquilibrationProblem
from molpot.forcefield.container import ForceField
from molpot.ir.scaling import NonbondedScaling
from molpot.manifest import ElectrostaticsRecord, SolverRecord

__all__ = ["AtomicChargeModel"]


class _Equilibration(Protocol):
    def solve(
        self,
        *,
        chi: torch.Tensor,
        eta: torch.Tensor,
        problem: ChargeEquilibrationProblem,
    ) -> torch.Tensor: ...


class AtomicChargeModel(nn.Module):
    """Domain object: features → partial charges via charge equilibration.

    Owns:
      * a param head producing ``χ``, ``η``
      * an equilibration solver (e.g. :class:`ElectronegativityEqualization`)

    Does **not** evaluate Coulomb energy — only returns ``charge`` or writes them
    onto a :class:`~molpot.forcefield.ForceField`.
    """

    def __init__(
        self,
        param_head: nn.Module,
        equilibration: _Equilibration,
        *,
        total_charge: float = 0.0,
    ) -> None:
        super().__init__()
        self.param_head = param_head
        self.equilibration = equilibration  # type: ignore[assignment]
        self.total_charge = float(total_charge)

    def charges(
        self,
        atom_features: torch.Tensor,
        batch: Mapping[str, Any] | TensorDict,
        *,
        total_charge: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Solve for partial charges ``(N,)``."""
        params = self.param_head(atom_features)
        chi, eta = params["chi"], params["eta"]
        qtot = self.total_charge if total_charge is None else total_charge
        problem = ChargeEquilibrationProblem.from_batch(batch, total_charge=qtot)
        return self.equilibration.solve(chi=chi, eta=eta, problem=problem)

    def write_forcefield(
        self,
        forcefield: ForceField,
        atom_features: torch.Tensor,
        batch: Mapping[str, Any] | TensorDict,
        *,
        total_charge: float | torch.Tensor | None = None,
    ) -> ForceField:
        """Copy-on-write ForceField with ``atoms["charge"]`` (and keep other atom params)."""
        q = self.charges(atom_features, batch, total_charge=total_charge)
        atoms = dict(forcefield.params("atoms") or {})
        atoms["charge"] = q
        # alias for heads that use "charge"
        atoms["charge"] = q
        return forcefield.with_params(atoms=atoms)

    def electrostatics_record(self, *, permanent_scaling: NonbondedScaling) -> ElectrostaticsRecord:
        """Name the charge constraint and the solve that produced ``charge`` (S2).

        :meth:`ForceField.manifest <molpot.forcefield.ForceField.manifest>`
        reports only the electrostatics a container owns — point charges and
        the permanent exclusion scaling — because a ``pair:coulomb`` style is
        not a charge model and constrains nothing. The constraint and the
        solver are this model's, and the caller composes the two records::

            import dataclasses

            manifest = ff.manifest(profile="A0")
            manifest = dataclasses.replace(
                manifest,
                electrostatics=charge_model.electrostatics_record(
                    permanent_scaling=ff.nonbonded_scaling
                ),
            )

        Args:
            permanent_scaling: The 1–2 / 1–3 / 1–4 scales of the force field
                this record will be composed with. Not the charge model's to
                invent — it is echoed back unchanged so the composed record
                stays complete.

        Returns:
            The record. The constraint is the per-graph total charge imposed
            through the KKT row of the equilibration system
            (``"total-charge-kkt"`` — the Karush-Kuhn-Tucker system that adds
            the constraint to the energy minimisation as one extra row and
            column, whose multiplier is the equalized electronegativity). The
            solve is *direct* — one bordered linear system, not an iterated
            fixed point — so it reports ``iterations=1``, ``tolerance=0.0`` and
            ``preconditioner="none"`` rather than the numbers an iterative
            solver would carry. ``multipole_order`` is
            ``0`` (the model produces point charges) and there is no induced
            response to describe.
            :attr:`~molpot.manifest.ElectrostaticsRecord.total_charge` stays
            ``None``: the constrained total is a per-batch input
            (``graphs.total_charge``), not model identity, and this model's own
            :attr:`total_charge` is only the fallback for a batch that omits
            it. The solver fields describe
            :class:`~molpot.charge.ElectronegativityEqualization`, the only
            ``equilibration`` implementation in tree; they are written as
            literals rather than read off ``self.equilibration``, so a second
            implementation has to extend this method instead of inheriting an
            answer that would then be wrong.
        """
        return ElectrostaticsRecord(
            multipole_order=0,
            polarization_scheme=None,
            damping=None,
            charge_constraint="total-charge-kkt",
            total_charge=None,
            permanent_scaling=permanent_scaling,
            induced_scaling=None,
            solver=SolverRecord(
                kind="electronegativity-equalization",
                iterations=1,
                tolerance=0.0,
                preconditioner="none",
            ),
        )

    def forward(
        self,
        atom_features: torch.Tensor,
        batch: Mapping[str, Any] | TensorDict,
        *,
        forcefield: ForceField | None = None,
        total_charge: float | torch.Tensor | None = None,
    ) -> torch.Tensor | ForceField:
        """If ``forcefield`` is given, return filled FF; else return ``charge`` only."""
        if forcefield is None:
            return self.charges(atom_features, batch, total_charge=total_charge)
        return self.write_forcefield(forcefield, atom_features, batch, total_charge=total_charge)
