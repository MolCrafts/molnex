"""Atomic charge assignment: param head + equilibration → q."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import torch
import torch.nn as nn
from tensordict import TensorDict

from molpot.charge.problem import ChargeEquilibrationProblem
from molpot.forcefield.container import ForceField

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

    Does **not** evaluate Coulomb energy — only returns ``q`` or writes them
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
        """Copy-on-write ForceField with ``atoms["q"]`` (and keep other atom params)."""
        q = self.charges(atom_features, batch, total_charge=total_charge)
        atoms = dict(forcefield.params("atoms") or {})
        atoms["q"] = q
        # alias for heads that use "charge"
        atoms["charge"] = q
        return forcefield.with_params(atoms=atoms)

    def forward(
        self,
        atom_features: torch.Tensor,
        batch: Mapping[str, Any] | TensorDict,
        *,
        forcefield: ForceField | None = None,
        total_charge: float | torch.Tensor | None = None,
    ) -> torch.Tensor | ForceField:
        """If ``forcefield`` is given, return filled FF; else return ``q`` only."""
        if forcefield is None:
            return self.charges(atom_features, batch, total_charge=total_charge)
        return self.write_forcefield(
            forcefield, atom_features, batch, total_charge=total_charge
        )
