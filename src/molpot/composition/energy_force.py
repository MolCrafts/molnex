"""Generic energy + force model wrapper (physics home for force derivation).

Encoders live in ``molzoo``; force derivation lives in ``molpot``. Subclass
:class:`EnergyForceModel` and implement :meth:`energy_forward` to attach a
:class:`~molpot.derivation.ForceDerivation` without inventing a third force
path. ``molzoo.pinet.PiNetPotential`` is the primary consumer.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from tensordict import TensorDict

from molpot.derivation import ForceDerivation


class EnergyForceModel(nn.Module):
    """Energy model + optional forces; **one backend per instance, no mixing**.

    Subclasses implement :meth:`energy_forward` only. Force composition is
    entirely determined by ``force_method``:

    * **``functorch``** — always one energy evaluation via
      ``ForceDerivation(..., has_aux=True)`` (train and eval). Never calls
      ``torch.autograd.grad``.
    * **``autograd``** — always one ``energy_forward`` on a ``requires_grad``
      position leaf, then
      :meth:`ForceDerivation.forces_from_energy` (autograd-only). Never calls
      ``torch.func.grad``.

    Args:
        force_method: ``"functorch"`` or ``"autograd"`` — fixed for the life
            of the module; see :class:`~molpot.derivation.ForceDerivation`.
        compute_forces: Default for :meth:`forward` when ``compute_forces`` is
            not passed explicitly.
    """

    def __init__(
        self,
        *,
        force_method: Literal["functorch", "autograd"] = "functorch",
        compute_forces: bool = False,
    ) -> None:
        super().__init__()
        self.force_derivation = ForceDerivation(method=force_method)
        self.compute_forces_default = compute_forces
        self._compiled_energy_forward = None

    def energy_forward(self, batch: TensorDict) -> dict[str, torch.Tensor]:
        """Return at least ``{"energy": (B,)}``; may include auxiliaries."""
        raise NotImplementedError

    def forward(
        self, batch: TensorDict, *, compute_forces: bool | None = None
    ) -> dict[str, torch.Tensor]:
        if compute_forces is None:
            compute_forces = self.compute_forces_default
        if compute_forces:
            return self._forward_with_forces(batch)
        energy_forward = self._compiled_energy_forward or self.energy_forward
        return energy_forward(batch)

    def _forward_with_forces(self, batch: TensorDict) -> dict[str, torch.Tensor]:
        pos = batch["atoms", "pos"].detach()
        base = batch.clone()
        base["atoms", "pos"] = pos
        method = self.force_derivation.method

        if method == "functorch":
            return self._forces_functorch(base, pos)
        if method == "autograd":
            return self._forces_autograd(base, pos)
        raise RuntimeError(f"unknown force method {method!r}")  # pragma: no cover

    def _forces_functorch(
        self, base: TensorDict, pos: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Functorch-only 1-pass: ``grad(..., has_aux=True)``."""

        def energy_fn_aux(p: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            b = base.clone()
            b["atoms", "pos"] = p
            out = self.energy_forward(b)
            return out["energy"].sum(), out

        forces, out = self.force_derivation(energy_fn_aux, pos, has_aux=True)
        out["forces"] = forces
        return out

    def _forces_autograd(
        self, base: TensorDict, pos: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Autograd-only 1-pass: energy on ``p``, then ``forces_from_energy``."""
        p = pos.detach().requires_grad_(True)
        base["atoms", "pos"] = p
        out = self.energy_forward(base)
        # create_graph only in train so force-supervised loss reaches θ;
        # eval keeps value-only forces (MD / inference).
        out["forces"] = self.force_derivation.forces_from_energy(
            out["energy"], p, create_graph=self.training
        )
        return out

    def compile_energy(self, *, backend: str = "inductor", **kwargs) -> None:
        """Compile the energy-only forward (not the force training path)."""
        self._compiled_energy_forward = torch.compile(
            self.energy_forward, backend=backend, **kwargs
        )
