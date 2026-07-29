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
    """Energy model + optional forces via the shared :class:`ForceDerivation`.

    Subclasses implement :meth:`energy_forward` only. Force paths:

    * **eval + functorch**: single ``grad(..., has_aux=True)`` pass.
    * **train + functorch**: eager energy (params connected) + force pass.
    * **autograd**: always ``ForceDerivation(method="autograd")`` (cuEq-safe).

    Args:
        force_method: ``"functorch"`` or ``"autograd"`` (see
            :class:`molpot.derivation.ForceDerivation`).
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

        if self.force_derivation.method == "functorch" and not self.training:

            def energy_fn_aux(p: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
                b = base.clone()
                b["atoms", "pos"] = p
                out = self.energy_forward(b)
                return out["energy"].sum(), out

            forces, out = self.force_derivation(energy_fn_aux, pos, has_aux=True)
            out["forces"] = forces
            return out

        def energy_fn(p: torch.Tensor) -> torch.Tensor:
            b = base.clone()
            b["atoms", "pos"] = p
            return self.energy_forward(b)["energy"].sum()

        # Training (or autograd): keep an eager energy pass so parameters stay
        # connected for a force-supervised loss when using functorch's separate
        # force transform; autograd_forces already handles create_graph.
        if self.force_derivation.method == "functorch":
            out = self.energy_forward(base.clone())
            out["forces"] = self.force_derivation(energy_fn, pos)
            return out

        out = self.energy_forward(base.clone())
        out["forces"] = self.force_derivation(energy_fn, pos)
        return out

    def compile_energy(self, *, backend: str = "inductor", **kwargs) -> None:
        """Compile the energy-only forward (not the force training path)."""
        self._compiled_energy_forward = torch.compile(
            self.energy_forward, backend=backend, **kwargs
        )
