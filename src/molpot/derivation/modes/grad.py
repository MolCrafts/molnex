"""GradMode — energy/force peers via ``torch.autograd`` only.

In-place batch contract. Sequential Energy then Force is one model forward when
``backward=True`` left a live ``E ← pos`` graph.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from molpot.derivation.protocol import (
    ENERGY_KEY,
    POS_KEY,
    absorb_model_output,
    call_energy,
    has_energy,
    write_forces,
)


class GradMode:
    """Autograd-only mode for sequential Energy/Force readouts."""

    name: str = "grad"

    def run_energy(
        self,
        deriv: object,
        batch: TensorDict,
        *,
        backward: bool,
    ) -> TensorDict:
        model = deriv.model  # type: ignore[attr-defined]
        if model is None:
            raise RuntimeError("session model is not set")

        pos = batch[POS_KEY]
        if backward:
            pos = pos.detach().requires_grad_(True)
            batch[POS_KEY] = pos
            deriv._pos_leaf = pos  # type: ignore[attr-defined]
            deriv._backward = True  # type: ignore[attr-defined]
        else:
            deriv._backward = False  # type: ignore[attr-defined]
            deriv._pos_leaf = None  # type: ignore[attr-defined]

        out = call_energy(model, batch)
        batch = absorb_model_output(batch, out)
        if not has_energy(batch):
            raise RuntimeError(
                "model.forward must write batch['graphs','energy'] (or return a dict with 'energy')"
            )
        deriv._energy_ready = True  # type: ignore[attr-defined]
        deriv._lazy_func = False  # type: ignore[attr-defined]
        return batch

    def run_forces(self, deriv: object, batch: TensorDict) -> TensorDict:
        if not getattr(deriv, "_energy_ready", False) or not getattr(deriv, "_backward", False):
            batch = self.run_energy(deriv, batch, backward=True)

        energy = batch[ENERGY_KEY]
        pos = getattr(deriv, "_pos_leaf", None)
        if pos is None:
            pos = batch[POS_KEY]
        if not pos.requires_grad:
            raise RuntimeError(
                "ForceReadout (grad mode) needs positions with requires_grad; "
                "call EnergyReadout(..., backward=True) first"
            )

        create_graph = bool(getattr(deriv.model, "training", False))  # type: ignore[attr-defined]
        with torch.enable_grad():
            (g,) = torch.autograd.grad(
                energy.sum(), pos, create_graph=create_graph, retain_graph=create_graph
            )
        write_forces(batch, -g)
        return batch
