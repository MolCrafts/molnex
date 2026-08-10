"""GradMode — energy/force peers via ``torch.autograd`` only.

In-place batch contract. Sequential Energy then Force is one model forward when
``backward=True`` left a live ``E ← pos`` graph.
"""

from __future__ import annotations

from tensordict import TensorDict

from molpot.derivation.kernels import grad_force_pass
from molpot.derivation.protocol import (
    POS_KEY,
    absorb_model_output,
    call_energy,
    has_energy,
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
        model = deriv.model
        if model is None:
            raise RuntimeError("session model is not set")

        pos = batch[POS_KEY]
        if backward:
            pos = pos.detach().requires_grad_(True)
            batch[POS_KEY] = pos
            deriv._pos_leaf = pos
            deriv._backward = True
        else:
            deriv._backward = False
            deriv._pos_leaf = None

        out = call_energy(model, batch)
        batch = absorb_model_output(batch, out)
        if not has_energy(batch):
            raise RuntimeError(
                "model.forward must write batch['graphs','energy'] (or return a dict with 'energy')"
            )
        deriv._energy_ready = True
        deriv._lazy_func = False
        return batch

    def run_forces(self, deriv: object, batch: TensorDict) -> TensorDict:
        if not getattr(deriv, "_energy_ready", False) or not getattr(deriv, "_backward", False):
            batch = self.run_energy(deriv, batch, backward=True)

        pos = getattr(deriv, "_pos_leaf", None)
        if pos is None:
            pos = batch[POS_KEY]
        if not pos.requires_grad:
            raise RuntimeError(
                "ForceReadout (grad mode) needs positions with requires_grad; "
                "call EnergyReadout(..., backward=True) first"
            )

        # Energy is already materialised on the leaf run_energy(backward=True)
        # installed — energy_core=None keeps the pair at one model forward.
        return grad_force_pass(
            None,
            batch,
            create_graph=bool(getattr(deriv.model, "training", False)),
            detach_energy=False,
        )
