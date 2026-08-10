"""FuncMode — energy/force peers via ``torch.func`` only.

In-place batch contract. ``EnergyReadout(backward=True)`` is **lazy**
(no model call); ``ForceReadout`` flushes a single ``grad(..., has_aux=True)``
pass that writes energy and forces. Energy-only uses ``backward=False``.
"""

from __future__ import annotations

from functools import partial

from tensordict import TensorDict

from molpot.derivation.kernels import func_force_pass
from molpot.derivation.protocol import (
    absorb_model_output,
    call_energy,
    has_energy,
)


class FuncMode:
    """torch.func-only mode for sequential Energy/Force readouts."""

    name: str = "func"

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

        if backward:
            if getattr(deriv, "_energy_ready", False) and not getattr(deriv, "_lazy_func", False):
                raise RuntimeError(
                    "Energy already materialised without lazy-func state; "
                    "for method='func' use EnergyReadout(backward=True) "
                    "then ForceReadout (lazy+flush), or Energy only with "
                    "backward=False"
                )
            deriv._lazy_func = True
            deriv._backward = True
            deriv._energy_ready = False
            return batch

        deriv._lazy_func = False
        deriv._backward = False
        out = call_energy(model, batch)
        batch = absorb_model_output(batch, out)
        if not has_energy(batch):
            raise RuntimeError(
                "model.forward must write batch['graphs','energy'] (or return a dict with 'energy')"
            )
        deriv._energy_ready = True
        return batch

    def run_forces(self, deriv: object, batch: TensorDict) -> TensorDict:
        model = deriv.model
        if model is None:
            raise RuntimeError("session model is not set")

        batch = func_force_pass(partial(call_energy, model), batch)

        deriv._lazy_func = False
        deriv._energy_ready = True
        deriv._backward = True
        return batch
