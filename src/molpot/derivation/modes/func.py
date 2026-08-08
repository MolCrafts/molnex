"""FuncMode — energy/force peers via ``torch.func`` only.

In-place batch contract. ``EnergyReadout(backward=True)`` is **lazy**
(no model call); ``ForceReadout`` flushes a single ``grad(..., has_aux=True)``
pass that writes energy and forces. Energy-only uses ``backward=False``.
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
        model = deriv.model  # type: ignore[attr-defined]
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
            deriv._lazy_func = True  # type: ignore[attr-defined]
            deriv._backward = True  # type: ignore[attr-defined]
            deriv._energy_ready = False  # type: ignore[attr-defined]
            return batch

        deriv._lazy_func = False  # type: ignore[attr-defined]
        deriv._backward = False  # type: ignore[attr-defined]
        out = call_energy(model, batch)
        batch = absorb_model_output(batch, out)
        if not has_energy(batch):
            raise RuntimeError(
                "model.forward must write batch['graphs','energy'] (or return a dict with 'energy')"
            )
        deriv._energy_ready = True  # type: ignore[attr-defined]
        return batch

    def run_forces(self, deriv: object, batch: TensorDict) -> TensorDict:
        model = deriv.model  # type: ignore[attr-defined]
        if model is None:
            raise RuntimeError("session model is not set")

        pos = batch[POS_KEY].detach()
        base = batch.clone()
        base[POS_KEY] = pos

        def energy_fn_aux(p: torch.Tensor) -> tuple[torch.Tensor, TensorDict]:
            b = base.clone()
            b[POS_KEY] = p
            out = call_energy(model, b)
            b = absorb_model_output(b, out)
            if not has_energy(b):
                raise RuntimeError(
                    "model.forward must write batch['graphs','energy'] inside the func energy path"
                )
            return b[ENERGY_KEY].sum(), b

        grad, filled = torch.func.grad(energy_fn_aux, has_aux=True)(pos)
        batch[ENERGY_KEY] = filled[ENERGY_KEY]
        if "atoms" in filled.keys() and "energy" in filled["atoms"].keys():
            batch["atoms", "energy"] = filled["atoms", "energy"]
        write_forces(batch, -grad)

        deriv._lazy_func = False  # type: ignore[attr-defined]
        deriv._energy_ready = True  # type: ignore[attr-defined]
        deriv._backward = True  # type: ignore[attr-defined]
        return batch
