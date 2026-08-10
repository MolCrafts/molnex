"""Internal readout session (mode + model + sequential state).

Not part of the public call shape — created inside :class:`EnergyReadout` /
:class:`ForceReadout` and stashed on the batch.
"""

from __future__ import annotations

from typing import Any, Literal

from tensordict import TensorDict

from molpot.derivation.modes.func import FuncMode
from molpot.derivation.modes.grad import GradMode


class Derivative:
    """Session: one mode, one model, state for Energy → Force on a batch."""

    def __init__(
        self,
        method: Literal["func", "grad"],
        model: Any,
    ) -> None:
        if method not in ("func", "grad"):
            raise ValueError(f"method must be 'func' or 'grad', got {method!r}")
        self.method = method
        self.model = model
        self.mode: FuncMode | GradMode = FuncMode() if method == "func" else GradMode()
        self._backward: bool = False
        self._energy_ready: bool = False
        self._lazy_func: bool = False
        self._pos_leaf = None

    def run_energy(self, batch: TensorDict, *, backward: bool = False) -> TensorDict:
        return self.mode.run_energy(self, batch, backward=backward)

    def run_forces(self, batch: TensorDict) -> TensorDict:
        return self.mode.run_forces(self, batch)
