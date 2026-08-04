"""EnergyReadout — peer step: write energy on batch (in-place).

``__init__`` binds a **monomorphic** ``__call__`` from ``method`` / ``backward``.
Energy-only never touches a Derivative session.
"""

from __future__ import annotations

from typing import Any, Literal

from tensordict import TensorDict

from molpot.derivation.derivative import Derivative
from molpot.derivation.protocol import (
    absorb_model_output,
    attach_session,
    call_energy,
    get_session,
    has_energy,
)


class EnergyReadout:
    """Materialise energy on ``batch``.

    Signature::

        EnergyReadout(model, *, method="func", backward=False)

    Args:
        model: Potential (energy core via ``_write_energy`` or ``forward``).
        method: ``"func"`` or ``"grad"`` — must match a following ForceReadout
            when ``backward=True``.
        backward: If ``True``, prepare for a following ForceReadout (one energy
            evaluation for the pair). If ``False``, energy-only — **no session**.
    """

    def __init__(
        self,
        model: Any,
        *,
        method: Literal["func", "grad"] = "func",
        backward: bool = False,
    ) -> None:
        self.model = model
        self.method = method
        self.backward = backward

        # Init-time kernel selection — __call__ is a single bound path.
        if not backward:
            self._run = self._energy_only
        elif method == "func":
            self._run = self._func_lazy
        elif method == "grad":
            self._run = self._grad_with_graph
        else:  # pragma: no cover
            raise ValueError(f"method must be 'func' or 'grad', got {method!r}")

    def __call__(self, batch: TensorDict) -> TensorDict:
        return self._run(batch)

    def _energy_only(self, batch: TensorDict) -> TensorDict:
        """Static energy path — no Derivative, no session (compile-friendly)."""
        out = call_energy(self.model, batch)
        batch = absorb_model_output(batch, out)
        if not has_energy(batch):
            raise RuntimeError(
                "energy core must write batch['graphs','energy'] (or return "
                "a dict with 'energy')"
            )
        return batch

    def _func_lazy(self, batch: TensorDict) -> TensorDict:
        """Mark batch for a following ForceReadout (func has_aux flush)."""
        session = self._session_for(batch)
        return session.run_energy(batch, backward=True)

    def _grad_with_graph(self, batch: TensorDict) -> TensorDict:
        """Energy on a requires_grad pos leaf for a following ForceReadout."""
        session = self._session_for(batch)
        return session.run_energy(batch, backward=True)

    def _session_for(self, batch: TensorDict) -> Derivative:
        session = get_session(batch)
        if session is None or session.method != self.method or session.model is not self.model:
            session = Derivative(method=self.method, model=self.model)
            attach_session(batch, session)
        else:
            session.model = self.model
        return session
