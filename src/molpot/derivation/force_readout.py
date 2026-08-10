"""ForceReadout — peer step: write forces on batch (in-place).

``__init__`` binds a monomorphic ``__call__`` from ``method``. Prefer a prior
``EnergyReadout(..., backward=True)`` so the pair is one potential forward.
"""

from __future__ import annotations

from typing import Any, Literal

from tensordict import TensorDict

from molpot.derivation.derivative import Derivative
from molpot.derivation.protocol import attach_session, detach_session, get_session


class ForceReadout:
    """Materialise forces on ``batch``.

    Signature::

        ForceReadout(model, *, method="func")

    Prefer a prior ``EnergyReadout(model, method=..., backward=True)`` so the
    pair is a single potential forward.
    """

    def __init__(
        self,
        model: Any,
        *,
        method: Literal["func", "grad"] = "func",
    ) -> None:
        self.model = model
        self.method = method
        if method == "func":
            self._run = self._run_func
        elif method == "grad":
            self._run = self._run_grad
        else:  # pragma: no cover
            raise ValueError(f"method must be 'func' or 'grad', got {method!r}")

    def __call__(self, batch: TensorDict) -> TensorDict:
        return self._run(batch)

    def _run_func(self, batch: TensorDict) -> TensorDict:
        session = self._session_for(batch)
        try:
            return session.run_forces(batch)
        finally:
            detach_session(batch)

    def _run_grad(self, batch: TensorDict) -> TensorDict:
        session = self._session_for(batch)
        try:
            return session.run_forces(batch)
        finally:
            detach_session(batch)

    def _session_for(self, batch: TensorDict) -> Derivative:
        session = get_session(batch)
        if session is None or session.method != self.method or session.model is not self.model:
            session = Derivative(method=self.method, model=self.model)
            attach_session(batch, session)
        else:
            session.model = self.model
        return session
