"""Force derivation: ``F = -∂E/∂pos``.

Single responsibility: atomic forces as the negative gradient of energy w.r.t.
atomic positions. Two **explicit, non-mixing** backends — the caller picks one
per model; there is no auto-detection and no silent fallback between them:

* **functorch** (``torch.func.grad``) — pure-PyTorch graphs (e.g. PiNet).
  Traced into the forward graph; ``energy → force → loss`` is one backward and
  composes with ``torch.compile(fullgraph=True)``. Supports ``has_aux=True`` so
  energy side-outputs come back with a single energy evaluation. Does **not**
  work on cuEquivariance *fused* kernels (legacy ``autograd.Function`` without
  ``setup_context`` — pytorch#170834).

* **autograd** (``torch.autograd.grad``) — cuEq / MACE-style graphs. Eager
  correct, ``create_graph`` keeps the force connected to parameters for a
  force-supervised double-backward, and is ``torch.compile``-able via
  ``torch._dynamo.allow_in_graph(torch.autograd.grad)`` with the outer
  ``loss.backward()`` run eagerly. Also exposes
  :func:`autograd_forces_from_energy` for the 1-pass composition when energy
  was already materialised on a ``requires_grad`` position leaf.

:class:`ForceDerivation` only **dispatches** to the chosen backend. It never
routes a ``functorch`` model through ``autograd`` helpers, or the reverse.

Pick the backend explicitly: ``ForceDerivation(method="functorch")`` for PiNet,
``ForceDerivation(method="autograd")`` for MACE. Default ``"autograd"`` (safe
for every model; choose ``"functorch"`` only for pure-torch graphs you want to
``torch.compile(fullgraph)``).

Example:
    >>> deriv = ForceDerivation(method="autograd")
    >>> pos = torch.randn(5, 3)
    >>> forces = deriv(lambda p: p.pow(2).sum(), pos)  # energy_fn(pos) -> scalar
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, overload

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# functorch backend (only)
# ---------------------------------------------------------------------------


def functorch_forces(
    energy_fn: Callable[[torch.Tensor], torch.Tensor],
    pos: torch.Tensor,
) -> torch.Tensor:
    """``F = -∂E/∂pos`` via ``torch.func.grad`` (pure-PyTorch models).

    Traced into the forward graph → single backward, ``torch.compile(fullgraph)``
    friendly. Raises on cuEquivariance fused kernels (legacy ``autograd.Function``
    without ``setup_context`` — pytorch#170834).
    """
    return -torch.func.grad(energy_fn)(pos)


def functorch_forces_with_aux(
    energy_fn: Callable[[torch.Tensor], tuple[torch.Tensor, Any]],
    pos: torch.Tensor,
) -> tuple[torch.Tensor, Any]:
    """``F = -∂E/∂pos`` plus auxiliary outputs via ``torch.func.grad(..., has_aux=True)``.

    ``energy_fn`` must return ``(scalar_energy, aux)``. Returns ``(forces, aux)``.
    One energy evaluation — the 1-pass path for the functorch backend.
    """
    grad, aux = torch.func.grad(energy_fn, has_aux=True)(pos)
    return -grad, aux


# ---------------------------------------------------------------------------
# autograd backend (only)
# ---------------------------------------------------------------------------


def autograd_forces(
    energy_fn: Callable[[torch.Tensor], torch.Tensor],
    pos: torch.Tensor,
) -> torch.Tensor:
    """``F = -∂E/∂pos`` via ``torch.autograd.grad`` (cuEq / MACE).

    Re-runs ``energy_fn`` on a fresh ``requires_grad`` leaf. ``create_graph``
    follows the ambient grad state: training (grad enabled) keeps the force
    connected to parameters (mixed 2nd derivative ``∂²E/∂pos∂θ``) so a
    force-loss ``.backward()`` reaches them; pure inference (``torch.no_grad()``)
    detaches. Works on cuEq fused ops (ordinary double backward).
    """
    create_graph = torch.is_grad_enabled()
    with torch.enable_grad():
        p = pos.detach().requires_grad_(True)
        (grad,) = torch.autograd.grad(energy_fn(p), p, create_graph=create_graph)
    return -grad


def autograd_forces_from_energy(
    energy: torch.Tensor,
    pos: torch.Tensor,
    *,
    create_graph: bool | None = None,
) -> torch.Tensor:
    """``F = -∂E/∂pos`` when energy was already computed on ``pos`` (autograd 1-pass).

    Autograd-backend only. Pair with one ``energy_forward`` that used
    ``pos`` as a ``requires_grad`` leaf — no second energy evaluation.

    Args:
        energy: Per-graph energies ``(B,)`` or a scalar total.
        pos: Position leaf that ``energy`` depends on; must require grad.
        create_graph: Keep force connected to parameters for a force-supervised
            ``loss.backward()``. Defaults to ambient ``torch.is_grad_enabled()``.

    Returns:
        Atomic forces ``(N, 3)``.
    """
    if create_graph is None:
        create_graph = torch.is_grad_enabled()
    scalar = energy if energy.ndim == 0 else energy.sum()
    with torch.enable_grad():
        (grad,) = torch.autograd.grad(scalar, pos, create_graph=create_graph)
    return -grad


_BACKENDS: dict[str, Callable] = {
    "functorch": functorch_forces,
    "autograd": autograd_forces,
}


class ForceDerivation(nn.Module):
    """Dispatch ``F = -∂E/∂pos`` to exactly one backend — never both.

    Args:
        method: ``"functorch"`` or ``"autograd"``. Fixed at construction; every
            call on this instance stays on that backend.
    """

    def __init__(self, method: Literal["functorch", "autograd"] = "autograd"):
        super().__init__()
        if method not in _BACKENDS:
            raise ValueError(f"method must be one of {sorted(_BACKENDS)}, got {method!r}")
        self.method = method

    @overload
    def forward(
        self,
        energy_fn: Callable[[torch.Tensor], torch.Tensor],
        pos: torch.Tensor,
        *,
        has_aux: Literal[False] = False,
    ) -> torch.Tensor: ...

    @overload
    def forward(
        self,
        energy_fn: Callable[[torch.Tensor], tuple[torch.Tensor, Any]],
        pos: torch.Tensor,
        *,
        has_aux: Literal[True],
    ) -> tuple[torch.Tensor, Any]: ...

    def forward(
        self,
        energy_fn: Callable,
        pos: torch.Tensor,
        *,
        has_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """Forces as the negative gradient of energy w.r.t. positions.

        Args:
            energy_fn: Maps positions ``(N, 3)`` to a **scalar** total energy
                (or ``(scalar, aux)`` when ``has_aux=True``), closing over the
                model parameters and the rest of the batch. Must recompute
                position-derived geometry inside itself so the gradient flows
                through it.
            pos: Atomic positions ``(N, 3)``. Does not need ``requires_grad``.
            has_aux: If ``True``, ``energy_fn`` returns ``(energy, aux)`` and this
                method returns ``(forces, aux)``. **functorch only** (1-pass
                energy + side outputs).

        Returns:
            Atomic forces ``(N, 3)``, or ``(forces, aux)`` when ``has_aux=True``.
        """
        if has_aux:
            if self.method != "functorch":
                raise ValueError(
                    f"has_aux=True is functorch-only; got ForceDerivation(method={self.method!r})"
                )
            return functorch_forces_with_aux(energy_fn, pos)
        return _BACKENDS[self.method](energy_fn, pos)

    def forces_from_energy(
        self,
        energy: torch.Tensor,
        pos: torch.Tensor,
        *,
        create_graph: bool | None = None,
    ) -> torch.Tensor:
        """Autograd-backend 1-pass: energy already on ``pos``, return forces.

        Raises:
            ValueError: If this instance is not ``method="autograd"``.
        """
        if self.method != "autograd":
            raise ValueError(
                "forces_from_energy is autograd-only; "
                f"got ForceDerivation(method={self.method!r}). "
                "For functorch use forward(..., has_aux=True)."
            )
        return autograd_forces_from_energy(energy, pos, create_graph=create_graph)
