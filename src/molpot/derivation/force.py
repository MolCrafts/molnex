"""Force derivation: ``F = -∂E/∂pos``.

Single responsibility: atomic forces as the negative gradient of energy w.r.t.
atomic positions. Two **explicit** backends — the caller picks one per model;
there is no auto-detection and no fallback:

* **functorch** (``torch.func.grad``) — for pure-PyTorch models (e.g. PiNet).
  The transform is traced into the forward graph, so ``energy → force → loss``
  is a single backward and composes with ``torch.compile(fullgraph=True)``.
  It does NOT work on cuEquivariance *fused* kernels: those register a legacy
  ``autograd.Function`` without a ``setup_context`` staticmethod, which the
  functorch transforms reject (pytorch#170834).

* **autograd** (``torch.autograd.grad``) — for models built on cuEq fused ops
  (e.g. MACE), matching the upstream MACE library. Eager-correct, trains (the
  force stays connected to parameters for a double-backward force loss), and is
  ``torch.compile``-able via ``torch._dynamo.allow_in_graph(torch.autograd.grad)``
  with the outer ``loss.backward()`` run eagerly.

Pick the backend explicitly: ``ForceDerivation(method="functorch")`` for PiNet,
``ForceDerivation(method="autograd")`` for MACE. The default is ``"autograd"``
(correct for every model; only pure-torch graphs gain anything from functorch).

Example:
    >>> deriv = ForceDerivation(method="autograd")
    >>> pos = torch.randn(5, 3)
    >>> forces = deriv(lambda p: p.pow(2).sum(), pos)  # energy_fn(pos) -> scalar
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
import torch.nn as nn


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


def autograd_forces(
    energy_fn: Callable[[torch.Tensor], torch.Tensor],
    pos: torch.Tensor,
) -> torch.Tensor:
    """``F = -∂E/∂pos`` via ``torch.autograd.grad`` (cuEq / MACE).

    ``create_graph`` follows the ambient grad state: training (grad enabled)
    keeps the force connected to the parameters (mixed 2nd derivative
    ``∂²E/∂pos∂θ``) so a force-loss ``.backward()`` reaches them; pure inference
    (``torch.no_grad()``) detaches. Works on cuEq fused ops (they support
    ordinary double backward).
    """
    create_graph = torch.is_grad_enabled()
    with torch.enable_grad():
        p = pos.detach().requires_grad_(True)
        (grad,) = torch.autograd.grad(energy_fn(p), p, create_graph=create_graph)
    return -grad


_BACKENDS: dict[str, Callable] = {
    "functorch": functorch_forces,
    "autograd": autograd_forces,
}


class ForceDerivation(nn.Module):
    """Compute forces as ``F = -∂E/∂pos`` with an explicit backend (no fallback).

    Args:
        method: ``"functorch"`` (``torch.func.grad``; pure-torch models such as
            PiNet — compile-friendly single backward) or ``"autograd"``
            (``torch.autograd.grad``; cuEq/MACE models). Default ``"autograd"``,
            correct for every model; choose ``"functorch"`` only for a pure-torch
            energy graph you want to ``torch.compile(fullgraph)``.
    """

    def __init__(self, method: Literal["functorch", "autograd"] = "autograd"):
        super().__init__()
        if method not in _BACKENDS:
            raise ValueError(f"method must be one of {sorted(_BACKENDS)}, got {method!r}")
        self.method = method

    def forward(
        self,
        energy_fn: Callable[[torch.Tensor], torch.Tensor],
        pos: torch.Tensor,
    ) -> torch.Tensor:
        """Forces as the negative gradient of energy w.r.t. positions.

        Args:
            energy_fn: Maps positions ``(N, 3)`` to a **scalar** total energy,
                closing over the model parameters and the rest of the batch.
                Must recompute position-derived geometry inside itself so the
                gradient flows through it.
            pos: Atomic positions ``(N, 3)``. Does not need ``requires_grad``.

        Returns:
            Atomic forces ``(N, 3)``.
        """
        return _BACKENDS[self.method](energy_fn, pos)
