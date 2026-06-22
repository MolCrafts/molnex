"""Force derivation: ``F = -∂E/∂pos``.

Single responsibility: compute atomic forces as the negative gradient of energy
with respect to atomic positions.

Two backends, auto-selected:

* **functorch** (``torch.func.grad``) — preferred. The transform is *traced into
  the forward graph*, so the inner force derivative becomes ordinary forward ops
  and ``energy → force → loss`` needs only ONE regular backward. No
  double-backward barrier, so it composes with ``torch.compile(fullgraph=True)``.
* **autograd fallback** (``torch.autograd.grad``) — used transparently when the
  energy graph contains an op that functorch cannot trace (e.g. cuEquivariance's
  *fused* kernels are legacy ``autograd.Function``\\s without a ``setup_context``
  staticmethod, so ``torch.func.grad`` raises on them). Those ops *do* support
  ordinary autograd double-backward, so the fallback is eager-correct and trains
  (force-supervised losses backprop through the force into parameters). It is
  NOT ``torch.compile(fullgraph)``-able (aot_autograd does not currently support
  double backward), but that path was uncompilable anyway, so falling back is
  strictly better than crashing.

Example:
    >>> deriv = ForceDerivation()
    >>> pos = torch.randn(5, 3)
    >>> forces = deriv(energy_fn, pos)  # (5, 3); energy_fn(pos) -> scalar
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import torch
import torch.nn as nn

# Substrings identifying the functorch-vs-legacy-autograd.Function incompatibility
# raised by ``torch.func.grad`` when the graph contains a fused custom op without
# a ``setup_context`` staticmethod (e.g. cuEquivariance fused kernels).
_FUNCTORCH_INCOMPAT_MARKERS = ("setup_context", "functorch transforms")


def _is_functorch_incompat(err: RuntimeError) -> bool:
    msg = str(err)
    return any(m in msg for m in _FUNCTORCH_INCOMPAT_MARKERS)


def _autograd_forces(
    energy_fn: Callable[[torch.Tensor], torch.Tensor],
    pos: torch.Tensor,
) -> torch.Tensor:
    """``F = -autograd.grad(E, pos)`` with double-backward when grad is enabled.

    ``create_graph`` follows the ambient grad state: training steps run with grad
    enabled → the force stays connected to the parameters (mixed 2nd derivative
    ∂²E/∂pos∂θ), so a force-loss ``.backward()`` reaches them; pure inference
    (``torch.no_grad()``) → ``create_graph=False`` → detached force values.
    """
    create_graph = torch.is_grad_enabled()
    with torch.enable_grad():
        p = pos.detach().requires_grad_(True)
        energy = energy_fn(p)
        (grad,) = torch.autograd.grad(energy, p, create_graph=create_graph)
    return -grad


def functorch_or_autograd_forces(
    energy_fn: Callable[[torch.Tensor], torch.Tensor],
    pos: torch.Tensor,
    *,
    warn: bool = True,
) -> torch.Tensor:
    """``F = -∂E/∂pos`` via functorch, falling back to autograd on incompatibility.

    Prefers ``torch.func.grad`` (compile-friendly, single backward). If the energy
    graph contains a functorch-incompatible op (e.g. a fused cuEquivariance kernel
    without a ``setup_context`` staticmethod), transparently uses
    ``torch.autograd.grad`` so eager execution never crashes. Stateless — callers
    that differentiate the same graph repeatedly (e.g. training loops) should use
    :class:`ForceDerivation`, which remembers the incompatibility.
    """
    try:
        return -torch.func.grad(energy_fn)(pos)
    except RuntimeError as err:
        if not _is_functorch_incompat(err):
            raise
        if warn:
            warnings.warn(_FALLBACK_MSG, RuntimeWarning, stacklevel=2)
        return _autograd_forces(energy_fn, pos)


_FALLBACK_MSG = (
    "Force derivation: the energy graph contains a functorch-incompatible op "
    "(e.g. a fused cuEquivariance kernel without a functorch setup_context). "
    "Falling back to torch.autograd.grad — eager-correct and trainable, but not "
    "torch.compile(fullgraph)-able. Use a pure-torch TP method (e.g. "
    "Allegro(tp_method='naive')) if you need fullgraph compilation of forces."
)


class ForceDerivation(nn.Module):
    """Compute forces as ``F = -∂E/∂pos``.

    Prefers ``torch.func.grad`` (compile-friendly, single backward). If the
    energy graph contains a functorch-incompatible op, transparently falls back
    to ``torch.autograd.grad`` so eager execution never crashes. The fallback is
    sticky per-instance: once a functorch incompatibility is observed, subsequent
    calls skip the (doomed) functorch attempt to avoid the wasted partial
    forward each step.

    Validated by the Phase-3 spikes (``trainer-hardening-01-compile-strategy``):
    the functorch path is correct vs a plain ``autograd.grad`` reference to
    ~1e-8, fullgraph-compiles with zero graph breaks, and a single
    ``loss.backward()`` populates parameter grads — on PiNet (no cuEq), MACE
    (cuEq fallback), and naive-method Allegro. The autograd fallback is
    GH200-verified to give identical forces and 11/11 nonzero param grads with
    cuEquivariance *fused* kernels (``uniform_1d`` / ``fused_tp``), which
    functorch cannot trace.

    Note: neither path removes the second-order compute — the mixed
    second-derivative FLOPs/memory are still incurred.
    """

    def __init__(self):
        super().__init__()
        # Sticky flag: set once functorch is found to be unsupported for the
        # energy graph this instance differentiates. Stored as a buffer so it
        # rides along with ``.to()`` / state dict and is visible to torch.compile
        # guards.
        self._functorch_unsupported: bool = False

    def forward(
        self,
        energy_fn: Callable[[torch.Tensor], torch.Tensor],
        pos: torch.Tensor,
    ) -> torch.Tensor:
        """Compute forces as the negative gradient of energy w.r.t. positions.

        Args:
            energy_fn: Maps positions ``(N, 3)`` to a **scalar** total energy,
                closing over the model parameters and the rest of the batch.
                Must recompute any position-derived geometry (edge vectors,
                distances) *inside* itself so the gradient flows through them.
            pos: Atomic positions ``(N, 3)``. Does not need ``requires_grad`` —
                both backends track the input themselves.

        Returns:
            Atomic forces ``(N, 3)``.
        """
        if not self._functorch_unsupported:
            try:
                return -torch.func.grad(energy_fn)(pos)
            except RuntimeError as err:
                if not _is_functorch_incompat(err):
                    raise
                warnings.warn(_FALLBACK_MSG, RuntimeWarning, stacklevel=2)
                self._functorch_unsupported = True

        return _autograd_forces(energy_fn, pos)
