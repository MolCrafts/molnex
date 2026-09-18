"""Tensor-level force derivation: ``F = -∂E/∂pos`` of a raw ``energy_fn(pos)``.

Single responsibility: the two autograd primitives a caller holding a **raw
tensor** energy composes — :class:`~molpot.potentials.base.BasePotential`,
:class:`~molpot.composition.composer.PotentialComposer`,
:class:`~molpot.composition.sonata.Sonata` and
:class:`~molpot.derivation.hessian.HessianDerivation`, which differentiates
the first derivative a second time. A *batch* is differentiated by
:class:`molix.autograd.Derivative`, which owns the leaf, the geometry rebuild
and the writes; nothing here re-derives a gradient of its own beyond these two
functions.

One backend — ``torch.autograd.grad``. It works on every model, including
cuEquivariance's fused kernels (legacy ``autograd.Function``s without
``setup_context``, which ``torch.func`` rejects — pytorch#170834); it keeps the
force connected to the parameters for a force-supervised double backward when
``create_graph`` is on; and it traces in-graph under ``torch.compile`` once
``torch._dynamo.config.trace_autograd_ops`` is set, which
:class:`molix.compile.Compiler` does.

Two primitives, one shape each — ``create_graph`` always follows the ambient
grad mode unless a caller resolves it earlier:

* :func:`autograd_forces_from_energy` — ``E`` already on a leaf; ``F``.
* :func:`autograd_forces` — ``energy_fn(pos) -> E``; ``F`` on a fresh leaf
  (:func:`molix.autograd.leaf`).

Example:
    >>> deriv = ForceDerivation()
    >>> pos = torch.randn(5, 3)
    >>> forces = deriv(lambda p: p.pow(2).sum(), pos)  # energy_fn(pos) -> scalar
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from molix.autograd import leaf


def autograd_forces_from_energy(
    energy: torch.Tensor,
    pos: torch.Tensor,
    *,
    create_graph: bool | None = None,
) -> torch.Tensor:
    """``F = -∂E/∂pos`` when energy was already computed on ``pos`` (1-pass).

    Pair with one energy forward that used ``pos`` as a ``requires_grad`` leaf
    — no second energy evaluation.

    The ambient ``torch.no_grad()`` an MD loop installs is expected and
    supported: both the per-graph reduction and the backward run inside a local
    ``torch.enable_grad()``, so the energy graph is re-enabled here rather than
    required from the caller. ``create_graph`` still defaults to the *ambient*
    mode, so that no_grad loop pays no tape.

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
    # The per-graph reduction happens *inside* enable_grad: an MD loop calls
    # this under an ambient ``torch.no_grad()`` (expected — the force is not
    # part of any training tape there), and a ``.sum()`` taken outside would be
    # graphless, so ``autograd.grad`` would reject it with "element 0 of tensors
    # does not require grad". Re-enabling grad locally keeps the reduction
    # attached to the energy that ``pos`` already feeds.
    with torch.enable_grad():
        scalar = energy if energy.ndim == 0 else energy.sum()
        (grad,) = torch.autograd.grad(scalar, pos, create_graph=create_graph)
    return -grad


def autograd_forces(
    energy_fn: Callable[[torch.Tensor], torch.Tensor],
    pos: torch.Tensor,
) -> torch.Tensor:
    """``F = -∂E/∂pos`` from one energy forward on a fresh leaf and one backward.

    The leaf is :func:`molix.autograd.leaf` of ``pos``, the energy is evaluated
    on it inside a local ``torch.enable_grad()`` so an MD loop's ambient
    ``no_grad`` still yields a differentiable graph, and
    :func:`autograd_forces_from_energy` takes the derivative. ``create_graph``
    follows the ambient grad state: training (grad enabled) keeps the force
    connected to parameters so a force-loss ``.backward()`` reaches them; pure
    inference (``torch.no_grad()``) detaches.

    Args:
        energy_fn: Maps the position leaf ``(N, 3)`` to per-graph energies
            ``(B,)`` or a scalar total. Must recompute every position-derived
            quantity from the leaf it is handed.
        pos: Positions ``(N, 3)`` Å; need not require grad.

    Returns:
        Atomic forces ``(N, 3)``.
    """
    variable = leaf(pos)
    with torch.enable_grad():
        energy = energy_fn(variable)
    return autograd_forces_from_energy(energy, variable)


class ForceDerivation(nn.Module):
    """``F = -∂E/∂pos`` of a tensor-level energy function.

    The stateless module form of :func:`autograd_forces`, for callers that
    hold a raw ``energy_fn(pos) -> scalar`` rather than a batch —
    :class:`~molpot.potentials.base.BasePotential`,
    :class:`~molpot.composition.composer.PotentialComposer`,
    :class:`~molpot.composition.sonata.Sonata`. A batch goes through
    :class:`molix.autograd.Derivative` instead.
    """

    def __init__(self) -> None:
        super().__init__()

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
        return autograd_forces(energy_fn, pos)
