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

from molpot.derivation.stress import symmetrize_strain_grad

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


def position_leaf(pos: torch.Tensor) -> torch.Tensor:
    """The fresh ``requires_grad`` leaf an autograd force is differentiated against.

    One line, but it belongs to this module rather than to each caller, for the
    same reason ``autograd.grad`` does: it is the **seam**, not physics.
    Detaching drops whatever history the caller's positions carried (so the
    force is a gradient of this energy alone, not of how the positions were
    produced) and ``requires_grad_`` re-attaches them as the differentiation
    variable.

    Dynamo — PyTorch's tracing front-end for ``torch.compile`` — cannot trace
    either half: a leaf created inside a graph and used outside it is
    "requires_grad_() intermediate leaked as output", so the trace stops there
    (a *graph break*: the compiled region ends and the rest runs eagerly).
    Creating it here means that break, like the ``autograd.grad`` below it, is
    reported against ``molpot/derivation/force.py``, which owns the seam and its
    cost. A physics module that spells this inline instead puts an unexplained
    break site in the middle of an energy expression.

    That attribution is load-bearing for the compiled classical core:
    :meth:`molpot.forcefield.Potentials.compile_core` declares this file as the
    **one** break a classical field may show, and it is reached only through a
    declared ``pair:coulomb``, which has no closed-form force. A ``lj/cut`` +
    bonded field never creates a leaf and traces as a single graph.

    Args:
        pos: Positions ``(N, 3)`` in Å, attached or not.

    Returns:
        A detached copy of ``pos`` with ``requires_grad=True``, in Å and on
        ``pos``'s dtype and device.
    """
    return pos.detach().requires_grad_(True)


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
        p = position_leaf(pos)
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


def autograd_forces_and_virial(
    energy_fn: Callable[..., torch.Tensor],
    pos: torch.Tensor,
    cell: torch.Tensor,
    *cartesians: torch.Tensor,
    create_graph: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``F = -∂E/∂pos`` and ``W = ∂E/∂ε`` from one autograd backward (MACE/cuEq).

    ``energy_fn(pos_s, cell_s, *cartesians_s)`` is evaluated on symmetrically
    strained Cartesian tensors ``v @ (I + ε_sym)ᵀ`` at ``ε = 0``. Extra
    ``cartesians`` are imaged edge displacements (or any other row-vectors in
    Å) that must strain with the cell.

    Args:
        energy_fn: Maps strained ``(pos, cell, *cartesians)`` to a scalar
            energy. Must recompute geometry from those tensors.
        pos: Positions ``(N, 3)`` Å.
        cell: Lattice vectors ``(3, 3)`` Å, one vector per row.
        *cartesians: Extra ``(..., 3)`` Cartesian tensors to strain.
        create_graph: Keep the graph for a force-supervised backward.
            Defaults to ambient ``torch.is_grad_enabled()``.

    Returns:
        ``(energy, forces, virial)`` with virial ``W = ∂E/∂ε`` shape ``(3, 3)``
        in the energy unit (stress is ``W / V``).
    """
    if create_graph is None:
        create_graph = torch.is_grad_enabled()
    if tuple(cell.shape) != (3, 3):
        raise ValueError(f"cell must be (3, 3), got {tuple(cell.shape)}")
    pos_leaf = position_leaf(pos)
    strain = torch.zeros(3, 3, dtype=pos.dtype, device=pos.device, requires_grad=True)
    with torch.enable_grad():
        sym = 0.5 * (strain + strain.T)
        ft = (torch.eye(3, dtype=pos.dtype, device=pos.device) + sym).T
        pos_s = pos_leaf @ ft
        cell_s = cell.to(dtype=pos.dtype, device=pos.device) @ ft
        extra_s = tuple(tensor.to(dtype=pos.dtype, device=pos.device) @ ft for tensor in cartesians)
        energy = energy_fn(pos_s, cell_s, *extra_s)
        scalar = energy if energy.ndim == 0 else energy.sum()
        g_pos, g_strain = torch.autograd.grad(
            scalar,
            (pos_leaf, strain),
            create_graph=create_graph,
            allow_unused=True,
        )
    if g_pos is None:
        g_pos = torch.zeros_like(pos_leaf)
    if g_strain is None:
        g_strain = torch.zeros_like(strain)
    return energy, -g_pos, symmetrize_strain_grad(g_strain)


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
