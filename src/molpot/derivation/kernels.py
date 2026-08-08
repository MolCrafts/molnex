"""Batch-level force passes shared by potentials and sequential readout modes.

Where this sits relative to :mod:`molpot.derivation.force`:

* ``force.py`` is the **tensor-level** layer — ``energy_fn(pos) -> scalar`` in,
  ``forces (N, 3)`` out. It owns the only ``torch.autograd.grad`` /
  ``torch.func.grad`` calls in this package.
* ``kernels.py`` (this module) is the **batch-level** layer — an *energy core*
  ``energy_core(batch) -> batch`` in, the same batch back with
  ``graphs.energy`` and ``atoms.forces`` written in place. It adds the three
  things a batch pass needs and a tensor pass cannot express: position-leaf
  ownership, the post-collate key writes, and the ``detach_energy`` policy.

The kernels **compose** ``force.py``; they never re-derive a gradient. Adding a
third hand-rolled force path inside an encoder or potential is forbidden (see
``.claude/notes/notes.md`` §"Force derivation — dual explicit backends").

Backend boundary (pick one per model — no auto-detect, no fallback):

* :func:`grad_force_pass` — ``torch.autograd.grad`` behind
  :func:`~molpot.derivation.force.autograd_forces_from_energy`. One energy
  forward, then one backward on the position leaf. Works on cuEquivariance
  *fused* kernels (legacy ``autograd.Function`` without ``setup_context``,
  pytorch#170834) and is the MACE-shaped path.
* :func:`func_force_pass` — ``torch.func.grad(..., has_aux=True)`` behind
  :func:`~molpot.derivation.force.functorch_forces_with_aux`. The derivative is
  traced into the forward graph, so ``energy → force → loss`` is a single
  backward and the whole pass composes with ``torch.compile(fullgraph=True)``.
  Pure-PyTorch energy graphs only (PiNet). Rejects cuEq fused kernels.

Units (repo-wide): positions Å, energies eV, forces eV/Å. The kernels are
geometry- and unit-agnostic — they only differentiate whatever the energy core
wrote — so a core that emits kcal/mol yields kcal/(mol·Å) forces silently. Keep
the core on the repo convention.

Compile constraints (hard): :func:`func_force_pass` sits on PiNet's
``fullgraph=True`` path. It must stay free of ``set_non_tensor``, ``.item()``,
tensor-value-dependent branching and session-dict lookups. The Python-level
branches in :func:`grad_force_pass` (``energy_core is None``,
``detach_energy``) are static per call site and never appear in the func path.
Bind ``energy_core`` once at construction (a bound method, or
``functools.partial(call_energy, model)``) rather than building a fresh lambda
per call, so the traced callable stays stable.

State: neither kernel holds state, allocates modules, or touches the
``protocol._SESSIONS`` side channel — session bookkeeping stays in
:mod:`molpot.derivation.modes`. Both mutate ``batch`` in place and return it.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from tensordict import TensorDict

from molpot.derivation.force import autograd_forces_from_energy, functorch_forces_with_aux
from molpot.derivation.protocol import (
    ENERGY_KEY,
    POS_KEY,
    absorb_model_output,
    has_energy,
    write_forces,
)

__all__ = ["func_force_pass", "grad_force_pass"]

#: An energy core: reads ``atoms.pos`` off the batch and writes ``graphs.energy``
#: (optionally ``atoms.energy``). May return the batch, a ``dict`` with an
#: ``"energy"`` entry, or ``None`` when it wrote in place.
EnergyCore = Callable[[TensorDict], "TensorDict | dict | None"]


def grad_force_pass(
    energy_core: EnergyCore | None,
    batch: TensorDict,
    *,
    create_graph: bool | None = None,
    detach_energy: bool | None = None,
) -> TensorDict:
    """One energy forward + ``torch.autograd.grad`` on the position leaf.

    Writes ``graphs.energy`` ``(B,)`` in eV (via the core) and
    ``atoms.forces`` ``(N, 3)`` in eV/Å onto ``batch``, in place.

    Position-leaf ownership: if ``atoms.pos`` ``(N, 3)`` is not already a live
    ``requires_grad`` **leaf**, this kernel makes one
    (``pos.detach().requires_grad_(True)``) and writes it back to the batch, so
    the energy the core computes is differentiable w.r.t. it. A leaf supplied by
    the caller is reused as is — no second leaf, no lost graph.

    Args:
        energy_core: Energy core ``energy_core(batch)`` writing ``graphs.energy``.
            ``None`` means the energy is **already materialised** on a live
            position leaf (the sequential ``EnergyReadout(backward=True)`` →
            ``ForceReadout`` protocol): the forward is skipped and the existing
            graph is differentiated, which is what makes that pair a single
            model forward.
        batch: Post-collate batch. Must carry ``atoms.pos``; ``graphs.energy``
            must already be present when ``energy_core is None``.
        create_graph: Keep the force connected to the parameters (mixed second
            derivative ``∂²E/∂pos∂θ``) so a force-supervised ``loss.backward()``
            reaches them. ``None`` follows the ambient ``torch.is_grad_enabled()``
            *at entry* — resolved before the internal ``enable_grad`` scope, so
            calling under ``torch.no_grad()`` yields detached forces rather than
            silently building a graph.
        detach_energy: Leaf-ownership knob for the returned ``graphs.energy``.

            * ``False`` — never detach; energy stays attached for an energy loss
              (PiNet and ``GradMode``).
            * ``True`` — always detach (energy is a reported quantity only).
            * ``None`` — detach **iff this call created the position leaf**, i.e.
              the caller had no graph to lose. This is MACE's ``get_outputs``
              shape (``mace_matpes.py:368-377``).

    Returns:
        The same ``batch`` object, with ``atoms.forces`` ``(N, 3)`` eV/Å written.

    Raises:
        RuntimeError: If ``energy_core is None`` but ``atoms.pos`` is not a live
            ``requires_grad`` leaf (nothing to differentiate), or if the core ran
            and left no ``graphs.energy`` behind.

    Note:
        The energy forward runs inside ``torch.enable_grad()``, so the pass
        returns correct forces even under an ambient ``torch.no_grad()`` — the
        gradient is a *value*, not a side effect of the caller's grad mode.
    """
    pos = batch[POS_KEY]
    owns_leaf = not (pos.requires_grad and pos.is_leaf)

    if energy_core is None and owns_leaf:
        raise RuntimeError(
            "grad_force_pass(energy_core=None) needs atoms.pos to be a live "
            "requires_grad leaf carrying the energy graph; got a tensor that is "
            "detached or non-leaf. Pass an energy_core, or materialise the "
            "energy on a position leaf first."
        )

    # Resolve before the enable_grad scope: create_graph must reflect the
    # caller's grad mode, not the one this kernel forces on for the derivative.
    resolved_create_graph = torch.is_grad_enabled() if create_graph is None else create_graph

    if owns_leaf:
        pos = pos.detach().requires_grad_(True)
        batch[POS_KEY] = pos

    with torch.enable_grad():
        if energy_core is not None:
            batch = absorb_model_output(batch, energy_core(batch))
            if not has_energy(batch):
                raise RuntimeError(
                    "energy core must write batch['graphs','energy'] "
                    "(graphs.energy) or return a dict with 'energy'"
                )
        forces = autograd_forces_from_energy(
            batch[ENERGY_KEY], pos, create_graph=resolved_create_graph
        )

    should_detach = owns_leaf if detach_energy is None else detach_energy
    if should_detach:
        batch[ENERGY_KEY] = batch[ENERGY_KEY].detach()
    write_forces(batch, forces)
    return batch


def func_force_pass(energy_core: EnergyCore, batch: TensorDict) -> TensorDict:
    """Single ``torch.func.grad(..., has_aux=True)`` pass — energy and force at once.

    The energy core is evaluated **once**, inside the functorch transform, on a
    cloned batch whose ``atoms.pos`` is the differentiation variable; the filled
    clone comes back as ``aux`` so ``graphs.energy`` ``(B,)`` eV (and
    ``atoms.energy`` ``(N,)`` eV when the core writes it) can be copied onto the
    real batch alongside ``atoms.forces`` ``(N, 3)`` eV/Å.

    Because the derivative is traced into the forward graph, an
    ``energy → force → loss`` objective needs one ordinary ``backward()`` and the
    whole pass survives ``torch.compile(fullgraph=True)``. There is deliberately
    no ``detach_energy`` knob here: the only callers want the energy attached,
    and cuEq fused kernels — the ones that need leaf-detaching — cannot use
    functorch at all (pytorch#170834).

    Args:
        energy_core: Energy core ``energy_core(batch)`` writing ``graphs.energy``.
            Must be a pure-PyTorch graph and must recompute every
            position-derived quantity (edge vectors, distances) from the
            ``atoms.pos`` it is handed, or the gradient path is cut.
        batch: Post-collate batch carrying ``atoms.pos`` ``(N, 3)`` Å.
            ``requires_grad`` is not needed — ``torch.func.grad`` tracks the
            input itself.

    Returns:
        The same ``batch`` object, with ``graphs.energy``, optionally
        ``atoms.energy``, and ``atoms.forces`` written in place.

    Raises:
        RuntimeError: If the core leaves no ``graphs.energy`` on the batch.
    """
    pos = batch[POS_KEY].detach()
    base = batch.clone()
    base[POS_KEY] = pos

    def energy_fn_aux(p: torch.Tensor) -> tuple[torch.Tensor, TensorDict]:
        b = base.clone()
        b[POS_KEY] = p
        b = absorb_model_output(b, energy_core(b))
        if not has_energy(b):
            raise RuntimeError(
                "energy core must write batch['graphs','energy'] (graphs.energy) "
                "inside the func energy path"
            )
        return b[ENERGY_KEY].sum(), b

    # functorch_forces_with_aux already returns -grad — do not negate again.
    forces, filled = functorch_forces_with_aux(energy_fn_aux, pos)
    batch[ENERGY_KEY] = filled[ENERGY_KEY]
    if "atoms" in filled.keys() and "energy" in filled["atoms"].keys():
        batch["atoms", "energy"] = filled["atoms", "energy"]
    write_forces(batch, forces)
    return batch
