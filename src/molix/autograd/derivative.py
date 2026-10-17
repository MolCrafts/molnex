"""The one derivative op: ``∂(of)/∂(wrt)`` on a frame, in place.

Every batch-level derivative in molnex is this one module. A caller states
*which quantity with respect to which variable*, as TensorFrame keys, and
composes it around any energy::

    model = Derivative(MaceMatpes(...).energy, wrt=("atoms.pos",))     # E, F = -∂E/∂pos
    npt = Derivative(pots.energy, wrt=("atoms.pos", "graphs.strain"))  # + W = ∂E/∂ε
    frame = model(frame)      # graphs.energy, atoms.forces[, graphs.virial] on the frame
    model = Compiler(fullgraph=True)(model)                            # the only compile spelling

A potential builds the same op over its own ``energy`` method in ``__init__``
(``MaceMatpes(..., wrt=("atoms.pos",))``, :class:`molix.autograd.Potential`),
so ``pot(frame)`` and ``Derivative(pot.energy, wrt=...)(frame)`` are one code
path — there is no second kernel, no ``compute_forces`` flag and no
``compile_core``.

**Contract of** ``energy``. ``energy(frame) -> frame`` reads only the frame it
is handed — ``atoms.pos``, ``pairs.pair_index`` / ``pair_diff`` (live, imaged),
``graphs.cell`` — and writes ``of`` (per-graph ``(B,)`` or a scalar) and,
optionally, ``atoms.energy``. An ``nn.Module`` is registered as the submodule
``energy`` so its parameters flow through the op under an ``energy.`` prefix;
a plain callable such as a bound method is held without registration, which
is how a potential wraps its own method without a module cycle. The op calls
exactly what it is handed: a potential is passed as ``pot.energy``, never as
the object — its ``forward`` is an op of its own, and an op nested over an op
would put a second leaf between the energy and the first.

**Who owns the leaf.** The op makes a leaf unless the column already is one.
For a plain column — no ``requires_grad``, or attached to some history — the
``requires_grad`` leaf of every variable is made inside
:meth:`Derivative.forward` (:func:`~molix.autograd.variables.leaf`) and never
left in the frame: the caller gets the derivative of this evaluation alone,
whatever the column's history. A column that already *is* a live leaf
(``requires_grad``, no history) is the variable itself: the op differentiates
against it as is and leaves it in the frame, because it is the caller's. A
caller that wants the whole training step as one graph on stock torch marks
the columns before the traced region — :meth:`Derivative.mark_leaves`, which
the default training and evaluation steps run in ``forward_loss``.

**Geometry.** The frame may carry a cached neighbour geometry —
``pairs.pair_diff`` (imaged, ``pos[t] - pos[s] + S``), ``pairs.pair_dist``,
``pairs.shifts``. The op rebuilds it once from the perturbed positions and
lattice remainder for the energy call and puts the caller's buffers back
afterwards, so ``∂/∂pos`` (and, straining positions, cell and remainders
alike, ``∂/∂ε``) flows through the geometry the energy actually reads. The
remainder is ``pairs.shifts`` when the list stored it, otherwise
``pair_diff − (pos[t] − pos[s])``, a constant with respect to the positions
either way. An energy that recomputes its geometry from ``atoms.pos`` itself
(bonded terms, an open-system encoder) is served by the same leaf.

**Compile boundary.** The op is what :class:`molix.compile.Compiler` is
handed: leaves, strain, geometry, energy, ``autograd.grad`` and the writes are
one region. In inference (``torch.no_grad()``, every output detached) that
region is one dynamo graph with no break. In training the outputs come back
attached: a leaf made inside the region then leaks through them and stock
torch breaks at one site, :func:`~molix.autograd.variables.leaf`
(pytorch/pytorch#196653 lifts it), whereas a column marked before the region
is a graph input and the region stays one graph — which is what
:meth:`Derivative.mark_leaves` is for. It is also why every leaf is made
*before* the ``enable_grad`` block: a break inside it would leave the
``autograd.grad`` outside the compiled region, and the training backward
through the compiled energy would be a double backward.
Nothing in the body branches on a tensor value, reads a Python scalar out of a
tensor, or wraps the writes in ``try``/``finally`` — an exception while
tracing is a compile error, not a runtime state to restore from. What the
energy needs done eagerly before a trace (an element-table check, a k-space
grid sized off the lattice) is the potential's precondition: run one eager
forward, then compile the composed op.

Units are the energy's: eV and Å on the MLIP side (forces eV/Å, virial eV),
the ForceField's convention on the classical side. The op only differentiates
what the energy writes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast

import torch
import torch.nn as nn
from tensordict import TensorDictBase

from molix.autograd.variables import CELL_KEY, Key, _Variable, get_column, variable
from molix.schema import ENERGY_KEY, POS_KEY, STRAIN_KEY

__all__ = ["Derivative"]

PAIR_INDEX_KEY: Key = ("pairs", "pair_index")
PAIR_DIFF_KEY: Key = ("pairs", "pair_diff")
PAIR_DIST_KEY: Key = ("pairs", "pair_dist")
SHIFTS_KEY: Key = ("pairs", "shifts")

#: A key as a caller may spell it: a dotted string or a tuple of names.
KeySpec = str | Sequence[str]


def _key(spec: KeySpec) -> Key:
    """Normalise ``"atoms.pos"`` / ``("atoms", "pos")`` to the tuple form."""
    key = tuple(spec.split(".")) if isinstance(spec, str) else tuple(spec)
    if len(key) < 2 or not all(isinstance(name, str) and name for name in key):
        raise ValueError(
            f"a TensorFrame key names a namespace and a column, e.g. 'atoms.pos'; got {spec!r}"
        )
    return key


class Derivative(nn.Module):
    """``∂(of)/∂(wrt)`` written onto the frame, one energy forward, one ``autograd.grad``.

    Args:
        energy: ``energy(frame) -> frame``, writing ``of`` (per-graph ``(B,)``
            or a scalar) and optionally ``atoms.energy``. An
            :class:`~torch.nn.Module` is registered as the submodule
            ``energy``; any other callable (a bound method such as
            ``pot.energy``) is held as is. Called exactly as handed.
        of: The quantity differentiated, as a TensorFrame key. Defaults to
            ``graphs.energy``.
        wrt: The variables, in order — TensorFrame keys spelled as dotted
            strings or tuples. ``graphs.strain`` must follow ``atoms.pos``.
            Empty (the default) means "energy only": the op runs ``energy``
            and returns the frame, nothing else.
        outputs: Where each derivative lands, one entry per ``wrt`` key;
            ``None`` entries (or ``None`` altogether) take the key table of
            :mod:`molix.autograd.variables`. The sign is the variable's, not
            the key's: ``atoms.pos`` always writes ``-∂E/∂pos``.

    Attributes:
        of: The normalised ``of`` key.
        wrt: The normalised variable keys, in order.
        outputs: The normalised output keys, one per variable.
        energy: What was handed in — the registered submodule, or the callable.

    Raises:
        TypeError: ``energy`` is not callable.
        ValueError: A key is not ``namespace.column``; a variable repeats;
            ``graphs.strain`` is not preceded by ``atoms.pos``; ``outputs``
            does not pair with ``wrt``.

    The op makes a leaf for every variable unless the column already is a live
    leaf — then that leaf is the variable and stays in the frame, the
    caller's. A caller that wants the whole training step as one graph on
    stock torch marks the columns before the traced region with
    :meth:`mark_leaves`.

    Examples:
        >>> op = Derivative(pot.energy, wrt=("atoms.pos",))   # pot.energy writes graphs.energy
        >>> frame = op(frame)                                 # + atoms.forces (N, 3)
        >>> npt = Derivative(pot.energy, wrt=("atoms.pos", "graphs.strain"))
        >>> frame = npt(frame)                             # + graphs.virial (1, 3, 3)
        >>> compiled = Compiler(fullgraph=True)(npt)
        >>> frame = compiled(npt.mark_leaves(frame))       # training: one graph on stock torch
    """

    _call: Callable[..., TensorDictBase]

    def __init__(
        self,
        energy: Callable[[TensorDictBase], TensorDictBase],
        *,
        of: KeySpec = ENERGY_KEY,
        wrt: Sequence[KeySpec] = (),
        outputs: Sequence[KeySpec | None] | None = None,
    ) -> None:
        super().__init__()
        if not callable(energy):
            raise TypeError(
                "energy must be callable — energy(frame) -> frame, e.g. pot.energy; "
                f"got {type(energy).__name__}"
            )
        # ``nn.Module.__setattr__`` registers a module and stores anything else.
        self.energy = energy
        # Stored past ``nn.Module.__setattr__`` so a module energy is
        # registered once, under ``energy`` only.
        object.__setattr__(self, "_call", energy)
        self.of: Key = _key(of)
        keys = tuple(_key(spec) for spec in wrt)
        if len(set(keys)) != len(keys):
            raise ValueError(f"each variable may appear once in wrt; got {keys}")
        if STRAIN_KEY in keys and (
            POS_KEY not in keys or keys.index(POS_KEY) > keys.index(STRAIN_KEY)
        ):
            raise ValueError(
                "graphs.strain deforms the positions, so atoms.pos must precede it in wrt: "
                "wrt=('atoms.pos', 'graphs.strain')"
            )
        targets: Sequence[KeySpec | None] = (None,) * len(keys) if outputs is None else outputs
        if len(targets) != len(keys):
            raise ValueError(
                f"outputs must name one key per wrt variable ({len(keys)}); got {len(targets)}"
            )
        self._variables: tuple[_Variable, ...] = tuple(
            variable(key, None if target is None else _key(target))
            for key, target in zip(keys, targets, strict=True)
        )
        self.wrt: tuple[Key, ...] = keys
        self.outputs: tuple[Key, ...] = tuple(v.output for v in self._variables)

    def mark_leaves(self, frame: TensorDictBase) -> TensorDictBase:
        """Mark every variable's column as a fresh live leaf, in place — the eager prologue.

        Runs each variable's host-side preparation (``graphs.strain`` is
        created as a ``(1, 3, 3)`` zero column when absent, the cell is
        checked) and replaces the column by ``column.detach().requires_grad_(True)``,
        cast and shaped as the variable is differentiated (the strain column
        stays ``(1, 3, 3)``), so a following :meth:`forward` reuses it instead
        of making its own. Made *outside* a compiled region, that leaf is a
        graph input, which dynamo accepts where it refuses an in-region leaf
        escaping through the attached outputs (stock torch 2.15,
        pytorch/pytorch#196653): the training step is then one graph with zero
        breaks. Fresh objects on every call, so a ``.grad`` never accumulates
        across steps. Harmless under ``torch.no_grad()``, where nothing needs
        it; a no-op for ``wrt=()``.

        Args:
            frame: The batch :meth:`forward` is handed next.

        Returns:
            The same ``frame``, its variable columns now live leaves.

        Raises:
            ValueError: As :meth:`forward` — a strain variable without
                ``graphs.cell``, or on a batch of more than one graph.
        """
        if not self._variables:
            return frame
        for var in self._variables:
            var.prepare(frame, None)
        pos = cast(torch.Tensor, frame[POS_KEY])
        for var in self._variables:
            frame[var.key] = var.mark(cast(torch.Tensor, frame[var.key]), pos)
        return frame

    def _leaves(self, buffers: tuple[torch.Tensor, ...], pos: torch.Tensor) -> list[torch.Tensor]:
        """One ``requires_grad`` leaf per variable, before grad mode is forced on.

        The column itself when the caller marked it (:meth:`mark_leaves`),
        made here otherwise. A method of its own, called from :meth:`forward`
        as one plain statement, on purpose: on stock torch the leak break of an
        unmarked column lands at :func:`~molix.autograd.variables.leaf`, and
        dynamo can only split :meth:`forward` cleanly at a plain call site.
        Made inside the ``enable_grad`` block, or in a loop of :meth:`forward`
        itself, the break drops the rest of ``forward`` out of the traced
        region — the ``autograd.grad`` then runs eagerly over a compiled
        energy, and the training backward is a double backward
        (``aot_autograd does not currently support double backward``); a
        comprehension in ``forward`` is reported as a second break of its own.

        Args:
            buffers: The caller's columns, one per variable, in ``wrt`` order.
            pos: The caller's positions — dtype and device for a leaf that is
                not a position column.

        Returns:
            The leaves, in ``wrt`` order.
        """
        leaves: list[torch.Tensor] = []
        for var, buffer in zip(self._variables, buffers, strict=True):
            leaves.append(var.leaf(buffer, pos))
        return leaves

    def forward(self, frame: TensorDictBase, **energy_kwargs: Any) -> TensorDictBase:
        """Derive and write in place.

        Args:
            frame: Post-collate batch (or its :class:`~molix.data.TensorFrame`
                view) carrying ``atoms.pos`` ``(N, 3)`` Å, whatever geometry the
                energy reads, and ``graphs.cell`` ``(3, 3)`` or ``(1, 3, 3)`` Å
                when ``graphs.strain`` is a variable.
            **energy_kwargs: Handed to the energy unchanged —
                ``energy(frame, **energy_kwargs)``. A training step that has
                already predicted a model's parameter tables on this batch
                passes them here (``op(frame, tables=tables)``) so the energy
                does not predict them again; the derivative is still only
                with respect to ``wrt``, through whatever graph the kwargs
                carry.

        Returns:
            The same ``frame``, with ``of`` written by the energy and one
            output per variable: ``atoms.forces`` ``(N, 3)``, ``graphs.virial``
            ``(1, 3, 3)``, ``<ns>.<name>_grad`` shaped like its column. ``of``
            and ``atoms.energy`` are detached iff grad was disabled at entry.

        Raises:
            ValueError: ``graphs.strain`` is a variable and ``graphs.cell`` is
                absent, or the batch holds more than one graph — one strain
                leaf is one lattice.
        """
        if not self._variables:
            return self._call(frame, **energy_kwargs)
        # ``create_graph`` is the caller's grad mode, resolved before grad is
        # forced on for the derivative.
        create_graph = torch.is_grad_enabled()
        for var in self._variables:
            var.prepare(frame, None)
        buffers = tuple(cast(torch.Tensor, frame[var.key]) for var in self._variables)

        pos_buf = cast(torch.Tensor, frame[POS_KEY])
        # Force-only ``wrt`` must not probe ``graphs``: ``dict.get`` on that
        # namespace still makes dynamo guard *key order* (GH200 2367461:
        # ``graphs.keys()[1] == 'energy'`` vs ``'task'``). Cell is only
        # for strain.
        if any(var.key == STRAIN_KEY for var in self._variables):
            cell_buf = get_column(frame, CELL_KEY)
            has_cell = cell_buf is not None
        else:
            cell_buf = None
            has_cell = False
        diff_col = get_column(frame, PAIR_DIFF_KEY)
        has_diff = diff_col is not None
        has_dist = has_diff and get_column(frame, PAIR_DIST_KEY) is not None
        has_shifts = has_diff and get_column(frame, SHIFTS_KEY) is not None
        source = target = None
        diff_buf = dist_buf = shifts_buf = shifts = None
        if has_diff:
            pair_index = cast(torch.Tensor, frame[PAIR_INDEX_KEY])
            source, target = pair_index[:, 0], pair_index[:, 1]
            diff_buf = cast(torch.Tensor, frame[PAIR_DIFF_KEY])
            if has_dist:
                dist_buf = cast(torch.Tensor, frame[PAIR_DIST_KEY])
            if has_shifts:
                shifts_buf = cast(torch.Tensor, frame[SHIFTS_KEY])
                shifts = shifts_buf.detach()
            else:
                # The remainder the list imaged with: a constant w.r.t. pos,
                # whatever history the cached buffer happens to carry.
                shifts = (diff_buf - (pos_buf[target] - pos_buf[source])).detach()

        leaves = self._leaves(buffers, pos_buf)
        with torch.enable_grad():
            pos, cell = pos_buf, cell_buf
            for var, variable_leaf in zip(self._variables, leaves, strict=True):
                pos, cell, shifts = var.apply(frame, variable_leaf, pos, cell, shifts)
            frame[POS_KEY] = pos
            if has_cell:
                frame[CELL_KEY] = cell
            if has_diff:
                diff = pos[target] - pos[source] + shifts
                frame[PAIR_DIFF_KEY] = diff
                if has_dist:
                    frame[PAIR_DIST_KEY] = diff.norm(dim=-1)
                if has_shifts:
                    frame[SHIFTS_KEY] = shifts
            before = frozenset(frame.keys(include_nested=True, leaves_only=True))
            frame = self._call(frame, **energy_kwargs)
            energy = cast(torch.Tensor, frame[self.of])
            grads = torch.autograd.grad(
                energy.sum(), leaves, create_graph=create_graph, allow_unused=True
            )

        for var, variable_leaf, grad, buffer in zip(
            self._variables, leaves, grads, buffers, strict=True
        ):
            var.write(frame, torch.zeros_like(variable_leaf) if grad is None else grad)
            frame[var.key] = buffer
        # The caller's own objects back: a leaf made here leaves no trace, a
        # leaf the caller marked stays theirs.
        frame[POS_KEY] = pos_buf
        if has_cell:
            frame[CELL_KEY] = cell_buf
        if has_diff:
            frame[PAIR_DIFF_KEY] = diff_buf
            if has_dist:
                frame[PAIR_DIST_KEY] = dist_buf
            if has_shifts:
                frame[SHIFTS_KEY] = shifts_buf
        if not create_graph:
            # The single backward consumed the energy graph: every column the
            # energy attached to it goes back detached (one left attached is a
            # dynamo break and a later backward's error). The variables were
            # restored to the caller's own buffers above and are not touched.
            written = [
                key
                for key in frame.keys(include_nested=True, leaves_only=True)
                if key not in before
            ]
            for key in (self.of, *written):
                frame[key] = cast(torch.Tensor, frame[key]).detach()
        return frame
