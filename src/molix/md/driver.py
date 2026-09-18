"""``MD`` — compose a force field and an integrator, then run a system.

``MD`` owns only the **MD-side precision** (trajectory state). Everything
else is a component the caller builds and attaches:

    md = (
        MD(torch.float64)
        .set_forcefield(ff)
        .set_neighbors(nl)
        .set_integrator(VelocityVerletIntegrator(dt))
        .set_hooks([
            VelocityInitHook(MaxwellBoltzmann(300, seed=0)),
            TrajectoryHook(...),
        ])
        .to(device)
    )
    md.run(system, n_steps)

``system`` is a TensorDict. Mass, positions and velocities live under
``atoms`` (``molix.schema.MASS_KEY`` / ``POS_KEY`` / ``VEL_KEY``). The
integrator is constructed with ``dt`` only.

Two policies are ``MD``'s own and are set on it, not on a component:
:meth:`MD.set_compiler` (compile the derivative op composed over the potential
— once per run, after one eager evaluation) and ``MD(dtype,
potential_dtype=…)`` (evaluate the potential at a second precision while the
trajectory stays at ``dtype``).

Neighbour-list cadence is not ``MD``'s. Configure it on
:class:`~molix.md.neighbors.NeighborList` and mount with
:meth:`MD.set_neighbors`. For a compiled rollout with a frozen edge set,
omit :meth:`MD.set_neighbors` entirely — with no list mounted, ``pairs``
is never rebuilt and stays whatever was bound into ``system`` for the
whole run.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from tensordict import TensorDict

from molix.autograd import Derivative
from molix.frame import TensorFrame, as_tensor_frame
from molix.md.barostat import BerendsenBarostat
from molix.md.forcefield import ForceField, MDForceField
from molix.md.integrators import Integrator
from molix.md.neighbors import NeighborList
from molix.md.runner import MDHook, MDRunner
from molix.md.types import MDState
from molix.schema import (
    ATOMIC_ENERGY_KEY,
    ENERGY_KEY,
    FORCES_KEY,
    MASS_KEY,
    POS_KEY,
    STRAIN_KEY,
    VEL_KEY,
    VIRIAL_KEY,
    CellBound,
)
from molix.units import preset_energy_to_md, units

_KB_MD = (1.0 * units.k_B).to("amu * angstrom ** 2 / femtosecond ** 2 / kelvin").magnitude


class _TensorDictForce:
    """Private ``pos -> dict`` over ``to_potentials()``, and the precision seam.

    MD never owns an energy graph: the trajectory is stepped under
    ``torch.no_grad()``, and the one ``energy.detach()`` at this seam
    drops the autograd tape so the next step does not keep Polar's
    saved tensors alive. Forces and virial come from one
    :class:`molix.autograd.Derivative` composed over the pots at
    :meth:`bind_work` — ``wrt=("atoms.pos",)``, plus ``"graphs.strain"``
    under a barostat — and compiled there when a compiler is set.

    It is also the **dtype boundary** when :class:`MD` was built with a
    ``potential_dtype``: positions (and ``graphs.cell``) go in at the
    potential's dtype and energy / forces / virial come back at the MD dtype,
    so the integrator state never picks the potential's precision up. With
    ``pot_dtype=None`` no cast is performed anywhere on this path.

    Every one of those casts is made **off to the side**, never written back
    onto the caller's batch: the batch is the MD state's own storage
    (``MDState.cell`` *is* ``graphs.cell``, and the end-of-run writeback lands
    in it), so leaving a reduced-precision copy there would quietly move the
    trajectory to the potential's precision — the exact confusion the split
    exists to avoid. The cell is held as a private cast copy
    (:meth:`_pot_cell`) refreshed from the live storage each call and swapped
    out again before this adapter returns. The edge columns are **not** this
    adapter's per-step business: a mounted :class:`~molix.md.neighbors.NeighborList`
    writes them at its ``geometry_dtype`` (:class:`MD` sets it to the
    potential's), and edges frozen on the batch with no list mounted are cast
    **once** at :meth:`bind_work` and put back by :meth:`unbind_work`.

    **Static shapes.** The op is handed the batch as the list bound it — the
    *whole* capacity buffer, dead tail included, ``batch_size=[capacity]`` —
    on every call. Nothing here slices to the live prefix: a data-dependent
    ``E`` would recompile the op on each rebuild and forbid CUDA-graph capture,
    and the dead rows price exactly nothing (their shift is the dead-edge
    displacement, their ``pos[0] - pos[0]`` cancels). Under a CUDA-graph build
    (``compiler.cuda_graphs``, see :class:`molix.compile.Compiler`) the call
    also honours the replay contract: ``torch.compiler.cudagraph_mark_step_begin()``
    before the op, and the outputs it leaves on the batch cloned out of the
    graph's static pool, which the next replay overwrites.

    Args:
        pots: The object ``ForceField.to_potentials()`` returned.
        scale: Energy conversion from the ForceField's unit convention to MD
            units (amu·Å²/fs²).
        md_dtype: Dtype energy / forces / virial are returned at — :attr:`MD.dtype`.
            ``None`` returns whatever the potential produced.
        pot_dtype: Dtype the potential is evaluated at — :attr:`MD.potential_dtype`.
            ``None`` (the default) disables the whole cast path.
        compiler: Applied to the composed op at :meth:`bind_work`, after one
            eager evaluation; ``None`` runs eager. A compiler exposing a true
            ``cuda_graphs`` attribute gets the CUDA-graph step contract.
    """

    def __init__(
        self,
        pots: object,
        scale: float,
        *,
        md_dtype: torch.dtype | None = None,
        pot_dtype: torch.dtype | None = None,
        compiler: Callable[[Callable[..., Any]], Callable[..., Any]] | None = None,
    ) -> None:
        self._pots = pots
        self._scale = float(scale)
        self._md_dtype = md_dtype
        self._pot_dtype = pot_dtype
        self._compiler = compiler
        # Read once: the CUDA-graph replay contract (step mark + cloned
        # outputs) is applied iff the build says it records graphs.
        self._cuda_graphs = bool(getattr(compiler, "cuda_graphs", False))
        #: The composed derivative op of the current run — eager or compiled —
        #: or ``None`` before :meth:`bind_work`.
        self._op: Callable[[TensorDict], TensorDict] | None = None
        # The caller's batch and its typed view: the same storage under two
        # names, because the two have different jobs. ``_work`` is what the
        # potential and the neighbour list are handed — the container the caller
        # built, so a potential that refuses it names the container the caller
        # actually passed. ``_frame`` is what *this* adapter reads and swaps
        # through (CLAUDE.md § TensorDict Contract).
        self._work: TensorDict | None = None
        self._frame: TensorFrame | None = None
        self._cell_view: torch.Tensor | None = None
        # The potential-dtype copy of the cell and the batch tensor it stands in
        # for during a call. Both ``None`` without a ``pot_dtype`` — then the
        # potential is handed the batch's own storage and nothing is swapped.
        self._cell_pot: torch.Tensor | None = None
        self._cell_md: torch.Tensor | None = None
        # The caller's own ``pairs`` namespace while a once-per-run cast copy
        # stands in for it (no list mounted, ``pot_dtype`` set); ``None`` otherwise.
        self._edges_md: TensorDict | None = None
        self.want_virial = False

    @property
    def work(self) -> TensorDict | None:
        """The bound working batch — the caller's container; ``None`` before :meth:`bind_work`."""
        return self._work

    def bind_work(self, work: TensorDict) -> None:
        """Bind the working ``TensorDict`` for the run and compose the op.

        Called once per :meth:`MD.run`, never once per step. Stores ``work``,
        re-resolves the live cell via :meth:`rebind_cell`, then composes the
        one derivative op of the run — ``Derivative(pots,
        wrt=("atoms.pos",))``, with ``"graphs.strain"`` added when a barostat
        wants the virial (``want_virial``). With a compiler set, the op is
        evaluated **once eagerly** on ``work`` first — the topology bind, the
        element-table check and any k-space grid sizing are host-side work a
        trace cannot do — and the compiler is applied to it afterwards.

        The batch is wrapped **once here** as its typed view
        (:func:`~molix.data.as_tensor_frame`, CLAUDE.md § TensorDict Contract)
        and every read and swap below addresses that view — the wrap is a view,
        so the caller's batch is the storage throughout and one rollout pays
        one wrap rather than one per step. This is the **only** place the two
        fields are assigned: ``_work`` and ``_frame`` are one storage under two
        names, and a path that set one without the other would desynchronise
        them silently — so no other method assigns either, and readers outside
        this class reach the batch through :attr:`work`.

        Args:
            work: The system ``TensorDict`` that positions are written into
                and potentials are evaluated on for the remainder of the run.
        """
        self._work = work
        self._frame = as_tensor_frame(work)
        self.rebind_cell()
        self._bind_edges(self._frame)
        wrt = (POS_KEY, STRAIN_KEY) if self.want_virial else (POS_KEY,)
        self._op = Derivative(self._pots, wrt=wrt)  # type: ignore[arg-type]
        if self._compiler is not None:
            with torch.no_grad():
                self(work[POS_KEY])
            self._op = self._compiler(self._op)

    def _bind_edges(self, frame: TensorFrame) -> None:
        """Once per run: frozen edge columns at ``pot_dtype``, if any need casting.

        With a neighbour list mounted the columns already arrive at the
        potential's precision (the list's ``geometry_dtype``) and nothing
        happens here — the list-bound namespace must stay the object it is, or
        the in-place rebuilds would become invisible. With no list the edges are
        the caller's frozen buffers, so a cast copy stands in for the run and
        :meth:`unbind_work` puts the caller's namespace back.
        """
        dtype = self._pot_dtype
        if dtype is None or "pairs" not in frame.keys():
            return
        pairs = frame["pairs"]
        columns = {key: pairs[key] for key in pairs.keys()}
        if all(not c.is_floating_point() or c.dtype == dtype for c in columns.values()):
            return
        self._edges_md = pairs
        frame["pairs"] = TensorDict(
            {k: c.to(dtype) if c.is_floating_point() else c for k, c in columns.items()},
            batch_size=pairs.batch_size,
        )

    def unbind_work(self) -> None:
        """Put back what :meth:`bind_work` swapped onto the batch for the run.

        Called by :meth:`MD.run` when the run ends, however it ends. Today that
        is the frozen-edges cast copy (:meth:`_bind_edges`); the batch is then
        exactly the caller's again.
        """
        frame = self._frame
        pairs = self._edges_md
        if frame is not None and pairs is not None:
            frame["pairs"] = pairs
        self._edges_md = None

    def rebind_cell(self, new_cell: torch.Tensor | None = None) -> None:
        """Re-resolve ``graphs.cell``, optionally write into it, then hand it to the pots.

        The **single owner of cell-layout resolution on the path to the
        potential**: both :meth:`bind_work` (once, from :meth:`MD.run`) and
        :meth:`MD._sync_cell` (once per barostat update) go through here, so
        the ``(1, 3, 3)`` vs ``(3, 3)`` decision and the
        :class:`~molix.schema.CellBound` dispatch exist in exactly one place.
        (``MDRunner.run`` resolves the same layout for the integrator's
        :class:`~molix.md.types.MDState`; that path does not reach the pots and
        is deliberately left alone.)

        The lookup is redone on **every** call and the previous view is never
        reused. The derivative op's strain variable replaces ``graphs.cell``
        with a graph tensor for the duration of the pass and restores the
        buffer only on the no-exception path, so a cached view can point at
        orphaned storage — copying a barostat update into it would land
        nowhere and leave the potential on the stale lattice.

        Under a ``pot_dtype`` what reaches the pots is the private cast copy
        (:meth:`_pot_cell`), not the batch's own tensor: the barostat update is
        copied into the MD-dtype storage — which is ``MDState.cell`` — and the
        copy is refreshed from it, so the trajectory's lattice keeps its
        precision and the potential is still handed one lattice in one dtype.

        Both of those hold only **between** evaluations, which is where the two
        callers sit. During one, :meth:`_cast_in` has the potential-dtype copy
        standing under ``graphs.cell`` and the MD-dtype tensor parked on
        :attr:`_cell_md`, so a call landing here would resolve the fp32 copy as
        if it were the trajectory's lattice and copy the barostat's update into
        a buffer :meth:`_restore_cell` is about to discard — the update silently
        lost, the potential left on the pre-barostat cell. There is no correct
        thing to do in that window, so it is asserted against rather than
        handled.

        Args:
            new_cell: ``(3, 3)`` cell to copy *into* the live storage, or
                ``None`` to only re-resolve and re-dispatch.

        Raises:
            AssertionError: Called while a ``pot_dtype`` cell swap is in flight
                (see above) — a caller reaching the adapter from inside an
                evaluation, which no in-tree path does.
        """
        assert self._cell_md is None, (
            "rebind_cell during an in-flight potential-dtype cell swap: graphs.cell is "
            "currently the pot_dtype copy (_cast_in), so this would resolve that copy as "
            "the live lattice and lose the update into a buffer _restore_cell discards. "
            "Rebind between evaluations, not inside one"
        )
        frame = self._frame
        cell = None if frame is None else self._live_cell(frame)
        self._cell_view = None if cell is None else (cell[0] if cell.dim() == 3 else cell)
        cell_view = self._cell_view
        if new_cell is not None and cell_view is not None:
            cell_view.copy_(new_cell)
        bound = self._pot_cell(cell_view)
        if bound is not None and isinstance(self._pots, CellBound):
            self._pots.bind_cell(bound)

    @staticmethod
    def _live_cell(frame: TensorFrame) -> torch.Tensor | None:
        """``graphs.cell`` as it stands right now, or ``None`` if there is none.

        The one read of that key on this path, so the "never cache a cell view"
        rule :meth:`rebind_cell` states has one place to hold: both that method
        and :meth:`_cast_in` come through here rather than indexing the batch
        themselves. Returns the batch's ``(3, 3)`` or ``(1, 3, 3)`` storage
        untouched — resolving the layout is the caller's step, because the two
        callers want it at different depths.
        """
        if ("graphs", "cell") not in frame.keys(include_nested=True):
            return None
        cell = frame["graphs", "cell"]
        return cell if isinstance(cell, torch.Tensor) else None

    def _pot_cell(self, cell_view: torch.Tensor | None) -> torch.Tensor | None:
        """The lattice at the potential's precision — the batch's own, or a copy.

        Without a ``pot_dtype`` (or when the batch already holds that dtype)
        this is the batch's tensor itself, so the no-seam path stays a plain
        pass-through and an mlip that retains the reference keeps seeing the
        barostat's in-place updates.

        Otherwise it is a copy this adapter owns, and it is **the same storage
        every call**: refreshed with ``copy_`` from the live cell rather than
        rebuilt, so a term that kept the reference at its ``bind_cell`` is
        current without a re-bind, exactly as it would be with the batch's own
        tensor. Writing the cast back onto the batch instead — as this seam
        once did — forks ``graphs.cell`` away from ``MDState.cell``, drops the
        trajectory's lattice to the potential's precision and leaves the
        end-of-run writeback landing in the wrong dtype.

        Args:
            cell_view: The live ``(3, 3)`` cell resolved off the batch, or
                ``None`` when the system carries no cell.

        Returns:
            The ``(3, 3)`` lattice to hand the potential, or ``None``.
        """
        dtype = self._pot_dtype
        if cell_view is None or dtype is None or cell_view.dtype == dtype:
            self._cell_pot = None
            return cell_view
        copy = self._cell_pot
        if copy is None or copy.shape != cell_view.shape or copy.device != cell_view.device:
            copy = torch.empty_like(cell_view, dtype=dtype)
            self._cell_pot = copy
        copy.copy_(cell_view)
        return copy

    def to_pots(self, device: torch.device) -> None:
        to_fn = getattr(self._pots, "to", None)
        if callable(to_fn):
            moved = to_fn(device)
            if moved is not None:
                self._pots = moved

    def __call__(self, pos: torch.Tensor) -> dict[str, torch.Tensor]:
        """Evaluate the potential at ``pos`` and return MD-unit results.

        Args:
            pos: Positions ``(N, 3)`` in Å at the MD dtype; cast to
                ``pot_dtype`` on the way in when the precision seam is active.

        Returns:
            ``{"energy", "forces"}`` — energy amu·Å²/fs², forces amu·Å/fs² —
            plus ``"virial"`` ``(3, 3)`` in amu·Å²/fs² on the barostat path.
            The batch is left with ``pos`` — the MD-dtype tensor handed in —
            under ``atoms.pos``: under a ``pot_dtype`` the cast copy stands
            there only for the duration of the call, like the cell, so the
            integrator state built off the batch afterwards (``MDRunner.run``
            after the eager warm-up in :meth:`bind_work`) reads the trajectory's
            precision, never the potential's.

        Raises:
            RuntimeError: :meth:`bind_work` has not run, so there is no batch to
                evaluate on.
        """
        work = self._work
        frame = self._frame
        op = self._op
        if work is None or frame is None or op is None:
            raise RuntimeError("bind the working TensorDict before evaluating forces")
        # Read and swapped through the view, evaluated on the caller's own
        # container: the two are one storage, so the potential is handed the
        # cast geometry either way — and a potential that refuses the batch
        # names the container the caller built rather than a wrapper this
        # adapter made. The edges go in exactly as the list bound them.
        frame[POS_KEY] = self._cast_in(frame, pos)
        if self._cuda_graphs:
            # Attribute lookup at call time, on purpose: the previous replay's
            # outputs are dead from here, and a test can observe the mark.
            torch.compiler.cudagraph_mark_step_begin()
        try:
            op(work)
            if self._cuda_graphs:
                self._own_outputs(work)
            out = self._pack(work[ENERGY_KEY], work[FORCES_KEY])
            if self.want_virial:
                out["virial"] = self._to_md(work[VIRIAL_KEY][0] * self._scale)
            return out
        finally:
            self._restore_cell(frame)
            if self._pot_dtype is not None:
                frame[POS_KEY] = pos

    @staticmethod
    def _own_outputs(work: TensorDict) -> None:
        """Clone what the op left on the batch out of the CUDA graph's static pool.

        A replayed graph writes its outputs into memory the *next* replay
        reuses, and reading such a tensor afterwards is an error torch raises
        (or, worse, a stale value a hook read too late). The packed results
        handed to the integrator are fresh tensors already (a multiply); this
        gives the batch-side keys the same ownership.
        """
        nested = work.keys(include_nested=True)
        for key in (ENERGY_KEY, FORCES_KEY, VIRIAL_KEY, ATOMIC_ENERGY_KEY):
            if key in nested:
                work[key] = work[key].clone()

    def _cast_in(self, frame: TensorFrame, pos: torch.Tensor) -> torch.Tensor:
        """Put the potential's geometry on ``pot_dtype`` — positions and the cell.

        Returns ``pos`` untouched when no ``potential_dtype`` was asked for, so
        the single-precision path is byte-identical to having no seam at all.

        The cell is refreshed into the private cast copy (:meth:`_pot_cell`) and
        that copy is shown under ``graphs.cell`` **for the duration of the
        call** — the terms read the lattice off the batch, not off their
        ``bind_cell``, so it has to be there — and swapped back out by
        :meth:`_restore_cell` before this adapter returns. Never written back:
        ``graphs.cell`` is ``MDState.cell``'s own storage, so replacing it
        forks the trajectory's lattice onto the potential's precision and the
        end-of-run writeback lands in the wrong dtype. Re-reading the live
        tensor every call is also what keeps a barostat update visible, since
        :meth:`MD._sync_cell` copies *into* that storage.

        The edge columns are the other half of the geometry and are the
        neighbour list's to write at this precision (its ``geometry_dtype``) —
        or, with no list, cast once at :meth:`bind_work`. The parameter tables
        are :class:`MD`'s half (``potential_dtype``).
        """
        dtype = self._pot_dtype
        if dtype is None:
            return pos
        cell = self._live_cell(frame)
        if cell is not None:
            batched = cell.dim() == 3
            pot_cell = self._pot_cell(cell[0] if batched else cell)
            if pot_cell is not None and pot_cell.dtype != cell.dtype:
                self._cell_md = cell
                frame["graphs", "cell"] = pot_cell.unsqueeze(0) if batched else pot_cell
        return pos.to(dtype)

    def _restore_cell(self, frame: TensorFrame) -> None:
        """Put the MD-dtype cell back under ``graphs.cell`` after an evaluation."""
        cell = self._cell_md
        if cell is None:
            return
        frame["graphs", "cell"] = cell
        self._cell_md = None

    def _pack(self, energy: torch.Tensor, forces: torch.Tensor) -> dict[str, torch.Tensor]:
        scale = self._scale
        return {
            # Detach is the MD/training seam: MDState must not keep the tape.
            "energy": self._to_md(energy.detach().sum() * scale),
            "forces": self._to_md(forces * scale),
        }

    def _to_md(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return a potential-side result at the MD dtype (no-op without the seam)."""
        dtype = self._md_dtype
        if self._pot_dtype is None or dtype is None or tensor.dtype == dtype:
            return tensor
        return tensor.to(dtype)


class MaxwellBoltzmann:
    """Maxwell-Boltzmann velocity distribution. Not a hook — draw only.

    Attach it with :class:`~molix.md.runner.VelocityInitHook`::

        md.set_hooks([VelocityInitHook(MaxwellBoltzmann(300, seed=0))])
    """

    def __init__(
        self,
        temperature: float,
        *,
        seed: int = 0,
        remove_com: bool = True,
    ) -> None:
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.remove_com = bool(remove_com)

    def velocities(self, pos: torch.Tensor, mass: torch.Tensor | float) -> torch.Tensor:
        """Draw ``(N, 3)`` velocities in Å/fs at :attr:`temperature`."""
        n = int(pos.shape[0])
        mass_t = torch.as_tensor(mass, dtype=torch.float64, device=pos.device)
        if mass_t.dim() == 0:
            mass_t = mass_t.expand(n)
        elif int(mass_t.shape[0]) != n:
            raise ValueError(f"atoms.mass length {mass_t.shape[0]} disagrees with n_atoms={n}")
        mass_col = mass_t.reshape(-1, 1)
        generator = torch.Generator().manual_seed(self.seed)
        noise = torch.randn((n, 3), generator=generator, dtype=torch.float64).to(device=pos.device)
        vel = noise * torch.sqrt(_KB_MD * self.temperature / mass_col)
        if self.remove_com:
            vel = vel - (mass_col * vel).sum(0) / mass_col.sum()
        return vel


class MD:
    """Compose ``ForceField`` + ``Integrator`` at a chosen MD-state dtype.

    **Two precisions, deliberately separable.** ``dtype`` is the trajectory's:
    positions, velocities and masses, and everything the integrator does with
    them. ``potential_dtype`` is the force field's: its parameter tables are
    moved there and it is handed geometry there, while its energy and forces
    come back at ``dtype``. Equal (the default — ``potential_dtype=None``) is
    one precision everywhere and no cast on the force path at all.

    The split exists for the quantization-as-thermal-noise study
    (:mod:`molix.quant`), whose operating point is an fp64 integrator over an
    fp32 potential: reducing the *potential's* precision injects a
    force perturbation whose effect on the trajectory is the object of study,
    and it can only be read against an integrator that is not itself the noise
    source. Running the whole loop at fp32 instead confounds the two.

    Args:
        dtype: MD-side precision — trajectory state after :meth:`run` casts
            ``atoms.pos`` / ``atoms.vel`` / ``atoms.mass``. ``None`` keeps the
            system tensors' dtype. Never calls ``.to`` on the molpy container.
        potential_dtype: Precision the potential is *evaluated* at. ``None``
            (default) leaves the potential exactly as ``to_potentials()`` built
            it and adds no cast anywhere.

            Otherwise the whole evaluation moves, in three places, because a
            field's numbers live in three: :meth:`_wire` assembles it from
            ``force.to_dtype(potential_dtype)``, whose parameter tables — the
            per-atom ``epsilon`` / ``sigma`` / ``charge`` a ``Potentials`` reads live
            off its container on every call — are restated there; it then calls
            ``.to(potential_dtype)`` on the assembled module for what the
            module itself owns (the bonded buffers, an mlip's weights); the
            force adapter casts the positions and ``graphs.cell`` it hands
            over per step and casts energy / forces / virial back to ``dtype``
            on the way out; and a mounted neighbour list writes the edge
            geometry at this precision (its ``geometry_dtype``, set by
            :meth:`set_neighbors`), refreshing its cast shifts in place on each
            rebuild rather than per step. Partial casts are not an option:
            molpot's Coulomb kernels compare the dtype of the charges and the
            pair distances against the positions' and raise rather than
            promote.

            The *state* keeps ``dtype`` throughout, and so do the neighbour
            list's own buffers: it owns ``pairs``, and validates the positions
            it is updated with against its own build precision.

    Raises:
        TypeError: ``dtype`` or ``potential_dtype`` is not a ``torch.dtype`` —
            most often a force field passed positionally, which
            :meth:`set_forcefield` takes instead. Also raised later, from
            :meth:`_wire`, when a ``potential_dtype`` was asked for but the
            attached force field cannot restate its parameter tables
            (no ``to_dtype``).

    References:
        ``.claude/notes/composable-potentials.md`` — finding **F28** (an ``MD``
        with neither a compile seam nor a precision policy), closed by
        :meth:`set_compiler` and ``potential_dtype``.
        :mod:`molix.quant` — the quantization-as-thermal-noise study the split
        precision exists for.
    """

    def __init__(
        self,
        dtype: torch.dtype | None = None,
        *,
        potential_dtype: torch.dtype | None = None,
    ) -> None:
        if dtype is not None and not isinstance(dtype, torch.dtype):
            raise TypeError(
                "MD(dtype) takes a torch.dtype; attach the force field with set_forcefield"
            )
        if potential_dtype is not None and not isinstance(potential_dtype, torch.dtype):
            raise TypeError("MD(potential_dtype=) takes a torch.dtype")
        self.dtype = dtype
        self.potential_dtype = potential_dtype
        self.device: torch.device | None = None
        self.force: MDForceField | None = None
        self.neighbors: NeighborList | None = None
        self.integrator: Integrator | None = None
        self.hooks: list[MDHook | tuple[MDHook, int]] = []
        self.barostat: BerendsenBarostat | None = None
        self.runner: MDRunner | None = None
        self._adapter: _TensorDictForce | None = None
        self._compiler: Callable[[Callable[..., Any]], Callable[..., Any]] | None = None

    def set_forcefield(self, force: MDForceField) -> MD:
        """Attach a :class:`molpy.ForceField` (including Class-I / OPLS subclasses).

        Energy and forces from ``force.to_potentials()`` are converted from
        ``force.units`` (LAMMPS style name) to amu·Å²/fs² via
        :func:`molix.units.preset_energy_to_md`.

        Args:
            force: Force field with a nullary ``to_potentials`` and a
                ``units`` label (:class:`~molix.md.forcefield.MDForceField`)
                — e.g. molnex's realized ``molpot.forcefield.ForceField``.

        Returns:
            ``self``.

        Raises:
            TypeError: If ``force`` is not a :class:`~molpy.ForceField`.
        """
        if not isinstance(force, ForceField):
            raise TypeError(f"set_forcefield expects a ForceField, got {type(force).__name__}")
        self.force = force
        self._wire()
        return self

    def set_neighbors(self, neighbors: NeighborList) -> MD:
        """Attach a rebuildable neighbour list and return ``self``.

        When both a force field and an integrator are already set, the
        integrator's :attr:`~molix.md.integrators.Integrator.rebuild` switch
        turns on. :meth:`run` calls :meth:`NeighborList.build` once; each
        :meth:`~molix.md.driver.MD.compute_force` then asks
        :meth:`NeighborList.update`.

        Args:
            neighbors: Fixed-capacity MD neighbour list. ``build`` writes
                ``pair_index``, ``shifts``, ``pair_diff`` and ``pair_dist`` over
                the whole capacity, at :attr:`potential_dtype` when one is set.

        Returns:
            ``self``.
        """
        self.neighbors = neighbors
        # The list writes the geometry the potential reads, so it writes it at
        # the potential's precision; ``None`` is its own buffer dtype.
        neighbors.geometry_dtype = self.potential_dtype
        self._wire()
        return self

    def set_integrator(self, integrator: Integrator) -> MD:
        """Attach a constructed integrator (``VelocityVerletIntegrator(dt)``)."""
        self.integrator = integrator
        if self.dtype is not None:
            self.integrator.cast_state(self.dtype)
        self._wire()
        return self

    def set_hooks(self, hooks: Sequence[MDHook | tuple[MDHook, int]] | None) -> MD:
        """Replace the observation hooks. Not a dynamics parameter."""
        self.hooks = list(hooks or [])
        self._wire()
        return self

    def set_barostat(self, barostat: BerendsenBarostat) -> MD:
        """Attach an isotropic Berendsen barostat and return ``self``.

        Requires ``graphs.cell`` on the system; the composed op then derives
        ``graphs.virial`` from a ``graphs.strain`` variable beside the forces.
        After each integrator step the cell and positions are scaled and a
        mounted neighbour list is rebuilt against the new lattice.

        Raises:
            TypeError: ``barostat`` is not a
                :class:`~molix.md.barostat.BerendsenBarostat`.
        """
        if not isinstance(barostat, BerendsenBarostat):
            raise TypeError(
                f"set_barostat expects a BerendsenBarostat, got {type(barostat).__name__}"
            )
        self.barostat = barostat
        self._wire()
        return self

    def set_compiler(
        self, compiler: Callable[[Callable[..., Any]], Callable[..., Any]] | None
    ) -> MD:
        """Compile the derivative op composed over the potential, once per run.

        The one compile seam of an MD run. :meth:`run` binds the working batch
        and composes ``Derivative(pots, wrt=("atoms.pos"[, "graphs.strain"]))``
        over the freshly assembled potential; with a compiler set, that op is
        evaluated once eagerly — the topology bind, an mlip's element-table
        check and any k-space grid sizing are host-side work a trace cannot do
        — and ``compiler(op)`` is what every later step calls. Held on ``MD``
        rather than installed by hand because every ``set_*`` call rebuilds
        the potential (``force.to_potentials()`` returns a fresh container),
        and a build installed on one container would survive exactly until the
        next setter; that is why this is a setter too, and why the order of
        the setters does not matter::

            md = (
                MD(torch.float64)
                .set_forcefield(ff)
                .set_compiler(Compiler(fullgraph=True))
                .set_integrator(VelocityVerletIntegrator(1.0))   # rebuild → same policy
            )

        ``compiler`` is structural — any ``callable -> callable``.
        :class:`molix.compile.Compiler` is the molnex one, and
        ``Compiler(cuda_graphs=True)`` is the MD build of choice: the op is
        handed the list's full capacity buffer every step, so its shapes are
        static, and a compiler whose ``cuda_graphs`` attribute is true gets the
        replay contract (step mark before the op, batch-side outputs cloned)
        from the adapter. The neighbour-list rebuild stays eager, outside the
        captured region. NVE and NPT compose the same energy through the same
        op; there is no barostat-specific core anywhere.

        Args:
            compiler: The callable-to-callable compiler, or ``None`` to run
                eager from the next run on.

        Returns:
            ``self``.

        References:
            ``.claude/notes/composable-potentials.md`` — finding **F28**: the
            benchmark arms that had no seam to ride on until this setter
            existed. Finding **F34**: the NPT path that now differentiates the
            same compiled op through a strain variable.
        """
        self._compiler = compiler
        self._wire()
        return self

    def to(self, device: torch.device | str) -> MD:
        """Place the bound integrator, neighbour list, and pots on ``device``.

        Does not call ``.to`` on the molpy ForceField container.
        """
        self.device = torch.device(device)
        if self.integrator is not None:
            self.integrator.to(self.device)
        if self.neighbors is not None:
            self.neighbors.to(self.device)
        if self._adapter is not None:
            self._adapter.to_pots(self.device)
        return self

    def _wire(self) -> None:
        """Rebuild the potential, the force adapter and the runner.

        Called by every setter, because ``force.to_potentials()`` is what
        assembles the terms and a changed component can change them. Two
        things ride on that rebuild rather than on a one-shot call, so they
        cannot be lost by a later setter:

        * :attr:`potential_dtype` — applied in **two** places, because a field's
          numbers live in two: ``force.to_dtype(dtype)`` restates the container's
          parameter tables (the per-atom ε / σ / q a ``Potentials`` reads live
          off its container on every call, which no module cast can see), and
          ``pots.to(dtype)`` moves what the assembled module itself owns — the
          bonded buffers and any mlip weights. The adapter casts the third part,
          the per-step geometry.
        * :attr:`_compiler` — handed to the adapter, which applies it to the
          op it composes at :meth:`run` over the container that is actually
          about to run, after placing it on :attr:`device`.

        Raises:
            TypeError: A :attr:`potential_dtype` was asked for but the attached
                force field has no ``to_dtype`` to restate its parameter tables
                with (``molpot.forcefield.ForceField`` has one). Casting only
                the module's own buffers would leave the per-atom ``ε`` / ``σ``
                / ``charge`` — which a ``Potentials`` reads live off its container —
                at the other precision, and molpot's Coulomb kernels refuse a
                dtype mismatch rather than promote.
        """
        if self.force is None or self.integrator is None:
            self.runner = None
            self._adapter = None
            return
        source = self.force
        if self.potential_dtype is not None:
            # Two halves, because a field's numbers live in two places: the
            # container's parameter tables (restated by ``to_dtype``, which the
            # assembled Potentials then reads live) and the module's own buffers
            # and mlip weights (moved by ``.to`` below).
            to_dtype = getattr(source, "to_dtype", None)
            if to_dtype is None:
                raise TypeError(
                    "MD(potential_dtype=…) needs a force field that can restate its "
                    f"parameter tables at another dtype; {type(source).__name__} has no "
                    "to_dtype (molpot.forcefield.ForceField does). Drop potential_dtype, "
                    "or declare the tables at the wanted dtype and evaluate at one precision"
                )
            source = to_dtype(self.potential_dtype)
        pots = source.to_potentials()
        if isinstance(pots, torch.nn.Module):
            pots.eval()
            pots.requires_grad_(False)
            if self.potential_dtype is not None:
                pots.to(self.potential_dtype)
        scale = preset_energy_to_md(self.force.units)
        self._adapter = _TensorDictForce(
            pots,
            scale,
            md_dtype=self.dtype,
            pot_dtype=self.potential_dtype,
            compiler=self._compiler,
        )
        self._adapter.want_virial = self.barostat is not None
        if self.device is not None:
            # A setter called *after* ``to(device)`` rebuilds the potential; the
            # rebuild has to be placed too, or the run evaluates on whichever
            # device the ForceField's tables happen to be on.
            self._adapter.to_pots(self.device)
        self.integrator.bind_force(self.compute_force)
        self.runner = MDRunner(
            self.integrator,
            hooks=self.hooks,
            barostat=self.barostat,
            sync_state=self._sync_cell,
        )

    def compute_force(self, pos: torch.Tensor) -> dict[str, torch.Tensor]:
        """Update the neighbour list on the working batch, then evaluate.

        Writes ``pos`` into the work TensorDict (Å), calls
        :meth:`NeighborList.update` when a list is mounted, then hands the batch
        to the force adapter, which runs the one derivative op composed at
        :meth:`_TensorDictForce.bind_work` — forces, plus the virial under a
        barostat — and reads ``graphs.energy`` / ``atoms.forces`` /
        ``graphs.virial`` back off the batch.

        Going through the adapter is also what makes the two ``MD``-level
        policies apply: the build :meth:`set_compiler` chose is the op the
        adapter holds, and the :attr:`potential_dtype` boundary is the
        adapter's — ``pos`` arrives here at the MD dtype (which is what the
        neighbour list validates against), is handed to the potential at the
        potential's, and comes back at the MD dtype whatever it ran at.

        Args:
            pos: Positions ``(N, 3)`` in Å, at the MD dtype.

        Returns:
            ``{"energy", "forces"}`` in MD units — energy amu·Å²/fs², forces
            amu·Å/fs² — plus ``"virial"`` ``(3, 3)`` in amu·Å²/fs² when a
            barostat is attached.

        Raises:
            RuntimeError: No force field or integrator has been attached
                (nothing was wired), or :meth:`run` has not bound the working
                batch yet — this is a per-step entry point and does not bind one
                of its own.
        """
        if self._adapter is None:
            raise RuntimeError("set_forcefield and set_integrator before compute_force")
        work = self._adapter.work
        if work is None:
            raise RuntimeError("bind the working TensorDict before evaluating forces")
        work[POS_KEY] = pos
        if self.neighbors is not None:
            self.neighbors.update(work)
        return self._adapter(pos)

    def run(self, system: TensorDict, n_steps: int, *, chunk: int | None = None) -> MDState:
        """Integrate ``n_steps`` on ``system``, writing ``atoms.pos`` / ``atoms.vel``.

        Args:
            system: TensorDict with ``atoms.pos``, ``atoms.vel``, ``atoms.mass``.
            n_steps: Number of steps.
            chunk: Steps advanced between hook firings. Default 1.
        """
        if self.runner is None or self._adapter is None:
            raise RuntimeError("set_forcefield and set_integrator before run")
        if chunk is None:
            chunk = 1
        with torch.no_grad():
            system = self._cast_system(system)
            if self.neighbors is not None:
                self.neighbors.build(system)
            self._adapter.bind_work(system)
            try:
                return self.runner.run(system, n_steps, chunk=chunk)
            finally:
                self._adapter.unbind_work()

    def _cast_system(self, system: TensorDict) -> TensorDict:
        """Bring ``atoms.pos`` / ``vel`` / ``mass`` (and cell) onto the run dtype/device."""
        for key in (POS_KEY, VEL_KEY, MASS_KEY):
            tensor = system[key]
            if self.dtype is not None:
                tensor = tensor.to(self.dtype)
            if self.device is not None:
                tensor = tensor.to(self.device)
            system[key] = tensor
        if ("graphs", "cell") in system.keys(include_nested=True):
            cell = system["graphs", "cell"]
            if self.dtype is not None:
                cell = cell.to(self.dtype)
            if self.device is not None:
                cell = cell.to(self.device)
            system["graphs", "cell"] = cell
        return system

    def _sync_cell(self, state: MDState) -> None:
        """Write a barostat-scaled cell back onto the work batch and neighbour list.

        The copy is delegated to
        :meth:`_TensorDictForce.rebind_cell`, which re-reads ``graphs.cell``
        first, so it always lands in the tensor *currently* stored there. A
        virial strain pass must not leave a replacement tensor here; if it
        did, a cached view would copy into orphaned storage and Polar
        k-space would keep the pre-barostat lattice. ``MDRunner`` calls this
        once per chunk while a barostat is attached, not once per step.
        """
        if state.cell is None or self._adapter is None:
            return
        work = self._adapter.work
        if work is None:
            return
        self._adapter.rebind_cell(state.cell)
        work[POS_KEY] = state.pos
        if self.neighbors is not None:
            self.neighbors.scale_isotropic(state.cell, state.pos)
