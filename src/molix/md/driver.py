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
:meth:`MD.set_compiler` (compile the potential's hot core — re-applied on every
rebuild) and ``MD(dtype, potential_dtype=…)`` (evaluate the potential at a
second precision while the trajectory stays at ``dtype``).

Neighbour-list cadence is not ``MD``'s. Configure it on
:class:`~molix.md.neighbors.NeighborList` and mount with
:meth:`MD.set_neighbors`. For a compiled rollout with a frozen edge set,
omit :meth:`MD.set_neighbors` entirely — with no list mounted, ``edges``
is never rebuilt and stays whatever was bound into ``system`` for the
whole run.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from tensordict import TensorDict

from molix.batch import TensorFrame, as_tensor_frame
from molix.md.barostat import BerendsenBarostat
from molix.md.forcefield import ForceField, MDForceField
from molix.md.integrators import Integrator
from molix.md.neighbors import NeighborList
from molix.md.runner import MDHook, MDRunner
from molix.md.types import MDState
from molix.schema import ENERGY_KEY, MASS_KEY, POS_KEY, VEL_KEY, CellBound
from molix.units import preset_energy_to_md, units

_KB_MD = (1.0 * units.k_B).to("amu * angstrom ** 2 / femtosecond ** 2 / kelvin").magnitude


class _TensorDictForce:
    """Private ``pos -> dict`` over ``to_potentials()``, and the precision seam.

    MD never owns an energy graph: the trajectory is stepped under
    ``torch.no_grad()``, and the one ``energy.detach()`` at this seam
    drops the autograd tape so the next step does not keep Polar's
    saved tensors alive. Forces and virial come from a single
    ``autograd.grad(..., create_graph=False)`` inside the pots.

    It is also the **dtype boundary** when :class:`MD` was built with a
    ``potential_dtype``: positions (and ``graphs.cell``) go in at the
    potential's dtype and energy / forces / virial come back at the MD dtype,
    so the integrator state never picks the potential's precision up. With
    ``pot_dtype=None`` no cast is performed anywhere on this path.

    Every one of those casts is made **per call and off to the side**, never
    written back onto the caller's batch: the batch is the MD state's own
    storage (``MDState.cell`` *is* ``graphs.cell``, and the end-of-run
    writeback lands in it), so leaving a reduced-precision copy there would
    quietly move the trajectory to the potential's precision — the exact
    confusion the split exists to avoid. The cell is held as a private
    cast copy (:meth:`_pot_cell`) refreshed from the live storage each call,
    the edge columns are re-cast each call (:meth:`_expose_live_edges`), and
    both are swapped out again before this adapter returns.

    Args:
        pots: The object ``ForceField.to_potentials()`` returned.
        scale: Energy conversion from the ForceField's unit convention to MD
            units (amu·Å²/fs²).
        md_dtype: Dtype energy / forces / virial are returned at — :attr:`MD.dtype`.
            ``None`` returns whatever the potential produced.
        pot_dtype: Dtype the potential is evaluated at — :attr:`MD.potential_dtype`.
            ``None`` (the default) disables the whole cast path.
    """

    def __init__(
        self,
        pots: object,
        scale: float,
        *,
        md_dtype: torch.dtype | None = None,
        pot_dtype: torch.dtype | None = None,
    ) -> None:
        self._pots = pots
        self._scale = float(scale)
        self._md_dtype = md_dtype
        self._pot_dtype = pot_dtype
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
        self._edges_full: TensorDict | None = None
        self.n_live: int | None = None
        self.want_virial = False
        self._eval = self._eval_classical

    @property
    def work(self) -> TensorDict | None:
        """The bound working batch — the caller's container; ``None`` before :meth:`bind_work`."""
        return self._work

    def bind_work(self, work: TensorDict) -> None:
        """Bind the working ``TensorDict`` for the run and pick the eval kernel.

        Called once per :meth:`MD.run`, never once per step. Stores ``work``,
        re-resolves the live cell via :meth:`rebind_cell`, then selects the
        potential-call path for the rest of the run: :meth:`_eval_npt` when a
        barostat wants the virial (``want_virial``), else :meth:`_eval_mlip`
        when the pots expose an ``mlip`` attribute, else :meth:`_eval_classical`.

        The batch is wrapped **once here** as its typed view
        (:func:`~molix.batch.as_tensor_frame`, CLAUDE.md § TensorDict Contract)
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
        if self.want_virial:
            self._eval = self._eval_npt
        elif getattr(self._pots, "mlip", None) is not None:
            self._eval = self._eval_mlip
        else:
            self._eval = self._eval_classical

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
        reused. ``molpot.forcefield.potentials``' virial strain closure
        replaces ``graphs.cell`` with a graph tensor for the duration of the
        pass and restores the buffer only on the no-exception path, so a cached
        view can point at orphaned storage — copying a barostat update into it
        would land nowhere and leave the potential on the stale lattice.

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

        Raises:
            RuntimeError: :meth:`bind_work` has not run, so there is no batch to
                evaluate on; or, on the barostat path, the potential exposes no
                ``calc_energy_forces_virial``.
        """
        work = self._work
        frame = self._frame
        if work is None or frame is None:
            raise RuntimeError("bind the working TensorDict before evaluating forces")
        # Read and swapped through the view, evaluated on the caller's own
        # container: the two are one storage, so the potential is handed the
        # narrowed, cast geometry either way — and a potential that refuses the
        # batch names the container the caller built rather than a wrapper this
        # adapter made.
        frame[POS_KEY] = self._cast_in(frame, pos)
        self._expose_live_edges(frame)
        try:
            return self._eval(work)
        finally:
            self._restore_edges(frame)
            self._restore_cell(frame)

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

        The edge columns are the other half of the geometry and are cast in
        :meth:`_expose_live_edges` on the same per-call, swap-and-restore
        terms, because the list refills its buffers in place on a rebuild. The
        parameter tables are :class:`MD`'s half (``potential_dtype``).
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

    def _expose_live_edges(self, frame: TensorFrame) -> None:
        """Hand the potential the edge rows it should price, at its own dtype.

        Two independent narrowings, one swap (undone by :meth:`_restore_edges`):

        **Rows.** NeighborList buffers are capacity-sized (CUDA-graph
        contract). Dead rows sit at ``DEAD_EDGE_CUTOFF_FACTOR * cutoff`` so the
        envelope is zero, but a fallback tensor product still materialises
        them, so the live prefix ``[0, n_live)`` is what the potential is handed.

        **Dtype.** Under a ``potential_dtype`` the float columns (``shifts`` /
        ``edge_diff`` / ``edge_dist``) are cast, because the geometry has to
        reach the potential in one precision: molpot's Coulomb kernels compare
        the dtype of the pair distances against the positions' and raise rather
        than promote. The cast is redone **every call** and never cached: the
        list refills ``edge_index`` / ``shifts`` *in place* on a rebuild, so a
        cast copy left on the batch would freeze the connectivity at the
        geometry of the last rebuild — a silently wrong PES, which is precisely
        the failure the in-place buffers exist to avoid. That is one copy of
        the edge columns per step, in this arm only.
        """
        if "edges" not in frame.keys():
            return
        full = frame["edges"]
        cap = int(full.batch_size[0]) if full.batch_size else 0
        n = self.n_live
        rows = n if n is not None and 0 < n < cap else cap
        dtype = self._pot_dtype
        if rows == cap and dtype is None:
            return
        columns = {key: full[key] for key in full.keys()}
        if rows != cap:
            columns = {key: column[:rows] for key, column in columns.items()}
        if dtype is not None:
            columns = {
                key: column.to(dtype) if column.is_floating_point() else column
                for key, column in columns.items()
            }
        self._edges_full = full
        frame["edges"] = TensorDict(columns, batch_size=[rows])

    def _restore_edges(self, frame: TensorFrame) -> None:
        full = self._edges_full
        if full is None:
            return
        frame["edges"] = full
        self._edges_full = None

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

    def _eval_npt(self, work: TensorDict) -> dict[str, torch.Tensor]:
        calc = getattr(self._pots, "calc_energy_forces_virial", None)
        if calc is None:
            raise RuntimeError(
                f"a barostat needs pots.calc_energy_forces_virial (got {type(self._pots).__name__})"
            )
        energy, forces, virial = calc(work)
        out = self._pack(energy, forces)
        out["virial"] = self._to_md(virial * self._scale)
        return out

    def _eval_mlip(self, work: TensorDict) -> dict[str, torch.Tensor]:
        forces = self._pots.calc_forces(work)
        return self._pack(work[ENERGY_KEY], forces)

    def _eval_classical(self, work: TensorDict) -> dict[str, torch.Tensor]:
        energy, forces = self._pots.calc_energy_forces(work)
        return self._pack(energy, forces)


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
            per-atom ``epsilon`` / ``sigma`` / ``q`` a ``Potentials`` reads live
            off its container on every call — are restated there; it then calls
            ``.to(potential_dtype)`` on the assembled module for what the
            module itself owns (the bonded buffers, an mlip's weights); and the
            force adapter casts the batch geometry it hands over (positions,
            ``graphs.cell``, the edge columns) and casts energy / forces /
            virial back to ``dtype`` on the way out. Partial casts are not an
            option: molpot's Coulomb kernels compare the dtype of the charges
            and the pair distances against the positions' and raise rather than
            promote.

            The *state* keeps ``dtype`` throughout, and so does the neighbour
            list: it owns ``edges``, and validates the positions it is updated
            with against its own build precision.

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
        :meth:`set_compiler` and ``potential_dtype`` together with
        :meth:`molpot.forcefield.Potentials.compile_core`.
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
                ``edge_index``, ``shifts``, live ``edge_diff`` and ``edge_dist``.

        Returns:
            ``self``.
        """
        self.neighbors = neighbors
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

        Requires ``graphs.cell`` on the system and a potential that can
        return a virial (``calc_virial`` or ``graphs.virial``). After each
        integrator step the cell and positions are scaled and a mounted
        neighbour list is rebuilt against the new lattice.

        Raises:
            TypeError: ``barostat`` is not a
                :class:`~molix.md.barostat.BerendsenBarostat`.
            RuntimeError: A compiler is installed (:meth:`set_compiler`). The
                strained energy the virial comes from has no compiled build, so
                attaching a barostat here would silently demote the run to
                eager — the same gap, refused from the other side. Drop the
                compiler (``set_compiler(None)``) to run NPT eager. Finding
                **F34** in ``.claude/notes/composable-potentials.md``.
        """
        if not isinstance(barostat, BerendsenBarostat):
            raise TypeError(
                f"set_barostat expects a BerendsenBarostat, got {type(barostat).__name__}"
            )
        if self._compiler is not None:
            raise RuntimeError(
                "set_barostat: a compiler is installed, and the NPT force path "
                "(calc_energy_forces_virial) has no compiled build to route through — "
                "the installed core serves calc_energy_forces only, so this run would be "
                "eager while reporting itself compiled (F34). Call set_compiler(None) "
                "first to run NPT eager, or drop the barostat to keep the compiled NVE core"
            )
        self.barostat = barostat
        self._wire()
        return self

    def set_compiler(
        self, compiler: Callable[[Callable[..., Any]], Callable[..., Any]] | None
    ) -> MD:
        """Compile the potential's hot core, and keep it compiled across rebuilds.

        The one compile seam of an MD run. ``compiler`` is applied by
        :meth:`_wire` through ``pots.compile_core`` — molpot's classical
        ``(system, pos) -> (energy, forces)`` core, or an mlip's own — every
        time the potential is rebuilt, which is on **every** ``set_*`` call,
        because ``force.to_potentials()`` returns a fresh container each time.
        Installing a compiled build by hand instead would therefore survive
        exactly until the next setter; that is why this is a setter too, and why
        the order of the setters does not matter::

            md = (
                MD(torch.float64)
                .set_forcefield(ff)
                .set_compiler(Compiler(fullgraph=True))
                .set_integrator(VelocityVerletIntegrator(1.0))   # rebuild → recompiled
            )

        ``compiler`` is structural — any ``callable -> callable``.
        :class:`molix.compile.Compiler` is the molnex one
        (``Compiler(cuda_graphs=True)`` for the fixed-shape CUDA-graph preset,
        which an MD neighbour list's fixed capacity is built for). It is passed
        as a value rather than imported by molpot, which must not depend on
        ``molix.compile``.

        Args:
            compiler: The callable-to-callable compiler, or ``None`` to run
                eager from the next rebuild on.

        Returns:
            ``self``.

        An **mlip** field has one extra precondition, and it is the model's, not
        this method's: its ``compile_core`` refuses until one eager forward has
        validated the element table and sized any k-space grid, neither of which
        a traced call can do. Bind the cell and evaluate once before attaching
        the compiler (see the MD user guide). A classical field has no such
        step — ``Potentials.calc_energy_forces`` binds its topology outside the
        compiled region on the first call.

        **NVE only, for now.** The core this installs is the one
        :meth:`~molpot.forcefield.Potentials.calc_energy_forces` routes
        through; the NPT step calls ``calc_energy_forces_virial`` instead, whose
        strained backward has no compiled build. Attaching a compiler to a run
        that has a barostat would therefore compile something the run never
        calls, and report a compiled rollout that is eager end to end — so it is
        refused rather than accepted (and :meth:`set_barostat` refuses the same
        pair from the other side). Finding **F34**.

        Raises:
            RuntimeError: A barostat is attached — see above; ``set_compiler``
                is not a no-op that quietly does nothing on the NPT path.
                Or raised by ``compile_core`` from :meth:`_wire` when the
                attached potential has no core to compile, or when an mlip's
                core is not yet ready to be traced — all of them loud failures
                rather than an eager run reported as compiled.

        References:
            ``.claude/notes/composable-potentials.md`` — finding **F28**: the
            benchmark arms that had no seam to ride on until this setter and
            :meth:`molpot.forcefield.Potentials.compile_core` existed. Finding
            **F34**: the compiled NPT core this setter does not yet have.
        """
        if compiler is not None and self.barostat is not None:
            raise RuntimeError(
                "set_compiler: this MD has a barostat, and the NPT force path "
                "(calc_energy_forces_virial) has no compiled build to route through — "
                "the installed core would serve calc_energy_forces, which an NPT run "
                "never calls, so the run would be eager while reporting itself compiled "
                "(F34). Drop the barostat for a compiled NVE run, or run NPT eager"
            )
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
        * :attr:`_compiler` — re-applied through ``pots.compile_core``, so the
          compiled core belongs to the container that is actually about to run.

        Order matters for the second: the compiler is applied **after** the
        potential has been placed on :attr:`device`, so a setter called after
        :meth:`to` compiles the placed module rather than leaving the run to
        evaluate on whichever device the ForceField's tables happened to sit on.
        The dtype half runs earlier — the potential is assembled at
        :attr:`potential_dtype` and then placed.

        Raises:
            TypeError: A :attr:`potential_dtype` was asked for but the attached
                force field has no ``to_dtype`` to restate its parameter tables
                with (``molpot.forcefield.ForceField`` has one). Casting only
                the module's own buffers would leave the per-atom ``ε`` / ``σ``
                / ``q`` — which a ``Potentials`` reads live off its container —
                at the other precision, and molpot's Coulomb kernels refuse a
                dtype mismatch rather than promote.
            RuntimeError: A compiler was installed by :meth:`set_compiler` but
                the assembled potential exposes no ``compile_core`` seam; or
                raised by that seam itself when there is nothing to compile.
                Both are loud on purpose — an eager run reported as compiled is
                a silently wrong benchmark.
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
            pots, scale, md_dtype=self.dtype, pot_dtype=self.potential_dtype
        )
        self._adapter.want_virial = self.barostat is not None
        if self.device is not None:
            # A setter called *after* ``to(device)`` rebuilds the potential; the
            # rebuild has to be placed too, or the run evaluates on whichever
            # device the ForceField's tables happen to be on.
            self._adapter.to_pots(self.device)
        if self._compiler is not None:
            placed = self._adapter._pots
            compile_core = getattr(placed, "compile_core", None)
            if compile_core is None:
                raise RuntimeError(
                    "set_compiler needs a potential with a compile_core seam "
                    f"(got {type(placed).__name__}); molpot.forcefield.Potentials has one"
                )
            compile_core(self._compiler)
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
        to the force adapter. Which potential entry that adapter calls is fixed
        once per run, at :meth:`_TensorDictForce.bind_work`: with a barostat
        attached ``pots.calc_energy_forces_virial``, for an mlip field
        ``pots.calc_forces`` (the energy having been written onto the batch),
        and otherwise ``pots.calc_energy_forces`` — the classical analytic +
        autograd mix.

        Going through the adapter is also what makes the two ``MD``-level
        policies apply: the compiled core installed by :meth:`set_compiler` is
        inside ``calc_energy_forces``, and the :attr:`potential_dtype` boundary
        is the adapter's — ``pos`` arrives here at the MD dtype (which is what
        the neighbour list validates against), is handed to the potential at the
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
            self._adapter.n_live = self.neighbors.num_edges
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
                self._adapter.n_live = self.neighbors.num_edges
            self._adapter.bind_work(system)
            return self.runner.run(system, n_steps, chunk=chunk)

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
