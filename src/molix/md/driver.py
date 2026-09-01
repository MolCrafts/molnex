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

Neighbour-list cadence is not ``MD``'s. Configure it on
:class:`~molix.md.neighbors.NeighborList` and mount with
:meth:`MD.set_neighbors`. A frozen list for a compiled rollout is
omit :meth:`MD.set_neighbors` to freeze the list.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from tensordict import TensorDict

from molix.md.barostat import BerendsenBarostat
from molix.md.forcefield import ForceField, MDForceField
from molix.md.integrators import Integrator
from molix.md.neighbors import NeighborList
from molix.md.runner import MDHook, MDRunner
from molix.md.types import MDState
from molix.schema import ENERGY_KEY, MASS_KEY, POS_KEY, VEL_KEY
from molix.units import preset_energy_to_md, units

_KB_MD = (1.0 * units.k_B).to("amu * angstrom ** 2 / femtosecond ** 2 / kelvin").magnitude


class _TensorDictForce:
    """Private ``pos -> dict`` over ``to_potentials()``.

    MD never owns an energy graph: the trajectory is stepped under
    ``torch.no_grad()``, and the one ``energy.detach()`` at this seam
    drops the autograd tape so the next step does not keep Polar's
    saved tensors alive. Forces and virial come from a single
    ``autograd.grad(..., create_graph=False)`` inside the pots.
    """

    def __init__(self, pots: object, scale: float) -> None:
        self._pots = pots
        self._scale = float(scale)
        self._work: TensorDict | None = None
        self._cell_view: torch.Tensor | None = None
        self._edges_full: TensorDict | None = None
        self.n_live: int | None = None
        self.want_virial = False
        self._eval = self._eval_classical

    def bind_work(self, work: TensorDict) -> None:
        self._work = work
        self._cell_view = None
        if ("graphs", "cell") in work.keys(include_nested=True):
            cell = work["graphs", "cell"]
            self._cell_view = cell[0] if cell.dim() == 3 else cell
        if self.want_virial:
            self._eval = self._eval_npt
        elif getattr(self._pots, "mlip", None) is not None:
            self._eval = self._eval_mlip
        else:
            self._eval = self._eval_classical

    def to_pots(self, device: torch.device) -> None:
        to_fn = getattr(self._pots, "to", None)
        if callable(to_fn):
            moved = to_fn(device)
            if moved is not None:
                self._pots = moved

    def __call__(self, pos: torch.Tensor) -> dict[str, torch.Tensor]:
        work = self._work
        if work is None:
            raise RuntimeError("bind the working TensorDict before evaluating forces")
        work[POS_KEY] = pos
        self._expose_live_edges(work)
        try:
            return self._eval(work)
        finally:
            self._restore_edges(work)

    def _expose_live_edges(self, work: TensorDict) -> None:
        """Hand the potential ``[0, n_live)`` only; padding stays on the list.

        NeighborList buffers are capacity-sized (CUDA-graph contract). Dead
        rows sit at ``DEAD_EDGE_CUTOFF_FACTOR * cutoff`` so the envelope is
        zero, but a fallback tensor product still materialises them. Polar
        MD is eager: a live view is correct and cheaper.
        """
        n = self.n_live
        if n is None or "edges" not in work.keys():
            return
        full = work["edges"]
        cap = int(full.batch_size[0]) if full.batch_size else 0
        if n <= 0 or n >= cap:
            return
        self._edges_full = full
        work["edges"] = TensorDict({key: full[key][:n] for key in full.keys()}, batch_size=[n])

    def _restore_edges(self, work: TensorDict) -> None:
        full = self._edges_full
        if full is None:
            return
        work["edges"] = full
        self._edges_full = None

    def _pack(self, energy: torch.Tensor, forces: torch.Tensor) -> dict[str, torch.Tensor]:
        scale = self._scale
        return {
            # Detach is the MD/training seam: MDState must not keep the tape.
            "energy": energy.detach().sum() * scale,
            "forces": forces * scale,
        }

    def _eval_npt(self, work: TensorDict) -> dict[str, torch.Tensor]:
        calc = getattr(self._pots, "calc_energy_forces_virial", None)
        if calc is None:
            raise RuntimeError(
                f"a barostat needs pots.calc_energy_forces_virial (got {type(self._pots).__name__})"
            )
        energy, forces, virial = calc(work)
        out = self._pack(energy, forces)
        out["virial"] = virial * self._scale
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

    Args:
        dtype: MD-side precision — trajectory state after :meth:`run` casts
            ``atoms.pos`` / ``atoms.vel`` / ``atoms.mass``. ``None`` keeps the
            system tensors' dtype. Never calls ``.to`` on the molpy container.
    """

    def __init__(self, dtype: torch.dtype | None = None) -> None:
        if dtype is not None and not isinstance(dtype, torch.dtype):
            raise TypeError(
                "MD(dtype) takes a torch.dtype; attach the force field with set_forcefield"
            )
        self.dtype = dtype
        self.device: torch.device | None = None
        self.force: MDForceField | None = None
        self.neighbors: NeighborList | None = None
        self.integrator: Integrator | None = None
        self.hooks: list[MDHook | tuple[MDHook, int]] = []
        self.barostat: BerendsenBarostat | None = None
        self.runner: MDRunner | None = None
        self._adapter: _TensorDictForce | None = None

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
        """
        if not isinstance(barostat, BerendsenBarostat):
            raise TypeError(
                f"set_barostat expects a BerendsenBarostat, got {type(barostat).__name__}"
            )
        self.barostat = barostat
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
        if self.force is None or self.integrator is None:
            self.runner = None
            self._adapter = None
            return
        pots = self.force.to_potentials()
        if isinstance(pots, torch.nn.Module):
            pots.eval()
            pots.requires_grad_(False)
        scale = preset_energy_to_md(self.force.units)
        self._adapter = _TensorDictForce(pots, scale)
        self._adapter.want_virial = self.barostat is not None
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
        :meth:`NeighborList.update` when a list is mounted, then
        ``pots.calc_energy`` / ``calc_forces``.

        Args:
            pos: Positions ``(N, 3)`` in Å.

        Returns:
            ``{"energy", "forces"}`` in amu·Å²/fs²; ``"virial"`` when a
            barostat is attached.
        """
        if self._adapter is None:
            raise RuntimeError("set_forcefield and set_integrator before compute_force")
        work = self._adapter._work
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

        Always copy into the tensor currently stored at ``graphs.cell``. A
        virial strain pass must not leave a replacement tensor here; if it
        did, a cached view would copy into orphaned storage and Polar
        k-space would keep the pre-barostat lattice.
        """
        if state.cell is None or self._adapter is None:
            return
        work = self._adapter._work
        if work is None:
            return
        if ("graphs", "cell") in work.keys(include_nested=True):
            cell = work["graphs", "cell"]
            (cell[0] if cell.dim() == 3 else cell).copy_(state.cell)
        work[POS_KEY] = state.pos
        if self.neighbors is not None:
            self.neighbors.scale_isotropic(state.cell, state.pos)
