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
``VelocityVerletIntegrator(dt, rebuild=False)``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from tensordict import TensorDict

from molix.md.forcefield import ForceField
from molix.md.integrators import Integrator
from molix.md.neighbors import NeighborList
from molix.md.runner import MDHook, MDRunner
from molix.md.types import ForceOutput, MDState
from molix.schema import ENERGY_KEY, FORCES_KEY, MASS_KEY, POS_KEY, VEL_KEY
from molix.units import preset_energy_to_md, units

_KB_MD = (1.0 * units.k_B).to("amu * angstrom ** 2 / femtosecond ** 2 / kelvin").magnitude


class _TensorDictForce:
    """Private :class:`~molix.md.integrators.PotentialLike` over ``to_potentials()``."""

    def __init__(
        self,
        pots: object,
        neighbors: NeighborList | None,
        scale: float,
    ) -> None:
        self._pots = pots
        self._neighbors = neighbors
        self._scale = float(scale)
        self._work: TensorDict | None = None

    @property
    def rebuilds_neighbors(self) -> bool:
        return self._neighbors is not None

    def rebuild_neighbors(self, pos: torch.Tensor) -> None:
        if self._neighbors is not None:
            self._neighbors.update(pos)

    def bind_work(self, work: TensorDict) -> None:
        self._work = work

    def to_pots(self, device: torch.device) -> None:
        to_fn = getattr(self._pots, "to", None)
        if callable(to_fn):
            moved = to_fn(device)
            if moved is not None:
                self._pots = moved

    def __call__(self, pos: torch.Tensor) -> ForceOutput:
        work = self._work
        if work is None:
            raise RuntimeError("bind the working TensorDict before evaluating forces")
        work[POS_KEY] = pos
        calc_energy = getattr(self._pots, "calc_energy")
        calc_forces = getattr(self._pots, "calc_forces")
        try:
            energy = calc_energy(work)
        except TypeError as exc:
            raise TypeError(
                "calc_energy must accept a TensorDict; "
                f"{type(self._pots).__name__}.calc_energy does not take TensorDict"
            ) from exc
        forces = calc_forces(work)
        if not isinstance(energy, torch.Tensor) or not isinstance(forces, torch.Tensor):
            raise TypeError(
                "calc_energy must accept a TensorDict; "
                f"{type(self._pots).__name__}.calc_energy does not take TensorDict"
            )
        if "graphs" not in work.keys():
            work["graphs"] = TensorDict({}, batch_size=[])
        work[ENERGY_KEY] = energy
        work[FORCES_KEY] = forces
        scale = self._scale
        return ForceOutput(energy.detach().sum() * scale, forces.detach() * scale)


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
        self.force: ForceField | None = None
        self.neighbors: NeighborList | None = None
        self.integrator: Integrator | None = None
        self.hooks: list[MDHook | tuple[MDHook, int]] = []
        self.runner: MDRunner | None = None
        self._adapter: _TensorDictForce | None = None

    def set_forcefield(self, force: ForceField) -> MD:
        """Attach a :class:`molpy.ForceField` (including Class-I / OPLS subclasses).

        Energy and forces from ``force.to_potentials()`` are converted from
        ``force.units`` (LAMMPS style name) to amu·Å²/fs² via
        :func:`molix.units.preset_energy_to_md`.

        Args:
            force: molpy ForceField container.

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
        :meth:`~molix.md.integrators.Integrator.eval_force` then asks
        :meth:`NeighborList.update`.

        Args:
            neighbors: Fixed-capacity MD neighbour list. ``build`` still writes
                only ``edge_index`` and ``shifts``.

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
        scale = preset_energy_to_md(self.force.units)
        self._adapter = _TensorDictForce(pots, self.neighbors, scale)
        self.integrator.bind_force(self._adapter)
        self.runner = MDRunner(self.integrator, hooks=self.hooks)

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
        system = self._cast_system(system)
        if self.neighbors is not None:
            self.neighbors.build(system)
        self._adapter.bind_work(system)
        return self.runner.run(system, n_steps, chunk=chunk)

    def _cast_system(self, system: TensorDict) -> TensorDict:
        """Bring ``atoms.pos`` / ``vel`` / ``mass`` onto the run dtype/device."""
        for key in (POS_KEY, VEL_KEY, MASS_KEY):
            tensor = system[key]
            if self.dtype is not None:
                tensor = tensor.to(self.dtype)
            if self.device is not None:
                tensor = tensor.to(self.device)
            system[key] = tensor
        return system
