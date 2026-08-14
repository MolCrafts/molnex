"""``MD`` — compose a force field and an integrator, then run a system.

``MD`` owns only the **MD-side precision** (trajectory state). Everything
else is a component the caller builds and attaches:

    md = (
        MD(torch.float64)
        .set_forcefield(ff)
        .set_integrator(VelocityVerletIntegrator(dt))
        .set_hooks([TrajectoryHook(...)])
        .to(device)
    )
    MaxwellBoltzmann().sample(system, 300.0, seed=0)
    md.run(system, n_steps)

``system`` is a TensorDict. Mass, positions and velocities live under
``atoms`` (``molix.schema.MASS_KEY`` / ``POS_KEY`` / ``VEL_KEY``). The
integrator is constructed with ``dt`` only.

Neighbour-list cadence is not ``MD``'s. Configure it on
:class:`~molix.md.neighbors.NeighborList`. A frozen list for a compiled
rollout is ``VelocityVerletIntegrator(dt, rebuild=False)``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from tensordict import TensorDict

from molix.md.forcefield import ForceField
from molix.md.integrators import Integrator
from molix.md.runner import MDHook, MDRunner
from molix.md.types import MDState
from molix.schema import MASS_KEY, POS_KEY, VEL_KEY
from molix.units import units

_KB_MD = (1.0 * units.k_B).to("amu * angstrom ** 2 / femtosecond ** 2 / kelvin").magnitude


class MaxwellBoltzmann:
    """Maxwell-Boltzmann velocities written onto ``atoms.vel``.

    Reads ``atoms.mass`` from the system TensorDict. No constructor mass —
    the system already carries it.
    """

    def sample(
        self,
        system: TensorDict,
        temperature: float,
        *,
        seed: int = 0,
        remove_com: bool = True,
    ) -> TensorDict:
        """Draw velocities at ``temperature`` (K) into ``system["atoms", "vel"]``.

        Args:
            system: TensorDict with ``atoms.mass`` ``(N,)`` or scalar, and
                ``atoms.pos`` ``(N, 3)`` (for the atom count).
            temperature: Kelvin.
            seed: Generator seed.
            remove_com: Remove the centre-of-mass momentum (NVE dof ``3N-3``).

        Returns:
            The same ``system``, with ``atoms.vel`` written in Å/fs (float64).
        """
        pos = system[POS_KEY]
        mass = system[MASS_KEY]
        n = int(pos.shape[0])
        mass_t = torch.as_tensor(mass, dtype=torch.float64)
        if mass_t.dim() == 0:
            mass_t = mass_t.expand(n)
        elif int(mass_t.shape[0]) != n:
            raise ValueError(f"atoms.mass length {mass_t.shape[0]} disagrees with n_atoms={n}")
        mass_col = mass_t.reshape(-1, 1)
        generator = torch.Generator().manual_seed(int(seed))
        noise = torch.randn((n, 3), generator=generator, dtype=torch.float64)
        vel = noise * torch.sqrt(_KB_MD * float(temperature) / mass_col)
        if remove_com:
            vel = vel - (mass_col * vel).sum(0) / mass_col.sum()
        system[VEL_KEY] = vel
        return system


class MD:
    """Compose ``ForceField`` + ``Integrator`` at a chosen MD-state dtype.

    Args:
        dtype: MD-side precision — trajectory state after :meth:`run` casts
            ``atoms.pos`` / ``atoms.vel`` / ``atoms.mass``. ``None`` keeps the
            system tensors' dtype. Never touches the force field.
    """

    def __init__(self, dtype: torch.dtype | None = None) -> None:
        if dtype is not None and not isinstance(dtype, torch.dtype):
            raise TypeError(
                "MD(dtype) takes a torch.dtype; attach the force field with set_forcefield"
            )
        self.dtype = dtype
        self.device: torch.device | None = None
        self.force: ForceField | None = None
        self.integrator: Integrator | None = None
        self.hooks: list[MDHook | tuple[MDHook, int]] = []
        self.runner: MDRunner | None = None

    def set_forcefield(self, force: ForceField) -> MD:
        """Attach the force field. Its dtype is whatever the caller built."""
        self.force = force
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
        """Place the bound integrator and force field on ``device``."""
        self.device = torch.device(device)
        if self.integrator is not None:
            self.integrator.to(self.device)
        if self.force is not None:
            self.force.to(self.device)
        return self

    def _wire(self) -> None:
        if self.force is None or self.integrator is None:
            self.runner = None
            return
        self.integrator.bind_force(self.force)
        self.runner = MDRunner(self.integrator, hooks=self.hooks)

    def run(self, system: TensorDict, n_steps: int, *, chunk: int | None = None) -> MDState:
        """Integrate ``n_steps`` on ``system``, writing ``atoms.pos`` / ``atoms.vel``.

        Args:
            system: TensorDict with ``atoms.pos``, ``atoms.vel``, ``atoms.mass``.
            n_steps: Number of steps.
            chunk: Steps advanced between hook firings. Default 1.
        """
        if self.runner is None:
            raise RuntimeError("set_forcefield and set_integrator before run")
        if chunk is None:
            chunk = 1
        system = self._cast_system(system)
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
