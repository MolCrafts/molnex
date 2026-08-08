"""``MD`` — the molecular-dynamics component.

A domain type with methods, in the sense the project's design rules mean it:
"I have a force field and a system; run dynamics at this precision." It owns the
three things a trajectory needs held together and that no lower layer can decide
alone — the **integrator**, the **MD-side precision**, and the
**neighbour-list cadence** — and delegates the loop to
:class:`~molix.md.runner.MDRunner`.

Precision is split in two, deliberately. ``MD(dtype=)`` governs the **MD side
only** — trajectory state (positions/velocities), the integrator's step
constants, and the mass — never the potential. The potential's precision is an
independent axis set explicitly via :meth:`MD.set_potential_dtype` (or by
constructing the force field at the desired dtype), because studying the MD
process and the inference process separately is the point: an fp64 trajectory
over an fp32 model is a meaningful, supported configuration. At the component
boundary the integrator casts the force field's output back into the state
dtype (``Integrator.eval_force``), so the two precisions never silently
promote mid-step::

    MD(force, mass=m, dt=0.5, dtype=torch.float64)         # fp64 trajectory
    md.set_potential_dtype(torch.float32)                  # fp32 inference
    MD(force, mass=m, dt=0.5, autocast_dtype=torch.bfloat16)  # bf16-mixed model

``autocast_dtype`` leaves parameters alone and wraps each force evaluation in
``torch.autocast``, which is the only mixed-precision form :mod:`molix.config`
supports (pure fp16/bf16 parameters are deliberately unsupported there, and
that stance is kept here).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from molix.md.forcefield import ForceField
from molix.md.integrators import Integrator, LangevinVerletIntegrator
from molix.md.runner import MDHook, MDRunner
from molix.md.types import ForceOutput, MDState
from molix.units import KB_AMU_A_FS


class _AutocastForceField(ForceField):
    """Wrap a force field so each evaluation runs under ``torch.autocast``.

    Applied around the *force field* rather than the integrator so the reduced
    precision covers the model only: the BAOAB arithmetic and the accumulated
    positions/velocities stay in the state dtype, which is what keeps a long
    trajectory from losing the low bits of its own coordinates.
    """

    def __init__(self, inner: ForceField, dtype: torch.dtype) -> None:
        super().__init__()
        self.inner = inner
        self.autocast_dtype = dtype

    def rebuild_neighbors(self, pos: torch.Tensor) -> None:
        """Delegate: connectivity is not a precision concern."""
        self.inner.rebuild_neighbors(pos)

    def forward(self, pos: torch.Tensor) -> ForceOutput:
        device_type = pos.device.type
        with torch.autocast(device_type=device_type, dtype=self.autocast_dtype):
            out = self.inner(pos)
        # Hand the integrator back its own precision; autocast is the model's
        # business, not the trajectory's.
        return ForceOutput(out.energy.to(pos.dtype), out.forces.to(pos.dtype))


class MaxwellBoltzmann:
    """Maxwell-Boltzmann initial-velocity sampler over a mass profile.

    Its own tiny type rather than a method on :class:`MD`: sampling initial
    conditions is not the run driver's responsibility, and a sampler only
    needs the masses. Velocities come back in float64 on CPU; :meth:`MD.run`
    casts them onto the run's dtype/device.

    Args:
        mass: Per-atom mass ``(N,)`` in amu, or a scalar with ``n_atoms``.
        n_atoms: Atom count — required when ``mass`` is a scalar (a scalar
            mass carries no system size); checked against a per-atom mass.
    """

    def __init__(self, mass: float | torch.Tensor, *, n_atoms: int | None = None) -> None:
        mass_t = torch.as_tensor(mass, dtype=torch.float64)
        if mass_t.dim() == 0:
            if n_atoms is None:
                raise ValueError("scalar mass carries no system size; pass n_atoms")
            mass_t = mass_t.expand(int(n_atoms))
        elif n_atoms is not None and int(mass_t.shape[0]) != int(n_atoms):
            raise ValueError(f"n_atoms={n_atoms} disagrees with mass length {mass_t.shape[0]}")
        self.mass = mass_t.detach().cpu()

    def sample(self, temperature: float, *, seed: int = 0, remove_com: bool = True) -> torch.Tensor:
        """Draw velocities ``(N, 3)`` in Å/fs at ``temperature``.

        Args:
            temperature: Kelvin.
            seed: Generator seed, so a run is reproducible from its arguments.
            remove_com: Remove the centre-of-mass momentum, which is what makes
                the NVE degrees of freedom ``3N - 3``.
        """
        mass_col = self.mass.reshape(-1, 1)
        generator = torch.Generator().manual_seed(int(seed))
        noise = torch.randn((mass_col.shape[0], 3), generator=generator, dtype=torch.float64)
        vel = noise * torch.sqrt(KB_AMU_A_FS * float(temperature) / mass_col)
        if remove_com:
            vel = vel - (mass_col * vel).sum(0) / mass_col.sum()
        return vel


class MD:
    """Molecular dynamics over a :class:`~molix.md.forcefield.ForceField`.

    Args:
        force: The force field. Its dtype is **not** touched by ``dtype`` —
            use :meth:`set_potential_dtype` (or construct it at the precision
            you want) to control the inference side separately.
        mass: Per-atom mass ``(N,)`` or a scalar, in amu.
        dt: Timestep in fs. Required unless ``integrator`` is given.
        gamma: Langevin friction γ in fs⁻¹. ``0`` (default) integrates NVE —
            BAOAB's O step becomes the identity.
        kbt: Thermal energy for the Langevin O step, in the integrator's
            (amu, Å, fs) energy unit. Ignored at ``gamma=0``.
        temperature: Convenience alternative to ``kbt``, in kelvin.
        integrator: A constructed :class:`~molix.md.integrators.Integrator`
            over ``force`` — the seam for Nosé–Hoover, NPT, or any
            non-Langevin scheme. Mutually exclusive with
            ``dt``/``gamma``/``kbt``/``temperature``/``seed``, which
            parameterise the default :class:`LangevinVerletIntegrator`.
        dtype: MD-side precision — trajectory state, integrator constants,
            mass. ``None`` keeps the caller tensors' dtype. The force field is
            deliberately left alone (see the module docstring).
        autocast_dtype: Run each force evaluation under ``torch.autocast`` at
            this dtype, leaving parameters alone. This is the mixed-precision
            path (e.g. ``torch.bfloat16``). Not combinable with an explicit
            ``integrator`` (wrap the force field yourself in that case).
        rebuild_every: Rebuild the neighbour list every N **force evaluations**
            (one per MD step under BAOAB), at the positions being evaluated.
            ``None`` freezes the list — correct only for open systems or runs
            short enough that the initial list stays valid. ``1`` is the
            accurate NVE setting without a Verlet skin.

        hooks: Extra hooks, appended after the neighbour-list hook.
        seed: Seed for the Langevin noise.
        device: Device to place the force field and state on (device, unlike
            dtype, must be shared by both sides).
    """

    def __init__(
        self,
        force: ForceField,
        *,
        mass: float | torch.Tensor,
        dt: float | None = None,
        gamma: float = 0.0,
        kbt: float | None = None,
        temperature: float | None = None,
        integrator: Integrator | None = None,
        dtype: torch.dtype | None = None,
        autocast_dtype: torch.dtype | None = None,
        rebuild_every: int | None = None,
        hooks: Sequence[MDHook | tuple[MDHook, int]] | None = None,
        seed: int = 0,
        device: torch.device | str | None = None,
    ) -> None:
        self.dtype = dtype
        self.device = torch.device(device) if device is not None else None

        mass_t = torch.as_tensor(mass)
        if dtype is not None:
            mass_t = mass_t.to(dtype)
        if self.device is not None:
            mass_t = mass_t.to(self.device)
        self.mass = mass_t

        if integrator is not None:
            if dt is not None or kbt is not None or temperature is not None:
                raise ValueError(
                    "integrator= is mutually exclusive with dt/gamma/kbt/temperature/seed — "
                    "those parameterise the default LangevinVerletIntegrator; a constructed "
                    "integrator already owns them"
                )
            if autocast_dtype is not None:
                raise ValueError(
                    "autocast_dtype cannot wrap a constructed integrator's force field; "
                    "wrap the force field before building the integrator"
                )
            if integrator.force is not force:
                raise ValueError("integrator.force must be the force field given to MD")
        else:
            if dt is None:
                raise ValueError("dt is required when no integrator is given")
            if kbt is not None and temperature is not None:
                raise ValueError("give kbt or temperature, not both")
            if kbt is None:
                kbt = KB_AMU_A_FS * float(temperature) if temperature is not None else 0.0
            if gamma > 0.0 and kbt == 0.0:
                raise ValueError("gamma > 0 needs kbt or temperature (a thermostat at 0 K freezes)")

        # Order matters: the neighbour-list hook must see the *inner* force
        # field's rebuild, and the autocast wrapper forwards it, so wrapping
        # first and hooking the wrapper is equivalent and keeps one owner.
        if autocast_dtype is not None:
            force = _AutocastForceField(force, autocast_dtype)
        self.autocast_dtype = autocast_dtype

        if integrator is None:
            integrator = LangevinVerletIntegrator(
                force, dt=dt, gamma=gamma, kbt=kbt, mass=mass_t, seed=seed
            )
        # MD-side precision: the integrator's own step constants follow the
        # run dtype; the force field inside it is deliberately not cast.
        if dtype is not None:
            integrator = integrator.cast_state(dtype)
        # Device, unlike dtype, is shared — a cross-device force call cannot
        # work — so the move recurses through the force field too.
        if self.device is not None:
            integrator = integrator.to(self.device)
        self.force = force
        self.integrator = integrator

        # Neighbour-list rebuild belongs in Integrator.eval_force (at the
        # force-evaluation positions), NOT in a step-start hook: BAOAB/VV
        # evaluates F at the *end* of the step, so a step-start rebuild leaves
        # the list one displacement behind the positions in F = -∇E and also
        # leaves the cached half-kick force inconsistent with the new list.
        if rebuild_every is not None:
            if int(rebuild_every) < 1:
                raise ValueError(f"rebuild_every must be >= 1, got {rebuild_every}")
            integrator.rebuild_every = int(rebuild_every)
            integrator._force_eval_count = 0
        self.rebuild_every = rebuild_every
        run_hooks: list[MDHook | tuple[MDHook, int]] = list(hooks or [])
        self.runner = MDRunner(integrator, mass=mass_t, hooks=run_hooks)

    def set_potential_dtype(self, dtype: torch.dtype) -> "MD":
        """Cast the force field (the inference side) to ``dtype``.

        The explicit counterpart of the constructor's ``dtype``: MD-side and
        potential-side precision are independent axes, so casting the model is
        never a side effect of setting the trajectory precision. The
        integrator's boundary cast keeps the state in the MD dtype regardless
        of what the model computes in.
        """
        self.force.to(dtype)
        return self

    def run(
        self, pos: torch.Tensor, vel: torch.Tensor, n_steps: int, *, chunk: int | None = None
    ) -> MDState:
        """Integrate ``n_steps``, returning the final typed :class:`MDState`.

        Trajectory capture is a hook's job — pass a
        :class:`~molix.md.runner.TrajectoryHook`.

        Args:
            pos: Initial positions ``(N, 3)``, Angstrom.
            vel: Initial velocities ``(N, 3)``, Å/fs.
            n_steps: Number of steps.
            chunk: Steps advanced between hook firings (dynamics are
                bit-identical; only observation cadence changes). Default 1.
                Neighbour rebuild no longer couples to ``chunk`` — it runs
                inside each force evaluation when ``rebuild_every`` is set.
        """
        if chunk is None:
            chunk = 1
        pos, vel = self._cast(pos), self._cast(vel)
        return self.runner.run(pos, vel, n_steps, chunk=chunk)

    def _cast(self, tensor: torch.Tensor) -> torch.Tensor:
        """Bring a caller tensor onto the run's dtype/device."""
        if self.dtype is not None:
            tensor = tensor.to(self.dtype)
        if self.device is not None:
            tensor = tensor.to(self.device)
        return tensor
