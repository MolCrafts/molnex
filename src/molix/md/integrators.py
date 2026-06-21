"""Model-agnostic Langevin velocity-Verlet integrator (BAOAB splitting).

The integrator is decoupled from any potential: it accepts a ``force_fn`` mapping
positions ``(N, 3)`` to ``(energy, forces)`` so it can be unit-tested against an
analytic toy potential (e.g. a harmonic oscillator) without PiNet.

BAOAB ordering per step (Leimkuhler & Matthews): B (half kick) → A (half drift) →
O (Ornstein-Uhlenbeck friction + noise) → A (half drift) → B (half kick). With
``gamma = 0`` the O step is the identity and BAOAB reduces to plain velocity-Verlet
(NVE). The O step satisfies the per-degree-of-freedom fluctuation-dissipation
relation, so the long-time kinetic energy obeys equipartition ⟨½ m v²⟩ = ½ k_B T.

Units are caller-defined but must be mutually consistent (e.g. forces eV/Å, mass
amu, Δt fs, γ 1/fs, ``kbt`` = k_B·T in eV).

``torch.compile``. The hot per-step update is factored into a pure tensor-in /
tensor-out core, :meth:`LangevinVerletIntegrator._step_core`, that takes the
entry force AND the O-step noise as explicit tensor arguments — no Python
``Generator`` object, no lazy state mutation — so it compiles to a single graph
(``fullgraph=True``) provided ``force_fn`` itself is traceable (a PiNet forward
with fixed-shape edges is, verified graph-break-free). Pass ``compile=True`` to
wrap the core in :func:`torch.compile`; noise is still drawn eagerly outside the
compiled region. The loop also caches the entry force across steps (one
``force_fn`` evaluation per step instead of two), numerically identical to the
naive two-evaluation form because the end-of-step force at ``x_{n+1}`` is exactly
the start-of-step force of step ``n+1``.

Reference:
    Leimkuhler & Matthews, "Rational Construction of Stochastic Numerical
    Methods for Molecular Sampling", Appl. Math. Res. Express 2013.
    https://doi.org/10.1093/amrx/abs010
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

ForceFn = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


class LangevinVerletIntegrator:
    """Langevin velocity-Verlet (BAOAB) integrator over an injected force function.

    Args:
        force_fn: Maps positions ``(N, 3)`` to ``(energy, forces (N, 3))``.
        dt: Timestep Δt.
        gamma: Langevin friction γ (``0`` → NVE).
        kbt: Thermal energy k_B·T (energy units).
        mass: Particle mass — scalar or per-atom tensor broadcastable to ``(N, 1)``.
        seed: Seed for the noise generator (reproducible trajectories).
        compile: When ``True``, wrap the per-step core in :func:`torch.compile`
            (``fullgraph=True, dynamic=False``) the first time it runs. Requires
            ``force_fn`` to be traceable and fixed-shape across steps; for PiNet
            that means a fixed-shape edge list. Default ``False`` (eager).
    """

    def __init__(
        self,
        force_fn: ForceFn,
        *,
        dt: float,
        gamma: float,
        kbt: float,
        mass: float | torch.Tensor,
        seed: int = 0,
        compile: bool = False,
    ) -> None:
        self.force_fn = force_fn
        self.dt = float(dt)
        self.gamma = float(gamma)
        self.kbt = float(kbt)
        self._mass = mass
        self._seed = int(seed)
        self._c1 = math.exp(-self.gamma * self.dt)
        self._c2 = math.sqrt(max(0.0, 1.0 - self._c1 * self._c1))
        self._generator: torch.Generator | None = None
        self._compile = bool(compile)
        self._stepper: ForceFn | None = None  # lazily-built (maybe compiled) core driver

    def _mass_col(self, ref: torch.Tensor) -> torch.Tensor:
        """Mass reshaped to broadcast against ``(N, 3)`` in ref's dtype/device."""
        if isinstance(self._mass, torch.Tensor):
            m = self._mass.to(dtype=ref.dtype, device=ref.device)
            return m.reshape(-1, 1) if m.dim() == 1 else m
        return torch.as_tensor(self._mass, dtype=ref.dtype, device=ref.device)

    def _noise(self, ref: torch.Tensor) -> torch.Tensor:
        if self._generator is None or self._generator.device != ref.device:
            self._generator = torch.Generator(device=ref.device).manual_seed(self._seed)
        return torch.randn(ref.shape, generator=self._generator, dtype=ref.dtype, device=ref.device)

    def _step_core(
        self,
        pos: torch.Tensor,
        vel: torch.Tensor,
        force: torch.Tensor,
        noise: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One BAOAB step from a *cached entry force* and a *pre-drawn noise*.

        Pure tensor-in / tensor-out (no Generator, no lazy state) so it is
        ``torch.compile``-able. ``force`` must equal ``force_fn(pos)[1]``; the
        returned force is ``force_fn`` at the new position, ready to feed back as
        the next step's ``force`` (force caching → one evaluation per step). The
        ``noise`` argument is consumed only when ``gamma > 0`` (the O step);
        callers pass a dummy for NVE.

        Args:
            pos: Positions ``(N, 3)``.
            vel: Velocities ``(N, 3)``.
            force: Entry force ``(N, 3)`` ``= force_fn(pos)[1]``.
            noise: Standard-normal tensor ``(N, 3)`` for the O step.

        Returns:
            ``(pos, vel, energy, force)`` at the advanced position.
        """
        mass = self._mass_col(pos)
        half_dt = 0.5 * self.dt
        vel = vel + half_dt * force / mass  # B (uses cached force at pos)
        pos = pos + half_dt * vel  # A
        if self.gamma > 0.0:  # O
            sigma = math.sqrt(self.kbt) * mass.rsqrt()  # sqrt(kbt/m) per DoF
            vel = self._c1 * vel + self._c2 * sigma * noise
        pos = pos + half_dt * vel  # A
        energy, force = self.force_fn(pos)
        vel = vel + half_dt * force / mass  # B
        return pos, vel, energy, force

    def _draw_noise(self, vel: torch.Tensor) -> torch.Tensor:
        """O-step noise ``(N, 3)``: real draw under Langevin, zeros under NVE.

        Drawn *outside* the compiled core to keep the ``Generator`` out of the
        graph. Under NVE (``gamma == 0``) the O step is skipped, so a cheap zero
        tensor is returned without touching the RNG (preserving the no-RNG NVE
        contract the legacy :meth:`step` had)."""
        if self.gamma > 0.0:
            return self._noise(vel)
        return torch.zeros_like(vel)

    def _get_stepper(self) -> ForceFn:
        """Return the per-step driver, building (and optionally compiling) once."""
        if self._stepper is None:
            self._stepper = (
                torch.compile(self._step_core, fullgraph=True, dynamic=False)
                if self._compile
                else self._step_core
            )
        return self._stepper

    def initial_force(self, pos: torch.Tensor) -> torch.Tensor:
        """Entry force ``force_fn(pos)[1]`` to seed a force-cached stepping loop."""
        return self.force_fn(pos)[1]

    def step_cached(
        self, pos: torch.Tensor, vel: torch.Tensor, force: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One step from a cached entry force; returns ``(pos, vel, energy, force)``.

        The advance primitive shared by :meth:`run` and :class:`molix.md.MDRunner`:
        draws the O-step noise eagerly, then calls the (possibly compiled) core.
        Pair with :meth:`initial_force` to seed the first ``force``.
        """
        return self._get_stepper()(pos, vel, force, self._draw_noise(vel))

    def step(
        self, pos: torch.Tensor, vel: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance one BAOAB step, computing the entry force internally.

        Kept for direct callers and unit tests that drive the integrator one
        step at a time. :meth:`run` uses :meth:`step_cached` (force caching) for
        a single ``force_fn`` evaluation per step.
        """
        _, force = self.force_fn(pos)
        return self._step_core(pos, vel, force, self._draw_noise(vel))

    def run(self, pos: torch.Tensor, vel: torch.Tensor, n_steps: int) -> dict[str, torch.Tensor]:
        """Integrate ``n_steps`` steps, recording per-step pos/vel/energy.

        Uses force caching (one ``force_fn`` evaluation per step) and, when the
        integrator was built with ``compile=True``, a :func:`torch.compile`-d
        step core. Numerically identical to the legacy per-step :meth:`step`
        loop.

        Returns:
            Dict with ``pos`` / ``vel`` of shape ``(n_steps, N, 3)`` and ``energy``
            of shape ``(n_steps,)`` (detached).
        """
        pos_hist: list[torch.Tensor] = []
        vel_hist: list[torch.Tensor] = []
        energy_hist: list[torch.Tensor] = []
        force = self.initial_force(pos)  # seed the cache; reused as each step's entry force
        for _ in range(n_steps):
            pos, vel, energy, force = self.step_cached(pos, vel, force)
            pos_hist.append(pos.detach().clone())
            vel_hist.append(vel.detach().clone())
            energy_hist.append(energy.detach().reshape(()).clone())
        return {
            "pos": torch.stack(pos_hist),
            "vel": torch.stack(vel_hist),
            "energy": torch.stack(energy_hist),
        }
