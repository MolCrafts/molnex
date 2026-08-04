"""Integrator components: advance an :class:`~molix.md.types.MDState` over a
:class:`~molix.md.forcefield.ForceField` (BAOAB Langevin velocity-Verlet).

An ``Integrator`` is an :class:`torch.nn.Module` that *holds a ``ForceField``
component* (not a closure) and advances state. The hot step is static and
typed — no ``isinstance`` on mass, no ``if gamma > 0`` branch (the O step is
written as an identity-at-γ=0 update), no ``None`` noise — so
:meth:`LangevinVerletIntegrator.step` and :meth:`LangevinVerletIntegrator.rollout`
``torch.compile(fullgraph=True)`` to a single graph *including* a traceable
force field (PiNet's functorch force path is graph-break-free).

BAOAB ordering (Leimkuhler & Matthews): B (half kick) → A (half drift) → O
(Ornstein-Uhlenbeck) → A → B. The O step ``v ← c1·v + c2·σ·ξ`` with
``c1 = e^{-γΔt}``, ``c2 = √(1-c1²)``, ``σ = √(k_BT/m)`` satisfies the
per-DoF fluctuation-dissipation relation. At ``γ = 0``: ``c1 = 1``, ``c2 = 0`` →
the O step is the identity (``0·σ·ξ = 0``) and BAOAB reduces to velocity-Verlet
(NVE) — which is exactly why the branch can be dropped.

Units — one self-consistent system. The arithmetic ``v += Δt·F/m`` needs
``[F] = [m][length]/[time]²``, so eV/Å + amu + fs is **not** consistent. The
canonical system is (amu, Å, fs) with energy in amu·Å²/fs² (= ``EV_PER_AMU_A2_FS2``
eV); drive an eV/Å potential by converting at the force field
(``PotentialForceField(..., energy_scale=1/EV_PER_AMU_A2_FS2)``).

Reference:
    Leimkuhler & Matthews, "Rational Construction of Stochastic Numerical
    Methods for Molecular Sampling", Appl. Math. Res. Express 2013.
    https://doi.org/10.1093/amrx/abs010
"""

from __future__ import annotations

import math

import torch
from torch import nn

from molix.md.forcefield import ForceField
from molix.md.types import MDState

#: Energy-unit bridge: 1 amu·Å²/fs² = 103.6426965638 eV.
EV_PER_AMU_A2_FS2 = 103.6426965638


def as_mass_col(mass: float | torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Mass reshaped to broadcast against ``(N, 3)`` in ``ref``'s dtype/device.

    A per-atom ``(N,)`` tensor becomes ``(N, 1)``; a scalar stays scalar. Shared
    by the integrator and :class:`molix.md.MDRunner` so both use one mass
    convention for kinetic energy.
    """
    if isinstance(mass, torch.Tensor):
        m = mass.to(dtype=ref.dtype, device=ref.device)
        return m.reshape(-1, 1) if m.dim() == 1 else m
    return torch.as_tensor(mass, dtype=ref.dtype, device=ref.device)


class Integrator(nn.Module):
    """Abstract integrator over a :class:`~molix.md.forcefield.ForceField` component.

    Args:
        force: The force-field component supplying ``forward(pos) -> ForceOutput``.
    """

    def __init__(self, force: ForceField) -> None:
        super().__init__()
        self.force = force

    def initial(self, pos: torch.Tensor, vel: torch.Tensor) -> MDState:
        """Seed an :class:`~molix.md.types.MDState`, evaluating the entry force."""
        out = self.force(pos)
        return MDState(pos, vel, out.forces, out.energy)

    def step(self, state: MDState, noise: torch.Tensor) -> MDState:  # noqa: D102
        raise NotImplementedError

    def rollout(self, state: MDState, n_steps: int) -> MDState:  # noqa: D102
        raise NotImplementedError


class LangevinVerletIntegrator(Integrator):
    """Langevin velocity-Verlet (BAOAB) over a force-field component.

    Args:
        force: Force-field component (``forward(pos) -> ForceOutput``).
        dt: Timestep Δt.
        gamma: Langevin friction γ (``0`` → NVE; the O step becomes the identity).
        kbt: Thermal energy k_B·T (energy units).
        mass: Particle mass — scalar or per-atom ``(N,)`` tensor, strictly positive.
        seed: Seed for the eager noise generator (reproducible :meth:`advance` /
            :meth:`run`). :meth:`rollout` uses global RNG so it stays compilable.

    Scalar parameters are immutable after construction (baked into the compiled
    graph). Buffers ``dt``/``c1``/``c2``/``mass_col``/``inv_mass``/``sigma`` carry
    the precomputed step constants; nothing is recomputed per step.
    """

    def __init__(
        self,
        force: ForceField,
        *,
        dt: float,
        gamma: float,
        kbt: float,
        mass: float | torch.Tensor,
        seed: int = 0,
    ) -> None:
        super().__init__(force)
        if isinstance(mass, torch.Tensor):
            if not bool((mass > 0).all()):
                raise ValueError("mass must be strictly positive")
        elif mass <= 0:
            raise ValueError("mass must be strictly positive")
        self.gamma = float(gamma)
        self._seed = int(seed)
        c1 = math.exp(-float(gamma) * float(dt))
        c2 = math.sqrt(max(0.0, 1.0 - c1 * c1))
        ref = torch.zeros(())  # CPU fp32 reference for buffer construction
        mass_col = as_mass_col(mass, ref)
        self.register_buffer("dt", torch.as_tensor(float(dt)))
        self.register_buffer("c1", torch.as_tensor(c1))
        self.register_buffer("c2", torch.as_tensor(c2))
        self.register_buffer("mass_col", mass_col)
        self.register_buffer("inv_mass", mass_col.reciprocal())
        self.register_buffer("sigma", math.sqrt(float(kbt)) * mass_col.rsqrt())
        self._generator: torch.Generator | None = None

    def step(self, state: MDState, noise: torch.Tensor) -> MDState:
        """One BAOAB step from the cached entry force and a pre-drawn ``noise``.

        Pure and static (no Python branch on γ, no ``isinstance``): the O step is
        applied unconditionally and is the identity when γ = 0 (c2 = 0). Returns
        the advanced state with the end-of-step force/energy cached for the next
        step (one force-field evaluation per step). ``torch.compile``-able.
        """
        half_dt = 0.5 * self.dt
        vel = state.vel + half_dt * state.force * self.inv_mass  # B (cached force)
        pos = state.pos + half_dt * vel  # A
        vel = self.c1 * vel + self.c2 * self.sigma * noise  # O (identity at γ=0)
        pos = pos + half_dt * vel  # A
        out = self.force(pos)
        vel = vel + half_dt * out.forces * self.inv_mass  # B
        return MDState(pos, vel, out.forces, out.energy)

    def draw_noise(self, ref: torch.Tensor) -> torch.Tensor:
        """Reproducible O-step noise ``(N, 3)`` from a seeded generator (eager).

        Used by :meth:`advance` / :meth:`run`; kept out of :meth:`rollout` so the
        compiled path has no ``Generator`` object in the graph.
        """
        if self._generator is None or self._generator.device != ref.device:
            self._generator = torch.Generator(device=ref.device).manual_seed(self._seed)
        return torch.randn(ref.shape, generator=self._generator, dtype=ref.dtype, device=ref.device)

    def advance(self, state: MDState) -> MDState:
        """Eager single step: draw reproducible noise, then :meth:`step`."""
        return self.step(state, self.draw_noise(state.vel))

    def rollout(self, state: MDState, n_steps: int) -> MDState:
        """Advance ``n_steps`` and return the final state (compile-friendly).

        Draws noise with global ``torch.randn_like`` inside the loop so the whole
        rollout — including the force field — is ``torch.compile(fullgraph=True)``
        traceable. Seed the global RNG (``torch.manual_seed``) for reproducibility.
        """
        for _ in range(n_steps):
            state = self.step(state, torch.randn_like(state.vel))
        return state

    def run(
        self, pos: torch.Tensor, vel: torch.Tensor, n_steps: int, *, stride: int = 1
    ) -> dict[str, torch.Tensor]:
        """Eager trajectory: record every ``stride``-th frame's pos/vel/energy.

        History is detached and moved to CPU as recorded, so device memory does
        not grow with ``n_steps`` (host memory grows O(T·N/stride); raise
        ``stride`` or use :class:`molix.md.TrajectoryHook` for long runs).
        Reproducible via the seeded generator.

        Returns:
            ``pos`` / ``vel`` ``(⌈n_steps/stride⌉, N, 3)`` and ``energy``
            ``(⌈n_steps/stride⌉,)`` (detached, on CPU).
        """
        stride = max(1, int(stride))
        pos_hist: list[torch.Tensor] = []
        vel_hist: list[torch.Tensor] = []
        energy_hist: list[torch.Tensor] = []
        state = self.initial(pos, vel)
        for i in range(n_steps):
            state = self.advance(state)
            if (i + 1) % stride == 0:
                pos_hist.append(state.pos.detach().to("cpu"))
                vel_hist.append(state.vel.detach().to("cpu"))
                energy_hist.append(state.energy.detach().reshape(()).to("cpu"))
        return {
            "pos": torch.stack(pos_hist),
            "vel": torch.stack(vel_hist),
            "energy": torch.stack(energy_hist),
        }
