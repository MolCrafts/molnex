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
canonical system is (amu, Å, fs) with energy in amu·Å²/fs²
(= :data:`molix.units.EV_PER_AMU_A2_FS2` eV); drive an eV/Å potential by
converting at the force field
(``PotentialForceField(..., energy_scale=1/EV_PER_AMU_A2_FS2)``).

Precision boundary: the force field owns its own dtype, independently of the
trajectory state's (``MD(dtype=)`` governs the state; ``MD.set_potential_dtype``
the model). :meth:`Integrator.eval_force` casts the force field's output back to
the state dtype so the two precisions never silently promote mid-step.

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
from molix.md.types import ForceOutput, MDState


def _as_mass_col(mass: float | torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
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

    The contract :class:`~molix.md.runner.MDRunner` drives — a conforming
    subclass implements :meth:`advance` (one eager step) and :meth:`rollout`
    (the compile-friendly fixed-length loop) and inherits the rest:
    :meth:`initial` seeds the state, :meth:`advance_n` chunks eager steps
    between observations, and :attr:`removed_dof` states the
    temperature-estimator convention.

    Args:
        force: The force-field component supplying ``forward(pos) -> ForceOutput``.
    """

    def __init__(self, force: ForceField) -> None:
        super().__init__()
        self.force = force

    @property
    def removed_dof(self) -> int:
        """Degrees of freedom the temperature estimator must not count.

        ``3`` by default — a deterministic integrator conserves the (removed)
        centre-of-mass momentum, leaving ``3N - 3``. Thermostatted integrators
        that agitate all ``3N`` DoF (Langevin's O step includes the COM)
        override this to ``0``.
        """
        return 3

    def eval_force(self, pos: torch.Tensor) -> ForceOutput:
        """Evaluate the force field, casting its output to the state dtype.

        The force field owns its own precision (deliberately independent of the
        trajectory's — see :class:`molix.md.driver.MD`); the state must not
        silently promote, so energy/forces come back in ``pos``'s dtype. A
        same-dtype ``.to`` is the identity, so the matched case costs nothing.
        """
        out = self.force(pos)
        return ForceOutput(out.energy.to(pos.dtype), out.forces.to(pos.dtype))

    def initial(self, pos: torch.Tensor, vel: torch.Tensor) -> MDState:
        """Seed an :class:`~molix.md.types.MDState`, evaluating the entry force."""
        out = self.eval_force(pos)
        return MDState(pos, vel, out.forces, out.energy)

    def advance(self, state: MDState) -> MDState:
        """One eager step."""
        raise NotImplementedError

    def advance_n(self, state: MDState, n_steps: int) -> MDState:
        """Advance ``n_steps`` eagerly; subclasses may specialise the loop."""
        for _ in range(n_steps):
            state = self.advance(state)
        return state

    def rollout(self, state: MDState, n_steps: int) -> MDState:
        """Advance ``n_steps`` and return the final state (compile-friendly)."""
        raise NotImplementedError

    def cast_state(self, dtype: torch.dtype) -> "Integrator":
        """Cast this integrator's own step-constant buffers to ``dtype``.

        Unlike ``.to(dtype)`` this does **not** recurse into the force field:
        the MD-side precision and the potential's precision are independent
        concerns (:class:`molix.md.driver.MD` casts the two separately).
        """
        for name, buf in self.named_buffers(recurse=False):
            if buf.is_floating_point():
                self._buffers[name] = buf.to(dtype)
        return self


class LangevinVerletIntegrator(Integrator):
    """Langevin velocity-Verlet (BAOAB) over a force-field component.

    Args:
        force: Force-field component (``forward(pos) -> ForceOutput``).
        dt: Timestep Δt in fs.
        gamma: Langevin friction γ in fs⁻¹ (``0`` → NVE; the O step becomes the
            identity).
        kbt: Thermal energy k_B·T (energy units).
        mass: Particle mass — scalar or per-atom ``(N,)`` tensor, strictly positive.
        seed: Seed for the eager noise generator (reproducible :meth:`advance`).
            :meth:`rollout` uses global RNG so it stays compilable.

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
        mass_col = _as_mass_col(mass, ref)
        self.register_buffer("dt", torch.as_tensor(float(dt)))
        self.register_buffer("c1", torch.as_tensor(c1))
        self.register_buffer("c2", torch.as_tensor(c2))
        self.register_buffer("mass_col", mass_col)
        self.register_buffer("inv_mass", mass_col.reciprocal())
        self.register_buffer("sigma", math.sqrt(float(kbt)) * mass_col.rsqrt())
        self._generator: torch.Generator | None = None

    @property
    def removed_dof(self) -> int:
        """``0`` under the thermostat (γ>0 agitates all 3N DoF, COM included); ``3`` NVE."""
        return 0 if self.gamma > 0.0 else 3

    def step(self, state: MDState, noise: torch.Tensor) -> MDState:
        """One BAOAB step from the cached entry force and a pre-drawn ``noise``.

        Pure and static (no Python branch on γ, no ``isinstance``): the O step is
        applied unconditionally and is the identity when γ = 0 (c2 = 0). Returns
        the advanced state with the end-of-step force/energy cached for the next
        step (one force-field evaluation per step). ``torch.compile``-able.
        """
        half_dt = 0.5 * self.dt
        vel = state.vel + half_dt * state.forces * self.inv_mass  # B (cached force)
        pos = state.pos + half_dt * vel  # A
        vel = self.c1 * vel + self.c2 * self.sigma * noise  # O (identity at γ=0)
        pos = pos + half_dt * vel  # A
        out = self.eval_force(pos)
        vel = vel + half_dt * out.forces * self.inv_mass  # B
        return MDState(pos, vel, out.forces, out.energy)

    def step_nve(self, state: MDState) -> MDState:
        """One γ=0 step: BAOAB with the identity O step elided.

        Bit-identical to ``step(state, noise)`` at ``gamma=0`` — there
        ``c1=1, c2=0``, so ``v ← 1.0·v + 0.0·σ·ξ`` is the identity in floating
        point too (multiply by 1.0 and add of +0.0 are exact). The two half
        drifts are kept as **separate adds** to preserve that bit identity;
        fusing them into one full drift would reassociate. Used by
        :meth:`advance_n` so long NVE runs skip the per-step noise draw; the
        compiled :meth:`rollout` keeps the branch-free :meth:`step` per the
        md-component-engine spec.
        """
        half_dt = 0.5 * self.dt
        vel = state.vel + half_dt * state.forces * self.inv_mass  # B (cached force)
        pos = state.pos + half_dt * vel  # A
        pos = pos + half_dt * vel  # A (O step elided: identity at γ=0)
        out = self.eval_force(pos)
        vel = vel + half_dt * out.forces * self.inv_mass  # B
        return MDState(pos, vel, out.forces, out.energy)

    def advance_n(self, state: MDState, n_steps: int) -> MDState:
        """Advance ``n_steps`` eagerly with no per-step host work.

        The γ selection is a construction-time Python branch out here in the
        eager driver — :meth:`step` itself stays branch-free (spec invariant).
        At γ=0 this also skips the per-step ``randn`` whose contribution the
        O step would multiply by ``c2=0`` anyway.
        """
        if self.gamma == 0.0:
            for _ in range(n_steps):
                state = self.step_nve(state)
            return state
        for _ in range(n_steps):
            state = self.step(state, self.draw_noise(state.vel))
        return state

    def draw_noise(self, ref: torch.Tensor) -> torch.Tensor:
        """Reproducible O-step noise ``(N, 3)`` from a seeded generator (eager).

        Used by :meth:`advance` / :meth:`advance_n`; kept out of :meth:`rollout`
        so the compiled path has no ``Generator`` object in the graph.
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
