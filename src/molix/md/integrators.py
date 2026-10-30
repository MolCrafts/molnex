"""Integrator components: advance an :class:`~molix.md.types.MDState`.

An ``Integrator`` is constructed with its own step parameters (``dt``, and
whatever else that scheme needs) and does **not** take a force field — the
caller binds a :class:`PotentialLike` with :meth:`Integrator.bind_force` (or
lets :class:`~molix.md.driver.MD` do it via ``set_forcefield`` /
``set_integrator``). The hot step is static and typed so
``torch.compile(fullgraph=True)`` can include a traceable force provider.

Neighbour-list cadence is **not** the integrator's: the list owns it (``skin`` /
``every`` / ``delay`` / ``check`` on :class:`~molix.md.neighbors.NeighborList`)
and :meth:`Integrator.eval_force` merely asks, once per force evaluation, at the
positions being evaluated. What the integrator keeps is a construction-time
on/off, :attr:`Integrator.rebuild`, derived from the bound provider.
Over a live list that question is an eager, host-syncing decision, so a
compiled rollout is built with ``rebuild=False`` (frozen list) while production
compiles the force path and keeps the loop eager.

Two schemes, two types — no ``gamma=0`` switch. Both are constructed with
``dt`` (and Langevin's γ / kT); mass is **not** an integrator parameter —
it lives on ``atoms.mass`` and is bound with :meth:`Integrator.bind_mass`.

* :class:`VelocityVerletIntegrator` — NVE (B-A-A-B; the two half-drifts stay
  as separate adds).
* :class:`LangevinVerletIntegrator` — BAOAB Langevin (γ > 0). Ordering
  (Leimkuhler & Matthews): B (half kick) → A (half drift) → O
  (Ornstein-Uhlenbeck) → A → B. The O step ``v ← c1·v + c2·σ·ξ`` with
  ``c1 = e^{-γΔt}``, ``c2 = √(1-c1²)``, ``σ = √(k_BT/m)``.

Units — one self-consistent system. The arithmetic ``v += Δt·F/m`` needs
``[F] = [m][length]/[time]²``, so eV/Å + amu + fs is **not** consistent. The
canonical system is (amu, Å, fs) with energy in amu·Å²/fs²
(``(1.0 * units.eV).to("amu * angstrom ** 2 / femtosecond ** 2")``).
:class:`~molix.md.driver.MD` converts ``ForceField.units`` through
:func:`molix.units.preset_energy_to_md`.

Precision boundary: the force provider owns its own dtype; ``MD(dtype)``
governs the trajectory state. :meth:`Integrator.eval_force` casts the
output back to the state dtype so the two precisions never silently promote
mid-step.

Reference:
    Leimkuhler & Matthews, "Rational Construction of Stochastic Numerical
    Methods for Molecular Sampling", Appl. Math. Res. Express 2013.
    https://doi.org/10.1093/amrx/abs010
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import torch
from torch import nn
from torch.utils._pytree import tree_map

from molix.md.types import MDState

#: Force-provider return: ``energy`` and ``forces`` required, ``virial`` optional.
ForceDict = dict[str, torch.Tensor]


@runtime_checkable
class PotentialLike(Protocol):
    """Internal ``pos -> dict`` provider bound on an integrator.

    Not a public type and not :class:`molpy.ff.forcefield.ForceField`. Neighbour-list
    cadence lives on :class:`~molix.md.driver.MD.compute_force`, not here.
    """

    def __call__(self, pos: torch.Tensor) -> ForceDict:
        """Energy and forces at ``pos`` ``(N, 3)`` Å.

        Args:
            pos: Positions ``(N, 3)`` in Å.

        Returns:
            ``{"energy": (), "forces": (N, 3)}`` in amu·Å²/fs²; optional
            ``"virial": (3, 3)``. Extra keys are ignored by the integrator.
        """
        ...


def _state_from(
    pos: torch.Tensor,
    vel: torch.Tensor,
    out: ForceDict,
    cell: torch.Tensor | None,
) -> MDState:
    """Pack energy / forces / virial from the force dict onto the state."""
    return MDState(pos, vel, out["forces"], out["energy"], cell, out.get("virial"))


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


def _require_positive_mass(mass: float | torch.Tensor) -> float | torch.Tensor:
    if isinstance(mass, torch.Tensor):
        if not bool((mass > 0).all()):
            raise ValueError("mass must be strictly positive")
        return mass
    if mass <= 0:
        raise ValueError("mass must be strictly positive")
    return mass


class Integrator(nn.Module):
    """Abstract integrator. Constructed without a force field.

    The contract :class:`~molix.md.runner.MDRunner` drives — a conforming
    subclass implements :meth:`advance` (one eager step) and :meth:`rollout`
    (the compile-friendly fixed-length loop) and inherits the rest:
    :meth:`initial` seeds the state, :meth:`advance_n` chunks eager steps
    between observations, and :attr:`removed_dof` states the
    temperature-estimator convention.

    """

    def __init__(self, dt: float) -> None:
        super().__init__()
        self.force: PotentialLike | None = None
        #: Time step, fs — a buffer so it moves with the module.
        self.register_buffer("dt", torch.as_tensor(float(dt)))

    def bind_force(self, force: PotentialLike) -> Integrator:
        """Attach the :class:`PotentialLike` this integrator will evaluate.

        Args:
            force: Internal ``pos -> dict`` provider (MD adapter or an
                unnamed test helper). Not a :class:`molpy.ff.forcefield.ForceField`.

        Returns:
            ``self``.
        """
        self.force = force
        return self

    def bind_mass(self, mass: float | torch.Tensor) -> Integrator:
        """Attach per-atom (or scalar) mass from ``atoms.mass``.

        Mass is system data, not a scheme parameter. The runner binds it from
        the TensorDict at the start of a run; direct integrator tests bind it
        the same way.
        """
        mass = _require_positive_mass(mass)
        if isinstance(mass, torch.Tensor):
            ref = mass
        else:
            ref = self.dt
        mass_col = _as_mass_col(mass, ref)
        self.mass: float | torch.Tensor = mass
        self.register_buffer("mass_col", mass_col)
        self.register_buffer("inv_mass", mass_col.reciprocal())
        return self

    @property
    def removed_dof(self) -> int:
        """Degrees of freedom the temperature estimator must not count.

        ``3`` by default — a deterministic integrator conserves the (removed)
        centre-of-mass momentum, leaving ``3N - 3``. Thermostatted integrators
        that agitate all ``3N`` DoF override this to ``0``.
        """
        return 3

    def eval_force(self, pos: torch.Tensor) -> ForceDict:
        """Evaluate the force field, casting every floating tensor to ``pos.dtype``.

        Neighbour-list cadence is :meth:`~molix.md.driver.MD.compute_force`'s
        job. Tree-maps the bound provider's dict so ``virial`` (and any extra
        keys) keep the state dtype; the integrator only reads ``energy``,
        ``forces``, and ``virial``.
        """
        if self.force is None:
            raise RuntimeError("bind a force field before evaluating")
        out = self.force(pos)
        dtype = pos.dtype

        def _cast(value: object) -> object:
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                return value.to(dtype)
            return value

        return tree_map(_cast, out)

    def initial(
        self,
        pos: torch.Tensor,
        vel: torch.Tensor,
        cell: torch.Tensor | None = None,
    ) -> MDState:
        """Seed an :class:`~molix.md.types.MDState`, evaluating the entry force."""
        out = self.eval_force(pos)
        return _state_from(pos, vel, out, cell)

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

    def cast_state(self, dtype: torch.dtype) -> Integrator:
        """Cast this integrator's own step-constant buffers to ``dtype``.

        Unlike ``.to(dtype)`` this does **not** recurse into the force field:
        MD-side precision and the potential's precision are independent.
        """
        for name, buf in self.named_buffers(recurse=False):
            if buf.is_floating_point():
                self._buffers[name] = buf.to(dtype)
        return self


class VelocityVerletIntegrator(Integrator):
    """NVE velocity-Verlet (B-A-A-B). Constructed with ``dt`` only.

    Mass comes from ``atoms.mass`` via :meth:`Integrator.bind_mass`.

    Args:
        dt: Timestep Δt in fs.
    """

    def __init__(self, dt: float) -> None:
        super().__init__(dt)

    def step(self, state: MDState) -> MDState:
        """One NVE step from the cached entry force.

        The two half-drifts stay as separate adds (not fused into one full
        drift) so the arithmetic matches the BAOAB A-A pair with an identity
        O step bit-for-bit.
        """
        half_dt = 0.5 * self.dt
        vel = state.vel + half_dt * state.forces * self.inv_mass  # B (cached force)
        pos = state.pos + half_dt * vel  # A
        pos = pos + half_dt * vel  # A
        out = self.eval_force(pos)
        vel = vel + half_dt * out["forces"] * self.inv_mass  # B
        return _state_from(pos, vel, out, state.cell)

    def advance(self, state: MDState) -> MDState:
        return self.step(state)

    def rollout(self, state: MDState, n_steps: int) -> MDState:
        for _ in range(n_steps):
            state = self.step(state)
        return state


class LangevinVerletIntegrator(Integrator):
    """Langevin velocity-Verlet (BAOAB). γ must be strictly positive.

    NVE is :class:`VelocityVerletIntegrator` — not this type with ``gamma=0``.
    Mass comes from ``atoms.mass`` via :meth:`Integrator.bind_mass`.

    Args:
        dt: Timestep Δt in fs.
        gamma: Langevin friction γ in fs⁻¹, strictly positive.
        kbt: Thermal energy k_B·T in the integrator's energy unit.
        seed: Seed for the eager noise generator (reproducible :meth:`advance`).
            :meth:`rollout` uses global RNG so it stays compilable.
    """

    def __init__(
        self,
        dt: float,
        *,
        gamma: float,
        kbt: float,
        seed: int = 0,
    ) -> None:
        super().__init__(dt)
        if float(gamma) <= 0.0:
            raise ValueError(
                "LangevinVerletIntegrator requires gamma > 0; use VelocityVerletIntegrator for NVE"
            )
        if float(kbt) <= 0.0:
            raise ValueError("LangevinVerletIntegrator requires kbt > 0")
        self.gamma = float(gamma)
        self._kbt = float(kbt)
        self._seed = int(seed)
        c1 = math.exp(-float(gamma) * float(dt))
        c2 = math.sqrt(max(0.0, 1.0 - c1 * c1))
        self.register_buffer("c1", torch.as_tensor(c1))
        self.register_buffer("c2", torch.as_tensor(c2))
        self._generator: torch.Generator | None = None

    def bind_mass(self, mass: float | torch.Tensor) -> Integrator:
        """Bind ``atoms.mass`` and rebuild the fluctuation-dissipation σ."""
        super().bind_mass(mass)
        self.register_buffer("sigma", math.sqrt(self._kbt) * self.mass_col.rsqrt())
        return self

    @property
    def removed_dof(self) -> int:
        """``0`` — the O step agitates all 3N DoF, COM included."""
        return 0

    def step(self, state: MDState, noise: torch.Tensor) -> MDState:
        """One BAOAB step from the cached entry force and a pre-drawn ``noise``."""
        half_dt = 0.5 * self.dt
        vel = state.vel + half_dt * state.forces * self.inv_mass  # B (cached force)
        pos = state.pos + half_dt * vel  # A
        vel = self.c1 * vel + self.c2 * self.sigma * noise  # O
        pos = pos + half_dt * vel  # A
        out = self.eval_force(pos)
        vel = vel + half_dt * out["forces"] * self.inv_mass  # B
        return _state_from(pos, vel, out, state.cell)

    def advance_n(self, state: MDState, n_steps: int) -> MDState:
        for _ in range(n_steps):
            state = self.step(state, self.draw_noise(state.vel))
        return state

    def draw_noise(self, ref: torch.Tensor) -> torch.Tensor:
        """Reproducible O-step noise ``(N, 3)`` from a seeded generator (eager)."""
        if self._generator is None or self._generator.device != ref.device:
            self._generator = torch.Generator(device=ref.device).manual_seed(self._seed)
        return torch.randn(ref.shape, generator=self._generator, dtype=ref.dtype, device=ref.device)

    def advance(self, state: MDState) -> MDState:
        """Eager single step: draw reproducible noise, then :meth:`step`."""
        return self.step(state, self.draw_noise(state.vel))

    def rollout(self, state: MDState, n_steps: int) -> MDState:
        """Advance ``n_steps`` (compile-friendly; global ``torch.randn_like``)."""
        for _ in range(n_steps):
            state = self.step(state, torch.randn_like(state.vel))
        return state
