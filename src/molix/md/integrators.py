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

    def step(
        self, pos: torch.Tensor, vel: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance one BAOAB step; returns ``(pos, vel, energy, forces)``."""
        mass = self._mass_col(pos)
        half_dt = 0.5 * self.dt

        _, force = self.force_fn(pos)
        vel = vel + half_dt * force / mass  # B
        pos = pos + half_dt * vel  # A
        if self.gamma > 0.0:  # O
            sigma = math.sqrt(self.kbt) * mass.rsqrt()  # sqrt(kbt/m) per DoF
            vel = self._c1 * vel + self._c2 * sigma * self._noise(vel)
        pos = pos + half_dt * vel  # A
        energy, force = self.force_fn(pos)
        vel = vel + half_dt * force / mass  # B
        return pos, vel, energy, force

    def run(self, pos: torch.Tensor, vel: torch.Tensor, n_steps: int) -> dict[str, torch.Tensor]:
        """Integrate ``n_steps`` steps, recording per-step pos/vel/energy.

        Returns:
            Dict with ``pos`` / ``vel`` of shape ``(n_steps, N, 3)`` and ``energy``
            of shape ``(n_steps,)`` (detached).
        """
        pos_hist: list[torch.Tensor] = []
        vel_hist: list[torch.Tensor] = []
        energy_hist: list[torch.Tensor] = []
        for _ in range(n_steps):
            pos, vel, energy, _ = self.step(pos, vel)
            pos_hist.append(pos.detach().clone())
            vel_hist.append(vel.detach().clone())
            energy_hist.append(energy.detach().reshape(()).clone())
        return {
            "pos": torch.stack(pos_hist),
            "vel": torch.stack(vel_hist),
            "energy": torch.stack(energy_hist),
        }
