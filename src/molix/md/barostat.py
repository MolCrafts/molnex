"""Barostats for :class:`~molix.md.driver.MD`.

Attach with :meth:`~molix.md.driver.MD.set_barostat`. The barostat is **not**
an integrator and **not** a hook: after each integrator step it rescales
positions and the cell, then the driver rebuilds the neighbour list.

Units: pressure is accepted in bar and converted to the MD system
(amu Å⁻¹ fs⁻²) through pint. ``tau`` is in fs, matching ``dt``.
Compressibility ``κ`` is in bar⁻¹ (ASE default ``4.5e-5``) and converted
the same way. The isotropic scale matches ASE ``NPTBerendsen``::

    μ = 1 − (Δt/τ) (κ/3) (P_target − P)

Without ``κ`` the prefactor has the wrong dimension and a 1 bar target
leaves the cell frozen until kinetic pressure explodes.

Instantaneous pressure::

    P = 2K / (3V) − Tr(W) / (3V)

with virial ``W = ∂E/∂ε`` (same energy unit as the integrator).

Reference:
    Berendsen et al., J. Chem. Phys. 81, 3684 (1984).
    https://doi.org/10.1063/1.448118
"""

from __future__ import annotations

import torch

from molix.md.integrators import _as_mass_col
from molix.md.types import MDState
from molix.units import units

_MD_PRESSURE = "amu / angstrom / femtosecond ** 2"


def pressure_bar_to_md(bar: float) -> float:
    """One bar expressed in amu Å⁻¹ fs⁻²."""
    return float((float(bar) * units.bar).to(_MD_PRESSURE).value)


def instantaneous_pressure(
    state: MDState,
    mass: float | torch.Tensor,
) -> torch.Tensor:
    """Scalar pressure from kinetic energy and ``state.virial``.

    Args:
        state: Live MD state with ``cell`` and ``virial``.
        mass: Per-atom (or scalar) mass in amu.

    Returns:
        Pressure ``()`` in amu Å⁻¹ fs⁻².

    Raises:
        RuntimeError: If ``cell`` or ``virial`` is missing.
    """
    if state.cell is None or state.virial is None:
        raise RuntimeError("instantaneous pressure needs cell and virial on MDState")
    mass_col = _as_mass_col(mass, state.vel)
    kinetic = 0.5 * (mass_col * state.vel * state.vel).sum()
    volume = torch.linalg.det(state.cell).abs()
    return (2.0 * kinetic - state.virial.trace()) / (3.0 * volume)


#: ASE ``NPTBerendsen`` default for water-like molecular liquids, bar⁻¹.
_DEFAULT_COMPRESSIBILITY_PER_BAR = 4.5e-5


def compressibility_bar_to_md(kappa_per_bar: float) -> float:
    """Compressibility ``κ`` in bar⁻¹ expressed in MD pressure⁻¹.

    ``κ_md · P_md = κ_bar · P_bar``, so divide by one bar in MD units.
    """
    return float(kappa_per_bar) / pressure_bar_to_md(1.0)


class BerendsenBarostat:
    """Isotropic Berendsen barostat.

    After each step, scale ``pos`` and ``cell`` by
    ``μ = 1 − (Δt/τ)(κ/3)(P_target − P)`` (ASE ``NPTBerendsen``). Velocities
    are not scaled.

    Args:
        pressure: Target pressure in bar.
        tau: Coupling time τ_p in fs, strictly positive.
        compressibility: Isothermal compressibility in bar⁻¹. Default
            ``4.5e-5`` (ASE water-like). ``0`` disables volume coupling.
    """

    def __init__(
        self,
        pressure: float,
        tau: float,
        *,
        compressibility: float = _DEFAULT_COMPRESSIBILITY_PER_BAR,
    ) -> None:
        if float(tau) <= 0.0:
            raise ValueError(f"tau must be > 0 fs, got {tau}")
        if float(compressibility) < 0.0:
            raise ValueError(f"compressibility must be >= 0 /bar, got {compressibility}")
        self.pressure = pressure_bar_to_md(pressure)
        self.pressure_bar = float(pressure)
        self.tau = float(tau)
        self.compressibility_bar = float(compressibility)
        self.kappa = compressibility_bar_to_md(compressibility)

    def scale(
        self,
        state: MDState,
        *,
        mass: float | torch.Tensor,
        dt: float,
    ) -> MDState:
        """Return a state with isotropically scaled ``pos`` and ``cell``."""
        if state.cell is None or state.virial is None:
            raise RuntimeError(
                "BerendsenBarostat needs cell and virial on MDState; "
                "put graphs.cell on the system and attach a potential that "
                "returns a force dict with a 'virial' tensor"
            )
        pressure = instantaneous_pressure(state, mass)
        dt_t = torch.as_tensor(float(dt), dtype=state.pos.dtype, device=state.pos.device)
        tau = torch.as_tensor(self.tau, dtype=state.pos.dtype, device=state.pos.device)
        target = torch.as_tensor(self.pressure, dtype=state.pos.dtype, device=state.pos.device)
        kappa = torch.as_tensor(self.kappa, dtype=state.pos.dtype, device=state.pos.device)
        mu = (1.0 - (dt_t / tau) * (kappa / 3.0) * (target - pressure)).clamp(min=1.0e-6)
        return state._replace(pos=state.pos * mu, cell=state.cell * mu)
