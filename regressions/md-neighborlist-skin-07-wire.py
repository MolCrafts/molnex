"""Regression: the list-owned rebuild policy wired through the public MD path.

Spec md-neighborlist-skin-07-wire — the chain's integration link. One policy
owner remains: ``NeighborList(skin=, every=, delay=, check=)`` decides,
``Integrator.eval_force`` asks once per force evaluation, and the removed
owners (``MD(rebuild_every=)``, ``NeighborListHook``) stay removed.

Capture:
    command: PYTHONPATH=src python regressions/md-neighborlist-skin-07-wire.py
    commit:  639c9df (+ md-neighborlist-skin-07-wire working tree)
    torch:   2.12.1+cpu (python 3.14.5)
    date:    2026-08-09
    device:  cpu, float64
    oracle:  none — every literal is hand-derived or captured once from this
             repo's own public API at implementation time.

Goldens:
    * ``rebuild_count(skin=0) == 100`` — one policy call per force evaluation
      over 100 steps; the entry evaluation sits at the build positions
      (``max_d2 == 0``, strict ``>``) and declines. The anti-vacuity pin: a
      wiring that never calls the policy cannot produce it.
    * ``ndanger(skin=0) == 99`` — the declined entry evaluation ticks ``ago``
      once, so step 1's rebuild lands at ``ago == 2`` (not dangerous); the
      remaining 99 rebuilds each fire at ``ago == 1 == max(every, delay)``.
    * ``rebuild_count(skin=1.0) == 6`` with ``ndanger == 0`` — the skin's
      entire point, captured once from this scenario.
    * Final total energies of the two arms agree to 1e-10 (the skin changes
      cost, never physics) and match the recorded float64 literal to
      rtol 1e-9 (a pure function of the lattice, the LJ parameters and the
      seeded velocities).

Drift policy: a runtime disagreement with any literal below is a DEFECT REPORT
against src/molix/md/ — never a reason to edit the literal. A wrong
``rebuild_count`` means the seam is not wired (or wired twice); a wrong
``ndanger`` means the gate arithmetic drifted; an energy mismatch means the
policy moved the physics.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

import molix.md
from molix.md import (
    MD,
    HarmonicForceField,
    LennardJonesCutForceField,
    MaxwellBoltzmann,
    NeighborList,
    VelocityVerletIntegrator,
)
from molix.units import units

_E_EV_TO_MD = (1.0 * units.eV).to("amu * angstrom ** 2 / femtosecond ** 2").magnitude

N_STEPS = 100
EXPECTED_REBUILDS = {0.0: 100, 1.0: 6}
EXPECTED_NDANGER = {0.0: 99, 1.0: 0}
EXPECTED_E_TOT_FINAL = 1.512422884343174e-02  # amu·Å²/fs², captured 2026-08-09
E_TOT_RTOL = 1e-9
CROSS_ARM_ATOL = 1e-10

_checks: list[tuple[str, bool, str]] = []


def _check(name: str, ok: bool, detail: str = "") -> None:
    _checks.append((name, bool(ok), detail))
    print(f"  {name:<48} {'ok' if ok else 'FAIL  ' + detail}")


def _lattice() -> tuple[torch.Tensor, torch.Tensor]:
    grid = torch.arange(4, dtype=torch.float64) * 3.0
    pos = torch.stack(torch.meshgrid(grid, grid, grid, indexing="ij"), dim=-1)
    return pos.reshape(-1, 3), torch.eye(3, dtype=torch.float64) * 12.0


def _argon_arm(skin: float, *, frozen: bool = False) -> tuple[NeighborList, float]:
    """Run 100 NVE steps through the public MD path; return (list, E_tot)."""
    pos, cell = _lattice()
    nl = NeighborList(cell=cell, cutoff=3.5, positions=pos, skin=skin, capacity_factor=2.5)
    ff = LennardJonesCutForceField(epsilon=0.0103 * _E_EV_TO_MD, sigma=2.5, neighbors=nl).to(
        torch.float64
    )
    ig = VelocityVerletIntegrator(4.0, rebuild=False if frozen else None)
    md = MD(torch.float64).set_forcefield(ff).set_integrator(ig)
    n = int(pos.shape[0])
    atoms = TensorDict(
        {
            "atoms": TensorDict(
                {
                    "pos": pos,
                    "vel": torch.zeros_like(pos),
                    "mass": torch.full((n,), 39.95, dtype=pos.dtype),
                },
                batch_size=[n],
            )
        },
        batch_size=[],
    )
    MaxwellBoltzmann().sample(atoms, 300.0, seed=0)
    state = md.run(atoms, N_STEPS)
    kinetic = 0.5 * 39.95 * (state.vel * state.vel).sum()
    return nl, float(state.energy + kinetic)


def main() -> int:
    print("Wiring (derived switch, frozen route, removed owners)")
    pos, cell = _lattice()
    nl = NeighborList(cell=cell, cutoff=3.5, positions=pos, skin=1.0, capacity_factor=2.5)
    ff = LennardJonesCutForceField(epsilon=0.0103 * _E_EV_TO_MD, sigma=2.5, neighbors=nl).to(
        torch.float64
    )
    md = MD(torch.float64).set_forcefield(ff).set_integrator(VelocityVerletIntegrator(4.0))
    _check("derived.lj_cut_asks_the_policy", md.integrator.rebuild is True)
    harmonic = (
        MD(torch.float64)
        .set_forcefield(HarmonicForceField(1.0))
        .set_integrator(VelocityVerletIntegrator(0.01))
    )
    _check("derived.listless_ff_never_asks", harmonic.integrator.rebuild is False)

    frozen_nl, _ = _argon_arm(1.0, frozen=True)
    _check(
        "frozen.rebuild_count_stays_zero",
        frozen_nl.rebuild_count == 0,
        f"got {frozen_nl.rebuild_count}",
    )

    try:
        MD(torch.float64, rebuild_every=1)  # type: ignore[call-arg]
        _check("removed.rebuild_every_kwarg", False, "did not raise")
    except TypeError:
        _check("removed.rebuild_every_kwarg", True)
    _check("removed.neighbor_list_hook", not hasattr(molix.md, "NeighborListHook"))

    print("Rebuild accounting (policy-gated, entry evaluation declined)")
    energies: dict[float, float] = {}
    for skin in (0.0, 1.0):
        arm_nl, e_tot = _argon_arm(skin)
        energies[skin] = e_tot
        _check(
            f"accounting.rebuilds_at_skin_{skin}",
            arm_nl.rebuild_count == EXPECTED_REBUILDS[skin],
            f"got {arm_nl.rebuild_count}, want {EXPECTED_REBUILDS[skin]}",
        )
        _check(
            f"accounting.ndanger_at_skin_{skin}",
            arm_nl.ndanger == EXPECTED_NDANGER[skin],
            f"got {arm_nl.ndanger}, want {EXPECTED_NDANGER[skin]}",
        )

    print("Physics (the skin changes cost, never the trajectory)")
    cross = abs(energies[1.0] - energies[0.0])
    _check("physics.arms_agree_to_1e-10", cross <= CROSS_ARM_ATOL, f"|dE| = {cross:.3e}")
    rel = abs(energies[0.0] - EXPECTED_E_TOT_FINAL) / abs(EXPECTED_E_TOT_FINAL)
    _check("physics.e_tot_matches_the_literal", rel <= E_TOT_RTOL, f"rel = {rel:.3e}")

    failed = [name for name, ok, _ in _checks if not ok]
    if failed:
        print(f"FAILED ({len(failed)}): {failed}")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
