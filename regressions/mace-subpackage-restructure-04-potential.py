"""Public-API energy/force scenario for chain step mace-subpackage-restructure-04.

Chain step 04 merges the two flat foundation models
(``molzoo.mace_matpes.MACEMatpes`` / ``molzoo.mace_omol.MACEOMol``) into the one
spec-driven :class:`molzoo.mace.MACEPotential`. Step 07 then **deletes** the flat
modules, at which point this file becomes the only frozen memory of what those
models computed — so it must land before 07 and it must never import them.

Scenario (public API only): ``MACEMatpesSpec`` / ``MACEOMolSpec`` →
``MACEPotential`` → the four public seams a caller actually has, on one fixed
5-atom cluster at fp64 on CPU:

1. ``potential(batch)`` — energy ``(B,)`` eV + forces ``(N, 3)`` eV/Å;
2. ``potential(batch)`` with ``edges.shifts`` — the periodic / MD path;
3. ``potential.energy_core(...)`` — the flat, compilable seam (raw tensors in,
   ``(B,)`` out), including the OMOL ``total_charge`` / ``total_spin``
   conditioning and the MatPES refusal of it;
4. ``MACEPotential(spec, compute_forces=False)`` and
   ``molpot.derivation.protocol.call_energy`` — the two energy-only entries;
   "energy only" is a *construction*, never a per-call flag.

Physics leg: ``Σ_i F_i = 0`` on an isolated cluster (the energy depends on the
positions only through ``r_ij``), asserted for both variants.

Goldens
-------
Every literal in :data:`MATPES` / :data:`OMOL` was captured by running this
file's own builders at the commit below and printing ``.tolist()``. At capture
time each number was **also** cross-checked against the flat model it replaces:
a fresh ``MACEMatpes`` / ``MACEOMol`` built with the same hyper-parameters,
handed the potential's ``state_dict()`` under ``load_state_dict(strict=True)``
("All keys matched successfully" for both), then called through its public
``energy_forces(...)``. The flat modules are deliberately **not** imported here
— after 07 removes them this script must still run unchanged — so that parity
lives in this comment only:

    capture script  : scratchpad ``capture_04_goldens.py`` — this file's
                      builders plus the flat-model cross-check described above
    capture command : PYTHONPATH=src python capture_04_goldens.py
    commit          : c2ccd6e (c2ccd6e57b4d05ab7e177154e0e8bb63831801d0)
    torch           : 2.12.1+cpu
    date            : 2026-08-08
    device / dtype  : CPU, float64 (``config.set_precision("fp64")``)
    oracle          : molzoo.mace_matpes.MACEMatpes / molzoo.mace_omol.MACEOMol
                      at c2ccd6e — in-repo self-oracle. No ASE / e3nn /
                      mace-torch, no network, no subprocess, no RNG beyond the
                      two fixed seeds below.
    observed        : energies bit-identical to the flat models (deviation
                      exactly 0.0) for MatPES free, MatPES periodic and OMOL;
                      OMOL forces bit-identical; MatPES forces agree to
                      7.3e-17 eV/Å (free) and 1.2e-16 eV/Å (periodic) — the
                      backward runs over an equal-valued but not
                      operation-identical graph, ~2 ulp at these magnitudes,
                      far inside the 1e-12 band the spec asks for.

Determinism: ``torch.manual_seed(0)`` immediately before each construction
(module init is the only RNG consumer), plus a private
``torch.Generator().manual_seed(0)`` for the OMOL readout — see
:func:`wake_zero_init_readout`. Two consecutive runs were byte-identical at
capture time.

Run:
    PYTHONPATH=src python regressions/mace-subpackage-restructure-04-potential.py
"""

from __future__ import annotations

import sys
from typing import NamedTuple

import torch

from molix import config

config.set_precision("fp64")  # before any module construction: cuEq bakes in dtype

from tensordict import TensorDict  # noqa: E402

from molpot.derivation.protocol import call_energy  # noqa: E402
from molzoo.mace import MACEMatpesSpec, MACEOMolSpec, MACEPotential  # noqa: E402

#: Golden band on energies (eV) and forces (eV/Å). The spec's numerical
#: invariant for the move is ``max|Δ| ≤ 1e-12``; measured deviations at capture
#: were 0.0 (energies) and ≤ 1.2e-16 (forces).
GOLDEN_ATOL = 1e-12

#: Cross-seam band: ``energy_core`` / ``forward`` / ``compute_forces=False`` /
#: ``call_energy`` must return the *same* energy, not a close one. Measured
#: bit-identical at capture, so the band is exactly zero.
SEAM_ATOL = 0.0

#: Newton's third law on an isolated cluster (measured ≤ 3.2e-17 eV/Å).
NET_FORCE_ATOL = 1e-10

# ---------------------------------------------------------------------------
# System: a fixed 5-atom cluster, all 20 ordered pairs as directed edges.
# Literal coordinates in Å — no RNG, so the geometry is reproducible anywhere.
# Same cluster as regressions/mace-subpackage-restructure-02-core.py.
# ---------------------------------------------------------------------------
ATOMIC_NUMBERS: list[int] = [1, 6, 8]
ATOMIC_ENERGIES: list[float] = [-13.6, -1029.0, -2041.0]  # eV/atom

POSITIONS = torch.tensor(
    [
        [0.00, 0.00, 0.00],
        [1.09, 0.00, 0.00],
        [1.70, 1.15, 0.00],
        [-0.40, 0.95, 0.30],
        [2.10, -0.85, -0.50],
    ],
    dtype=torch.float64,
)
Z = torch.tensor([1, 6, 8, 1, 1], dtype=torch.long)
BATCH = torch.zeros(5, dtype=torch.long)
NUM_GRAPHS = 1
EDGE_INDEX = torch.tensor(  # (E, 2): [:, 0] = source, [:, 1] = target
    [(i, j) for i in range(5) for j in range(5) if i != j], dtype=torch.long
)

#: ``unit_shifts @ cell`` on the first two (mutually reverse) edges, in Å: the
#: minimal periodic image that still exercises the ``edges.shifts`` seam.
SHIFTS = torch.zeros(EDGE_INDEX.shape[0], 3, dtype=torch.float64)
SHIFTS[0, 0] = 2.0
SHIFTS[1, 0] = -2.0

#: Neutral closed-shell singlet — the OMOL defaults. Spin ``0`` would index an
#: untrained embedding row, so the conditioning is always spelled out.
TOTAL_CHARGE = torch.zeros(NUM_GRAPHS, dtype=torch.long)
TOTAL_SPIN = torch.ones(NUM_GRAPHS, dtype=torch.long)
CATION_CHARGE = torch.ones(NUM_GRAPHS, dtype=torch.long)

#: Tiny MatPES hyper-parameters (``l_max=1``, 16 channels) — CPU-fast and
#: structurally identical to the shipped model.
MATPES_KWARGS = dict(
    r_max=5.0,
    num_bessel=4,
    num_polynomial_cutoff=5,
    l_max=1,
    num_features=16,
    max_hidden_l=1,
    num_interactions=2,
    correlation=2,
    mlp_dim=8,
    radial_mlp=[8],
)
#: Tiny OMOL hyper-parameters. ``use_fallback`` is passed to the *constructor*
#: rather than the spec (the flat ``MACEOMol`` hard-coded the fused path; on CPU
#: without ``cuequivariance-ops-torch`` that degrades to the same naive
#: contraction, so the two agree bit-for-bit).
OMOL_KWARGS = dict(
    r_max=5.0,
    num_bessel=4,
    num_polynomial_cutoff=5,
    l_max=1,
    num_features=16,
    num_interactions=2,
    correlation=2,
    mlp_dim=8,
    edge_channels=8,
)

#: ``molrep.readout.mace._ScalarO3Linear`` zero-initialises weight *and* bias,
#: so an untrained OMOL ``NonLinearBiasReadout`` emits a constant per-atom
#: energy and every OMOL force is identically zero. These four ``state_dict``
#: entries are re-seeded (see :func:`wake_zero_init_readout`) so the OMOL
#: goldens below are not a vacuous ``0 == 0``.
ZERO_INIT_READOUT_KEYS = (
    "readout.linear_mid.weight",
    "readout.linear_mid.bias",
    "readout.linear_2.weight",
    "readout.linear_2.bias",
)


# ---------------------------------------------------------------------------
# Goldens — see the module docstring for provenance.
# ---------------------------------------------------------------------------


class Golden(NamedTuple):
    """One variant's captured energy/force signature for this cluster."""

    energy: list[float]  # eV, (B,)
    forces: list[list[float]]  # eV/Å, (N, 3)
    force_abs_sum: float  # eV/Å, Σ|F| — a single-number checksum


MATPES = Golden(
    energy=[-3110.9265191001523],
    forces=[
        [0.019021511971043834, 0.06740306449657524, 0.0042655930165905525],
        [-0.03884100764534142, -0.14330968688519968, 0.03275558451993726],
        [-0.025188575897160226, 0.0664617371393798, -0.004729317224920349],
        [-0.005148555478052416, 0.02124111057338372, -0.005410511456735142],
        [0.05015662704951024, -0.011796225324139067, -0.02688134885487232],
    ],
    force_abs_sum=0.5226104575328413,
)

#: The same MatPES potential with ``edges.shifts`` on the first two edges. Only
#: the total energy and the force checksum are pinned: the point of this leg is
#: that the shift *reaches* the edge vectors (it moves the energy by 1.0e-2 eV),
#: not a second full force table.
MATPES_PERIODIC_ENERGY: list[float] = [-3110.936741536619]
MATPES_PERIODIC_FORCE_ABS_SUM = 0.6035586909434021

OMOL = Golden(
    energy=[-3143.014313842871],
    forces=[
        [0.0835761466096715, -0.027661029924360432, -0.014209789809885562],
        [-0.053411295138196244, -0.0595154032557213, -0.013578152445751224],
        [-0.004522088767538401, 0.019486164597877845, -0.0013636477030469518],
        [0.05307925742031967, -0.0006937421127873157, -0.0044739342544147884],
        [-0.07872202012425653, 0.0683840106949912, 0.03362552421309853],
    ],
    force_abs_sum=0.5163022070719174,
)

#: The same OMOL potential and geometry at total charge +1 (spin still 1): the
#: conditioning must reach the joint embedding, and it moves the energy by
#: 15.8 eV.
OMOL_CATION_ENERGY: list[float] = [-3127.2188877088856]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def wake_zero_init_readout(potential: MACEPotential) -> None:
    """Re-seed the OMOL readout so its forces are not identically zero.

    ``_ScalarO3Linear`` zero-initialises both its weight and its bias, which
    makes an *untrained* OMOL readout emit a constant per-atom energy — every
    force would then be exactly zero and the goldens would assert nothing. The
    four scalar-linear entries are refilled from a private, fixed generator
    (write-back through the public ``state_dict`` / ``load_state_dict`` pair, no
    private attribute touched). Checkpoint use is unaffected: official weights
    overwrite these entries.

    Args:
        potential: OMOL potential to modify in place.

    Raises:
        RuntimeError: If a key is missing (the readout was renamed) or is
            already non-zero (the initialisation changed) — either way the
            goldens below no longer describe the model that produced them.
    """
    state = potential.state_dict()
    generator = torch.Generator().manual_seed(0)
    for key in ZERO_INIT_READOUT_KEYS:
        if key not in state:
            raise RuntimeError(f"OMOL readout key {key!r} is gone — re-capture the goldens")
        if float(state[key].abs().max()) != 0.0:
            raise RuntimeError(f"OMOL readout key {key!r} is no longer zero-init at construction")
        state[key] = torch.empty_like(state[key]).normal_(generator=generator)
    potential.load_state_dict(state, strict=True)


def matpes_potential(*, compute_forces: bool = True) -> MACEPotential:
    """Seed-0 MatPES potential (``use_fallback=True``: CPU has no ops wheel)."""
    spec = MACEMatpesSpec(
        atomic_numbers=ATOMIC_NUMBERS, atomic_energies=ATOMIC_ENERGIES, **MATPES_KWARGS
    )
    torch.manual_seed(0)
    return MACEPotential(spec, compute_forces=compute_forces, use_fallback=True).eval()


def omol_potential(*, compute_forces: bool = True) -> MACEPotential:
    """Seed-0 OMOL potential, with the zero-init readout woken up."""
    spec = MACEOMolSpec(
        atomic_numbers=ATOMIC_NUMBERS, atomic_energies=ATOMIC_ENERGIES, **OMOL_KWARGS
    )
    torch.manual_seed(0)
    potential = MACEPotential(spec, compute_forces=compute_forces, use_fallback=True).eval()
    wake_zero_init_readout(potential)
    return potential


def make_batch(
    *,
    shifts: torch.Tensor | None = None,
    graphs: dict[str, torch.Tensor] | None = None,
) -> TensorDict:
    """Post-collate batch for :data:`POSITIONS` (atoms / edges / graphs).

    A fresh object every call: ``forward`` writes in place and swaps
    ``atoms.pos`` for its own differentiation leaf.

    Args:
        shifts: Optional periodic shift vectors ``(E, 3)`` in Å, folded into
            ``edge_diff`` and published as ``edges.shifts``.
        graphs: Optional per-graph fields (``total_charge`` / ``total_spin``).

    Returns:
        A ``TensorDict`` with ``edge_diff = pos[target] - pos[source] (+ S)``
        and ``edge_dist = ‖edge_diff‖``.
    """
    pos = POSITIONS.clone()
    edge_diff = pos[EDGE_INDEX[:, 1]] - pos[EDGE_INDEX[:, 0]]
    if shifts is not None:
        edge_diff = edge_diff + shifts
    edges = TensorDict(
        edge_index=EDGE_INDEX,
        edge_diff=edge_diff,
        edge_dist=edge_diff.norm(dim=-1).clamp(min=1e-6),
        batch_size=[EDGE_INDEX.shape[0]],
    )
    if shifts is not None:
        edges["shifts"] = shifts
    graph_data = TensorDict(
        num_atoms=torch.tensor([pos.shape[0]], dtype=torch.long), batch_size=[NUM_GRAPHS]
    )
    for key, value in (graphs or {}).items():
        graph_data[key] = value
    return TensorDict(
        atoms=TensorDict(Z=Z, pos=pos, batch=BATCH, batch_size=[pos.shape[0]]),
        edges=edges,
        graphs=graph_data,
        batch_size=[],
    )


# ---------------------------------------------------------------------------
# Checking
# ---------------------------------------------------------------------------


class Checker:
    """Collects every deviation so one run reports all failures, not the first."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def close(self, name: str, got: torch.Tensor, want: list, atol: float) -> None:
        """Assert ``max|got - want| <= atol`` against a literal nested list."""
        expected = torch.tensor(want, dtype=torch.float64)
        if got.shape != expected.shape:
            self.failures.append(f"{name}: shape {tuple(got.shape)} != {tuple(expected.shape)}")
            return
        deviation = float((got - expected).abs().max())
        if not deviation <= atol:
            self.failures.append(f"{name}: max|Δ| = {deviation:.6e} > atol {atol:.1e}")
        print(f"  {name:<34} max|Δ| {deviation:.3e}")

    def scalar(self, name: str, got: float, want: float, atol: float) -> None:
        """Assert ``|got - want| <= atol`` on a single number."""
        deviation = abs(got - want)
        if not deviation <= atol:
            self.failures.append(
                f"{name}: got {got!r}, want {want!r} (|Δ| = {deviation:.6e} > atol {atol:.1e})"
            )
        print(f"  {name:<34} |Δ| {deviation:.3e}")

    def truth(self, name: str, holds: bool, message: str) -> None:
        """Assert a boolean contract (key present / absent, error raised)."""
        if not holds:
            self.failures.append(f"{name}: {message}")
        print(f"  {name:<34} {'ok' if holds else 'FAILED'}")


def check_matpes(checker: Checker) -> None:
    """MatPES: energy + forces, the periodic seam, and Newton's third law."""
    print("MACEMatpes variant (density interactions, ZBL, per-layer readouts)")
    potential = matpes_potential()

    out = potential(make_batch())
    energy = out["graphs", "energy"].detach()
    forces = out["atoms", "forces"].detach()
    checker.close("matpes.energy", energy, MATPES.energy, GOLDEN_ATOL)
    checker.close("matpes.forces", forces, MATPES.forces, GOLDEN_ATOL)
    checker.scalar(
        "matpes.force_abs_sum", float(forces.abs().sum()), MATPES.force_abs_sum, GOLDEN_ATOL
    )
    checker.scalar("matpes.net_force", float(forces.sum(0).abs().max()), 0.0, NET_FORCE_ATOL)

    periodic = potential(make_batch(shifts=SHIFTS))
    periodic_forces = periodic["atoms", "forces"].detach()
    checker.close(
        "matpes.periodic.energy",
        periodic["graphs", "energy"].detach(),
        MATPES_PERIODIC_ENERGY,
        GOLDEN_ATOL,
    )
    checker.scalar(
        "matpes.periodic.force_abs_sum",
        float(periodic_forces.abs().sum()),
        MATPES_PERIODIC_FORCE_ABS_SUM,
        GOLDEN_ATOL,
    )


def check_matpes_energy_seams(checker: Checker) -> None:
    """The three energy-only entries must return the *same* energy, not a close one."""
    print("\nEnergy-only seams (energy_core / compute_forces=False / call_energy)")
    potential = matpes_potential()

    with torch.no_grad():
        core = potential.energy_core(POSITIONS.clone(), Z, EDGE_INDEX, BATCH, NUM_GRAPHS)
    checker.close("energy_core.energy", core, MATPES.energy, SEAM_ATOL)

    energy_only = matpes_potential(compute_forces=False)(make_batch())
    checker.close(
        "compute_forces=False.energy",
        energy_only["graphs", "energy"].detach(),
        MATPES.energy,
        SEAM_ATOL,
    )
    checker.truth(
        "compute_forces=False.no_forces",
        ("atoms", "forces") not in energy_only.keys(include_nested=True),
        "an energy-only potential must not write atoms.forces",
    )

    protocol = call_energy(potential, make_batch())
    checker.close(
        "call_energy.energy", protocol["graphs", "energy"].detach(), MATPES.energy, SEAM_ATOL
    )
    checker.truth(
        "call_energy.no_forces",
        ("atoms", "forces") not in protocol.keys(include_nested=True),
        "call_energy must reach the energy core, never the force pipeline",
    )

    rejected = False
    try:
        potential.energy_core(
            POSITIONS.clone(), Z, EDGE_INDEX, BATCH, NUM_GRAPHS, total_charge=TOTAL_CHARGE
        )
    except ValueError:
        rejected = True
    checker.truth(
        "matpes.rejects_total_charge",
        rejected,
        "a MatPES potential silently ignoring total_charge returns a wrong energy",
    )


def check_omol(checker: Checker) -> None:
    """OMOL: energy + forces under charge/spin conditioning, and the cation shift."""
    print("\nMACEOMol variant (residual interactions, charge/spin conditioning)")
    potential = omol_potential()
    conditioning = {"total_charge": TOTAL_CHARGE, "total_spin": TOTAL_SPIN}

    out = potential(make_batch(graphs=conditioning))
    energy = out["graphs", "energy"].detach()
    forces = out["atoms", "forces"].detach()
    checker.close("omol.energy", energy, OMOL.energy, GOLDEN_ATOL)
    checker.close("omol.forces", forces, OMOL.forces, GOLDEN_ATOL)
    checker.scalar("omol.force_abs_sum", float(forces.abs().sum()), OMOL.force_abs_sum, GOLDEN_ATOL)
    checker.scalar("omol.net_force", float(forces.sum(0).abs().max()), 0.0, NET_FORCE_ATOL)
    checker.truth(
        "omol.forces_are_not_vacuous",
        float(forces.abs().max()) > 0.0,
        "every OMOL force is exactly zero — the readout wake-up no longer works",
    )

    with torch.no_grad():
        cation = potential.energy_core(
            POSITIONS.clone(),
            Z,
            EDGE_INDEX,
            BATCH,
            NUM_GRAPHS,
            total_charge=CATION_CHARGE,
            total_spin=TOTAL_SPIN,
        )
    checker.close("omol.cation.energy", cation, OMOL_CATION_ENERGY, GOLDEN_ATOL)


def main() -> int:
    """Run every seam of both variants against the embedded goldens."""
    checker = Checker()
    check_matpes(checker)
    check_matpes_energy_seams(checker)
    check_omol(checker)

    if checker.failures:
        print("\nFAILED — MACEPotential no longer reproduces the pre-restructure numbers:")
        for failure in checker.failures:
            print(f"  {failure}")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
