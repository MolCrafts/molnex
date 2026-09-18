#!/usr/bin/env python3
"""Classical Class-I export parity: molnex ``Potentials`` vs a real ``lmp`` run.

The neural gate (``pair_style molnex``) asks whether LAMMPS can *call* molnex.
This one asks the opposite question — whether a Class-I force field **exported**
by :mod:`molix.ff_export` still means the same energy once LAMMPS, not molnex,
evaluates it. Nothing here loads a model: the deck is written by
:class:`~molix.ff_export.LammpsWriter` from the same ``ForceField`` object the
molnex side evaluates, and the two numbers are then compared.

It lives under ``interface/`` because it drives an external ``lmp`` binary in a
subprocess, which CLAUDE.md's third-party-surface rule confines to this
directory (no ``import lammps`` anywhere, here included). It is registered as
two ctest cases; see ``interface/tests/CMakeLists.txt``.

Cases
-----
``butane`` — *bonded-only, open boundary.*
    A synthetic OPLS-shaped field (bond / angle / OPLS torsion, **no** pair
    style) over a butane-like C4 chain. Single-point energy and forces against
    ``run 0``, then a short NVE on both sides for a ms/step figure. Because the
    field declares no pair style, the deck supplies ``pair_style zero`` itself —
    that line is script-local scaffolding, not something the exporter emits.

``periodic`` — *the whole nonbonded stack, single point only.*
    8 TIP3P-geometry waters plus Na⁺ and Cl⁻ in a 12 Å cube: switched
    Lennard-Jones, Ewald-split Coulomb under the lattice, and 1-2 / 1-3 / 1-4
    exclusions on both classes. It is the case that can fail for a *convention*
    reason — a taper window transferred wrongly, an excluded pair subtracted in
    real space instead of against the reciprocal sum, a mixing rule that differs
    between the two sides — so its deck is written entirely by the exporter:
    ``write_data`` (box from ``graphs.cell``), ``write_settings(skip_pair_style=
    True)`` for the bonded include and ``write_pair_block`` for
    ``pair_style hybrid/overlay lj/mdf … coul/long …`` + ``kspace_style pppm`` +
    ``special_bonds``. All three read one ``ForceField.assign_types`` result, so
    the 26 atoms reach LAMMPS as the four types their parameters actually make
    (O, H, Na⁺, Cl⁻) and the pair block is ten ``pair_coeff`` lines.

Both sides must be *converged* before a disagreement can mean anything, so the
periodic case reports its own convergence first and only then its parity:

* the Ewald split ``α = 1/(σ√2)`` is **pinned on the LAMMPS side too**
  (``kspace_modify gewald``), so the ``erfc(α r_c) ≈ 4e-8`` both sides drop at
  the cutoff cancels instead of surfacing as a term-independent offset;
* the molnex reciprocal sum is computed **twice**, once by the assembled ``pme``
  mesh and once by the explicit ``ewald`` k-vector sum, and their difference is
  printed. A mesh error hiding inside the parity residual is the one failure
  mode a single evaluation cannot distinguish from a convention bug — and it is
  why the *gate* is taken against the ``ewald`` sum, which is converged to fp64
  here while no affordable mesh is (see :func:`run_periodic`).

Goldens
-------
The captured literals are the **molnex** energy and largest force component of
each case, not the LAMMPS ones. That split is deliberate. LAMMPS is a moving
external dependency: a new binary legitimately shifts its own numbers, and a
golden of the LAMMPS side would fail for that reason and say nothing about
molnex. What must not move without someone noticing is the molnex side, so that
is what is pinned — at ``rel 1e-10``, i.e. tight enough that any change in the
classical kernels is caught, and checked on **every** run, including runs where
the LAMMPS comparison is what moved. ``--capture`` prints a ready-to-paste
``GOLDENS`` block; the values are recomputed from the same builders the gate
uses, so a capture can never disagree with a verify.

Tolerances
----------
``rel 1e-6`` on energy (``|E_molnex − E_lammps| / |E_lammps|``) and
``1e-5 kcal/mol/Å`` absolute on the largest force-component deviation, for both
cases. These are convention tolerances, not convergence ones: a wrong ½ on a
harmonic constant, a degree/radian slip, a missing exclusion or an unmatched
taper each move the energy by percent, and the per-term breakdown printed on
every run is what localises them.

Exit codes
----------
- ``0`` — every gate met.
- ``1`` — a gate failed (a real parity or regression failure).
- ``2`` — the check could not be performed (no captured goldens, ``lmp`` present
  but the run failed).
- ``77`` — no ``lmp`` binary. ctest reads this as SKIP
  (``SKIP_RETURN_CODE 77``). Unlike the neural gate, which never skips because
  its whole reason to exist is a plugin someone has to build, this one is a
  plain LAMMPS run and a machine without LAMMPS has nothing to say about it.

Usage
-----
::

    PYTHONPATH=src python interface/tests/parity_classical.py --case butane
    PYTHONPATH=src python interface/tests/parity_classical.py --case periodic \
        --lmp /path/to/lmp
    PYTHONPATH=src python interface/tests/parity_classical.py --case periodic \
        --capture            # prints the GOLDENS block to paste

``--lmp`` falls back to ``$LMP`` and then to ``lmp`` on ``PATH``.

References:
    LAMMPS docs: pair_style lj/mdf, pair_style coul/long, pair_style
    hybrid/overlay, kspace_style pppm, special_bonds, pair_modify
    Spec: composable-a0-05-export
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from molpy import Box, Frame

from molix.data import TotalCharge
from molix.data.collate import collate_molecules
from molix.ff_export import LammpsAdapter, LammpsWriter
from molix.md import (
    MD,
    MDHook,
    MDObservables,
    MDRunner,
    VelocityVerletIntegrator,
)
from molix.units import units
from molpot.forcefield import ForceField
from molpot.ir import ExclusionScales, NonbondedScaling

#: Energy gate: ``|E_molnex − E_lammps| / |E_lammps|``.
ENERGY_RTOL = 1e-6

#: Force gate: largest per-component deviation, kcal/mol/Å.
FORCE_ATOL = 1e-5

#: Golden gate: the molnex side must not move at all between runs.
GOLDEN_RTOL = 1e-10

#: Convergence target for ``pme`` against the explicit ``ewald`` k-vector sum.
KSPACE_SELF_RTOL = 1e-8

#: One kcal/mol expressed in molnex's MD energy unit (amu Å² fs⁻²). The MD
#: driver reports energies in that unit, so the butane NVE trace is divided by
#: this to reach the kcal/mol LAMMPS thermo prints — multiplying converts the
#: wrong way and silently reports ``0.000000``.
_MD_PER_KCAL = (
    (1.0 * units.kilocalorie_per_mole).to("amu * angstrom ** 2 / femtosecond ** 2").magnitude
)


class ParityError(RuntimeError):
    """The parity check could not be performed at all (exit code 2)."""


_CAPTURE_HINT = (
    "GOLDENS not captured — run with --capture on a machine with lmp and paste\n"
    "the printed block between the GOLDENS BEGIN / GOLDENS END markers in this file.\n"
    "This tool will not invent values."
)

# --- GOLDENS BEGIN ---
#: molnex-side single-point literals, per case. See "Goldens" in the module
#: docstring for why the LAMMPS side is deliberately *not* pinned here.
GOLDENS: dict[str, dict[str, float] | dict[str, str]] = {
    "butane": {"energy": 0.1309516320367863, "force_max": 4.921673210243161},
    "periodic": {"energy": -47.23502966626038, "force_max": 89.37654718130673},
    "provenance": {
        "butane": "C4 chain, bonded-only OPLS form, boundary f f f, pair_style zero",
        "periodic": (
            "8 TIP3P-geometry waters + Na+ + Cl-, 26 atoms, boundary p p p, cubic L = 12.0 A, "
            "seed 20260908"
        ),
        "nonbonded": (
            "lj/cut switch 4.5-5.5 A, Lorentz-Berthelot mixing; coulomb smearing 1.0 A, "
            "real-space cutoff 5.5 A, molnex side the converged ewald sum (lambda_min 0.25 A), "
            "deck kspace_style pppm 1e-10 with the split pinned; special_bonds lj 0 0 0.5 coul 0 "
            "0 0.5"
        ),
        "lammps_version": "LAMMPS (30 Mar 2026 - Development - patch_30Mar2026-1074-g7f680de296)",
        "torch_version": "2.14.0+cpu",
        "platform": "Linux-5.14.0-687.29.1.el9_8.x86_64-x86_64-with-glibc2.34",
        "capture_command": (
            "interface/tests/parity_classical.py --case periodic --lmp "
            "/home/jicli594/.local/x86_64/bin/lmp --capture"
        ),
        "date": "2026-09-08",
        "tolerance": (
            "molnex side pinned at rel 1e-10; LAMMPS comparison at rel 1e-06 (energy) and 1e-05 "
            "kcal/mol/A (max force component)"
        ),
    },
}
# --- GOLDENS END ---


# ---------------------------------------------------------------------------
# butane case — bonded-only, open boundary
# ---------------------------------------------------------------------------


class _PosEnergyHook(MDHook):
    """Record positions and energies of every NVE step of the molnex side."""

    def __init__(self) -> None:
        self.pos: list[torch.Tensor] = []
        self.pe: list[float] = []
        self.etot: list[float] = []

    def on_step_end(self, runner: MDRunner, step: int, obs: MDObservables) -> None:
        self.pos.append(obs.pos.detach().cpu().clone())
        self.pe.append(float(obs.potential.detach().cpu()))
        self.etot.append(float(obs.total.detach().cpu()))


#: The C4 chain of the butane case, Å. One row per atom, in atom order.
BUTANE_XYZ = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.53, 0.0, 0.0],
        [2.05, 1.43, 0.0],
        [3.58, 1.43, 0.0],
    ],
    dtype=float,
)


def butane_frame() -> Frame:
    """The butane-like C4 chain as one molpy ``Frame`` — this case's data at rest.

    3 bonds, 2 angles, 1 proper torsion, and a declared neutral
    ``total_charge``. A ``Frame`` is the unit of data at rest in molnex and
    :func:`~molix.data.collate.collate_molecules` is the only conversion to the
    batch a model reads, so this tool builds the same object a dataset would
    rather than assembling batch tensors by hand.

    Returns:
        The frame.
    """
    frame = Frame(meta={"total_charge": 0.0})
    frame["atoms"] = {
        "number": np.full(4, 6, dtype=np.int64),
        "x": BUTANE_XYZ[:, 0].copy(),
        "y": BUTANE_XYZ[:, 1].copy(),
        "z": BUTANE_XYZ[:, 2].copy(),
    }
    # A per-atom float column on the block the line above built: this field
    # carries no ``atoms`` parameter table, so the chain's (zero) charges are
    # the system's own column and the deck's charge column comes from them.
    frame["atoms"]["charge"] = np.zeros(4, dtype=np.float64)
    frame["bonds"] = {
        "atomi": np.array([0, 1, 2], dtype=np.int64),
        "atomj": np.array([1, 2, 3], dtype=np.int64),
        "type_id": np.zeros(3, dtype=np.int64),
    }
    frame["angles"] = {
        "atomi": np.array([0, 1], dtype=np.int64),
        "atomj": np.array([1, 2], dtype=np.int64),
        "atomk": np.array([2, 3], dtype=np.int64),
    }
    frame["propers"] = {
        "atomi": np.array([0], dtype=np.int64),
        "atomj": np.array([1], dtype=np.int64),
        "atomk": np.array([2], dtype=np.int64),
        "atoml": np.array([3], dtype=np.int64),
    }
    return frame


def butane_batch(dtype: torch.dtype = torch.float64) -> TensorDict:
    """The butane case's batch: :func:`butane_frame` through the real collate.

    No list is placed: the field is bonded-only, so there is no pair term to
    price and nothing to build one at — ``ForceField.neighbor_list`` refuses a
    field with no pair style, which is the right answer here. The collate's
    empty ``pairs`` namespace is left as it comes, unread.

    Open-boundary: the frame carries no box, so the batch carries no
    ``graphs.cell``. The batch is built at *dtype* by the collate itself.

    Args:
        dtype: Floating dtype the batch is priced in.

    Returns:
        The batch.
    """
    batch = collate_molecules([butane_frame()], dtype=dtype)
    # The frame declared its net charge; this only normalises the dtype the
    # collate wrote it in — no fixture states a charge the data does not have.
    TotalCharge().execute(batch)
    return batch


def synthetic_opls_ff(dtype: torch.dtype = torch.float64) -> ForceField:
    """Bonded OPLS form; no ``improper:periodic`` (the writer is harmonic-only)."""
    return (
        ForceField(name="OPLS", units="real")
        .def_atomstyle("full")
        .def_bondstyle("harmonic")
        .def_anglestyle("harmonic")
        .def_dihedralstyle("opls")
        .with_params(
            bonds={
                "k": torch.tensor([200.0, 200.0, 200.0], dtype=dtype),
                "r0": torch.tensor([1.53, 1.53, 1.53], dtype=dtype),
            },
            angles={
                "k": torch.tensor([100.0, 100.0], dtype=dtype),
                "theta0": torch.tensor([math.radians(112.0), math.radians(112.0)], dtype=dtype),
            },
            propers={
                "k1": torch.tensor([1.5], dtype=dtype),
                "k2": torch.tensor([3.0], dtype=dtype),
                "k3": torch.tensor([0.5], dtype=dtype),
                "k4": torch.tensor([0.1], dtype=dtype),
            },
        )
    )


def run_molnex_nve(
    ff: ForceField,
    batch: TensorDict,
    *,
    vel: torch.Tensor,
    mass: torch.Tensor,
    dt: float,
    n_steps: int,
    dtype: torch.dtype,
) -> tuple[_PosEnergyHook, float]:
    """Run the molnex NVE side and return its trace plus wall time in seconds."""
    hook = _PosEnergyHook()
    md = MD(dtype).set_forcefield(ff).set_integrator(VelocityVerletIntegrator(dt)).set_hooks([hook])
    batch["atoms", "pos"] = batch["atoms", "pos"].to(dtype).clone()
    batch["atoms", "vel"] = vel.to(dtype)
    batch["atoms", "mass"] = mass.to(dtype)
    start = time.perf_counter()
    md.run(batch, n_steps)
    return hook, time.perf_counter() - start


def write_butane_job(
    out_dir: Path,
    ff: ForceField,
    batch: TensorDict,
    *,
    masses: torch.Tensor,
    vel: torch.Tensor,
    dt: float,
    n_steps: int,
) -> None:
    """Write ``system.data`` / ``system.in.settings`` and the two NVE decks."""
    writer = LammpsWriter()
    # One assignment for both files: the data file's type labels and the
    # include's coeff names are the same strings by construction. All four
    # carbons carry the same mass and the field declares no pair table, so they
    # condense onto a single atom type, the three bonds onto one and the two
    # angles onto one.
    types = ff.assign_types(batch, masses=masses)
    writer.write_data(out_dir / "system.data", batch, types=types, masses=masses)
    writer.write_settings(out_dir / "system.in.settings", ff, batch, types=types, masses=masses)
    thermo_every = max(1, n_steps // 10)
    # The field declares no pair style, so nothing is exported for the nonbonded
    # section and the deck has to name one: ``zero`` is LAMMPS's "no pair
    # interaction at all", which is exactly what a bonded-only field means.
    velocities = "\n".join(
        line
        for i, v in enumerate(vel.detach().cpu().tolist(), start=1)
        for line in (
            f"group a{i} id {i}",
            f"velocity a{i} set {v[0]:.16g} {v[1]:.16g} {v[2]:.16g} units box",
        )
    )
    deck = f"""# butane NVE deck (script-local; not part of LammpsWriter)
units real
atom_style full
boundary f f f
newton on

read_data system.data
include system.in.settings
pair_style zero 10.0
pair_coeff * *

neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes

# explicit initial velocities (Å/fs)
{velocities}

fix 1 all nve
timestep {dt}

thermo_style custom step pe ke etotal temp
# full precision: LAMMPS' default ~9 significant digits is coarser than the gate
thermo_modify format float %.16g
thermo {thermo_every}
dump 1 all custom {thermo_every} dump.lammpstrj id type x y z fx fy fz
dump_modify 1 sort id
dump_modify 1 format float %.16g

run {n_steps}
"""
    (out_dir / "in.nve").write_text(deck)
    (out_dir / "in.run0").write_text(
        deck.replace(f"run {n_steps}", "run 0").replace(
            f"dump 1 all custom {thermo_every}", "dump 1 all custom 1"
        )
    )


# ---------------------------------------------------------------------------
# periodic case — water + ions, the whole nonbonded stack
# ---------------------------------------------------------------------------

#: Cubic lattice edge in Å. The real-space cutoff below must stay under L/2 or
#: the minimum-image convention both sides assume stops being one.
BOX_A = 12.0

#: Real-space cutoff (Å) of both nonbonded terms, and of the placed list.
CUTOFF_A = 5.5

#: Start of the Lennard-Jones taper window (Å).
SWITCH_START_A = 4.5

#: Ewald splitting width σ (Å). ``erfc(r_c / (σ√2)) ≈ 4e-8`` at ``CUTOFF_A``, so
#: what the real-space sum drops at the cutoff is far below the parity gate.
SMEARING_A = 1.0

#: PME mesh spacing (Å) and Ewald shortest wavelength (Å). Both are convergence
#: knobs, checked against each other on every run.
MESH_SPACING_A = 0.0625
LR_WAVELENGTH_A = 0.25

#: PPPM relative accuracy asked of LAMMPS, via ``LammpsAdapter(kspace_accuracy=)``.
#: Two decades below the adapter's own 1e-6 default and below the force gate:
#: at 1e-8 the LAMMPS mesh error alone is ~1e-5 kcal/mol/Å, i.e. the gate would
#: be measuring PPPM's grid rather than the export's conventions.
KSPACE_ACCURACY = 1e-10

#: TIP3P internal geometry: O–H bond length (Å) and H–O–H angle (deg).
OH_A = 0.9572
HOH_DEG = 104.52

#: Harmonic constants, molnex's ``½k`` convention: kcal/mol/Å² and kcal/mol/rad².
BOND_K = 1000.0
ANGLE_K = 200.0

#: Per-species Lennard-Jones ``(ε kcal/mol, σ Å)``, charge ``e``, mass amu and Z.
#: TIP3P water with OPLS-shaped ions; H carries no dispersion at all, which is
#: why every LJ pair this system excludes is identically zero (see the gate
#: report — the exclusion evidence here is electrostatic, not dispersive).
SPECIES: dict[str, tuple[float, float, float, float, int]] = {
    "O": (0.1521, 3.1507, -0.834, 15.999, 8),
    "H": (0.0, 0.0, 0.417, 1.008, 1),
    "Na": (0.0028, 3.328, 1.0, 22.990, 11),
    "Cl": (0.1, 4.417, -1.0, 35.453, 17),
}

#: Placement seed, water count, closest approach of two heavy sites (Å), the
#: margin keeping every atom strictly inside the box (Å) and the amplitude of
#: the deterministic distortion (Å).
SEED = 20260908
N_WATER = 8
MIN_HEAVY_A = 2.8
MARGIN_A = 1.2
DISTORT_A = 0.03


def _rotation(rng: np.random.Generator) -> np.ndarray:
    """A uniformly random rotation matrix ``(3, 3)`` from a normalised quaternion."""
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def water_ion_geometry() -> tuple[
    list[str], np.ndarray, list[tuple[int, int]], list[tuple[int, int, int]]
]:
    """Place ``N_WATER`` waters plus Na⁺ / Cl⁻ in the cubic box, deterministically.

    Heavy sites are drawn by reject-and-resample until no two are closer than
    ``MIN_HEAVY_A`` under the minimum image, inside a margin that keeps every
    hydrogen strictly within the box — LAMMPS then never has to remap an atom,
    so the coordinates molnex differentiates and the ones in the data file are
    the same numbers. Every atom is then displaced by a small fixed-seed
    Gaussian: at exactly the equilibrium geometry the bond and angle energies
    would both be zero, and a wrong ``½k`` or a radian/degree slip would be
    invisible in a term that is zero on both sides.

    Returns:
        ``(labels, positions Å (N, 3), bonds, angles)``. Angles are
        ``(H, O, H)`` — the vertex is the middle index, as both molnex's
        ``angles.atomj`` and LAMMPS's second angle atom require.
    """
    rng = np.random.default_rng(SEED)
    heavy: list[np.ndarray] = []
    span = BOX_A - 2 * MARGIN_A
    while len(heavy) < N_WATER + 2:
        candidate = MARGIN_A + rng.random(3) * span
        deltas = [candidate - other for other in heavy]
        if all(
            float(np.linalg.norm(d - BOX_A * np.round(d / BOX_A))) >= MIN_HEAVY_A for d in deltas
        ):
            heavy.append(candidate)

    labels: list[str] = []
    positions: list[np.ndarray] = []
    bonds: list[tuple[int, int]] = []
    angles: list[tuple[int, int, int]] = []
    half = math.radians(HOH_DEG) / 2.0
    for w in range(N_WATER):
        rotation = _rotation(rng)
        oxygen = heavy[w]
        arm1 = rotation @ (OH_A * np.array([math.cos(half), math.sin(half), 0.0]))
        arm2 = rotation @ (OH_A * np.array([math.cos(half), -math.sin(half), 0.0]))
        index = len(positions)
        positions += [oxygen, oxygen + arm1, oxygen + arm2]
        labels += ["O", "H", "H"]
        bonds += [(index, index + 1), (index, index + 2)]
        angles.append((index + 1, index, index + 2))
    for offset, name in enumerate(("Na", "Cl")):
        positions.append(heavy[N_WATER + offset])
        labels.append(name)

    xyz = np.asarray(positions, dtype=float)
    xyz += rng.normal(scale=DISTORT_A, size=xyz.shape)
    return labels, xyz, bonds, angles


def water_ion_frame() -> Frame:
    """The whole periodic box as **one** molpy ``Frame`` — this case's data at rest.

    **One frame, not twenty-six.** A frame per molecule collates to a batch of
    ten graphs, and a periodic Coulomb sum is defined per *lattice*:
    :class:`~molpot.forcefield.Potentials` refuses ``num_graphs > 1`` under a
    ``graphs.cell`` outright (ledger finding **F20**) rather than price the whole
    box in graph 0's cell and hand that number to every graph. The box is
    therefore one system, and :func:`water_ion_geometry` already numbers its
    bonds and angles in box-global indices, so the collate has nothing to rebase.

    The species tables ride as per-atom columns (``charge`` in ``e``, ``mass`` in
    amu) and the declared net charge as a ``meta`` scalar, so both are the
    frame's data rather than something bolted onto the batch afterwards.

    Returns:
        The frame.
    """
    labels, xyz, bonds, angles = water_ion_geometry()
    frame = Frame(meta={"total_charge": 0.0})
    # The lattice is the frame's own: molpy holds the lattice vectors as the
    # columns of ``Box.matrix`` and the collate transposes them into the rows
    # of ``graphs.cell``.
    frame.box = Box(np.eye(3) * BOX_A)
    frame["atoms"] = {
        "number": np.array([SPECIES[t][4] for t in labels], dtype=np.int64),
        "x": np.ascontiguousarray(xyz[:, 0]),
        "y": np.ascontiguousarray(xyz[:, 1]),
        "z": np.ascontiguousarray(xyz[:, 2]),
    }
    # Per-atom float columns on the block the line above built.
    frame["atoms"]["charge"] = np.array([SPECIES[t][2] for t in labels], dtype=np.float64)
    frame["atoms"]["mass"] = np.array([SPECIES[t][3] for t in labels], dtype=np.float64)
    frame["bonds"] = {
        "atomi": np.array([b[0] for b in bonds], dtype=np.int64),
        "atomj": np.array([b[1] for b in bonds], dtype=np.int64),
        "type_id": np.zeros(len(bonds), dtype=np.int64),
    }
    frame["angles"] = {
        "atomi": np.array([a[0] for a in angles], dtype=np.int64),
        "atomj": np.array([a[1] for a in angles], dtype=np.int64),
        "atomk": np.array([a[2] for a in angles], dtype=np.int64),
    }
    return frame


def water_ion_batch(dtype: torch.dtype = torch.float64) -> TensorDict:
    """The periodic batch: :func:`water_ion_frame` through the real collate, plus the list.

    Two things the collate cannot carry yet are repaired on the way out, each
    The list is placed by :meth:`~molpot.forcefield.ForceField.neighbor_list`,
    so its radius is the field's own (finding **F21**) rather than a number this
    builder picks; ``symmetry=True`` is that method's, and it is load-bearing —
    the half-pair mode emits ``pair_index[:, 0] > pair_index[:, 1]``, which
    ``Potentials`` reads as an entirely dead list (**F22**). The ``pme`` and
    ``ewald`` spellings declare the same real-space radius, so which one asks
    for the list does not matter.

    Args:
        dtype: Floating dtype the batch is priced in.

    Returns:
        The batch, with ``pairs`` placed at the field's radius under the
        minimum-image convention.
    """
    batch = collate_molecules([water_ion_frame()], dtype=dtype)
    # The frame declared its net charge; this only normalises the dtype.
    TotalCharge().execute(batch)
    neighbors = water_ion_ff(batch, dtype=dtype).neighbor_list(pbc=True)
    assert neighbors.cutoff == CUTOFF_A, (
        f"the field's own list radius is {neighbors.cutoff} A but this case's real-space "
        f"cutoff is {CUTOFF_A} A: the deck and the molnex side would be summing different "
        "neighbour sets"
    )
    neighbors.execute(batch)
    return batch


def water_ion_ff(
    batch: TensorDict,
    *,
    calculator: str = "pme",
    dtype: torch.dtype = torch.float64,
) -> ForceField:
    """The Class-I field of the periodic case, per atom of ``batch``.

    Parameters are stated per atom and per interaction — the tables are as long
    as the system — and condensed onto the four species
    (:meth:`~molpot.forcefield.ForceField.assign_types`) only where the deck
    needs types.

    Args:
        batch: The batch built by :func:`water_ion_batch`; its ``atoms.charge`` and
            topology counts size the parameter tables. Read before the
            neighbour list is placed — that list's radius is *this* field's, so
            the field has to exist first.
        calculator: ``"pme"`` (what the deck exports) or ``"ewald"`` (the
            convergence reference the mesh is checked against).
        dtype: Parameter dtype.

    Returns:
        The filled :class:`~molpot.forcefield.ForceField`.
    """
    labels = _labels_of(batch)
    # ``bonds`` is the schema's ``(2, N)`` COO under ``batch_size=[]``, so the
    # bond count is that block's second axis; the valence namespaces keep their
    # 1-D columns and are sized by their own batch size.
    n_bonds = int(batch["bonds", "bond_index"].shape[1])
    n_angles = int(batch["angles"].batch_size[0])
    return (
        ForceField(
            name="water-ion",
            units="real",
            nonbonded_scaling=NonbondedScaling(charge=ExclusionScales(0.0, 0.0, 0.5),
                lj=ExclusionScales(0.0, 0.0, 0.5),
            ),
        )
        .def_atomstyle("full")
        .def_bondstyle("harmonic")
        .def_anglestyle("harmonic")
        .def_pairstyle("lj/cut", cutoff=CUTOFF_A, switch_start=SWITCH_START_A)
        .def_pairstyle(
            "coulomb",
            calculator=calculator,
            smearing=SMEARING_A,
            real_space_cutoff=CUTOFF_A,
            mesh_spacing=MESH_SPACING_A,
            lr_wavelength=LR_WAVELENGTH_A,
        )
        .with_params(
            bonds={
                "k": torch.full((n_bonds,), BOND_K, dtype=dtype),
                "r0": torch.full((n_bonds,), OH_A, dtype=dtype),
            },
            angles={
                "k": torch.full((n_angles,), ANGLE_K, dtype=dtype),
                "theta0": torch.full((n_angles,), math.radians(HOH_DEG), dtype=dtype),
            },
            atoms={
                "charge": batch["atoms", "charge"],
                "epsilon": torch.tensor([SPECIES[t][0] for t in labels], dtype=dtype),
                "sigma": torch.tensor([SPECIES[t][1] for t in labels], dtype=dtype),
            },
        )
    )


def _labels_of(batch: TensorDict) -> list[str]:
    """Species labels of the periodic batch, recovered from ``atoms.Z``."""
    by_z = {spec[4]: name for name, spec in SPECIES.items()}
    return [by_z[int(z)] for z in batch["atoms", "Z"].tolist()]


def write_periodic_job(out_dir: Path, ff: ForceField, batch: TensorDict) -> None:
    """Write the periodic deck: data + bonded include + exported pair block.

    Everything physical is included, not written here: ``pair_style``, every
    ``pair_coeff``, ``kspace_style``, ``kspace_modify gewald`` and
    ``special_bonds`` all come out of
    :meth:`~molix.ff_export.LammpsWriter.write_pair_block`, and the bonded
    coefficients out of :meth:`~molix.ff_export.LammpsWriter.write_settings`.
    The deck adds only ``pair_modify table 16`` and the output-format lines —
    numerics and printing, no model.

    That includes the Ewald splitting parameter, which used to be a deck-local
    line here: the exporter now states it, so this tool no longer has to know
    that LAMMPS would otherwise pick its own α from the requested accuracy and
    solve a neighbouring decomposition.
    """
    writer = LammpsWriter()
    masses = batch["atoms", "mass"]
    # One assignment for all three files. The 26 atoms condense onto the four
    # species (O, H, Na+, Cl-) because a LAMMPS type is an equivalence class of
    # (mass, charge, epsilon, sigma); the pair block is then 4*5/2 = 10
    # pair_coeff lines instead of 26*27/2 = 351, and the 16 O-H bonds and 8
    # H-O-H angles one coeff line each.
    types = ff.assign_types(batch, masses=masses)
    writer.write_data(out_dir / "system.data", batch, types=types, masses=masses)
    writer.write_settings(
        out_dir / "system.in.settings",
        ff,
        batch,
        types=types,
        masses=masses,
        skip_pair_style=True,
    )
    spec = LammpsAdapter(kspace_accuracy=KSPACE_ACCURACY).translate(ff.to_ir())
    writer.write_pair_block(out_dir / "system.in.pair", spec, types=types)
    (out_dir / "in.run0").write_text(
        """# water + ion single-point deck. Every pair_style / pair_coeff /
# kspace_style / kspace_modify / special_bonds / bonded line is INCLUDED from
# what LammpsWriter wrote, unmodified; the deck itself adds only pair_modify
# table 16 and the output-format lines below - numerics and printing, no model.
units real
atom_style full
boundary p p p
newton on

read_data system.data
include system.in.settings
include system.in.pair

# coul/long reads erfc from an interpolation table of 2^N entries; molnex
# evaluates it exactly. Measured on this system against an independent analytic
# sum of the same real-space half: N = 12 (the default) leaves ~1e-6 kcal/mol in
# ecoul, N = 16 converges it to ~2e-7 and N = 20 moves it no further. LAMMPS
# prints "WARNING: For better accuracy use pair_modify table 0" here, and that
# advice is wrong for this comparison — N = 0 selects the Abramowitz-Stegun
# five-term erfc polynomial, whose ~1e-7 relative error is an order of magnitude
# worse than the 16-bit table. Numerics, not convention.
pair_modify table 16

neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes

thermo_style custom step pe evdwl ecoul elong ebond eangle
# LAMMPS' default thermo format prints ~9 significant digits, which is coarser
# than the parity gate itself — the per-term deltas would then measure the
# printer, not the physics.
thermo_modify format float %.16g
thermo 1
dump 1 all custom 1 dump.lammpstrj id type x y z fx fy fz
dump_modify 1 sort id
dump_modify 1 format float %.16g

run 0
"""
    )


# ---------------------------------------------------------------------------
# LAMMPS driving and parsing
# ---------------------------------------------------------------------------


def run_lammps(lmp: str, workdir: Path, input_name: str) -> tuple[str, float]:
    """Run one deck and return ``(log text, wall seconds)``.

    Raises:
        ParityError: ``lmp`` exited non-zero; the message carries its stderr and
            the tail of the log, which is where LAMMPS puts the real reason.
    """
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    start = time.perf_counter()
    proc = subprocess.run(
        [lmp, "-in", input_name, "-log", "log.lammps", "-screen", "none"],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - start
    log_path = workdir / "log.lammps"
    log = log_path.read_text() if log_path.exists() else ""
    if proc.returncode != 0:
        raise ParityError(
            f"lmp failed ({proc.returncode}):\n{proc.stderr[-2000:]}\nlog tail:\n{log[-2000:]}"
        )
    return log, elapsed


def parse_thermo(log: str) -> list[dict[str, float]]:
    """Every thermo row of ``log``, keyed by the column headers above it."""
    rows: list[dict[str, float]] = []
    headers: list[str] | None = None
    for line in log.splitlines():
        parts = line.split()
        if not parts:
            headers = None
            continue
        if parts[0] == "Step":
            headers = parts
            continue
        if headers is None:
            continue
        if len(parts) != len(headers):
            headers = None
            continue
        try:
            values = [float(p) for p in parts]
        except ValueError:
            headers = None
            continue
        rows.append(dict(zip(headers, values, strict=True)))
    return rows


def parse_dump_forces(dump_path: Path) -> torch.Tensor | None:
    """Forces ``(N, 3)`` of the first dump frame, ordered by atom id."""
    if not dump_path.exists():
        return None
    lines = dump_path.read_text().splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("ITEM: ATOMS")), None)
    if start is None:
        return None
    columns = {name: k for k, name in enumerate(lines[start].split()[2:])}
    rows = []
    for line in lines[start + 1 :]:
        if line.startswith("ITEM:"):
            break
        rows.append(line.split())
    if not rows:
        return None
    rows.sort(key=lambda r: int(r[columns["id"]]))
    return torch.tensor(
        [[float(r[columns[c]]) for c in ("fx", "fy", "fz")] for r in rows],
        dtype=torch.float64,
    )


def lammps_version(log: str) -> str:
    """The version banner LAMMPS prints on the first line of its log."""
    for line in log.splitlines():
        if line.startswith("LAMMPS ("):
            return line.strip()
    return "unknown"


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


def check_goldens(case: str, energy: float, force_max: float, *, capturing: bool) -> bool:
    """Compare the molnex side against the captured literals at ``GOLDEN_RTOL``.

    Args:
        case: Which entry of :data:`GOLDENS` to read.
        energy: The molnex energy just computed, kcal/mol.
        force_max: The largest molnex force component just computed.
        capturing: ``--capture`` was asked for, i.e. this run exists *to*
            produce the literals — so their absence is the starting state and
            not a failure. Any literals that are present are still compared.

    Returns:
        Whether the molnex side matched (vacuously true while capturing with no
        literals yet).

    Raises:
        ParityError: ``case`` has no captured literals and this is a verify run.
            A run with nothing to compare against has not checked anything, so
            it reports "could not be performed" rather than a failed gate — and
            never invents a value.
    """
    golden = GOLDENS.get(case)
    if golden is None:
        if not capturing:
            raise ParityError(_CAPTURE_HINT)
        print("  not captured yet — this run is the capture")
        return True
    ok = True
    for name, got in (("energy", energy), ("force_max", force_max)):
        want = float(golden[name])
        rel = abs(got - want) / abs(want) if want else abs(got - want)
        verdict = "OK" if rel <= GOLDEN_RTOL else "FAIL"
        print(f"  molnex {name:<10} = {got!r}  golden {want!r}  rel {rel:.3e}  {verdict}")
        ok = ok and rel <= GOLDEN_RTOL
    return ok


#: ``[tool.ruff] line-length`` of this repository, which a captured block must
#: respect: a capture that leaves the lint gate red is a capture nobody pastes.
_LINE_LIMIT = 100


def _string_literal(text: str, indent: str) -> str:
    """``text`` as a double-quoted literal that fits inside :data:`_LINE_LIMIT`.

    Long values are emitted as implicitly concatenated chunks inside parentheses
    — the form ``ruff format`` itself produces — and the chunks are checked to
    rejoin into exactly ``text`` before being returned, so a wrapped provenance
    string is never a *different* string from the one that was measured.

    Raises:
        RuntimeError: The wrapped chunks do not rejoin to ``text`` (a value
            carrying whitespace ``textwrap`` normalises).
    """
    single = json.dumps(text)
    if len(indent) + len(single) + 1 <= _LINE_LIMIT:
        return single
    chunks = textwrap.wrap(
        text,
        width=_LINE_LIMIT - len(indent) - 8,
        break_on_hyphens=False,
        break_long_words=False,
    )
    pieces = [f"{chunk} " for chunk in chunks[:-1]] + chunks[-1:]
    if "".join(pieces) != text:
        raise RuntimeError(f"cannot wrap {text!r} without changing it")
    inner = "\n".join(f"{indent}    {json.dumps(piece)}" for piece in pieces)
    return f"(\n{inner}\n{indent})"


def capture_block(command: str, version: str) -> str:
    """The ``GOLDENS`` literal to paste, recomputed from the shared builders.

    Both cases are evaluated here, not just the one that was run: the molnex
    side needs no ``lmp``, and a block carrying one case's numbers next to the
    previous capture's other case is exactly how two halves of a golden drift
    apart.
    """
    butane = synthetic_opls_ff().to_potentials()
    butane_system = butane_batch()
    e_butane = float(butane.calc_energy(butane_system).detach())
    f_butane = float(butane.calc_forces(butane_system).detach().abs().max())

    periodic_system = water_ion_batch()
    periodic = water_ion_ff(periodic_system, calculator="ewald").to_potentials()
    e_periodic = float(periodic.calc_energy(periodic_system).detach())
    f_periodic = float(periodic.calc_forces(periodic_system).detach().abs().max())

    provenance = {
        "butane": "C4 chain, bonded-only OPLS form, boundary f f f, pair_style zero",
        "periodic": (
            f"{N_WATER} TIP3P-geometry waters + Na+ + Cl-, "
            f"{len(_labels_of(periodic_system))} atoms, boundary p p p, "
            f"cubic L = {BOX_A} A, seed {SEED}"
        ),
        "nonbonded": (
            f"lj/cut switch {SWITCH_START_A}-{CUTOFF_A} A, Lorentz-Berthelot mixing; "
            f"coulomb smearing {SMEARING_A} A, real-space cutoff {CUTOFF_A} A, molnex "
            f"side the converged ewald sum (lambda_min {LR_WAVELENGTH_A} A), deck "
            f"kspace_style pppm {KSPACE_ACCURACY:g} with the split pinned; "
            "special_bonds lj 0 0 0.5 coul 0 0 0.5"
        ),
        "lammps_version": version,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "capture_command": command,
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "tolerance": (
            f"molnex side pinned at rel {GOLDEN_RTOL:g}; LAMMPS comparison at rel "
            f"{ENERGY_RTOL:g} (energy) and {FORCE_ATOL:g} kcal/mol/A (max force component)"
        ),
    }
    body = ",\n".join(
        f'        "{key}": {_string_literal(value, "        ")}'
        for key, value in provenance.items()
    )
    return (
        "# --- GOLDENS BEGIN ---\n"
        '#: molnex-side single-point literals, per case. See "Goldens" in the module\n'
        "#: docstring for why the LAMMPS side is deliberately *not* pinned here.\n"
        "GOLDENS: dict[str, dict[str, float] | dict[str, str]] = {\n"
        f'    "butane": {{"energy": {e_butane!r}, "force_max": {f_butane!r}}},\n'
        f'    "periodic": {{"energy": {e_periodic!r}, "force_max": {f_periodic!r}}},\n'
        '    "provenance": {\n'
        f"{body},\n"
        "    },\n"
        "}\n"
        "# --- GOLDENS END ---"
    )


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------


def run_butane(args: argparse.Namespace, workdir: Path) -> tuple[bool, str]:
    """The bonded-only open-boundary case. Returns ``(ok, lammps version)``."""
    dtype = torch.float64
    torch.manual_seed(0)
    batch = butane_batch(dtype)
    ff = synthetic_opls_ff(dtype)
    pots = ff.to_potentials()

    energy = float(pots.calc_energy(batch).detach())
    forces = pots.calc_forces(batch).detach()
    force_max = float(forces.abs().max())
    print("=== molnex single point (kcal/mol, kcal/mol/Å) ===")
    print(f"  E        = {energy!r}")
    print(f"  |F|_max  = {force_max!r}")
    for name, term in pots.term_energies(batch).items():
        print(f"    {name:<14} {float(term.detach()):+.10f}")

    n_atoms = batch["atoms", "pos"].shape[0]
    mass = torch.full((n_atoms,), 12.011, dtype=dtype)
    vel = 1e-4 * torch.randn(n_atoms, 3, dtype=dtype)
    vel -= vel.mean(dim=0, keepdim=True)

    write_butane_job(workdir, ff, batch, masses=mass, vel=vel, dt=args.dt, n_steps=args.steps)
    log0, _ = run_lammps(args.lmp, workdir, "in.run0")
    version = lammps_version(log0)
    thermo0 = parse_thermo(log0)
    pe_lammps = thermo0[0]["PotEng"]
    f_lammps = parse_dump_forces(workdir / "dump.lammpstrj")

    print("\n=== LAMMPS run 0 ===")
    print(f"  {version}")
    print(f"  PE       = {pe_lammps!r}")
    rel_e = abs(pe_lammps - energy) / abs(pe_lammps) if pe_lammps else abs(pe_lammps - energy)
    df_max = float("nan") if f_lammps is None else float((f_lammps - forces.cpu()).abs().max())
    print(f"  |ΔE|/|E| = {rel_e:.3e}   (gate {ENERGY_RTOL:g})")
    print(f"  max|ΔF|  = {df_max:.3e}   (gate {FORCE_ATOL:g} kcal/mol/Å)")

    hook, t_molnex = run_molnex_nve(
        ff, batch, vel=vel, mass=mass, dt=args.dt, n_steps=args.steps, dtype=dtype
    )
    dump = workdir / "dump.lammpstrj"
    if dump.exists():
        dump.unlink()
    log, t_lammps = run_lammps(args.lmp, workdir, "in.nve")
    thermo = parse_thermo(log)
    print(f"\n=== NVE ({args.steps} steps, dt = {args.dt} fs) ===")
    print(f"  molnex : {t_molnex / args.steps * 1e3:.4f} ms/step ({args.steps / t_molnex:.0f}/s)")
    print(f"  lammps : {t_lammps / args.steps * 1e3:.4f} ms/step ({args.steps / t_lammps:.0f}/s)")
    if thermo and hook.pe:
        pe_molnex_final = hook.pe[-1] / _MD_PER_KCAL
        pe_lammps_final = thermo[-1]["PotEng"]
        print(f"  final PE molnex = {pe_molnex_final:.6f} kcal/mol")
        print(f"  final PE lammps = {pe_lammps_final:.6f} kcal/mol")
        print(f"  |ΔPE_final|     = {abs(pe_molnex_final - pe_lammps_final):.3e}")

    print("\n=== goldens (butane) ===")
    ok_golden = check_goldens("butane", energy, force_max, capturing=args.capture)
    ok = rel_e < ENERGY_RTOL and (math.isnan(df_max) or df_max < FORCE_ATOL) and ok_golden
    print(f"\n{'PASS' if ok else 'FAIL'}: butane")
    return ok, version


def run_periodic(args: argparse.Namespace, workdir: Path) -> tuple[bool, str]:
    """The periodic nonbonded case. Returns ``(ok, lammps version)``.

    **Which molnex number is the reference.** The exported deck names no
    reciprocal-space algorithm — :class:`~molix.ff_export.LammpsAdapter` emits
    ``kspace_style pppm`` for *every* lattice calculator, mesh or not — so the
    ``pme`` and ``ewald`` spellings of this force field export the identical
    pair block, which this function asserts rather than assumes. The molnex side
    of the comparison is therefore taken from the ``ewald`` twin: its k-vector
    sum is converged to fp64 at :data:`LR_WAVELENGTH_A` (halving it does not move
    a digit), while a mesh method's own truncation error is far larger than the
    force gate and would be the only thing the gate ever measured. The ``pme``
    evaluation is still run and reported, as the mesh diagnostic it is.
    """
    dtype = torch.float64
    batch = water_ion_batch(dtype)
    labels = _labels_of(batch)
    ff_mesh = water_ion_ff(batch, calculator="pme", dtype=dtype)
    ff_exact = water_ion_ff(batch, calculator="ewald", dtype=dtype)
    adapter = LammpsAdapter(kspace_accuracy=KSPACE_ACCURACY)
    if adapter.translate(ff_mesh.to_ir()).forces != adapter.translate(ff_exact.to_ir()).forces:
        raise ParityError(
            "the pme and ewald spellings of this force field no longer export the same "
            "force spec, so the converged sum can no longer stand in for the mesh one; "
            "compare against the calculator the deck actually names"
        )
    pots = ff_exact.to_potentials()

    n_atoms = len(labels)
    n_pairs = int(batch["pairs", "pair_index"].shape[0])
    print("=== periodic system ===")
    print(f"  {n_atoms} atoms ({N_WATER} water + Na+ + Cl-), cubic L = {BOX_A} Å, seed {SEED}")
    print(
        f"  placed edges = {n_pairs} rows (bidirectional, cutoff {CUTOFF_A} Å < L/2 = {BOX_A / 2})"
    )
    print(f"  Σq = {float(batch['atoms', 'q'].sum()):+.1e} e")

    print("\n=== convergence (both sides must be converged before parity means anything) ===")
    alpha = 1.0 / (SMEARING_A * math.sqrt(2.0))
    print(f"  splitting α = 1/(σ√2) = {alpha:.10f} 1/Å, pinned on the LAMMPS side too")
    truncated = math.erfc(alpha * CUTOFF_A)
    print(f"  real space  : erfc(α r_c) = erfc({alpha * CUTOFF_A:.4f}) = {truncated:.3e}")
    energy = float(pots.calc_energy(batch).detach())
    forces = pots.calc_forces(batch).detach()
    force_max = float(forces.abs().max())
    mesh = ff_mesh.to_potentials()
    e_mesh = float(mesh.calc_energy(batch).detach())
    f_mesh = mesh.calc_forces(batch).detach()
    rel_kspace = abs(e_mesh - energy) / abs(energy)
    df_kspace = float((f_mesh - forces).abs().max())
    print(f"  ewald (λ_min {LR_WAVELENGTH_A} Å) E = {energy!r}   ← the molnex side below")
    print(f"  pme   (mesh {MESH_SPACING_A} Å)   E = {e_mesh!r}")
    verdict = "OK" if rel_kspace <= KSPACE_SELF_RTOL else "ABOVE TARGET"
    print(f"  pme vs ewald: |ΔE|/|E| = {rel_kspace:.3e} (target {KSPACE_SELF_RTOL:g}) {verdict}")
    print(f"                max|ΔF|  = {df_kspace:.3e} kcal/mol/Å")
    if rel_kspace > KSPACE_SELF_RTOL:
        # Reported, not silently accepted: the mesh does not reach the target at
        # any mesh worth paying for in this 12 Å box, measured against the ewald
        # sum — 0.25 Å: 7.3e-5 / 1.1e-2, 0.125: 4.2e-6 / 1.4e-3, 0.0625: 2.5e-7
        # / 1.7e-4, 0.03125: 1.7e-8 / 2.0e-5 (|ΔE|/|E| and max|ΔF|), the last of
        # those costing ~60 s. That is why the parity above is taken against the
        # ewald sum and not this one; the number here is a molnex-internal mesh
        # diagnostic and no part of the gate.
        print(
            f"  → pme does NOT reach {KSPACE_SELF_RTOL:g} at a mesh worth paying for here; "
            "the gate is taken against the ewald sum, see this function's docstring"
        )

    terms = {name: float(term.detach()) for name, term in pots.term_energies(batch).items()}
    molnex_terms = {
        "bond": terms["bonds"],
        "angle": terms["angles"],
        "vdW": terms["lj_repulsion"] + terms["lj_dispersion"],
        "coulomb": terms["coulomb"],
    }
    print("\n=== molnex single point (kcal/mol, kcal/mol/Å) ===")
    print(f"  E        = {energy!r}")
    print(f"  |F|_max  = {force_max!r}")

    write_periodic_job(workdir, ff_mesh, batch)
    written = sorted(path.name for path in workdir.glob("system.*"))
    print(f"\n  deck written by LammpsWriter, consumed unmodified: {written}")

    log, _ = run_lammps(args.lmp, workdir, "in.run0")
    version = lammps_version(log)
    row = parse_thermo(log)[0]
    lammps_terms = {
        "bond": row["E_bond"],
        "angle": row["E_angle"],
        "vdW": row["E_vdwl"],
        "coulomb": row["E_coul"] + row["E_long"],
    }
    pe_lammps = row["PotEng"]
    f_lammps = parse_dump_forces(workdir / "dump.lammpstrj")

    print(f"\n=== LAMMPS run 0 ===\n  {version}")
    print(f"  {'term':<10}{'molnex':>24}{'lammps':>24}{'Δ':>14}")
    for name, mine in molnex_terms.items():
        theirs = lammps_terms[name]
        print(f"  {name:<10}{mine:>24.12f}{theirs:>24.12f}{mine - theirs:>14.3e}")
    print(f"  {'TOTAL':<10}{energy:>24.12f}{pe_lammps:>24.12f}{energy - pe_lammps:>14.3e}")
    print(f"  (LAMMPS ecoul {row['E_coul']:.12f} + elong {row['E_long']:.12f})")

    rel_e = abs(energy - pe_lammps) / abs(pe_lammps) if pe_lammps else abs(energy - pe_lammps)
    df_max = float("nan") if f_lammps is None else float((f_lammps - forces.cpu()).abs().max())
    print(f"\n  |ΔE|/|E| = {rel_e:.3e}   (gate {ENERGY_RTOL:g})")
    print(f"  max|ΔF|  = {df_max:.3e}   (gate {FORCE_ATOL:g} kcal/mol/Å)")

    print("\n=== goldens (periodic) ===")
    ok_golden = check_goldens("periodic", energy, force_max, capturing=args.capture)
    ok = rel_e < ENERGY_RTOL and df_max < FORCE_ATOL and ok_golden
    print(f"\n{'PASS' if ok else 'FAIL'}: periodic")
    return ok, version


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", choices=("butane", "periodic"), default="butane")
    parser.add_argument(
        "--lmp",
        default=None,
        help="path to the lmp binary; falls back to $LMP then to lmp on PATH",
    )
    parser.add_argument("--steps", type=int, default=200, help="NVE steps (butane only)")
    parser.add_argument("--dt", type=float, default=0.5, help="NVE timestep in fs (butane only)")
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--keep", action="store_true", help="keep the work directory")
    parser.add_argument(
        "--capture",
        action="store_true",
        help="print a ready-to-paste GOLDENS block after a passing run",
    )
    args = parser.parse_args()

    resolved = args.lmp or os.environ.get("LMP") or shutil.which("lmp")
    if resolved is None or not Path(resolved).exists():
        print(
            f"no lmp binary (--lmp / $LMP / PATH gave {resolved!r}) — skipping "
            f"the classical parity for --case {args.case}",
            file=sys.stderr,
        )
        return 77
    args.lmp = resolved

    # ``delete`` states the whole lifecycle up front (3.12+), so --keep is one
    # flag reaching one argument instead of a finalizer detached after the fact.
    tmp = (
        None
        if args.workdir
        else tempfile.TemporaryDirectory(prefix=f"molnex_parity_{args.case}_", delete=not args.keep)
    )
    workdir = Path(tmp.name) if tmp is not None else Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"workdir: {workdir}\nlmp    : {args.lmp}\n")

    try:
        case = run_butane if args.case == "butane" else run_periodic
        try:
            ok, version = case(args, workdir)
        except ParityError as failure:
            print(f"\ncannot run the parity check: {failure}", file=sys.stderr)
            return 2
        if args.capture and ok:
            command = " ".join([Path(sys.argv[0]).as_posix(), *sys.argv[1:]])
            print("\n=== paste between the GOLDENS markers ===")
            print(capture_block(command, version))
        elif args.capture:
            print("\nnot capturing: the gates above did not pass", file=sys.stderr)
        return 0 if ok else 1
    finally:
        # ``delete=False`` above only stops the *automatic* removal (the
        # finalizer at garbage collection or interpreter exit); an explicit
        # ``cleanup()`` deletes whatever ``delete`` said, so --keep has to skip
        # it too.
        if tmp is not None:
            if args.keep:
                print(f"kept workdir at {workdir}")
            else:
                tmp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
