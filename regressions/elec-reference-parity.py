"""Reference-value parity for the long-range electrostatics calculators.

Moved out of the unit suite (formerly ``tests/regression/``, ``pytest -m
regression``): every case here recomputes a full Ewald / PME / P3M sum and
compares it with a published or externally computed reference, which takes
minutes on CPU. The fast analytic checks (direct Coulomb on point-charge
molecules, dipole direct/short-range limits, a Madelung subset, tuner input
validation) stay in ``tests/test_molpot/test_elec/``.

Pins, on CPU:

1. Madelung constants of seven neutral crystals (cubic → triclinic, 1:1, 1:2,
   2:1) for Ewald, PME and P3M, three length scales, half and full lists.
2. Wigner-crystal energies (net-charge cells with neutralising background),
   Ewald, three smearings each.
3. Energy, forces and stress of two random 4 Na + 4 Cl cubic cells against
   GROMACS SPME, two rotations, two scales, all three calculators.
4. Dipolar Ewald: three parallel dipoles and three crystal frames against
   espressomd DipolarP3M (energies and forces).
5. ``tune_ewald`` / ``tune_pme`` / ``tune_p3m``: the tuned parameters reproduce
   the CsCl Madelung constant to the requested accuracy (1e-1, 1e-3, 1e-5).

Provenance
----------
    Madelung / Wigner : https://doi.org/10.1016/B978-0-12-814369-8.00007-8,
                        https://doi.org/10.1021/ic2023852, Coldwell-Horsfall &
                        Maradudin (1960) eq. (A21)
    GROMACS frames    : data/coulomb_test_frames.xyz — SPME, fourierspacing
                        0.01 nm, pme_order 8, rcoulomb 0.3 nm (from torch-pme)
    espressomd frames : data/dipoles_test_frames.xyz — DipolarP3M, mesh 64
                        (from torch-pme)
    capture command   : PYTHONPATH=src python regressions/elec-reference-parity.py
    goldens           : external references above, never captured from molnex
    verified at       : e89a92c + this change, torch 2.14.1+cpu, python 3.12
    device/precision  : CPU; float64 (1-3), float32 and float64 (4-5)
    date              : 2026-10-08
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import torch

DATA = Path(__file__).parent / "data"
SQRT3 = math.sqrt(3)
DTYPES = [torch.float32, torch.float64]


# ---------------------------------------------------------------------------
# Reference structures
# ---------------------------------------------------------------------------


def _wigner(madelung_ws: float, atoms_per_cubic_cell: int) -> float:
    """Rescale a Wigner-Seitz-radius Madelung constant to lattice parameter 1."""
    radius = (3 / (4 * math.pi * atoms_per_cubic_cell)) ** (1 / 3)
    return madelung_ws / radius


def define_crystal(name: str = "CsCl", dtype=torch.float64):
    """Return ``(positions, charges (N,1), cell, madelung_ref, formula_units)``."""
    if name == "CsCl":
        pos = [[0, 0, 0], [0.5, 0.5, 0.5]]
        q, cell, ref, units = [-1.0, 1.0], torch.eye(3), 2.0353610945260, 1
    elif name == "NaCl_primitive":
        pos = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        q = [1.0, -1.0]
        cell = torch.tensor([[0, 1.0, 1], [1, 0, 1], [1, 1, 0]])
        ref, units = 1.7475645946, 1
    elif name == "NaCl_cubic":
        pos = [
            [0.0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ]
        q = [+1.0, -1, -1, -1, +1, +1, +1, -1]
        cell, ref, units = 2 * torch.eye(3), 1.7475645946, 4
    elif name == "zincblende":
        pos = [[0, 0, 0], [0.5, 0.5, 0.5]]
        q = [1.0, -1]
        cell = torch.tensor([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
        ref, units = 2 * 1.6380550533 / SQRT3, 1
    elif name == "wurtzite":
        u = 3 / 8
        c = math.sqrt(1 / u)
        pos = [
            [0.5, 0.5 / SQRT3, 0.0],
            [0.5, 0.5 / SQRT3, u * c],
            [0.5, -0.5 / SQRT3, 0.5 * c],
            [0.5, -0.5 / SQRT3, (0.5 + u) * c],
        ]
        q = [1.0, -1, 1, -1]
        cell = torch.tensor([[0.5, -0.5 * SQRT3, 0], [0.5, 0.5 * SQRT3, 0], [0, 0, c]])
        ref, units = 1.64132 / (u * c), 2
    elif name == "fluorite":
        pos = [[1 / 4, 1 / 4, 1 / 4], [3 / 4, 3 / 4, 3 / 4], [0, 0, 0]]
        q = [-1.0, -1, 2]
        cell = torch.tensor([[1.0, 1, 0], [1, 0, 1], [0, 1, 1]]) / 2.0
        ref, units = 11.6365752270768, 1
    elif name == "cu2o":
        pos = [
            [0, 0, 0],
            [1 / 2, 1 / 2, 1 / 2],
            [1 / 4, 1 / 4, 1 / 4],
            [1 / 4, 3 / 4, 3 / 4],
            [3 / 4, 1 / 4, 3 / 4],
            [3 / 4, 3 / 4, 1 / 4],
        ]
        q, cell, ref, units = [-2.0, -2, 1, 1, 1, 1], torch.eye(3), 10.2594570330750, 2
    elif name == "wigner_sc":
        pos, q, cell = [[0, 0, 0]], [1.0], torch.eye(3)
        ref, units = _wigner(1.7601188, 1), 1
    elif name == "wigner_bcc":
        pos, q = [[0, 0, 0]], [1.0]
        cell = torch.tensor([[1.0, 0, 0], [0, 1, 0], [1 / 2, 1 / 2, 1 / 2]])
        ref, units = _wigner(1.791860, 2), 1
    elif name == "wigner_bcc_cubiccell":
        pos, q, cell = [[0, 0, 0], [1 / 2, 1 / 2, 1 / 2]], [1.0, 1.0], torch.eye(3)
        ref, units = _wigner(1.791860, 2), 2
    elif name == "wigner_fcc":
        pos, q = [[0, 0, 0]], [1.0]
        cell = torch.tensor([[1.0, 0, 1], [0, 1, 1], [1, 1, 0]]) / 2
        ref, units = _wigner(1.791753, 4), 1
    elif name == "wigner_fcc_cubiccell":
        pos = [[0.0, 0, 0], [0.5, 0, 0.5], [0.5, 0.5, 0], [0, 0.5, 0.5]]
        q, cell = [1.0, 1, 1, 1], torch.eye(3)
        ref, units = _wigner(1.791753, 4), 4
    else:
        raise ValueError(f"unknown crystal {name!r}")
    return (
        torch.tensor(pos, dtype=dtype),
        torch.tensor(q, dtype=dtype).reshape(-1, 1),
        cell.to(dtype),
        torch.tensor(ref, dtype=dtype),
        units,
    )


# ---------------------------------------------------------------------------
# Brute-force periodic neighbour list (oracle, float64)
# ---------------------------------------------------------------------------


def neighbor_list(positions, box, cutoff=None, full=False):
    """Return ``(pairs (E,2), shifts (E,3), distances (E,))`` in the caller's dtype.

    Distance vector of ``(i, j, S)`` is ``pos[j] + S @ box - pos[i]``; the trivial
    self pair is excluded; the half list keeps ``i < j`` or ``i == j`` with ``S``
    lexicographically positive. ``cutoff=None`` uses half the shortest cell row.
    """
    if cutoff is None:
        cutoff = float(torch.linalg.norm(box, dim=1).min() / 2 - 1e-6)
    pos = positions.detach().to(torch.float64)
    cell = box.detach().to(torch.float64)
    n = pos.shape[0]
    recip = torch.linalg.inv(cell)
    heights = 1.0 / torch.linalg.norm(recip, dim=0)
    frac = pos @ recip
    spread = frac.max(dim=0).values - frac.min(dim=0).values  # positions need not be wrapped
    reps = (torch.ceil(cutoff / heights + spread).long() + 1).tolist()
    shifts = torch.cartesian_prod(*[torch.arange(-r, r + 1) for r in reps]).to(torch.float64)
    ns = shifts.shape[0]
    dvec = pos.view(1, 1, n, 3) + (shifts @ cell).view(ns, 1, 1, 3) - pos.view(1, n, 1, 3)
    dist = dvec.norm(dim=-1)
    si = torch.arange(ns).view(ns, 1, 1).expand(ns, n, n)
    ii = torch.arange(n).view(1, n, 1).expand(ns, n, n)
    jj = torch.arange(n).view(1, 1, n).expand(ns, n, n)
    zero = (shifts.abs().sum(-1) == 0).view(ns, 1, 1)
    mask = (dist > 1e-12) & (dist < cutoff) & ~((ii == jj) & zero)
    i, j, s, d = ii[mask], jj[mask], shifts.long()[si[mask]], dist[mask]
    if not full:
        s0, s1, s2 = s.unbind(-1)
        s_pos = (s0 > 0) | ((s0 == 0) & (s1 > 0)) | ((s0 == 0) & (s1 == 0) & (s2 > 0))
        keep = (i < j) | ((i == j) & s_pos)
        i, j, s, d = i[keep], j[keep], s[keep], d[keep]
    return torch.stack([i, j], dim=1), s.to(positions.dtype), d.to(positions.dtype)


def frame_stresses(path: Path) -> list[torch.Tensor]:
    """Per-frame ``stress="s00 ... s22"`` (3x3, row-major) from extxyz comment lines.

    The in-tree extxyz parser keeps energy, forces and per-atom columns only.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    stresses, i = [], 0
    while i < len(lines) and lines[i].strip():
        n_atoms = int(lines[i])
        match = re.search(r'stress="([^"]+)"', lines[i + 1])
        assert match is not None, f"frame {len(stresses)} has no stress"
        values = [float(v) for v in match.group(1).split()]
        stresses.append(torch.tensor(values, dtype=torch.float64).reshape(3, 3))
        i += n_atoms + 2
    return stresses


def edge_vectors(positions, pairs, cell, shifts):
    return positions[pairs[:, 1]] - positions[pairs[:, 0]] + shifts @ cell


def rot_x(phi):
    c, s = math.cos(phi), math.sin(phi)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)


def rot_z(theta):
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=torch.float64)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def madelung_cases():
    from molpot.potentials.elec import (
        CoulombPotential,
        EwaldCalculator,
        InversePowerLawPotential,
        P3MCalculator,
        PMECalculator,
    )

    crystals = [
        "CsCl",
        "NaCl_primitive",
        "NaCl_cubic",
        "zincblende",
        "wurtzite",
        "cu2o",
        "fluorite",
    ]
    for name in crystals:
        for scale in (1 / 2.0353610, 1.0, 3.4951291):
            for calc_name in ("ewald", "pme", "p3m"):
                for full in (True, False):
                    pos, q, cell, ref, units = define_crystal(name)
                    pos, cell, ref = pos * scale, cell * scale, ref / scale
                    if calc_name == "ewald":
                        cutoff = scale
                        smearing = cutoff / 5.0
                        calc = EwaldCalculator(
                            InversePowerLawPotential(exponent=1, smearing=smearing),
                            lr_wavelength=0.5 * smearing,
                            full_neighbor_list=full,
                        )
                        rtol = 4e-6
                    else:
                        cutoff = 2 * scale
                        smearing = cutoff / 5.0
                        cls = PMECalculator if calc_name == "pme" else P3MCalculator
                        potential = (
                            InversePowerLawPotential(exponent=1, smearing=smearing)
                            if calc_name == "pme"
                            else CoulombPotential(smearing=smearing)
                        )
                        calc = cls(potential, mesh_spacing=smearing / 8, full_neighbor_list=full)
                        rtol = 9e-4
                    pairs, _, dist = neighbor_list(pos, cell, cutoff, full)
                    calc.to(torch.float64)
                    phi = calc.forward(
                        positions=pos,
                        charges=q,
                        cell=cell,
                        neighbor_indices=pairs,
                        neighbor_distances=dist,
                    )
                    madelung = -torch.sum(phi * q) / units
                    torch.testing.assert_close(
                        madelung,
                        ref,
                        atol=0.0,
                        rtol=rtol,
                        msg=lambda m, a=(name, scale, calc_name, full): f"madelung {a}: {m}",
                    )
                    yield


def wigner_cases():
    from molpot.potentials.elec import EwaldCalculator, InversePowerLawPotential

    factor = {
        "wigner_sc": 1.0,
        "wigner_fcc": 1 / math.sqrt(2),
        "wigner_fcc_cubiccell": 1 / math.sqrt(2),
        "wigner_bcc": math.sqrt(3) / 2,
        "wigner_bcc_cubiccell": math.sqrt(3) / 2,
    }
    for name, f in factor.items():
        for scale in (0.4325, 1.0, 2.0353610):
            pos, q, cell, ref, _ = define_crystal(name)
            pos, cell, ref = pos * scale, cell * scale, ref / scale
            pairs, _, dist = neighbor_list(pos, cell)
            for smearing in (0.1, 0.06, 0.019):
                eff = smearing * f * scale
                calc = EwaldCalculator(
                    InversePowerLawPotential(exponent=1, smearing=eff), lr_wavelength=eff / 2
                )
                calc.to(torch.float64)
                phi = calc.forward(
                    positions=pos,
                    charges=q,
                    cell=cell,
                    neighbor_indices=pairs,
                    neighbor_distances=dist,
                )
                torch.testing.assert_close(
                    phi * q,
                    -torch.ones_like(phi) * ref / 2,
                    atol=0.0,
                    rtol=4.2e-6,
                    msg=lambda m, a=(name, scale, smearing): f"wigner {a}: {m}",
                )
                yield


def gromacs_cases():
    from molix.datasets._extxyz import parse_extxyz_frames
    from molpot.potentials.elec import (
        CoulombPotential,
        EwaldCalculator,
        P3MCalculator,
        PMECalculator,
    )
    from molpot.potentials.elec.prefactors import eV_A

    frames = parse_extxyz_frames(DATA / "coulomb_test_frames.xyz")
    stresses = frame_stresses(DATA / "coulomb_test_frames.xyz")
    rotations = [rot_z(0.987654), rot_z(1.23456) @ rot_x(0.82321)]
    for frame_index in (0, 1):
        frame = frames[frame_index]
        for scale in (0.43, 1.33):
            for k, ortho in enumerate(rotations):
                for calc_name in ("ewald", "pme", "p3m"):
                    for full in (True, False):
                        f64 = torch.float64
                        pos = scale * torch.tensor(frame.pos, dtype=f64) @ ortho
                        cell = scale * torch.tensor(frame.cell, dtype=f64) @ ortho
                        q = torch.tensor(frame.arrays["initial_charges"], dtype=f64).reshape(-1, 1)
                        cutoff = 5.54 * scale
                        smearing = cutoff / 6.0
                        potential = CoulombPotential(smearing=smearing, prefactor=eV_A)
                        if calc_name == "ewald":
                            calc = EwaldCalculator(
                                potential, lr_wavelength=0.5 * smearing, full_neighbor_list=full
                            )
                        else:
                            cls = PMECalculator if calc_name == "pme" else P3MCalculator
                            calc = cls(
                                potential, mesh_spacing=smearing / 8.0, full_neighbor_list=full
                            )
                        calc.to(f64)
                        pairs, shifts, _ = neighbor_list(pos, cell, cutoff, full)

                        def energy_of(
                            strain, pos=pos, cell=cell, q=q, pairs=pairs, shifts=shifts, calc=calc
                        ):
                            p = pos + pos @ strain.T
                            c = cell + cell @ strain.T
                            d = edge_vectors(p, pairs, c, shifts).norm(dim=-1)
                            phi = calc(
                                charges=q,
                                cell=c,
                                positions=p,
                                neighbor_indices=pairs,
                                neighbor_distances=d,
                            )
                            return (phi * q).sum()

                        label = (frame_index, scale, k, calc_name, full)
                        pos.requires_grad_(True)
                        energy = energy_of(torch.zeros(3, 3, dtype=f64))
                        torch.testing.assert_close(
                            energy,
                            torch.tensor(frame.energy, dtype=f64) / scale,
                            atol=0.0,
                            rtol=1e-4,
                            msg=lambda m, a=label: f"gromacs energy {a}: {m}",
                        )
                        forces = torch.autograd.grad(-energy, pos)[0]
                        forces_ref = torch.tensor(frame.forces, dtype=f64) / scale**2 @ ortho
                        torch.testing.assert_close(
                            forces,
                            forces_ref,
                            atol=0.0,
                            rtol=5e-3,
                            msg=lambda m, a=label: f"gromacs forces {a}: {m}",
                        )
                        pos.requires_grad_(False)
                        strain = torch.zeros(3, 3, dtype=f64, requires_grad=True)
                        stress = torch.autograd.grad(energy_of(strain), strain)[0]
                        # x2: GROMACS reports the virial, not the stress
                        stress_ref = 2.0 * stresses[frame_index] / scale
                        stress_ref = torch.einsum("ab,aA,bB->AB", stress_ref, ortho, ortho)
                        torch.testing.assert_close(
                            stress,
                            stress_ref,
                            atol=0.0,
                            rtol=5e-3,
                            msg=lambda m, a=label: f"gromacs stress {a}: {m}",
                        )
                        yield


def dipole_cases():
    from molix.datasets._extxyz import parse_extxyz_frames
    from molpot.potentials.elec import CalculatorDipole, PotentialDipole
    from molpot.potentials.elec.prefactors import eV_A

    def espresso_smearing(alpha):
        return (1 / (2 * alpha**2)) ** 0.5

    frames = parse_extxyz_frames(DATA / "dipoles_test_frames.xyz")[:3]
    cutoffs = [3.9986718930, 4.0000000000, 4.7363281250]
    alphas = [0.8819831493, 0.8956299559, 0.7215211182]
    for dtype in DTYPES:
        # Three parallel dipoles along y in a 10 A cube.
        dipoles = torch.tensor([[1.0, 1.0, 0.0]] * 3, dtype=dtype)
        cell = 10.0 * torch.eye(3, dtype=dtype)
        positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 4.0, 0.0]], dtype=dtype)
        pairs = torch.tensor([[0, 1], [1, 2], [0, 2]])
        vectors = torch.tensor([[0.0, 2.0, 0.0], [0.0, 2.0, 0.0], [0.0, 4.0, 0.0]], dtype=dtype)
        calc = CalculatorDipole(
            potential=PotentialDipole(smearing=espresso_smearing(1.0)),
            full_neighbor_list=False,
            lr_wavelength=0.1,
        )
        calc.to(dtype=dtype)
        result = (calc(dipoles, cell, positions, pairs, vectors) * dipoles).sum()
        torch.testing.assert_close(
            result,
            torch.tensor(-0.30848574939287954, dtype=dtype),
            atol=1e-6,
            rtol=1e-4,
            msg=lambda m, d=dtype: f"dipolar ewald parallel {d}: {m}",
        )
        yield

        for idx, (frame, cutoff, alpha) in enumerate(zip(frames, cutoffs, alphas, strict=True)):
            calc = CalculatorDipole(
                potential=PotentialDipole(smearing=espresso_smearing(alpha), prefactor=eV_A),
                full_neighbor_list=False,
                lr_wavelength=0.1,
            )
            calc.to(dtype=dtype)
            positions = torch.tensor(frame.pos, dtype=dtype)
            dipoles = torch.tensor(frame.arrays["dipoles"], dtype=dtype)
            cell = torch.tensor(frame.cell, dtype=dtype)
            pairs, shifts, _ = neighbor_list(positions, cell, cutoff)
            positions.requires_grad_(True)
            vectors = edge_vectors(positions, pairs, cell, shifts)
            pot = calc(
                dipoles=dipoles,
                cell=cell,
                positions=positions,
                neighbor_indices=pairs,
                neighbor_vectors=vectors,
            )
            energy = (pot * dipoles).sum()
            torch.testing.assert_close(
                energy,
                torch.tensor(frame.energy, dtype=dtype),
                atol=1e-5,
                rtol=1e-4,
                msg=lambda m, a=(idx, dtype): f"dipolar crystal energy {a}: {m}",
            )
            forces = -torch.autograd.grad(energy, positions)[0]
            torch.testing.assert_close(
                forces,
                torch.tensor(frame.forces, dtype=dtype),
                atol=1e-5,
                rtol=1e-4,
                msg=lambda m, a=(idx, dtype): f"dipolar crystal forces {a}: {m}",
            )
            yield


def tuning_cases():
    from molpot.potentials.elec import (
        CoulombPotential,
        EwaldCalculator,
        P3MCalculator,
        PMECalculator,
    )
    from molpot.potentials.elec.tuning import tune_ewald, tune_p3m, tune_pme

    cutoff = 4.4
    combos = [
        (EwaldCalculator, tune_ewald, 1),
        (PMECalculator, tune_pme, 2),
        (P3MCalculator, tune_p3m, 2),
    ]
    for dtype in DTYPES:
        for cls, tune, n_params in combos:
            for accuracy in (1e-1, 1e-3, 1e-5):
                for full in (True, False):
                    pos, q, cell, ref, units = define_crystal(dtype=dtype)
                    pairs, _, dist = neighbor_list(pos, cell, cutoff, full)
                    smearing, params, _ = tune(
                        q,
                        cell,
                        pos,
                        cutoff,
                        neighbor_indices=pairs,
                        neighbor_distances=dist,
                        full_neighbor_list=full,
                        accuracy=accuracy,
                    )
                    assert len(params) == n_params, (tune.__name__, params)
                    calc = cls(
                        potential=CoulombPotential(smearing=smearing),
                        full_neighbor_list=full,
                        **params,
                    )
                    calc.to(dtype=dtype)
                    phi = calc.forward(
                        positions=pos,
                        charges=q,
                        cell=cell,
                        neighbor_indices=pairs,
                        neighbor_distances=dist,
                    )
                    torch.testing.assert_close(
                        -torch.sum(phi * q) / units,
                        ref,
                        atol=0,
                        rtol=accuracy,
                        msg=lambda m, a=(tune.__name__, dtype, accuracy, full): f"tuning {a}: {m}",
                    )
                    yield


def main() -> int:
    scenarios = {
        "madelung": madelung_cases,
        "wigner": wigner_cases,
        "gromacs": gromacs_cases,
        "dipole": dipole_cases,
        "tuning": tuning_cases,
    }
    for name, cases in scenarios.items():
        n = sum(1 for _ in cases())
        print(f"  {name:<9}: {n} cases OK")
    print("elec-reference-parity: all reference values OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 — standalone regression script
        print(f"FAIL: {exc}", file=sys.stderr)
        raise
