<!-- mol:note:topic:learnable-classical-ff -->
# Learnable classical FF — ForceField path (2026-08-12)

## Rule (binding)

### OOP, single responsibility, no glue

| Type | Owns |
|------|------|
| `ForceField` | **is** `molpy.ForceField`; styles + torch param tables; `def_*style`, `with_params`, **override** `to_potentials()`, `realize(system)`, `to_ir()` |
| `ClassIForceField` | `ForceField` subclass; harmonic bond/angle + Fourier dihedral |
| `OPLSForceField` | `ClassIForceField` subclass; OPLS `k1..k4` dihedral |
| `Potentials` | torch `BasePotential`; `calc_energy` / `calc_forces` / `forward` writes `graphs.energy` + `atoms.forces` |
| `ClassicalIMM` | encoder + heads → **returns ForceField**; `project` only |
| `LammpsWriter` | TensorDict → molpy `Frame`; `ff.clone().realize(system)` then `molpy.io` |
| `molpy.io.write_smarts` | graph → local SMARTS (molrs `write_local_smarts`; io surface) |
| `Condenser` | one type per SMARTS key; mean prototype; fail if the bucket is too wide |

No `to_molpy()`. No `ForceField.class_i()` / `ForceField.opls()` classmethod façades.
No second ForceField type for SMARTS (`SymbolicForceField` is hand-bind inspect only).
No free-function helpers as public API. No `fill` / parameterizer / composer façade.
No energy method on `ClassicalIMM` — energy is always on `Potentials`.

### User path

```python
ff = ClassIForceField()
model = ClassicalIMM(encoder, forcefield=ff, bond_head=..., ...)
ff_t = model(batch)                          # ClassIForceField (molpy.ForceField)
E = ff_t.to_potentials().calc_energy(batch)  # torch override
ff_opls = model.project(OPLSForceField())    # general → OPLS form only
ff_t.clone().realize(batch)                  # native types for molpy.io
envs = [[(write_smarts(mol, b.itom), write_smarts(mol, b.jtom)) for b in mol.bonds]]
result = Condenser().merge(
    [ff_t.params("bonds")], interaction=InteractionClass.BOND, environments=envs,
)
# register types with molpy: style.def_type(..., **rec.prototype)
```

### `project` contract

- **Does:** map continuous/general param tables into a concrete style set
  (e.g. multi-term Fourier → OPLS `k1..k4` with LAMMPS sign/½ convention).
- **Does not:** train, evaluate energy, read XML.
- **Fails loud:** missing required source params, unsupported style pair,
  non-zero Fourier residual outside n∈{1,2,3,4} or γ∉{0,π}.
- **Parity:** when the source is already on the OPLS manifold (n=1..4,
  γ=0,π,0,π), `E`/`F` after project match to numerical tolerance.

### Charge equilibration (P0)

Algorithm-named solvers under ``molpot.charge`` (no product brands):

```python
from molpot.charge import (
    AtomicChargeModel, ChiEtaHead,
    DiagonalHardnessKernel, ElectronegativityEqualization,
)

charge = AtomicChargeModel(
    ChiEtaHead(feature_dim=…),
    ElectronegativityEqualization(DiagonalHardnessKernel()),
)
model = ClassicalIMM(encoder, forcefield=ff, charge_model=charge)
ff_t = model(batch)   # writes atoms.q via EEM KKT; energy still to_potentials()
```

- **Produces ``q`` only** — Coulomb energy via ``pair:coulomb`` on ``Potentials``.
- Kernels: ``DiagonalHardnessKernel``, ``ShieldedCoulombKernel`` (Ohno).
- Nonbonded: **Potential terms own cutoffs**; ``Potentials`` only assembles them.
- Later: ACKS2 as a separate solver type.

### Units

`from molix.units import units` — pint-style `molpy.UnitSystem`.
Do not restate conversion floats.

### Status

Class-I and OPLS are subclasses of the same `ForceField` container
(`ClassIForceField`, `OPLSForceField`). IO stays in molpy.io (no
`ForceField.from_xml`).

### Library and experiment boundaries

| Layer | Owns |
|-------|------|
| **molexp** | `Workspace → Project → Experiment → Run`; exact training configuration, task scheduling, metrics, calibration, and evidence |
| **molhub** | Versioned datasets and collections; parsing, target provenance, units, physical grouping, identity, and splits |
| **molmcp molexp plane** | Scaffold + **knowledge** |
| **molnex** | Dataset-agnostic model, loss, packed-collation, optimization, and export primitives |

MolNex must never import a concrete MolHub dataset adapter or hard-code a
dataset coordinate in its library API. A Molexp experiment resolves MolHub
records and maps them into MolNex's generic sample schema.

### Supersedes

Composer / Parameterizer / IR bags as user path; glue free functions; `model.fill`;
`ForceField.to_molpy()` (the container **is** molpy.ForceField);
`ForceField.class_i()` / `ForceField.opls()` (use `ClassIForceField` /
`OPLSForceField`);
SMARTS induction via a parallel ForceField type (write molpy types instead);
`molix.units` float constants (`EV_PER_AMU_A2_FS2`, `KB_AMU_A_FS`, …).
