<!-- mol:note:topic:learnable-classical-ff -->
# Learnable classical FF — ForceField path (2026-08-20)

## Rule (binding)

### OOP, single responsibility, no glue

| Type | Owns |
|------|------|
| `ForceField` | **is** `molpy.ForceField`; styles + torch param tables; `def_*style`, `with_params`, **override** `to_potentials()`, `realize(system)`, `to_ir()` |
| `OPLSAAForceField` | `ForceField` subclass; harmonic bond/angle, OPLS `k1..k4` dihedral, periodic improper, LJ + Coulomb; default `f14=0.5` |
| `Potentials` | torch `BasePotential`; `calc_energy` / `calc_forces` / `forward` writes `graphs.energy` + `atoms.forces` |
| `Parameterizer` | encoder + heads → **returns ForceField**; no energy API; no `project` |
| `ForceFieldPredictions` | one hop model→per-task mapping (`energy` / `forces` / `hessian` / `q`) |
| `LammpsWriter` | TensorDict → molpy `Frame`; `ff.clone().realize(system)` then `molpy.io` |
| `molpy.io.write_smarts` | graph → local SMARTS (molrs `write_local_smarts`; io surface) |
| `Condenser` | one type per SMARTS key; mean prototype; fail if the bucket is too wide |

No `to_molpy()`. No `ForceField.class_i()` / `ForceField.opls()` classmethod façades.
No second ForceField type for SMARTS (`SymbolicForceField` is hand-bind inspect only).
No free-function helpers as public API. No `fill` / composer façade as the user path.
No energy method on `Parameterizer` — energy is always on `Potentials`.

### User path

```python
from molpot import Parameterizer, OPLSAAForceField, ForceFieldPredictions

ff = OPLSAAForceField()
model = Parameterizer(encoder, template=ff, bond_head=..., ...)
ff_t = model(batch)                          # filled OPLSAAForceField
E = ff_t.to_potentials().calc_energy(batch)  # torch override
pred = ForceFieldPredictions(model, keys=("energy", "forces"))
ff_t.clone().realize(batch)                  # native types for molpy.io
envs = [[(write_smarts(mol, b.itom), write_smarts(mol, b.jtom)) for b in mol.bonds]]
result = Condenser().merge(
    [ff_t.params("bonds")], interaction=InteractionClass.BOND, environments=envs,
)
# register types with molpy: style.def_type(..., **rec.prototype)
```

### Topology belongs to the molecule, not the conformer (2026-09-17)

**Landed 2026-09-17.** `molix.data.Topology` (13 unit tests) is the value
object; `class_i_fit.label_sample` and `espaloma_mm_protocol_train.label_frame`
both route through it, and the three `enumerate_*` copies in `scripts/` are
deleted — one owner for valence enumeration. A double-counting bug was caught
in the process: `i-j-k-l` and `l-k-j-i` are the same dihedral, so enumerating
both directions prices every torsion twice; ByteFF reduces with
`sorted_atomids` into a set, and `Topology` now does the same.

**Rule.** Valence topology — bonds, angles, propers, impropers — is a function
of the *bond graph*. It is computed from bonds and never from positions, once
per molecule, not once per batch. `molix.data.Topology` is that value object:
it takes `(bonds, numbers)`, refuses a `pos=` keyword, enumerates the three
families in its constructor, and exposes a Weisfeiler-Lehman `fingerprint`
that every conformer of one molecule shares.

**Why (ByteFF, `bytedance/byteff@90ef54a7`).** `byteff/mol/topology.py` —
`Topology.__init__(bonds, node_labels, bond_ids)` — takes no coordinate and
caches `_angles` / `_propers` / `_atoms_with_three_neighbors` as members.
`byteff/mol/molecule.py::_from_xyz` **raises** `MappedSmilesMissingException`
when an xyz file has no `mapped_smiles`, and uses conformer 0 only to infer
stereochemistry. `Molecule.merge` refuses to pool conformers unless the mapped
SMILES are identical. One topology, many conformers, chemistry-defined bonds.

**What was wrong here.** `ClassIPrepare.execute` (scripts) called
`molhub.dataset.omol25.infer_bonds(numbers, xyz)` per frame per epoch inside
the DataLoader worker, then enumerated the valence families from the result.
The criterion is `0.4 ≤ r ≤ 1.2·(R_i+R_j)`, i.e. about **+19 % of bond length**
for every common bond. Measured on butane: stretching one C-H from 1.09 to
1.30 Å takes the topology from 13 bonds / 24 angles / 27 propers / 0 impropers
to 12 / 21 / 24 / **1** — an improper appears from nowhere because the carbon
became trivalent. Two conformers of one molecule are parameterised as two
different molecules, and the energy is discontinuous across it. OMol25 is
off-equilibrium sampling; that amplitude is routine.

**`molecule_id` is the row index — and that is fine.** `OMol25Lmdb.__getitem__`
sets `str(idx)` and `OMol25.__getitem__`'s `setdefault` always fires, so the
field is the record number. OMol25 publishes ~100 M calculations over ~83 M
unique molecular systems (≈1.2 structures per system), i.e. a record *is*
essentially a system, so grouping by it is not the split leak an earlier draft
of this note claimed. Retracted.

**What does survive is single-structure misassignment.** The criterion decides
per structure, so a non-equilibrium geometry with one stretched bond is
parameterised as a *different molecule* than the same molecule relaxed — a
radical plus a loose hydrogen, with an improper that does not exist. That is
wrong on its own, without any appeal to conformer grouping.

**Enumeration is not a performance problem either.** Measured on real butane:
`infer_bonds` 21 us, `Topology` construction (all three families) 24 us —
about two minutes per 4 M-row epoch, spread across DataLoader workers. An
earlier estimate of hours came from synthetic random-walk "molecules" whose
distance-criterion graphs are far denser than any real one. Retracted.
`Topology.fingerprint` *is* expensive by comparison (~220 us, 10x the
enumeration) and is deliberately not taken on the per-frame path.

### Rondo — the named product (2026-09-17)

The learnable Class-I force field has a name and a home: **Rondo**, at
`src/molzoo/rondo/`, spec `src/molzoo/specs/rondo.md`.

| Symbol | Role |
|--------|------|
| `RondoForceField(ClassIForceField)` | the functional form being fitted; owns the nonbonded radius it prices |
| `RondoSpec` | the configuration contract (pydantic) |
| `Rondo(Parameterizer)` | `ChemEncoder` + the six molpot heads on that template |
| `RondoPotential(nn.Module)` | energy + forces; the host-side bind / traced-derivative split |

It descends from Espaloma (arXiv:2010.01196) and is contrasted against ByteFF
(arXiv:2408.12817). Three departures make it a distinct model, not a port —
EEM charges rather than additive BCCs, **learned** Lennard-Jones rather than
GAFF-2.2 inherited, and Cartesian `(a_n, b_n)` torsions rather than `(k, γ)`.
All three are argued in the spec §1.3.

**Supersedes** `espaloma_like_template()` / `build_espaloma_like_model()` /
`ForceAwareModel` in `scripts/mm_param_learning/`: a model does not live in
`scripts/`. Those three names are to be deleted, not kept as aliases.

**`RondoForceField` states the radius on both pair styles.** `neighbor_list()`
takes the largest pair cutoff on the field and `pair:lj/cut` carries a default
of its own, so a `cutoff` that reached only the Coulomb style was silently
ignored whenever it fell below that default — the field said 9 Å and priced
10 Å. The historical `espaloma_like_template()` hid this behind `cutoff=10.0`,
which happened to equal the default.

**Open risks, recorded in spec §6.1, not solved:** R1 — LJ σ/ε and the
long-range Coulomb term are free parameters trained only on isolated
gas-phase molecules, so the loss is nearly flat along those directions (the
`(σ/r)¹²` Huber and the 200 kcal/mol/Å force-outlier gate are symptoms, not
fixes). R2 — one combined E+F objective leaves the classic 1-4-LJ ↔ torsion
degeneracy unbroken; ByteFF's staged Hessian → torsion → off-equilibrium
schedule exists precisely to break it.

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
model = Parameterizer(encoder, template=ff, charge_model=charge, ...)
ff_t = model(batch)   # writes atoms.charge via EEM KKT; energy still to_potentials()
```

- **Produces ``charge`` only** — Coulomb energy via ``pair:coulomb`` on ``Potentials``.
- Kernels: ``DiagonalHardnessKernel``, ``ShieldedCoulombKernel`` (Ohno).
- Nonbonded: **Potential terms own cutoffs**; ``Potentials`` only assembles them.
- Later: ACKS2 as a separate solver type.

### Units

`from molix.units import units` — pint-style `molpy.UnitSystem`.
Do not restate conversion floats.

### Status

Named product template is `OPLSAAForceField(ForceField)`. Minimal MD / harmonic
cases may use `ForceField` + `def_*style` without a product subclass. IO stays
in molpy.io (no `ForceField.from_xml`).

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

Composer façade / IR bags as user path; glue free functions; `model.fill`;
`ForceField.to_molpy()` (the container **is** molpy.ForceField);
`ForceField.class_i()` / `ForceField.opls()` (use `OPLSAAForceField` or
`ForceField` + `def_*style`);
`Parameterizer.project` / continuous→OPLS projection as a public API;
SMARTS induction via a parallel ForceField type (write molpy types instead);
`molix.units` float constants (`EV_PER_AMU_A2_FS2`, `KB_AMU_A_FS`, …).
