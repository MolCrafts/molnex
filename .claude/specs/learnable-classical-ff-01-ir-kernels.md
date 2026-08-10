---
title: Learnable classical FF — Potential IR + Class-I kernels
status: done
created: 2026-08-10
revised: 2026-08-10
chain: learnable-classical-ff
depends_on: []
grilled: true
---

# Learnable classical FF — Potential IR + Class-I kernels

## Summary

Introduce a package-local **Potential Intermediate Representation (IR)** under
`molpot/ir/` that names Class-I molecular-mechanics interaction bags and their
canonical units, plus the missing **periodic torsion / improper** kernels
required to evaluate a full Class-I force field. Existing bonded and nonbonded
kernels (`BondHarmonic`, `AngleHarmonic`, `LJ126`, Coulomb) are reused
unchanged. `DihedralHarmonic` is documented as an improper-style harmonic form
only — it is **not** the Class-I proper torsion. Downstream sub-specs (03 heads,
05 parameterizer, 08 export) all speak this IR.

## Domain basis

Canonical Class-I (AMBER / GAFF / SMIRNOFF-compatible) energy terms, SI-free
internal units **kcal/mol, Å, elementary charge e, radians**:

| Term | Equation | Primary refs |
|------|----------|--------------|
| Bond | \(E = \tfrac12 k_b (r - r_0)^2\) | OpenMM §19.1; Cornell 1995 |
| Angle | \(E = \tfrac12 k_\theta (\theta - \theta_0)^2\) | OpenMM §19.2 |
| Proper torsion | \(E = \sum_n \frac{k_n}{s}\bigl[1 + \cos(n\phi - \gamma_n)\bigr]\) | OpenMM §19.4; SMIRNOFF |
| Improper (periodic) | same cosine form on improper torsion angle | OpenMM §19.5; AMBER |
| Improper (harmonic) | \(E = \tfrac12 k_\chi (\chi - \chi_0)^2\) | CHARMM / some SMIRNOFF |
| LJ 12-6 | \(E = 4\varepsilon\bigl[(\sigma/r)^{12}-(\sigma/r)^6\bigr]\) | OpenMM §19.6 |
| Coulomb | \(E = k_e q_i q_j / r\) | OpenMM §19.7 |

**Nonbonded scaling defaults** (AMBER/GAFF Class-I):

| Interaction | scale_q | scale_LJ |
|-------------|---------|----------|
| 1–2 (bonded) | 0 | 0 |
| 1–3 (angle) | 0 | 0 |
| 1–4 (proper torsion) | 5/6 | 0.5 |

Proper-torsion identity golden (regression): for a single term with
\(k=2.0\ \mathrm{kcal/mol}\), \(s=1\), \(n=1\), \(\gamma=0\), a cis
configuration \(\phi=0\) yields \(E = 2.0\ \mathrm{kcal/mol}\)
(\(\frac{k}{s}[1+\cos(0)] = 2k/s = 2.0\)).

References:

- OpenMM User Guide §19 "Forces" — https://docs.openmm.org/
- SMIRNOFF specification (OpenFF) — proper / improper / nonbonded sections
- Cornell et al., JACS 1995 DOI 10.1021/ja00124a002 (AMBER)
- Wang et al., JCC 2004 DOI 10.1002/jcc.20035 (GAFF)

## Design


### Placement supersede (2026-08-10 — binding)

Full rule: `.claude/notes/learnable-classical-ff.md`.

1. **Reuse first** — prefer molpy (≥0.13) / in-tree modules over new twins.
2. **No `Foo(method=…)` on new APIs** — if generalizing, one more general single-responsibility type (or two named peers), never a method/mode switch that selects unrelated implementations.
3. **Non-diff sinks to molpy/molrs** (import via `molpy` only): topology enumeration, SMARTS, classical non-torch E/F, FF style tables. This sub-spec must not reimplement those.
4. **Improper index** = molrs `Topology` layout `[center, i, j, k]` (center at **row 0**).

**01-specific:**
- IR bags / torch kernels are the **differentiable training view** of Class-I params; names and conventions must align with `molpy.potential` styles (`BondHarmonicStyle`, `DihedralFourierStyle` / `DihedralPeriodicStyle`, `ImproperPeriodicStyle`, `ImproperHarmonicStyle`, LJ/coul pair styles) — do not invent a second force-field model.
- Non-torch evaluation remains `ForceField.to_potentials()` in molpy; do not reimplement export math here.
- Prefer reusing geometry helpers if shared with existing dihedral code; keep `ProperTorsionPeriodic` as a **named** type (not `Dihedral(method="periodic")`).


### Reuse decision

| Symbol | Action | Rationale |
|--------|--------|-----------|
| `BondHarmonic` | **reuse** | Already \( \tfrac12 k(r-r_0)^2 \), COO `bond_index [2,N]` |
| `AngleHarmonic` | **reuse** | Already \( \tfrac12 k(\theta-\theta_0)^2 \) |
| `LJ126` | **reuse** | Standard 12-6 form |
| Coulomb / elec path | **reuse** | Existing `molpot.potentials.elec` Coulomb |
| `DihedralHarmonic` | **reuse + document** | Harmonic form only; rename-in-docs as improper-style; do **not** repurpose as proper |
| `ProperTorsionPeriodic` | **new** | Cosine multi-term Class-I proper |
| `ImproperPeriodic` | **new** | Cosine multi-term improper |
| `ImproperHarmonic` | **new** | Harmonic improper (or thin alias over documented DihedralHarmonic if shapes match) |
| Potential IR types | **new** | Typed bags + unit tags + CLASS_I_CANONICAL |

### Potential IR (`molpot/ir/`)

Plain dataclasses / NamedTuples (no torch dependency beyond Tensor fields where
parameters live). Package layout:

```
src/molpot/ir/
  __init__.py          # re-exports
  units.py             # UnitTag, CLASS_I_CANONICAL
  bags.py              # BondBag, AngleBag, ProperTorsionBag, …
  scaling.py           # NonbondedScaling
  potential_ir.py      # PotentialIR aggregate
```

**`UnitTag`** — enum / frozen str enum of physical dimensions:
`energy`, `length`, `charge`, `angle`, `force_const_bond`,
`force_const_angle`, `torsion_barrier`.

**`CLASS_I_CANONICAL`** — frozen mapping:

```python
{
  "energy": "kcal/mol",
  "length": "angstrom",
  "charge": "e",
  "angle": "radian",
}
```

IR bags (parameter containers, not evaluators):

| Bag | Required fields | Index convention |
|-----|-----------------|------------------|
| `BondBag` | `k`, `r0` (per type or per interaction) | `bond_index [2, N]` |
| `AngleBag` | `k`, `theta0` | `angle_index [3, N]` |
| `ProperTorsionBag` | `k` `(T, n_terms)`, `periodicity` `(n_terms,)`, `phase` `(n_terms,)`, `idivf`/`s` | `proper_index [4, N]` |
| `ImproperPeriodicBag` | same cosine fields | `improper_index [4, N]` |
| `ImproperHarmonicBag` | `k`, `chi0` | `improper_index [4, N]` |
| `LJBag` | `epsilon`, `sigma` (per atom or per type) | atom-level |
| `ChargeBag` | `q` | atom-level |
| `NonbondedScaling` | `scale_q_12/13/14`, `scale_lj_12/13/14` | defaults above |

**`PotentialIR`** — aggregate holding optional bags + scaling + `unit_system="class_i_canonical"`. Validation: missing bag is allowed (zero contribution); inconsistent units raise.

### New kernels

Place under existing tree to mirror current layout:

- `src/molpot/potentials/dihedrals/periodic.py` — `ProperTorsionPeriodic`
- `src/molpot/potentials/impropers/periodic.py` — `ImproperPeriodic`
- `src/molpot/potentials/impropers/harmonic.py` — `ImproperHarmonic`

Each inherits `BasePotential`, implements `forward(...) -> scalar energy Tensor`,
reads positions via `pos=` / `_get_positions` so `ForceDerivation` /
`BasePotential.calc_forces` works without a third force path.

**`ProperTorsionPeriodic` API sketch:**

```python
class ProperTorsionPeriodic(BasePotential):
    """Class-I multi-term cosine proper torsion.

    E = sum_n (k_n / s) * [1 + cos(n * phi - gamma_n)]
    """
    def forward(self, data=None, *, pos, proper_index, proper_types=None, **kw) -> Tensor:
        ...
```

- `proper_index`: COO-style `[4, N]` (i-j-k-l); reject `[N, 4]` and reject
  geometric `edge_index [E, 2]` with a loud `ValueError` (same anti-alias guard
  as `BondHarmonic`).
- Multi-term: per-type tables of shape `[n_types, n_terms]` for `k` and `phase`;
  shared integer `periodicity` per term column.
- `idivf` / `s` (AMBER scale factor, often 1): divide barrier as `k/s`.

**`ImproperPeriodic`**: same math, `improper_index [4, N]` in molrs layout
``[center, i, j, k]`` (**center at row 0**). Kernel evaluates φ on
``(i, center, j, k)``. OpenFF trefoil reordering is an **export adapter**, not
a second internal layout.

**`ImproperHarmonic`**: \( \tfrac12 k (\chi - \chi_0)^2 \). If implementation is
byte-equivalent to `DihedralHarmonic` with renamed kwargs, implement as thin
wrapper that calls the same geometry kernel and document the alias; public
name stays `ImproperHarmonic` for IR clarity.

### DihedralHarmonic documentation patch

Module + class docstring MUST state:

> Harmonic dihedral / improper-style form \(E=\tfrac12 k(\phi-\phi_0)^2\).
> **Not** a Class-I proper torsion. For AMBER/GAFF/SMIRNOFF propers use
> `ProperTorsionPeriodic`.

### Exports

`molpot.potentials` and `molpot` top-level re-export the three new kernels and
IR public types. No factory wrappers (`make_*` forbidden).

## Files to create or modify

- `src/molpot/ir/__init__.py` (new)
- `src/molpot/ir/units.py` (new)
- `src/molpot/ir/bags.py` (new)
- `src/molpot/ir/scaling.py` (new)
- `src/molpot/ir/potential_ir.py` (new)
- `src/molpot/potentials/dihedrals/periodic.py` (new)
- `src/molpot/potentials/impropers/__init__.py` (new)
- `src/molpot/potentials/impropers/periodic.py` (new)
- `src/molpot/potentials/impropers/harmonic.py` (new)
- `src/molpot/potentials/dihedrals/harmonic.py` (docstring patch only)
- `src/molpot/potentials/__init__.py` (exports)
- `src/molpot/__init__.py` (exports)
- `tests/test_molpot/test_ir/test_units.py` (new)
- `tests/test_molpot/test_ir/test_bags.py` (new)
- `tests/test_molpot/test_ir/test_potential_ir.py` (new)
- `tests/test_molpot/test_potentials/test_dihedrals/test_periodic.py` (new)
- `tests/test_molpot/test_potentials/test_impropers/test_periodic.py` (new)
- `tests/test_molpot/test_potentials/test_impropers/test_harmonic.py` (new)
- `tests/regression/test_class_i_kernels.py` (new; hard-coded goldens)

## Tasks

- [x] Write failing IR unit tests: UnitTag, CLASS_I_CANONICAL, bag field shapes, NonbondedScaling defaults (1-2/1-3 = 0, 1-4 q=5/6 LJ=0.5)
- [x] Implement `molpot/ir/` (units, bags, scaling, PotentialIR aggregate + validation)
- [x] Write failing kernel tests for ProperTorsionPeriodic (cis E=2.0 golden, multi-term sum, shape guards on proper_index)
- [x] Implement `ProperTorsionPeriodic` in `potentials/dihedrals/periodic.py`
- [x] Write failing tests for ImproperPeriodic + ImproperHarmonic (shape guards, energy formulas)
- [x] Implement ImproperPeriodic + ImproperHarmonic; document DihedralHarmonic as improper-style only
- [x] Wire package exports; add Google docstrings with shapes and paper/OpenMM refs
- [x] Add regression suite `tests/regression/test_class_i_kernels.py` with hard-coded cis E=2.0 and bond/angle ½k sanity goldens
- [x] Run project check + unit tests (regression marked separately)

## Testing strategy

- **IR (code):** `NonbondedScaling` defaults exactly `scale_q_12=0, scale_q_13=0, scale_q_14=5/6, scale_lj_12=0, scale_lj_13=0, scale_lj_14=0.5`; `CLASS_I_CANONICAL` keys/values frozen; `PotentialIR` rejects unknown unit_system.
- **Proper torsion (scientific):** single-term cis golden \(E=2.0\) kcal/mol as above; \(\phi=\pi\) (trans, \(\gamma=0, n=1\)) gives \(E=0\); multi-term sums componentwise.
- **Shape guards (code):** `proper_index` / `improper_index` must be `[4, N]`; wrong leading dim raises `ValueError` naming the expected shape (mirror BondHarmonic).
- **Force path (code):** energy depends on `pos` as live leaf; `BasePotential.calc_forces` / `ForceDerivation(method="autograd")` yields finite forces for a non-equilibrium geometry.
- **Regression (scientific):** hard-coded goldens only — no live OpenMM import required.
- Full ruff + pytest unit suite green.

## Out of scope

- Collate namespaces / topology ingestion (sub-spec 02).
- Parameter heads / ClassicalMMComposer (03).
- Chemical perception encoder (04).
- End-to-end neural parameterizer (05).
- Condensation / SMARTS / OpenMM export / provenance (06–09).
- Class-II (cubic/quartic) or Urey–Bradley cross terms.
- PME / long-range electrostatics changes — reuse existing elec path as-is.
- Unit conversion to kJ/mol·nm (owned by 08 export boundary).
