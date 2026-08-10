---
title: Learnable classical FF — Potential IR → OpenMM force-spec compiler
status: approved
created: 2026-08-10
revised: 2026-08-10
grilled: true
chain: learnable-classical-ff
depends_on:
  - learnable-classical-ff-01-ir-kernels
  - learnable-classical-ff-06-condensation
  - learnable-classical-ff-07-smarts
---

# Learnable classical FF — Potential IR → OpenMM force-spec compiler

## Summary

Compile Potential IR (+ optional condensed TypeSystem / SymbolicForceField
metadata) into a **backend-neutral force specification**, with an **OpenMM
adapter** implementing unit and functional-form translations. Export lives under
`molix.ff_export`. **No live OpenMM import is required** for unit/regression
tests — goldens are hard-coded numbers and serializable force-spec dicts/JSON.

## Domain basis

IR units (01): **kcal/mol, Å, e, rad**.
OpenMM standard MD units: **kJ/mol, nm, e, rad**.

Conversion factors (document as named constants):

| Quantity | IR → OpenMM |
|----------|-------------|
| Energy | \(1\ \mathrm{kcal/mol} = 4.184\ \mathrm{kJ/mol}\) |
| Length | \(1\ \mathrm{Å} = 0.1\ \mathrm{nm}\) |
| Bond \(k\) for \(E=\tfrac12 k (r-r_0)^2\) | \(k_\mathrm{OMM} = k_\mathrm{IR} \times 4.184\ \mathrm{kJ/mol} / (0.1\ \mathrm{nm})^2 = k_\mathrm{IR} \times 418.4\) … |

**Load-bearing goldens (regression):**

1. Bond force constant: \(k = 100\ \mathrm{kcal\,mol^{-1}\,Å^{-2}}\) →
   \(k_\mathrm{OMM} = 41840\ \mathrm{kJ\,mol^{-1}\,nm^{-2}}\)
   (because \(100 \times 4.184 / 0.01 = 41840\)).

2. Proper torsion amplitude: IR form
   \(E = \sum (k_n/s)[1+\cos(n\phi-\gamma)]\) with barrier-related \(V_n = 2\)
   (kcal/mol) maps to OpenMM `PeriodicTorsionForce` energy
   \(E = \sum k_\mathrm{omm}[1+\cos(n\phi-\gamma)]\) with
   \(k_\mathrm{omm} = 4.184\) when \(k_n/s = V_n\) in kcal maps by ×4.184 and
   OpenMM absorbs the same `[1+cos]` prefactor (document exact field mapping:
   if IR stores \(k\) such that \(E=(k/s)[1+\cos]\), then
   \(k_\mathrm{omm} = (k/s)\times 4.184\); golden case \(V_n=2 \Rightarrow
   k_\mathrm{omm}=4.184\) when \(V_n\) denotes the **half-barrier** or full
   coefficient as documented — **implementer MUST match OpenMM §19 and the
   golden \(V_n=2 \to k_\mathrm{omm}=4.184\)** exactly as specified in tests).

Clarify in `ConventionTable` comments which of \(k\) vs \(k/2\) OpenMM uses;
tests pin the golden, docs follow tests.

References: OpenMM User Guide §19 Forces; SMIRNOFF unit conventions.

## Design


### Placement supersede (2026-08-10 — binding)

Full rule: `.claude/notes/learnable-classical-ff.md`.

1. **Reuse first** — prefer molpy (≥0.13) / in-tree modules over new twins.
2. **No `Foo(method=…)` on new APIs** — if generalizing, one more general single-responsibility type (or two named peers), never a method/mode switch that selects unrelated implementations.
3. **Non-diff sinks to molpy/molrs** (import via `molpy` only): topology enumeration, SMARTS, classical non-torch E/F, FF style tables. This sub-spec must not reimplement those.
4. **Improper index** = molrs `Topology` layout `[center, i, j, k]` (center at **row 0**).

**08-specific:**
- Prefer molpy forcefield IO / existing export paths when they cover the target.
- Case matrix (Exact / Convention / Approximate / Unsupported) stays if molpy has no equivalent compiler; unit tables should share constants with molpy when present.
- No `Translator(method="openmm"|"gromacs")` — peer adapters (`OpenMMAdapter`, future `GromacsAdapter`) via registry of **types**, each single-purpose.


### Reuse decision

| Symbol | Action | Rationale |
|--------|--------|-----------|
| PotentialIR + bags (01) | **input** | Source of parameters |
| TypeSystem / SymbolicForceField (06/07) | **optional input** | Attach types/SMARTS to force-spec metadata |
| Live `openmm` package | **optional** | Adapter may offer `to_openmm_system` behind importorskip; core compile is pure Python |
| molpot energy kernels | **not invoked** | Export is static translation, not evaluation |

### Package layout

```
src/molix/ff_export/
  __init__.py
  cases.py           # TranslationCase enum (4-way)
  conventions.py     # ConventionTable (unit + form maps)
  force_spec.py      # ForceSpec dataclasses / dict schema
  adapter.py         # BackendAdapter Protocol + registry
  openmm_adapter.py  # OpenMMAdapter
  compiler.py        # ForceFieldCompiler
```

### TranslationCase (4-way)

Explicit enum covering how an IR term maps into the backend form:

1. **`DIRECT_UNIT_SCALE`** — same functional form; only unit conversion
   (harmonic bond/angle with matching \( \tfrac12 k x^2 \) convention).
2. **`FORM_REPARAMETERIZE`** — same physics, different parameter convention
   (e.g. torsion \(k\) vs \(k/2\), LJ ε/σ vs A/B, idivf absorption).
3. **`DECOMPOSE`** — one IR bag expands to multiple backend forces (e.g.
   multi-term proper → multiple PeriodicTorsion parameters / forces).
4. **`UNSUPPORTED`** — raise structured error with term name + reason (no
   silent drop).

Each IR term declares which case applies via `ConventionTable`.

### ConventionTable

Frozen table rows:

```python
@dataclass(frozen=True)
class ConventionRow:
    ir_term: str              # "bond_harmonic", "proper_periodic", ...
    case: TranslationCase
    unit_factors: Mapping[str, float]
    notes: str
```

Includes the bond k and torsion goldens as doctest-level comments + tested
helpers `scale_bond_k(k_kcal_per_A2) -> k_kj_per_nm2`,
`scale_torsion_k(k_kcal) -> k_kj`.

### BackendAdapter + registry

```python
class BackendAdapter(Protocol):
    name: str
    def translate(self, ir: PotentialIR, *, meta=None) -> ForceSpec: ...

_REGISTRY: dict[str, type[BackendAdapter]] = {}

def register_adapter(name: str):
    ...
```

No factory-as-primary-constructor for ForceSpec — build with `ForceSpec(...)`.
Registry is fine for backend plugins (not a `make_forcefield` wrapper).

### OpenMMAdapter

Produces a serializable `ForceSpec` (JSON-friendly nested dicts) describing:

- HarmonicBondForce entries: particles, k (kJ/nm²), r0 (nm)
- HarmonicAngleForce: k (kJ/rad²), theta0 (rad)
- PeriodicTorsionForce: per term n, phase, k (kJ/mol)
- NonbondedForce: q, σ (nm), ε (kJ/mol), exceptions with 1-4 scales from IR
  NonbondedScaling

Optional: `materialize_openmm(spec)` behind `pytest.importorskip("openmm")` —
**not** required for green CI.

### ForceFieldCompiler

```python
class ForceFieldCompiler:
    def __init__(self, adapter: BackendAdapter | str = "openmm"): ...
    def compile(self, ir: PotentialIR, *, type_systems=None, symbolic=None) -> ForceSpec: ...
```

Primitives: `compile` only; serialization helpers `ForceSpec.to_dict` /
`from_dict` allowed as alternate constructors with clear semantics.

## Files to create or modify

- `src/molix/ff_export/` (new package)
- `tests/test_molix/test_ff_export/test_conventions.py` (new)
- `tests/test_molix/test_ff_export/test_compiler.py` (new)
- `tests/test_molix/test_ff_export/test_openmm_adapter.py` (new)
- `tests/regression/test_ff_export_goldens.py` (new)

## Tasks

- [ ] Write failing unit tests for bond k golden 100 → 41840 and torsion Vn=2 → k_omm=4.184
- [ ] Implement ConventionTable + unit scale helpers + TranslationCase enum
- [ ] Write failing ForceSpec schema / to_dict round-trip tests
- [ ] Implement ForceSpec + BackendAdapter registry
- [ ] Implement OpenMMAdapter translations for bond, angle, proper, LJ/charge (+ 1-4 scales)
- [ ] Implement ForceFieldCompiler.compile; UNSUPPORTED raises clearly
- [ ] Add regression goldens file (no live openmm)
- [ ] Docstrings with OpenMM §19 refs; check + tests

## Testing strategy

- **Goldens (scientific):** exact float checks for 100→41840 and Vn=2→4.184
  (allow tiny fp noise only if using non-representable intermediates; prefer
  exact rational arithmetic in helpers).
- **Case coverage (code):** at least one IR term exercised per TranslationCase
  including UNSUPPORTED.
- **No live OpenMM (code):** default tests never import openmm; optional test
  marked importorskip.
- **Round-trip (code):** ForceSpec.to_dict/from_dict equality.
- Full unit suite green; regression marked separately.

## Out of scope

- Running OpenMM dynamics or comparing live energies to molpot (future
  regression may add importorskip path).
- LAMMPS / GROMACS adapters (registry allows later; not in this spec).
- Training / neural path.
- Provenance surfaces (09).
- Editing molpot kernels.
