---
title: Learnable classical FF — neural ClassicalMMParameterizer
status: approved
created: 2026-08-10
revised: 2026-08-10
grilled: true
chain: learnable-classical-ff
depends_on:
  - learnable-classical-ff-01-ir-kernels
  - learnable-classical-ff-02-valence-topology
  - learnable-classical-ff-03-mm-heads
  - learnable-classical-ff-04-chem-encoder
---

# Learnable classical FF — neural ClassicalMMParameterizer

## Summary

Compose the end-to-end **encoder → heads → Potential IR → classical energy/force**
path as `ClassicalMMParameterizer` in a **new** file
`src/molpot/composition/parameterizer.py`. This is the training-time nn.Module
that accepts a chemical-perception encoder **via Protocol** (so molpot never
imports molzoo), runs 03 heads into 01 IR, evaluates Class-I energy, and derives
forces only through `ForceDerivation`. Sibling of 03: **03 owns
`classical_mm.py` / ClassicalMMComposer; 05 does not claim that file as new.**

## Domain basis

- Internal MM parameters and classical energy are in **CLASS_I_CANONICAL units
  from 01: kcal/mol, Å, e, rad**.
- Optional ML loss surfaces may use eV / eV·Å for compatibility with existing
  molnex training — conversion happens **at the loss/boundary only**, never by
  silently redefining IR units. Document conversion factors if a helper is
  provided (e.g. `KCAL_TO_EV = 1/23.060547830619026` or project-standard
  constant); do not make eV the primary IR unit.
- Forces: \( F = -\partial E/\partial \mathbf{r} \) via
  `molpot.derivation.ForceDerivation` (default `method="autograd"`). No third
  hand-rolled force path inside the parameterizer.

## Design


### Placement supersede (2026-08-10 — binding)

Full rule: `.claude/notes/learnable-classical-ff.md`.

1. **Reuse first** — prefer molpy (≥0.13) / in-tree modules over new twins.
2. **No `Foo(method=…)` on new APIs** — if generalizing, one more general single-responsibility type (or two named peers), never a method/mode switch that selects unrelated implementations.
3. **Non-diff sinks to molpy/molrs** (import via `molpy` only): topology enumeration, SMARTS, classical non-torch E/F, FF style tables. This sub-spec must not reimplement those.
4. **Improper index** = molrs `Topology` layout `[center, i, j, k]` (center at **row 0**).

**05-specific:**
- Wire only. Topology and molpy ForceField tables are inputs; parameterizer does not own non-diff perception.
- Forces: single path via `BasePotential` / `ForceDerivation` already used by classical terms — do not add a new method switch.


### Reuse decision

| Symbol | Action | Rationale |
|--------|--------|-----------|
| `ClassicalMMComposer` + MM heads (03) | **reuse** | Parameterizer orchestrates; may own heads or wrap composer |
| Potential IR + kernels (01) | **reuse** | Energy evaluation |
| Valence batch namespaces (02) | **consume** | Topology inputs |
| Chem encoder (04) | **Protocol injection** | molpot ↛ molzoo / must not hard-import molrep.chem if that violates deps — use a narrow Protocol |
| `ForceDerivation` | **reuse only** | Forces |
| `PotentialComposer` / Sonata | **do not overload** | Separate Class-I path |

### File ownership (conflict fix)

```
src/molpot/composition/classical_mm.py   # owned by 03 — ClassicalMMComposer, not "new" here
src/molpot/composition/parameterizer.py  # NEW in 05 — ClassicalMMParameterizer
```

### Encoder Protocol

```python
@runtime_checkable
class ChemEncoderProtocol(Protocol):
    """Minimal encoder surface for classical MM parameterization.

    Implementations live in molrep/molzoo; molpot only sees this Protocol.
    """
    def forward(self, td: TensorDict) -> TensorDict: ...
    # and/or
    def embeddings(self, td: TensorDict) -> ChemEmbeddingsLike: ...
```

`ChemEmbeddingsLike` is a Protocol or TypedDict with atom/bond/angle/proper/
improper feature tensors — defined in molpot as a structural type **without**
importing molrep.chem (duck typing).

### ClassicalMMParameterizer

```python
class ClassicalMMParameterizer(nn.Module):
    """encoder (Protocol) -> MM heads -> PotentialIR -> classical E (+ optional F).

    Args:
        encoder: ChemEncoderProtocol module (parameters registered if nn.Module).
        composer: ClassicalMMComposer or head bundle.
        force_derivation: ForceDerivation | None
    """
```

Primitives (prefer explicit methods over one opaque façade):

1. `encode(batch) -> features` — runs encoder Protocol.
2. `parameterize(batch, features=None) -> PotentialIR` — heads → IR (kcal/mol…).
3. `energy(batch, ir=None, pos=None) -> Tensor` — classical sum.
4. `forward(batch, *, compute_forces=False) -> TensorDict | dict` — thin
   composition of 1–3; writes energy (and forces if requested) under documented
   keys (`graphs.energy` / `atoms.forces` or top-level — match existing molpot
   potential conventions such as Sonata / PiNet potential I/O).

Register encoder as submodule only if it is an `nn.Module` (it will be).

### Dependency hard rule

```
molpot ↛ molzoo
molpot may depend on molrep only if already established; prefer Protocol-only
coupling so 04 can evolve without molpot importing molrep.chem.
```

Verify with import tests / grep in CI-oriented unit tests.

### Units boundary

- `PotentialIR.unit_system == class_i_canonical` always inside parameterize.
- If training code needs eV, provide an **optional** explicit converter at the
  edge (function or flag on a loss helper), never mutate IR in place to eV.

## Files to create or modify

- `src/molpot/composition/parameterizer.py` (new)
- `src/molpot/composition/__init__.py` — export ClassicalMMParameterizer
- `src/molpot/__init__.py` — export if public
- `tests/test_molpot/test_composition/test_parameterizer.py` (new)

## Tasks

- [ ] Write failing tests for ChemEncoderProtocol structural typing with a FakeEncoder
- [ ] Write failing end-to-end test: FakeEncoder features → parameterize IR units kcal/mol → finite energy
- [ ] Implement ClassicalMMParameterizer in parameterizer.py reusing ClassicalMMComposer / heads
- [ ] Wire ForceDerivation path for compute_forces=True; assert forces shape (N,3)
- [ ] Add unit-boundary test: IR stays kcal/mol; optional eV conversion only at helper if present
- [ ] Assert parameterizer.py / composition path does not import molzoo
- [ ] Google docstrings; exports; run check + tests

## Testing strategy

- **Protocol (code):** a minimal FakeEncoder nn.Module satisfies the Protocol and
  drives parameterize without molzoo installed in the import graph of the test
  module (test file may still import molrep if needed for real encoder tests —
  keep at least one FakeEncoder-only test).
- **IR units (code):** parameterize output uses CLASS_I_CANONICAL (kcal/mol, Å, e, rad).
- **Energy smoke (scientific):** with identity/fake features and controlled head
  init or mocked composer, energy matches a hand-built IR evaluation.
- **Forces (code):** `compute_forces=True` yields (N,3) forces; finite-difference
  check on a tiny system optional but preferred.
- **Import boundary (code):** `molpot.composition.parameterizer` module AST/grep
  has no `molzoo` import.
- Full suite green.

## Out of scope

- Implementing chem encoder internals (04).
- Changing ClassicalMMComposer file ownership (03).
- Condensation / SMARTS / OpenMM export / provenance (06–09).
- Full trainer integration / logging hooks.
- Making eV the IR native unit.
