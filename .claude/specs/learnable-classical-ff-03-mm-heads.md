---
title: Learnable classical FF — continuous MM parameter heads + ClassicalMMComposer
status: done
created: 2026-08-10
revised: 2026-08-10
grilled: true
chain: learnable-classical-ff
depends_on:
  - learnable-classical-ff-01-ir-kernels
  - learnable-classical-ff-02-valence-topology
---

# Learnable classical FF — continuous MM parameter heads + ClassicalMMComposer

## Summary

Add continuous (non-discrete-type) **canonical MM parameter heads** that map
per-interaction or per-atom feature vectors to Class-I IR bags, and a
**`ClassicalMMComposer`** that wires features → heads → `PotentialIR` → existing
evaluators (01 kernels + reused bonded/nonbonded) → scalar energy. No chemical
encoder lives here — features are injected. Sibling of sub-spec 05
(`ClassicalMMParameterizer` in a separate file); 03 owns heads + composer only.

## Domain basis

Head outputs are **continuous parameters in CLASS_I_CANONICAL units**
(kcal/mol, Å, e, rad) matching IR from 01:

| Head | Outputs | Constraints |
|------|---------|-------------|
| Bond | `k > 0`, `r0 > 0` | softplus floors |
| Angle | `k > 0`, `theta0 ∈ (0, π)` | softplus on k; theta0 via scaled sigmoid or softplus+clamp |
| Proper torsion | per-term `k ≥ 0`, free `phase` / fixed or free periodicity | softplus on k; phase unconstrained (rad) |
| Improper | periodic: same as proper; harmonic: `k > 0`, free `chi0` | softplus on k |
| LJ (generalized) | `epsilon > 0`, `sigma > 0` | softplus (existing) |
| Charge | free `q` with optional neutrality | reuse ChargeHead |

Endpoint symmetry: bond/angle/proper features that reverse atom order
`(i,j)↔(j,i)`, `(i,j,k)↔(k,j,i)`, `(i,j,k,l)↔(l,k,j,i)` must yield **identical**
parameters (enforce by symmetric feature construction at the call site or by
averaging head inputs inside the head — document which; prefer call-site
symmetric pooling so heads stay pure MLPs).

## Design


### Placement supersede (2026-08-10 — binding)

Full rule: `.claude/notes/learnable-classical-ff.md`.

1. **Reuse first** — prefer molpy (≥0.13) / in-tree modules over new twins.
2. **No `Foo(method=…)` on new APIs** — if generalizing, one more general single-responsibility type (or two named peers), never a method/mode switch that selects unrelated implementations.
3. **Non-diff sinks to molpy/molrs** (import via `molpy` only): topology enumeration, SMARTS, classical non-torch E/F, FF style tables. This sub-spec must not reimplement those.
4. **Improper index** = molrs `Topology` layout `[center, i, j, k]` (center at **row 0**).

**03-specific:**
- Heads emit torch tensors keyed consistently with 01 IR / molpy style params.
- Do not invent SMARTS or typing here. No `ParamHead(method="bond"|"angle")` mega-head — keep one head type per interaction family (or one MultiHead of disjoint heads).


### Reuse decision

| Symbol | Action | Rationale |
|--------|--------|-----------|
| `LJParameterHead` | **generalize / keep** | Already softplus ε,σ; extend if per-pair needed, else reuse as-is for atom LJ |
| `ChargeHead` | **reuse** | Per-molecule neutrality already implemented |
| `MultiHead` | **reuse** | Merge disjoint head dicts |
| `PotentialComposer` | **pattern-reuse** | ClassicalMMComposer is Class-I specialist; do not overload generic composer |
| Bond/Angle/Proper/Improper kernels + IR | **reuse from 01** | Heads emit bags, composer builds PotentialIR, evaluators compute E |
| Chemical encoder | **out of scope** | 04 |

### New heads (`molpot/composition/heads.py` or `molpot/composition/mm_heads.py`)

Prefer **`mm_heads.py`** if `heads.py` is already crowded; re-export from
`composition/__init__.py`.

```python
class BondParamHead(nn.Module):
    """features (N_bonds, D) -> {k: (N_bonds,), r0: (N_bonds,)} in kcal/mol/Å², Å."""

class AngleParamHead(nn.Module):
    """features (N_angles, D) -> {k, theta0}."""

class ProperTorsionParamHead(nn.Module):
    """features (N_propers, D) -> multi-term k, phase; periodicity config-fixed or predicted."""

class ImproperParamHead(nn.Module):
    """features (N_impropers, D) -> periodic and/or harmonic params per config."""
```

Conventions:

- Explicit `nn.Linear` (no `LazyLinear`).
- Softplus + positive floor on force constants and σ/ε/r0.
- `dtype=config.ftype`.
- Google docstrings with shapes and unit tags referencing IR.
- No factory `make_*` constructors.

**LJParameterHead:** keep atom-level API; if ClassicalMM needs pair LJ, add
optional `LJPairParamHead` only at second call site (inline until then).

### ClassicalMMComposer (`src/molpot/composition/classical_mm.py`) — NEW file

```python
class ClassicalMMComposer(nn.Module):
    """features + topology -> heads -> PotentialIR -> Class-I energy.

    Does NOT own an encoder. Caller supplies feature tensors keyed by
    interaction class (or a single atom feature tensor + scatter recipe).
    """
```

Pipeline:

1. **Inputs:** atom features `(N, D_a)` and/or pre-pooled interaction features
   for bonds/angles/propers/impropers; topology from batch namespaces
   (`bonds`, `angles`, `propers`, `impropers`, `atoms`, `edges` as needed for
   nonbonded pairs).
2. **Heads:** run Bond/Angle/Proper/Improper/LJ/Charge heads (MultiHead for
   atom-level; dedicated calls for interaction-level).
3. **IR assemble:** build `PotentialIR` bags in CLASS_I units + default
   `NonbondedScaling`.
4. **Evaluate:** instantiate or hold kernel modules (`BondHarmonic` style
   type-indexed **or** per-interaction parameter mode — prefer **per-interaction
   continuous parameters** for the learnable path, avoiding a discrete type
   table inside the composer; document if kernels need a thin per-interaction
   adapter).
5. **Aggregate:** sum term energies → scalar graph energy; optional per-term
   breakdown dict for debugging.

Forces: **not** hand-rolled. Expose energy as a function of `pos` and use
`ForceDerivation` / `BasePotential.calc_forces` only (autograd default).

### Public API shape

Primitive, caller-composed:

```python
composer = ClassicalMMComposer(bond_head=..., angle_head=..., ...)
ir = composer.parameterize(features, batch)   # -> PotentialIR
energy = composer.energy(ir, batch, pos=...)  # or composer.forward(batch, features)
```

Prefer splitting `parameterize` vs `energy` as two primitives rather than a
single `run_everything`. `forward` may chain them for nn.Module convenience but
must remain a thin composition of the primitives.

### File ownership note (conflict fix)

- **This spec (03)** owns `src/molpot/composition/classical_mm.py` and
  `ClassicalMMComposer` plus MM heads.
- **Spec 05** owns `src/molpot/composition/parameterizer.py` and
  `ClassicalMMParameterizer` (encoder Protocol → composer/heads). Do **not**
  place the parameterizer in `classical_mm.py`.

## Files to create or modify

- `src/molpot/composition/mm_heads.py` (new) **or** extend `heads.py`
- `src/molpot/composition/classical_mm.py` (new) — ClassicalMMComposer
- `src/molpot/composition/__init__.py` — exports
- `src/molpot/__init__.py` — exports if public
- `tests/test_molpot/test_composition/test_mm_heads.py` (new)
- `tests/test_molpot/test_composition/test_classical_mm.py` (new)

## Tasks

- [x] Write failing tests for BondParamHead / AngleParamHead positivity (softplus) and shapes
- [x] Implement BondParamHead, AngleParamHead
- [x] Write failing tests for ProperTorsionParamHead multi-term outputs and ImproperParamHead
- [x] Implement ProperTorsionParamHead, ImproperParamHead; generalize LJParameterHead only if tests require
- [x] Write failing ClassicalMMComposer tests: features→PotentialIR bags; energy finite; endpoint symmetry when inputs symmetric
- [x] Implement ClassicalMMComposer.parameterize + energy using IR + 01 kernels; MultiHead+ChargeHead reuse
- [x] Export symbols; Google docstrings with units (kcal/mol, Å, e, rad)
- [x] Run check + unit tests

## Testing strategy

- **Positivity (code/scientific):** k, r0, epsilon, sigma always > floor after
  forward on random features (including large negative pre-activations).
- **Shapes (code):** head outputs match interaction counts from batch topology.
- **IR units (code):** composer.parameterize sets `unit_system` consistent with
  CLASS_I_CANONICAL.
- **Energy smoke (scientific):** harmonic bond-only system with known k,r0,pos
  recovers \( \tfrac12 k (r-r_0)^2 \) when heads are replaced by constant
  embeddings / mocked outputs.
- **Endpoint symmetry (code):** reversing bond endpoints in features yields
  identical parameters when the test supplies symmetric pooled features.
- **No encoder import (code):** classical_mm.py / mm_heads.py do not import
  molzoo or molrep.chem.
- Full suite green.

## Out of scope

- Chem encoder / perception (04).
- ClassicalMMParameterizer encoder→… path (05) — separate file.
- Discrete type condensation (06).
- SMARTS / export / provenance (07–09).
- Training loops / losses (molix trainer wiring).
- Functorch-only force path (use default autograd ForceDerivation).
