---
title: Learnable classical FF — physics-aware multi-system chemical class condensation
status: approved
created: 2026-08-10
revised: 2026-08-10
grilled: true
chain: learnable-classical-ff
depends_on:
  - learnable-classical-ff-03-mm-heads
  - learnable-classical-ff-04-chem-encoder
  - learnable-classical-ff-05-neural-parameterizer
---

# Learnable classical FF — physics-aware multi-system chemical class condensation

## Summary

Turn continuous per-interaction MM parameters into a **discrete, multi-system
chemical type system** via physics-aware condensation: merge interactions whose
parameters agree within **interaction-class-specific budgets**, producing a
`TypeSystem` per `InteractionClass` and a `Condenser` that greedily merges.
Generalize `TypeHead` / `Labeler` for multi-system labeling. **No SMARTS text**
— symbolic patterns are sub-spec 07.

## Domain basis

Condensation is a **lossy projection** from continuous parameter space to a
finite type table used by classical engines / export (08).

Merge only when parameter differences fall inside budgets that preserve
energetic fidelity at a stated tolerance:

| InteractionClass | Representative parameters | Example budget (tunable) |
|------------------|---------------------------|---------------------------|
| bond | k, r0 | Δr0 ≤ 0.01 Å; Δk / k ≤ 5% or absolute floor |
| angle | k, theta0 | Δθ0 ≤ 1° (in rad); relative k |
| proper | k_n, phase_n (per term) | per-term barrier and phase tolerances |
| improper | mode-dependent | same spirit |
| lj | epsilon, sigma | Δσ ≤ 0.01 Å; relative ε |
| charge | q | absolute Δq (e.g. 0.02 e) — often **not** condensed aggressively |

Budgets are config objects (`MergeCriterion`), not hard-coded magic alone —
defaults documented with physical rationale (export / force-field readability
vs energy error).

Greedy merge: sort by abundance or insertion order; assign each interaction to
an existing type if within criterion of the type centroid/prototype, else spawn
a new type. Deterministic given a documented sort key.

## Design


### Placement supersede (2026-08-10 — binding)

Full rule: `.claude/notes/learnable-classical-ff.md`.

1. **Reuse first** — prefer molpy (≥0.13) / in-tree modules over new twins.
2. **No `Foo(method=…)` on new APIs** — if generalizing, one more general single-responsibility type (or two named peers), never a method/mode switch that selects unrelated implementations.
3. **Non-diff sinks to molpy/molrs** (import via `molpy` only): topology enumeration, SMARTS, classical non-torch E/F, FF style tables. This sub-spec must not reimplement those.
4. **Improper index** = molrs `Topology` layout `[center, i, j, k]` (center at **row 0**).

**06-specific:**
- Condensation algorithms may live in molrep; residual metrics can call torch kernels or molpy evaluators via injected callbacks — no inlined second energy stack.
- No SMARTS emission (07).


### Reuse decision

| Symbol | Action | Rationale |
|--------|--------|-----------|
| `molrep.heads.TypeHead` / `molpot.heads.TypeHead` | **generalize carefully** | Prefer extending the molrep perception-side TypeHead for multi-system **or** add a thin multi-head wrapper; do not break existing atom-type API |
| `Labeler` / `ProxyLabeler` | **extend** | New `TypeSystemLabeler` implementing Labeler Protocol for condensed ids |
| Continuous heads (03) | **consume outputs** | Condensation runs on parameter tensors / IR bags, not raw chem features alone |
| SMARTS (07) | **out of scope** | Condensation emits integer type ids + optional prototype params only |

### Core types (`molrep/condensation/` or `molrep/chem/condensation.py`)

Prefer package:

```
src/molrep/condensation/
  __init__.py
  classes.py       # InteractionClass enum
  criterion.py     # MergeCriterion, default budgets
  type_system.py   # TypeSystem, TypeRecord
  condenser.py     # Condenser.greedy_merge
  labeler.py       # TypeSystemLabeler
```

**`InteractionClass`** — enum: `BOND`, `ANGLE`, `PROPER`, `IMPROPER`, `LJ`,
`CHARGE` (charge optional / often skipped).

**`MergeCriterion`** — per-class thresholds; method
`accepts(prototype_params, candidate_params) -> bool`.

**`TypeRecord`** — `{type_id: int, prototype: Tensor/dict, member_count: int,
support_ids: optional}`.

**`TypeSystem`** — holds ordered `TypeRecord`s for one InteractionClass;
methods: `assign(params) -> type_id`, `n_types`, `prototypes_table()`.

**`Condenser`**

```python
class Condenser:
    """Physics-aware greedy merge across one or many systems."""

    def merge(
        self,
        params_by_system: Sequence[Tensor | Mapping],
        *,
        interaction: InteractionClass,
        criterion: MergeCriterion,
    ) -> TypeSystem:
        ...
```

Multi-system: concatenate interactions with `system_id` tracking so type ids are
**global** across the chemical library, not per-molecule renumbered.

**`TypeHead` generalization**

- Multi-system: either one TypeHead per InteractionClass with `num_types` from
  a frozen TypeSystem, or a `MultiTypeHead` dict.
- Training mode (optional in this spec): classify continuous embeddings into
  discrete types with CE loss against condenser labels — if included, keep
  primitive (head.forward logits only; loss is caller's job).
- Reuse `decode` / `decode_with_confidence` where the molrep TypeHead already
  provides them (provenance 09 will need confidence).

**`TypeSystemLabeler`**

Implements `Labeler` Protocol (or a multi-interaction extension): maps batch
topology rows to condensed type ids via nearest prototype or stored assignment
table. `num_types` / `type_map` populated from TypeSystem.

### No SMARTS

Condensation outputs integer ids + numeric prototypes. Emitting SMARTS/SMIRKS
strings is **forbidden** here (07).

## Files to create or modify

- `src/molrep/condensation/` (new package as above)
- `src/molrep/heads/type.py` — multi-system / multi-class hooks if needed
- `src/molrep/heads/labeler.py` — TypeSystemLabeler
- `tests/test_molrep/test_condensation/test_criterion.py` (new)
- `tests/test_molrep/test_condensation/test_condenser.py` (new)
- `tests/test_molrep/test_condensation/test_type_system.py` (new)
- `tests/test_molrep/test_heads/test_type_system_labeler.py` (new)

## Tasks

- [ ] Write failing MergeCriterion tests (accept within budget, reject outside)
- [ ] Implement InteractionClass + MergeCriterion defaults with documented units
- [ ] Write failing Condenser tests: identical params -> 1 type; two far params -> 2 types; multi-system global ids
- [ ] Implement TypeSystem + Condenser.greedy_merge (deterministic sort key)
- [ ] Write failing TypeSystemLabeler tests implementing Labeler Protocol
- [ ] Implement TypeSystemLabeler; generalize TypeHead for multi-class num_types if required
- [ ] Ensure no SMARTS/string pattern generation in condensation package
- [ ] Docstrings + check + tests

## Testing strategy

- **Budget physics (scientific/code):** criterion uses Å / rad / kcal-scale
  differences as documented; flipping budget to 0 forces one type per unique
  param row.
- **Greedy determinism (code):** same inputs + same sort key → same TypeSystem
  (type count and prototype values).
- **Multi-system (code):** two systems sharing near-identical bond params collapse
  to one global type; distant params remain separate.
- **Labeler (code):** labels in `0..n_types-1`; type_map keys cover range.
- **No SMARTS (code):** condensation package has no smarts/smirks emitters.
- Full suite green.

## Out of scope

- SMARTS/SMIRKS emission and matching (07).
- OpenMM export of condensed tables (08) — consumes TypeSystem later.
- Provenance / coverage classifiers (09).
- End-to-end training loop with CE type loss (caller may add).
- Active learning loops.
