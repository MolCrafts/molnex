---
title: Learnable classical FF — SMARTS/SMIRKS symbolic interface + SymbolicForceField
status: approved
created: 2026-08-10
revised: 2026-08-10
grilled: true
chain: learnable-classical-ff
depends_on:
  - learnable-classical-ff-06-condensation
---

# Learnable classical FF — SMARTS/SMIRKS symbolic interface + SymbolicForceField

## Summary

Add a **symbolic chemical-perception interface** under `molrep.perception` that
binds discrete condensed classes (06) to SMARTS/SMIRKS patterns, matches them
via a `SmartsMatcher` Protocol (fake matcher for unit tests; `MolpySmartsMatcher`
backed by **molpy only**, never bare `molrs`), and exposes a
`SymbolicForceField` that pairs patterns with parameter prototypes for
export/human inspection. Matching is pure perception — no energy evaluation.

## Design


### Placement supersede (2026-08-10 — binding)

Full rule: `.claude/notes/learnable-classical-ff.md`.

1. **Reuse first** — prefer molpy (≥0.13) / in-tree modules over new twins.
2. **No `Foo(method=…)` on new APIs** — if generalizing, one more general single-responsibility type (or two named peers), never a method/mode switch that selects unrelated implementations.
3. **Non-diff sinks to molpy/molrs** (import via `molpy` only): topology enumeration, SMARTS, classical non-torch E/F, FF style tables. This sub-spec must not reimplement those.
4. **Improper index** = molrs `Topology` layout `[center, i, j, k]` (center at **row 0**).

**07-specific:**
- **Reuse** `molpy.typifier.smarts.SmartsTypifier` and `molrs.perceive.SmartsPattern` (via molpy). `FakeSmartsMatcher` for unit tests only.
- Forbidden: second SMARTS engine in molnex; bare `import molrs` in molnex tests/src.


### Reuse decision

| Symbol | Action | Rationale |
|--------|--------|-----------|
| TypeSystem / TypeRecord (06) | **consume** | Discrete class ids + prototypes |
| molpy Element/MolRec/SMARTS APIs | **use via molpy** | Project hard rule: never bare molrs under src/tests |
| RDKit | **forbid** | Not on allowed third-party surface |
| Continuous ChemEncoder (04) | **sibling** | Symbolic path is discrete/export-facing |

### Package layout

```
src/molrep/perception/
  __init__.py
  records.py       # DiscreteClassRecord
  patterns.py      # SymbolicPattern
  matcher.py       # SmartsMatcher Protocol, FakeSmartsMatcher, MolpySmartsMatcher
  registry.py      # ClassPatternRegistry
  forcefield.py    # SymbolicForceField
```

### Types

**`DiscreteClassRecord`**

```python
@dataclass(frozen=True)
class DiscreteClassRecord:
    interaction: InteractionClass  # from condensation
    type_id: int
    prototype: Mapping[str, float | tuple[float, ...]]
    smarts: str | None = None      # optional until bound
    smirks: str | None = None      # for parameter-bearing transforms if used
    label: str | None = None
```

**`SymbolicPattern`**

Holds SMARTS or SMIRKS string + arity (1 atom, 2 bond, 3 angle, 4 torsion) +
optional atom-map documentation. Validation: non-empty pattern string; arity in
{1,2,3,4}.

**`SmartsMatcher` Protocol**

```python
class SmartsMatcher(Protocol):
    def match_atoms(self, mol, pattern: str) -> LongTensor:
        """Return atom indices (K,) or (K, arity) matches."""
        ...
    def match_bonds(self, mol, pattern: str) -> LongTensor: ...
    # or a single match(mol, pattern, arity) -> LongTensor [arity, K]
```

Prefer one `match(mol, pattern: SymbolicPattern) -> Tensor` returning
`[arity, n_hits]` COO-style to align with valence topology conventions.

**`FakeSmartsMatcher`**

Deterministic test double: configured with an explicit dict
`pattern_str -> matches tensor`. No molpy required. Used in all unit tests that
do not need real chemistry.

**`MolpySmartsMatcher`**

```python
class MolpySmartsMatcher:
    """SMARTS matching via molpy (not molrs)."""
```

- `from molpy import ...` only.
- If molpy SMARTS surface is incomplete, implement the thinnest wrapper and
  mark integration tests `skipif` with a clear reason — but the class must exist
  and import molpy only.
- Never `import molrs` / `from molrs import`.

**`ClassPatternRegistry`**

Maps `(InteractionClass, type_id) -> SymbolicPattern` with reverse lookup
`pattern -> type_id`. Binding API:

```python
registry.bind(interaction, type_id, pattern: SymbolicPattern) -> None
registry.get(interaction, type_id) -> SymbolicPattern
```

Rejects duplicate conflicting binds.

**`SymbolicForceField`**

```python
class SymbolicForceField:
    """Discrete classes + patterns + prototypes for human/export consumption."""

    def __init__(self, type_systems: Mapping[InteractionClass, TypeSystem],
                 registry: ClassPatternRegistry): ...

    def records(self) -> list[DiscreteClassRecord]: ...
    def match_molecule(self, mol, matcher: SmartsMatcher) -> dict[...]: ...
```

`match_molecule` returns assigned type ids per interaction found (for
validation against continuous/condensed paths). **No energy.**

### molpy-only hard rule

All of `src/molrep/perception/` and its tests: molpy only, never molrs.
Add a unit test that greps or imports the matcher module and asserts no molrs.

## Files to create or modify

- `src/molrep/perception/` (new package)
- `tests/test_molrep/test_perception/test_patterns.py` (new)
- `tests/test_molrep/test_perception/test_matcher.py` (new)
- `tests/test_molrep/test_perception/test_registry.py` (new)
- `tests/test_molrep/test_perception/test_forcefield.py` (new)

## Tasks

- [ ] Write failing tests for SymbolicPattern validation and DiscreteClassRecord
- [ ] Implement records.py + patterns.py
- [ ] Write failing FakeSmartsMatcher tests (configured hits returned as [arity,K])
- [ ] Implement SmartsMatcher Protocol + FakeSmartsMatcher
- [ ] Write MolpySmartsMatcher with molpy-only imports; smoke test or skipif
- [ ] Implement ClassPatternRegistry bind/get/conflict tests
- [ ] Implement SymbolicForceField.records / match_molecule using FakeSmartsMatcher
- [ ] molrs-forbidden test; docstrings; check + tests

## Testing strategy

- **Fake matcher (code):** sole dependency for registry/forcefield unit tests.
- **Registry (code):** bind, get, conflict raises.
- **Force field (code):** records length equals sum of type counts with bound
  patterns; match_molecule returns expected type assignments for fake hits.
- **molpy matcher (code):** import path uses molpy; optional functional test if
  molpy SMARTS available.
- **Hard rule (code):** no molrs in perception package.
- No energy / molpot imports required.
- Full suite green.

## Out of scope

- Energy evaluation / OpenMM export (08).
- Automatic SMARTS *induction* from condensed clusters (research); v1 is
  bind/match only (human or external patterns).
- RDKit or OpenFF toolkit dependencies.
- Continuous encoder changes (04).
