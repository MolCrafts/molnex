---
title: Learnable classical FF — confidence, coverage, and provenance surfaces
status: approved
created: 2026-08-10
revised: 2026-08-10
grilled: true
chain: learnable-classical-ff
depends_on:
  - learnable-classical-ff-05-neural-parameterizer
  - learnable-classical-ff-06-condensation
  - learnable-classical-ff-07-smarts
  - learnable-classical-ff-08-ff-export
---

# Learnable classical FF — confidence, coverage, and provenance surfaces

## Summary

Expose **parameter provenance** and **chemical-space coverage** so consumers of
a learnable Class-I FF know whether a prediction is in-support, extrapolated, or
unknown. Split surfaces across packages by dependency rules: chemical support
indexing in **molrep**, coverage regimes + parameter provenance records in
**molpot**, reusing `TypeHead.decode_with_confidence`. Active-learning loops are
out of scope.

## Design


### Placement supersede (2026-08-10 — binding)

Full rule: `.claude/notes/learnable-classical-ff.md`.

1. **Reuse first** — prefer molpy (≥0.13) / in-tree modules over new twins.
2. **No `Foo(method=…)` on new APIs** — if generalizing, one more general single-responsibility type (or two named peers), never a method/mode switch that selects unrelated implementations.
3. **Non-diff sinks to molpy/molrs** (import via `molpy` only): topology enumeration, SMARTS, classical non-torch E/F, FF style tables. This sub-spec must not reimplement those.
4. **Improper index** = molrs `Topology` layout `[center, i, j, k]` (center at **row 0**).

**09-specific:**
- Thin audit surfaces only. No AL loop (stays external). Reuse `TypeHead.decode_with_confidence` without a parallel softmax helper.


### Reuse decision

| Symbol | Action | Rationale |
|--------|--------|-----------|
| `TypeHead.decode_with_confidence` (molrep.heads.type / molpot.heads.type) | **reuse** | Already returns `(indices, confidence)` via softmax max |
| TypeSystem / Condenser (06) | **consume** | Known support = condensed training types |
| SymbolicForceField (07) | **optional link** | Provenance may cite SMARTS pattern id |
| ForceFieldCompiler (08) | **optional attach** | Provenance metadata on ForceSpec entries |
| Active learning orchestration | **out of scope** | Surfaces only |

### molrep: ChemicalSupportIndex

```
src/molrep/provenance/
  __init__.py
  support.py     # ChemicalSupportIndex
```

**`ChemicalSupportIndex`**

- Built from training chemical fingerprints or type-id histograms / embedding
  centroids (v1: discrete type_id sets per InteractionClass + optional
  embedding ball radius).
- API:

```python
class ChemicalSupportIndex:
    def __init__(self, supported_type_ids: Mapping[InteractionClass, set[int]], ...): ...
    def contains(self, interaction: InteractionClass, type_id: int) -> bool: ...
    def coverage_fraction(self, predicted_ids: Tensor, interaction: InteractionClass) -> float: ...
```

Pure bookkeeping in molrep — no molpot import.

### molpot: coverage + provenance

```
src/molpot/provenance/
  __init__.py
  regime.py        # CoverageRegime enum
  classifier.py    # SupportClassifier
  parameter.py     # ParameterProvenance dataclass
```

**`CoverageRegime`** enum:

- `IN_SUPPORT` — type_id / embedding inside ChemicalSupportIndex (or high confidence)
- `NEAR_SUPPORT` — within a configured margin (confidence band or distance)
- `EXTRAPOLATING` — finite prediction but outside support
- `UNKNOWN` — missing topology / failed match / below confidence floor

**`SupportClassifier`**

```python
class SupportClassifier:
    """Map confidences + support index -> CoverageRegime per interaction row."""

    def __init__(
        self,
        support: ChemicalSupportIndex | None,
        *,
        conf_in: float = 0.8,
        conf_near: float = 0.5,
    ): ...

    def classify(
        self,
        type_ids: Tensor,
        confidence: Tensor,
        *,
        interaction: InteractionClass,
    ) -> list[CoverageRegime] | Tensor: ...
```

Accepts confidences from `TypeHead.decode_with_confidence(logits)`.

**`ParameterProvenance`**

```python
@dataclass(frozen=True)
class ParameterProvenance:
    interaction: str | InteractionClass
    type_id: int | None
    confidence: float | None
    regime: CoverageRegime
    source: str                  # "neural_continuous" | "condensed_type" | "symbolic" | ...
    pattern: str | None = None   # SMARTS if known
    ir_units: str = "class_i_canonical"
    notes: str = ""
```

Factory-free: construct with `ParameterProvenance(...)`. Helpers may attach a
list of provenances to a PotentialIR or ForceSpec as metadata dicts without
becoming a god context object — keep metadata **alongside**, not a mega blob
every layer reaches into.

### Integration points (thin)

- Parameterizer (05) **may** optionally return provenance list when
  `return_provenance=True` — additive flag, default off.
- Compiler (08) **may** copy provenance into ForceSpec metadata — optional
  follow-through; if not wired in this sub-spec, provide a pure function
  `attach_provenance(spec, records) -> ForceSpec` in molpot or molix.ff_export
  without forcing call sites.

Prefer implementing core types + classifier tests first; wiring flags are
tasks only if low-cost.

### Dependency rules

- molrep.provenance ↛ molpot
- molpot.provenance may import molrep.provenance **only if** package dependency
  already allows molpot→molrep; if not, redefine a structural Protocol for
  support lookup in molpot and keep ChemicalSupportIndex in molrep for the
  perception side. Document the chosen direction in the implementation PR.
- Neither owns AL loop / acquisition functions.

## Files to create or modify

- `src/molrep/provenance/__init__.py` (new)
- `src/molrep/provenance/support.py` (new)
- `src/molpot/provenance/__init__.py` (new)
- `src/molpot/provenance/regime.py` (new)
- `src/molpot/provenance/classifier.py` (new)
- `src/molpot/provenance/parameter.py` (new)
- `tests/test_molrep/test_provenance/test_support.py` (new)
- `tests/test_molpot/test_provenance/test_classifier.py` (new)
- `tests/test_molpot/test_provenance/test_parameter.py` (new)

## Tasks

- [ ] Write failing ChemicalSupportIndex contains / coverage_fraction tests
- [ ] Implement ChemicalSupportIndex in molrep.provenance
- [ ] Write failing SupportClassifier regime tests (in / near / extrapolating / unknown)
- [ ] Implement CoverageRegime + SupportClassifier using confidence thresholds
- [ ] Write failing ParameterProvenance construction / immutability tests
- [ ] Implement ParameterProvenance; wire decode_with_confidence in a small
      example unit test (TypeHead logits -> ids, conf -> classify)
- [ ] Optional: return_provenance flag or attach_provenance helper
- [ ] Docstrings; ensure no AL loop code; check + tests

## Testing strategy

- **Support index (code):** membership true/false; coverage_fraction on mixed
  predicted ids.
- **Classifier (scientific/code):** fixed confidences map to expected regimes
  under documented thresholds; missing support index degrades safely
  (confidence-only or UNKNOWN policy — document + test).
- **Provenance record (code):** frozen dataclass fields; JSON-friendly
  `asdict` if provided.
- **Reuse (code):** test calls real `TypeHead.decode_with_confidence` (molrep or
  molpot head — pick the one condensation uses) rather than reimplementing
  softmax max.
- **No AL (code):** provenance packages do not define acquisition/sample
  selection loops.
- Full suite green.

## Out of scope

- Active learning loops, uncertainty-driven acquisition, online fine-tuning.
- Changing condensation merge physics (06).
- Changing export goldens (08).
- UI / dashboard for coverage maps.
- Training metrics hooks in molix Trainer (may consume provenance later).
