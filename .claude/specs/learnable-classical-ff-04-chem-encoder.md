---
title: Learnable classical FF — continuous chemical perception encoder
status: approved
created: 2026-08-10
revised: 2026-08-10
grilled: true
chain: learnable-classical-ff
depends_on:
  - learnable-classical-ff-02-valence-topology
---

# Learnable classical FF — continuous chemical perception encoder

## Summary

Add a **continuous chemical-perception** stack in `molrep.chem` that embeds
atoms and bonds and builds **symmetry-aware interaction contexts** (bond /
angle / proper / improper feature vectors) for downstream MM heads (03). A thin
`molzoo.chem` recipe (`ChemPerception`) composes the blocks. **No energy, no
force, no molpot import** — pure representation.

## Design


### Placement supersede (2026-08-10 — binding)

Full rule: `.claude/notes/learnable-classical-ff.md`.

1. **Reuse first** — prefer molpy (≥0.13) / in-tree modules over new twins.
2. **No `Foo(method=…)` on new APIs** — if generalizing, one more general single-responsibility type (or two named peers), never a method/mode switch that selects unrelated implementations.
3. **Non-diff sinks to molpy/molrs** (import via `molpy` only): topology enumeration, SMARTS, classical non-torch E/F, FF style tables. This sub-spec must not reimplement those.
4. **Improper index** = molrs `Topology` layout `[center, i, j, k]` (center at **row 0**).

**04-specific:**
- Discrete chemical graph features (element, charge, aromaticity, ring, bond order) come from molpy `Atomistic` / Frame fields when available — pack tensors only; do not re-run SMARTS for basic features.
- Continuous GNN remains molrep (diff).


### Reuse decision

| Symbol | Action | Rationale |
|--------|--------|-----------|
| `JointEmbedding` | **reuse** | Discrete Z + continuous features already fused here |
| `molrep.embedding` node/edge primitives | **reuse** | MLP / Embedding blocks |
| Geometric MACE/PiNet equivariant stacks | **do not force** | Chem perception is topology-first; may use scalar MLPs only in v1 |
| molpot heads / IR | **forbid import** | One-way dep: molrep must not import molpot |
| molzoo encoder pattern | **reuse** | Thin recipe under `molzoo.chem` writing features into TensorDict |

### Package layout

```
src/molrep/chem/
  __init__.py
  atom.py          # AtomChemEmbedding
  bond.py          # BondChemEmbedding
  context.py       # BondContext, AngleContext, ProperContext, ImproperContext builders
  encoder.py       # ChemEncoder (TensorDict in/out)
  embeddings.py    # ChemEmbeddings first-class container (optional TypedDict / dataclass of tensors)

src/molzoo/chem/
  __init__.py
  perception.py    # ChemPerception recipe nn.Module
```

### Core types

**`AtomChemEmbedding`**

- Inputs: atomic numbers `Z (N,)`, optional formal charge / aromaticity /
  hybridization as discrete channels if present in batch, else Z-only.
- Implementation: `JointEmbedding` (or Embedding+MLP) → `h_atom (N, D_a)`.
- OOP method surface: `forward(td) -> td` writing under
  `atoms.chem_features` or returning tensor from a pure module — prefer
  TensorDictModuleBase-style `in_keys`/`out_keys` when integrating.

**`BondChemEmbedding`**

- Inputs: endpoint atom embeddings + bond order / type if available + optional
  distance (scalar).
- Symmetry: \( h_{ij} = h_{ji} \) — implement by summing or mean-pooling
  direction-specific MLPs on `(h_i, h_j)` and `(h_j, h_i)`, or by sorting
  endpoints. **Test symmetry.**
- Output: `h_bond (N_bonds, D_b)` aligned with `bonds.bond_index` columns.

**Context builders** (pure modules or methods on a `ContextBuilder` type):

| Context | Pool recipe (v1) | Symmetry |
|---------|------------------|----------|
| Bond | already `h_bond` | \(ij = ji\) |
| Angle | MLP/pool over `(h_i, h_j, h_k)` with reverse `(k,j,i)` | equal under reverse |
| Proper | pool `(h_i..h_l)` with reverse `(l,k,j,i)` | equal under reverse |
| Improper | pool four atom embeddings; **central atom at row 0** (molrs `[center,i,j,k]`, matches 01 IR) | invariant under documented outer-leg swaps with central fixed |

Contexts consume topology from batch namespaces supplied by 02
(`bonds`, `angles`, `propers`, `impropers`).

**`ChemEmbeddings`**

First-class container (dataclass or plain dict with fixed keys) holding:

```
atom: (N, D_a)
bond: (N_bonds, D_b)
angle: (N_angles, D_ang)
proper: (N_propers, D_p)
improper: (N_impropers, D_imp)
```

Used as the feature payload for ClassicalMMComposer (03) / Parameterizer (05).

**`ChemEncoder`**

```python
class ChemEncoder(nn.Module):
    """Topology-aware chemical perception encoder.

    Reads atoms.Z / bonds.* / angles.* / propers.* / impropers.* and writes
    ChemEmbeddings tensors back onto the batch (or returns ChemEmbeddings).
    """
```

Mutate TensorDict in place under a documented key prefix
(e.g. `atoms.chem_features`, `bonds.chem_features`, …) **or** return
`ChemEmbeddings` — pick one primary path; if both, `forward` returns the
TensorDict and a method `embeddings(td) -> ChemEmbeddings` views the fields.

**`molzoo.chem.ChemPerception`**

Thin recipe: config (Pydantic BaseModel) + constructs ChemEncoder with
default dims. Encoder-only: writes features; no energy head.

### Dependency rules

- `molrep.chem` → may use `molrep.embedding`, `molix` dtype config only if
  already patterned elsewhere in molrep; prefer torch + molrep only.
- `molzoo.chem` → may import `molrep.chem`.
- **Neither** imports `molpot` or energy kernels.

## Files to create or modify

- `src/molrep/chem/__init__.py` (new)
- `src/molrep/chem/atom.py` (new)
- `src/molrep/chem/bond.py` (new)
- `src/molrep/chem/context.py` (new)
- `src/molrep/chem/encoder.py` (new)
- `src/molrep/chem/embeddings.py` (new)
- `src/molrep/__init__.py` (optional re-exports)
- `src/molzoo/chem/__init__.py` (new)
- `src/molzoo/chem/perception.py` (new)
- `tests/test_molrep/test_chem/test_atom.py` (new)
- `tests/test_molrep/test_chem/test_bond.py` (new)
- `tests/test_molrep/test_chem/test_context.py` (new)
- `tests/test_molrep/test_chem/test_encoder.py` (new)
- `tests/test_molzoo/test_chem/test_perception.py` (new)

## Tasks

- [ ] Write failing tests for AtomChemEmbedding shapes and JointEmbedding reuse
- [ ] Implement AtomChemEmbedding
- [ ] Write failing BondChemEmbedding symmetry tests (h_ij == h_ji)
- [ ] Implement BondChemEmbedding
- [ ] Write failing context symmetry tests (angle reverse, proper reverse)
- [ ] Implement context builders + ChemEmbeddings container
- [ ] Write failing ChemEncoder TensorDict I/O tests with valence namespaces
- [ ] Implement ChemEncoder; add molzoo ChemPerception recipe + config
- [ ] Assert no molpot imports; Google docstrings; run check + tests

## Testing strategy

- **Shapes (code):** embeddings align with topology counts from a synthetic
  batch (atoms N, bonds Nb, angles Na, …).
- **Symmetry (code/scientific):** bond/angle/proper reverse-order invariance of
  context features (allclose).
- **Isolation (code):** `molrep.chem` and `molzoo.chem` source trees contain no
  `molpot` import strings.
- **Recipe (code):** ChemPerception constructs, runs forward on a mini batch,
  writes expected keys.
- No energy assertions.
- Full suite green.

## Out of scope

- Parameter heads / energy (03 / 05).
- Discrete type condensation (06).
- SMARTS / SMIRKS symbolic matching (07).
- Equivariant geometric message passing beyond what is needed for chem
  features (MACE/Allegro remain separate encoders).
- Training / losses.
