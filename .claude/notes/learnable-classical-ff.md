<!-- mol:note:topic:learnable-classical-ff -->
# Learnable classical FF — placement & reuse (2026-08-10)

## Why

The architecture must sit **between** continuous ML chemical perception and
classical MM, without re-implementing molpy/molrs force-field infrastructure.

## Rule (binding for `learnable-classical-ff-*` specs and all new code)

### 1. Prefer existing modules

Before adding a type in molnex, search molpy (≥0.13) / molrs (via molpy only) /
in-tree molpot/molrep/molix. Prefer **reuse** or **generalize**; invent only
when no existing owner fits.

Known homes (do not fork):

| Concern | Owner |
|---------|--------|
| Force-field model (styles, types, params) | `molpy.core.forcefield.ForceField` / `molpy.potential.*` (molrs-backed) |
| Classical non-torch E/F | `forcefield.to_potentials().calc_energy/forces(frame)` |
| Topology **enumerate** (optional offline) | molrs `Topology` via molpy — produces index **columns**, not a second store |
| Batch topology **storage** | **molix TensorDict only** — never park angles/propers in molpy Frame for the ML path |
| SMARTS match / typifier base | `molpy.typifier.smarts.SmartsTypifier`, `molrs.perceive.SmartsPattern` |
| Torch classical terms for **differentiable** training | `molpot.potentials.*` (align names/params with molpy styles) |
| Batch collate / rebase | `molix.data.collate` |
| Chem perception (learned continuous) | `molrep` + `molzoo` recipes |
| Learnable param heads / IR torch bags for training | `molpot.composition` / thin `molpot.ir` **only as torch-facing view** of Class-I params |

### TensorDict topology contract (molix — not molpy)

Post-collate (and flat sample) connectivity for classical MM uses **column
keys under a namespace**, same spirit as molpy/molrs Frame blocks
(`atomi` / `atomj` / …), nested in TensorDict:

```python
batch["bonds", "atomi"]      # (N_b,) long
batch["bonds", "atomj"]      # (N_b,)
batch["angles", "atomi"]     # (N_a,)  — user-facing: td["angles"]["atomi"]
batch["angles", "atomj"]
batch["angles", "atomk"]     # central atom for angles is atomj (i-j-k)
batch["propers", "atomi"]    # (N_p,)
batch["propers", "atomj"]
batch["propers", "atomk"]
batch["propers", "atoml"]
batch["impropers", "atomi"]  # molrs center-first: atomi = center
batch["impropers", "atomj"]
batch["impropers", "atomk"]
batch["impropers", "atoml"]
# optional type columns:
batch["angles", "type"]      # (N_a,)  or "angle_types" — pick one, document
```

**Not** a packed `angle_index [3, N]` as the primary batch schema (that may
exist only as a kernel-local stack at the potential call site:
`torch.stack([atomi, atomj, atomk], dim=0)`).

**Not** “store the batch in molpy”. molpy may *emit* columns when building a
sample; the live training/MD batch is TensorDict under molix.

Rebase on collate: each of `atomi`/`atomj`/`atomk`/`atoml` is an atom-index
1-D vector; add `atom_offset` to every present column under the valence
namespaces (register keys in `INDEX_KEYS` or a sibling column registry).

### 2. Generalize without multi-method switch

If two modules do similar work, **promote a single more general type** with one
clear responsibility — never a switch:

```python
# ❌ forbidden in new APIs
Foo(method="a" | "b")
Bar(mode="x")  # when mode selects unrelated implementations

# ✅ required
MoreGeneralFoo(...)          # one implementation, broader domain
# or two peer types with distinct names if both must exist:
AutogradForces / FunctorchForces
```

Pre-existing `ForceDerivation(method=…)` is **legacy**; do **not** copy this
pattern into new classical-FF surfaces. New force entry points for Class-I
training use one path (prefer `BasePotential.calc_forces` / autograd) unless a
second named type is justified.

### 3. Non-diff sinks to molpy / molrs

Anything that does **not** require PyTorch differentiation **must not** be
reimplemented in molnex:

- valence enumeration, ring/aromatic perception, SMARTS matching
- discrete FF table models, style registries, non-torch energy
- unit conversion tables for engine export (prefer molpy IO / conventions)
- graph chemical feature extraction from molpy `Atomistic` / Frame

Molnex owns:

- continuous chem encoder (diff)
- continuous → MM parameter heads (diff)
- torch classical energy for training + autograd forces
- collate of **already-built** valence index tensors into TensorDict batches
- thin adapters: molpy ForceField / Topology / SMARTS hits → torch IR bags

**Import hard rule (unchanged):** `from molpy import …` only under `src/` /
`tests/`; never bare `import molrs`.

### Improper index convention

**Source of truth:** molrs `Topology` impropers = `[center, i, j, k]` (center at
**row 0**). Torch `improper_index` matches that layout. OpenFF trefoil reordering
is an **export/import adapter**, not a second internal convention.

### Dependency pin

`molcrafts-molpy>=0.13.0` (molpy 0.13.x line; molrs major.minor paired by molpy).

## Supersedes

- Spec drafts that invented parallel SMARTS engines, valence enumerators, or
  OpenMM-only IR without molpy ForceField alignment
- Improper “central at row 1” chain-wide default (replaced by molrs center-first)

## Spec impact (chain)

| Sub-spec | Must change |
|----------|-------------|
| 01 | IR bags align to molpy Class-I styles; improper center row 0; no second FF model |
| 02 | Collate TensorDict namespaces `bonds`/`angles`/`propers`/`impropers` with **atomi… columns** (`td["angles"]["atomi"]`); enum optional upstream; never molpy as batch store |
| 03–05 | Stay molnex (diff path); inject topology/features from molpy-built tensors |
| 06 | Perception-side merge; physical_eval may call torch kernels or molpy for residuals |
| 07 | **Reuse** SmartsTypifier / SmartsPattern; no second matcher engine |
| 08 | Prefer molpy forcefield IO / conventions; case matrix still molnex if missing |
| 09 | Thin surfaces only |

**Status.** active (binding for learnable-classical-ff chain).
