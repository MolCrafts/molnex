---
title: Unexported EnergyResidual writes in-flight energy_residual without touching raw totals
status: code-complete
grilled: true
created: 2026-09-11
chain: cache-layers
depends: cache-layers-03-refs, cache-layers-04-dtype
supersedes: cache-layers-05-residual
---

# Unexported EnergyResidual writes in-flight energy_residual without touching raw totals

## Summary

A collated batch can carry an in-flight residual energy label without mutating the raw total: `molix.data.residual.EnergyResidual` is a keyword-only `BatchTask` that always writes `graphs.e0` (float64) and `graphs.energy_residual` (compute dtype) from a caller-supplied frozen Z-indexed table, never overwrites `graphs.energy`, never touches disk, and never constructs `AtomicReferenceEnergy`. Ghost rows whose placeholder `Z` is a real element (default hydrogen, `pad_atom_type=1`) contribute zero through `atoms.mask`. This slug’s tests are the current caller; the type stays off `molix.data.__all__` and is not named as a shipped `BatchTask` subclass in `task.py` until `cache-layers-07-protocol` mounts it and retires `class_i_fit.energy_force_terms` as a label residualizer.

## Domain basis

Elemental atomic-reference dressing is the per-graph sum of a frozen coefficient table over **real** atoms only:

\[
e_0(g)=\sum_{i\in g,\,m_i=1} c[Z_i],\qquad
E_{\mathrm{res}}(g)=E_{\mathrm{total}}(g)-e_0(g),\qquad
E_{\mathrm{total}}(g)=E_{\mathrm{res}}^{\mathrm{(pred)}}(g)+e_0(g).
\]

\(c[Z]\) is a dense Z-indexed table in the **same energy unit** as `graphs.energy` (eV on OMol25 / OC22). A zero slot is not support: a boolean mask \(s[Z]\) names which atomic numbers are defined. \(e_0\) depends only on composition, so

\[
\frac{\partial e_0}{\partial \mathbf{r}_i}=0
\qquad\Rightarrow\qquad
\mathbf{F}_i=-\frac{\partial E_{\mathrm{res}}}{\partial \mathbf{r}_i}=-\frac{\partial E_{\mathrm{total}}}{\partial \mathbf{r}_i}.
\]

The subtraction is carried out in IEEE float64. Only \(E_{\mathrm{res}}\) may then follow the batch compute dtype (`atoms.pos`). A late `.double()` cannot recover digits already lost in float32. Ghost atoms with a real-element placeholder \(Z\) (default hydrogen, \(Z=1\)) are excluded by the structural mask \(m_i\), not by inspecting \(Z\).

This spec implements the first two identities as an in-flight **label** transform. Restoration \(E_{\mathrm{total}}=E_{\mathrm{res}}^{\mathrm{(pred)}}+e_0\) is the scientific inverse; the inference adapter that applies it is out of scope here. Adding \(e_0\) onto a model prediction is not this transform: `MacePotential.energy_core` already does that add into `ENERGY_KEY`.

References:

- Levine et al., “The Open Molecules 2025 (OMol25) Dataset, Evaluations, and Models,” 2025. https://arxiv.org/abs/2505.08762
- Tran et al., “The Open Catalyst 2022 (OC22) Dataset and Challenges for Oxide Electrocatalysis,” *ACS Catal.* **13**, 3066–3084 (2023). https://doi.org/10.1021/acscatal.2c05426 https://arxiv.org/abs/2206.08917

Units: \(c\), \(e_0\), \(E_{\mathrm{total}}\), \(E_{\mathrm{res}}\) share the energy unit of `graphs.energy`. Positions stay Å. `EnergyResidual` does not convert units (`molix.units` stays with cache-layers-03-refs).

Hard-coded oracle (numpy float64, water-like \(\mathrm{H_2O}\), eV; used as the domain golden):

```text
Z = [8, 1, 1]
c[1] = -13.6
c[8] = -2043.8
E_total = -2071.1234567890123
e0      = -2043.8 + 2*(-13.6) = -2071.0
E_res   = E_total - e0        = -0.1234567890123
```

A `PadMolecularBatch(..., pad_atom_type=1)` ghost with the same \(c[1]\) must leave \(e_0\) and \(E_{\mathrm{res}}\) unchanged.

## Design

**Slice of `cache-layers-01-cache`.** This spec owns only the in-flight residual / mask / label-ownership contract from that document’s TensorFrame table (`graphs.energy` raw, `graphs.e0` always written here, `graphs.energy_residual` explicit, `atoms.mask` structural). Fit/freeze/checkpoint of \(c[Z]\) is cache-layers-03-refs. Column-aware collate/FrameStore precision and `E0_KEY` are cache-layers-04-dtype. Public façade export, `DataModule` mount, and retirement of the script-local label subtract are cache-layers-07-protocol. Cache, geometry, and `ForceOutlierWeight` stay on 01 or out of scope.

**Law findings from the previous draft (resolved here).** `architecture-first`: the type is **not** public `molix.data` surface in this slug. Tests of `molix.data.residual.EnergyResidual` are the current caller. `cache-layers-07-protocol` is the composition site (mount on `DataModule`, add to `__all__`, name in `task.py`). `conceptual-integrity`: one in-flight **label** residualizer — `EnergyResidual`. `scripts/mm_param_learning/class_i_fit.py` `energy_force_terms` (`E_DFT − e0` via `index_add_` / prediction `e0`) is **superseded by 07**; it must not remain a second label residualizer after that spec lands. This slug does not edit the script. `MacePotential` / `AtomicReferenceEnergy` remains a model-side **ADD** of \(e_0\) into `ENERGY_KEY` (prediction / inference total), not a second label residualizer. A training step must not subtract on the label and add on the prediction for the same loss. `earn-complexity`: no `write_e0` flag — `graphs.e0` is always written. `tests-owned-behavior`: `src/molix/core/losses/molecular.py` and `MSELoss` are not modified; `target_key="energy_residual"` is covered in `TestEnergyResidual` as a column presence / isolation assertion. `explicit-flow`: illegal tables fail at construction.

**Owning type.** `molix.data.residual.EnergyResidual(BatchTask)` — named after `graphs.energy_residual`, same pattern as `TotalCharge` → `graphs.total_charge`. Not a homonym of `ResidualInteraction`. Construction is keyword-only, matching `PadMolecularBatch`:

```python
EnergyResidual(*, coefficients: Tensor, support: Tensor)
```

No `write_e0`, no `make_residual`, no factory, no `nn.Module`. `coefficients` is a 1-D Z-indexed table; `support` is a 1-D boolean of the same length. `__init__` raises `ValueError` unless both are 1-D, they have the same length, and `support.dtype` is `torch.bool` (do not silently cast uint8). After those checks it clones both, detaches, stores coefficients as `float64` and support as `bool`. Caller mutation after construction is invisible. molix never imports molpot and never constructs `AtomicReferenceEnergy`; cache-layers-03-refs (or a 07 script) hands in the tensors.

**Not shipped this slug.** Do not add `EnergyResidual` to `molix.data.__all__`. Do not list it as a shipped subclass in `src/molix/data/task.py` (that file still names `PadMolecularBatch` and `TotalCharge` only). `__init__.py` may import the symbol so tests can reach it as a package attribute, but **must not** put it on `__all__`. Tests import `from molix.data.residual import EnergyResidual`. Public export is 07’s job.

**Lifecycle / collate order.** When 07 mounts it, it sits on `DataModule(batch_tasks=[...])` after collate. When the batch is padded, `PadMolecularBatch` must run first so `atoms.mask` exists:

```python
batch_tasks=[
    PadMolecularBatch(max_atoms=..., max_pairs=..., cutoff=...),
    EnergyResidual(coefficients=c, support=s),
]
```

This slug does not mount it. `execute` follows `TotalCharge`: refuse a plain `dict` (`TypeError` — a wrap would write a container the caller does not hold); wrap a `TensorDict` / `TensorFrame` with `as_tensor_frame`; mutate in place; return the caller’s container. Runs in the DataLoader collate, **not** inside `Compiler` / `Derivative` — host raises and `.item()` are allowed here. No compile test is owed. `_CollateFn` already runs tasks then `batch_to`; cache-layers-04-dtype keeps `E0_KEY` float64 across that move. `ENERGY_RESIDUAL_KEY` is **not** added to `FLOAT64_LABEL_KEYS` — the residual follows `atoms.pos` dtype after the float64 subtract.

**`execute` arithmetic.**

1. Read `atoms.Z`, `atoms.batch`, `graphs.energy` (`ENERGY_KEY`). Missing namespaces / missing `graphs.energy` raise `KeyError`.
2. Structural mask: `atoms.mask` if present, else all-True. Ghost rows are `False`.
3. Real atoms (`mask == True`) with `Z < 0`, `Z >= len(coefficients)`, or `support[Z] == False` raise `ValueError` naming the unique offending \(Z\) values. Ghost rows never raise, even when `pad_atom_type=1` and `support[1]` is False or `c[1]` is nonzero. A zero coefficient with `support[Z] == False` still raises on a real atom (zero is not support).
4. Per-atom reference `c[Z].to(float64) * mask.to(float64)` (ghosts contribute exact 0; no boolean gather). Out-of-range ghost `Z` must still contribute 0 without raising (index through a clamped gather, then multiply by the mask).
5. `graphs.e0` `(B,)` float64 = `molix.F.batch_add` of those per-atom values into `atoms.batch` with `dim_size = graphs.batch_size[0]`. **Always written** (no flag).
6. `graphs.energy_residual` = `(graphs.energy.to(float64) - e0).to(atoms.pos.dtype)`.
7. `graphs.energy` is not replaced: the same tensor object remains. `EnergyResidual` does not write `atoms.energy`, `ENERGY_KEY`, or `FORCES_KEY`.

**`task_id`.** `energy_residual:<digest>` where digest is a stable short hash of the frozen coefficients and support bytes, so two tables never share a collate identity (`TotalCharge` puts the declared charge in `task_id` for the same reason).

**Schema.** `molix.schema` gains **only** `ENERGY_RESIDUAL_KEY = ("graphs", "energy_residual")`. Reuse `E0_KEY = ("graphs", "e0")` from cache-layers-04-dtype — do not declare a second `E0_KEY`. `has_energy` continues to probe `ENERGY_KEY` only (raw totals). `execute` writes `frame[E0_KEY]` and `frame[ENERGY_RESIDUAL_KEY]`.

**Losses / `target_key`.** `energy_loss` / `energy_mse` already index `batch["graphs", target_key]`. This spec does **not** edit `molecular.py` or `MSELoss`. `TestEnergyResidual` asserts the residual column is present at `ENERGY_RESIDUAL_KEY` and that raw `graphs.energy` is unused for the residual identity (same tensor object as on entry, values not copied onto the residual). 07 passes `target_key="energy_residual"` at the composition site. `MSELoss` is a flat-mapping generic; it is not taught to walk `("graphs", "energy_residual")` here. There is no expression parser and no `"energy - e0"` target.

**Label isolation.** `Derivative` and potentials already write predictions to `ENERGY_KEY` / `FORCES_KEY`. Because `EnergyResidual` stores labels at `ENERGY_RESIDUAL_KEY`, an in-place energy write cannot clobber them. This spec does not change `Derivative`; it pins the column split and unit-tests that a write to `ENERGY_KEY` leaves `graphs.energy_residual` intact.

**One label residualizer.** After the chain, `EnergyResidual` is the only in-flight label residualizer. `class_i_fit.energy_force_terms` (file `scripts/mm_param_learning/class_i_fit.py`, the `E_DFT − e0` branch) is superseded by 07 — not left as a second owner. `AtomicReferenceEnergy` / `MacePotential.energy_core` (the `e0 +` readout sum written to `ENERGY_KEY`) stays the model-side add for total-energy inference. Coefficients still have one scientific home: the frozen table 03-refs checkpoints; this task only *applies* tensors it is given.

**Precision split with 04.** Residual always upcasts the subtract to float64 even if a caller handed float32 totals. Durable preservation of `graphs.energy` as float64 through collate / FrameStore / `batch_to` is cache-layers-04-dtype. `graphs.e0` is written float64 so 04’s `E0_KEY` skip keeps it.

### Reuse decision

- `reuse BatchTask` — `EnergyResidual` subclasses it; `execute` / `__call__` / `task_id` stay the extension point.
- `reuse PadMolecularBatch` — compose `atoms.mask`; do not reimplement padding or invent a second ghost convention.
- `reuse ENERGY_KEY` — read raw totals; never write them.
- `reuse E0_KEY` (cache-layers-04-dtype) — write `graphs.e0` at that constant; do not redefine it.
- `reuse energy_loss` / `MSELoss(target_key=)` — existing `target_key` contract; this spec does not edit those symbols. 07 passes `"energy_residual"`. Tests assert the column, not the loss modules.
- `reuse molix.F.batch_add` — per-graph sum of masked \(c[Z]\).
- `new — EnergyAggregation` — librarian tagged it `reuse`, but it lives in molpot; molix must not import molpot. `batch_add` is the molix scatter-sum that aggregation wraps.
- `reuse as_tensor_frame` — same view seam as `TotalCharge`.
- `reuse TensorFrame` — typed view, no storage.
- `reuse molix.units.units` — no conversion floats in this task.
- `pattern TotalCharge` — refuse plain dict, wrap, in-place write, return caller container, Google docstring with units, `task_id` names the column. **new EnergyResidual** — TotalCharge fills a declared scalar; EnergyResidual is a Z-lookup + masked scatter + float64 subtract. Do not generalize TotalCharge into a mode switch.
- `pattern Derivative` — predictions stay on `ENERGY_KEY` / `FORCES_KEY`; no Derivative edit.
- `AtomicReferenceEnergy` — **new — do not call it.** molix must not import molpot; the caller passes `c[Z]` and `s[Z]` as tensors (cache-layers-03-refs). ARE remains the model-side ADD.
- `AtomicReferenceEnergy.__init__`, `MACEEncoder.atomic_energies`, `Checkpoint`, `GlobalRescale` — out of scope (03-refs).
- `TokenBudgetBatchSampler`, `FrameStoreReader.atom_counts`, `FrameStoreWriter.seal`, `DataMixture._source_atom_counts`, `collate_molecules` / `batch_to` — out of scope (02 / 04).
- MD `NeighborList`, pipeline `NeighborList`, `CellBound.bind_cell` — out of scope (06).
- `OMol25Lmdb`, `BlobStore`, `PrepareJob`, `DownloadCache` — out of scope (01 / 02).
- `AtomicDress`, `AtomCountCache`, `_fit_e0_subsample`, `ForceAwareModel`, `LazyOMol25` — out of scope (07 deletes or retargets). This slug does not edit `class_i_fit.py`.
- `Cache`, scripts, `AtomicDress` as a writer — out of scope; not reused.

## Files to create or modify

- `src/molix/data/residual.py` (new)
- `src/molix/schema.py`
- `tests/test_molix/test_data/test_residual.py` (new)
- `tests/test_molix/test_schema.py`

## Tasks

- [x] Write failing unit tests for EnergyResidual (tests/test_molix/test_data/test_residual.py → TestEnergyResidual) (rework: type Residual → EnergyResidual; drop write_e0; construction ValueError; target_key column only — no loss-module tests)
- [x] Write failing unit tests for ENERGY_RESIDUAL_KEY (tests/test_molix/test_schema.py) (rework: do not add E0_KEY; 04 owns it)
- [x] Implement EnergyResidual in src/molix/data/residual.py (rework: unexported module; always write graphs.e0; no write_e0; construction raises unless 1-D same-length bool support)
- [x] Add ENERGY_RESIDUAL_KEY in src/molix/schema.py (rework: reuse 04’s E0_KEY; do not declare a second E0_KEY)
- [x] Add docstring per google style with units (eV) on EnergyResidual in src/molix/data/residual.py
- [x] Verify padded ghost-H float64 residual against the hard-coded oracle in tests/test_molix/test_data/test_residual.py
- [x] Run full check + test suite

## Testing strategy

Unit tests only, path-mirrored, one method per concern. This repo has no `regressions/` tier (deleted 2026-09-07; goldens live in the mirrored unit module). Unit green is `python -m pytest {path} -v` on that file alone. No DataLoader, no `Derivative`, no molpot, no `energy_loss` / `MSELoss` imports, no third-party energy code at test time. Tests import `from molix.data.residual import EnergyResidual`.

**`TestEnergyResidual` (`tests/test_molix/test_data/test_residual.py`)** — targets `EnergyResidual.__init__` / `EnergyResidual.execute` / `EnergyResidual.task_id` only. Unpadded batches from `collate_molecules` like `TestTotalCharge`. Ghost rows: `PadMolecularBatch(..., pad_atom_type=1).execute` or an equivalent `atoms.mask` with hydrogen placeholder `Z` — EnergyResidual must not inspect `Z` to drop ghosts.

Happy path:

- Keyword-only: `EnergyResidual(c, s)` is `TypeError`; `EnergyResidual(coefficients=c, support=s)` runs.
- Unpadded H2O: `graphs.e0` equals `-2071.0` (float64), and the float64 subtract `graphs.energy.to(float64) - graphs.e0` equals `-0.1234567890123` within `1e-12` eV of the Domain-basis oracle. `graphs.energy_residual.dtype` is `atoms.pos.dtype`; when `pos` is float64 the stored residual matches the oracle at `1e-12` eV; when `pos` is float32 the stored residual equals that float64 result `.to(float32)`.
- `graphs.e0` is always present after `execute` (no omit path).
- `graphs.energy` is the **same tensor object** as before `execute`.
- Stored coefficients are float64 clones: mutating the caller’s tensor does not change the next `execute`.
- `target_key`: `batch[ENERGY_RESIDUAL_KEY]` is populated; raw `graphs.energy` is unused for the residual identity (not equal to the residual tensor; not overwritten). No test constructs a key named `"energy - e0"`. No test imports `energy_loss` or `MSELoss`.

Edge cases:

- `__init__` raises `ValueError` when coefficients or support is not 1-D, when lengths differ, or when support is not `torch.bool` (uint8 `0/1` is not bool).
- After `PadMolecularBatch(..., pad_atom_type=1)`, ghost \(Z=1\) rows contribute 0 even though `c[1] != 0`; `e0` still `-2071.0`.
- Real atom with `support[Z] == False` raises `ValueError` naming that \(Z\); a ghost with the same \(Z\) does not.
- Real atom with `Z >= len(coefficients)` raises `ValueError`.
- A zero coefficient with `support[Z] == False` still raises on a real atom (zero is not support).
- Plain `dict` raises `TypeError`.
- A subsequent in-place write to `ENERGY_KEY` leaves `graphs.energy_residual` the same tensor / values (`FORCES_KEY` similarly untouched by EnergyResidual).
- `e0` is invariant under translating `atoms.pos` (Z-only table).
- Two different tables produce different `task_id`s.

Domain (hard-coded; comment: numpy float64, 2026-09-11, the Domain-basis H2O numbers plus this CH4 large-total case):

```text
Z = [6, 1, 1, 1, 1]
c[1] = -13.60569312299
c[6] = -1030.08
E_total = -1084.412345678901
e0      = -1084.50277249196
E_res   = 0.090426813059
```

On an fp32-`pos` batch, `graphs.e0` and `energy.to(float64) - e0` still match the float64 oracle to `1e-12` eV. A comment may note that `float32(E_total) - float32(e0)` does not.

**`tests/test_molix/test_schema.py`** — `ENERGY_RESIDUAL_KEY == ("graphs", "energy_residual")`. Do not re-assert `E0_KEY` here unless 04’s tests already own that constant; 05 must not define it.

**Regression example.** Hard-coded goldens in `tests/test_molix/test_data/test_residual.py` (H2O + padded ghost-H + CH4). No `regressions/` script.

## Out of scope

- Adding `EnergyResidual` to `molix.data.__all__` or naming it as a shipped subclass in `src/molix/data/task.py` (cache-layers-07-protocol).
- Retargeting or deleting `class_i_fit.energy_force_terms` (07; that subtract is superseded by 07, not by this slug’s Files).
- Editing `src/molix/core/losses/molecular.py`, `energy_loss`, `energy_mse`, or `MSELoss`.
- A `write_e0` flag; omitting `graphs.e0`.
- Declaring `E0_KEY` (cache-layers-04-dtype already owns it).
- Fitting, freezing, checkpointing, or loading \(c[Z]\) (cache-layers-03-refs).
- Collate / FrameStore / `batch_to` column-aware dtypes (cache-layers-04-dtype); EnergyResidual still subtracts in float64.
- `AtomicReferenceEnergy` edits; molix must not import molpot. ARE stays the model-side ADD to `ENERGY_KEY`.
- `Cache`, FrameStore persistence of `e0` / residual / masks / weights, dressed dataset copies.
- `NeighborList`, geometry binding, `Compiler` / CUDA-graph of EnergyResidual (collate-side).
- Inference restoration adapter (`E_total = E_res + e0` on predictions).
- `ForceOutlierWeight`, Huber energy loss, `200 kcal/mol/Å` inlier masks — do not add them here.
- `AtomicDress`, MolHub transforms, training scripts, OMol25 recipe defaults (cache-layers-02 / 07).
- Changing `energy_mse`’s default `target_key` from `"energy"`.
- `make_residual` or any factory alias.
- `regressions/` scripts (tier deleted; goldens are unit tests).
