---
title: Column-aware float64 graph totals at Frame collate and batch_to; DataMixture count snapshots
status: code-complete
grilled: true
created: 2026-09-11
chain: cache-layers
depends: cache-layers-03-refs
supersedes: cache-layers-01-cache (collate/device-move precision + DataMixture count snapshot)
---

# Column-aware float64 graph totals at Frame collate and batch_to; DataMixture count snapshots

## Summary

Frame collate emits `graphs.energy` and `graphs.e0` as IEEE-754 binary64 (the molpy Frame's rest precision) even when pipeline `ftype` is float32 or mixed, while positions and other compute leaves follow `ftype`. Store collate does not narrow those graph totals below the mmap column's dtype and does not upcast a float32 store leaf — a late `.double()` is not a repair, and `_FrameStoreCollateFn` does not preserve molpy Frame bits. `batch_to` still moves every tensor to device; on a TensorDict it skips dtype conversion for `FLOAT64_LABEL_KEYS` via `named_apply(..., nested_keys=True)` matching the full paths `ENERGY_KEY` and `E0_KEY`, never the leaf name `"energy"` (that spelling is also `ATOMIC_ENERGY_KEY`). `DataMixture` snapshots both `atom_counts` and `pair_counts` into owned cloned CPU long tensors at construction. `TensorFrame` stays a zero-copy typed view. Energy MSE on the mixed-dtype batches this spec produces is unusable until `cache-layers-05`. OMol-facing tests use eV goldens; the collate seam itself does not pick a unit.

## Domain basis

OMol25 reports energy error in meV/atom on raw DFT totals of many keV. Subtracting a per-graph elemental reference

```text
e0[g]     = sum_i c[Z_i]                 # same energy unit as graphs.energy; real atoms only
E_res[g]  = E_raw[g] − e0[g]             # must be evaluated in float64
```

is a cancellation of two large numbers whose difference is the training target. IEEE-754 binary32 has a 23-bit mantissa. For a Kr-scale total `|E| = 74800 eV` (`2^16 ≤ |E| < 2^17`) the unit in the last place is

```text
ulp32(74800 eV) = 2^(16−23) = 2^(−7) = 0.0078125 eV = 7.8125 meV
```

One ulp is already several meV, so a 1 meV/atom residual cannot be represented after a float32 cast of `E_raw` or `e0`. Binary64 (`ulp ≈ 1.4e-11 eV` at this magnitude) is the label dtype. PyTorch CUDA/CPU autocast downcasts float32, not float64: keeping the column as float64 is what excludes it from autocast; casting float32 → float64 after the fact pads zeros.

Published OMol `omol_elem_refs` Kr is `|c[36]| ≈ 74932.45 eV` (isolated-atom scale). The collate golden below is a synthetic Kr-scale total chosen so `ulp32` is exact.

Hard-coded goldens used by the unit tests (eV; OMol-facing only):

- `ulp32(74800.0) = 0.0078125`
- `E_raw = -74800.123456789` (Python float / float64 bit pattern) must survive Frame `collate_molecules(..., dtype=torch.float32)` exactly on the energy leaf
- `float32(E_raw)` differs from `E_raw` by more than `0.001` eV (one meV)

References:

- Levine et al., “The Open Molecules 2025 (OMol25) Dataset, Evaluations, and Models,” 2025. https://arxiv.org/abs/2505.08762 (DOI `10.48550/arXiv.2505.08762`). Errors in meV / meV atom⁻¹; source totals in eV.
- IEEE Std 754-2019, binary32 / binary64.
- FAIR fairchem commit `41d36dc6bf8ced6348dc73e6aaf54d0527ebd8c9`: reference arithmetic is float64 independent of backbone precision; energy labels are `DoubleTensor` (`configs/allscaip/omol_4m.yml`, `src/fairchem/core/modules/normalization/element_references.py`). Comment on the OMol tensor: “do to the scale of the numbers this needs to be a double”. Offline pin only — tests must not import fairchem.
- Tran et al., “The Open Catalyst 2022 (OC22) Dataset and Challenges for Oxide Electrocatalysis,” *ACS Catal.* **13**, 3066–3084 (2023). https://doi.org/10.1021/acscatal.2c05426 https://arxiv.org/abs/2206.08917. Linear composition subtract; Z-only `e0` does not change forces.

Units: a Frame's energy unit is the Frame's (FrameStoreReader still documents kcal mol⁻¹ on some blocks). This layer does not convert and does not inherit Class-I kcal mol⁻¹. Name eV only on OMol-facing tests and comments (Levine / ulp goldens). `molix.units.units` remains the conversion owner if a caller ever converts.

## Design

The seam is the existing integer-skip in `_column` / `_meta_scalar` / `batch_to`: integers already refuse `dtype`. Extend that skip to two graph float leaves, identified by one schema set — not a `PrecisionPolicy` type, not an `energy_dtype=` knob, not a residual column.

**Schema (address only).** In `src/molix/schema.py`, next to `ENERGY_KEY`:

```python
E0_KEY: tuple[str, str] = ("graphs", "e0")
FLOAT64_LABEL_KEYS: frozenset[tuple[str, str]] = frozenset({ENERGY_KEY, E0_KEY})
```

`E0_KEY` is the NestedKey for an optional graph reference sum that `cache-layers-03-refs` dressing / `cache-layers-05` Residual may write. This spec does not produce `graphs.e0` and does not add `ENERGY_RESIDUAL_KEY`. `has_energy` still probes `ENERGY_KEY` only. `collate_molecules` and `batch_to` both import `FLOAT64_LABEL_KEYS`; neither owns a private `("graphs", "energy")` / `"e0"` pair.

**`collate_molecules(dtype=)` / `MoleculeCollator`.** `dtype` still means “compute floating leaves” (`atoms.pos`, forces, mass, `graphs.cell`, other meta floats). When stacking frame `meta` into `graphs.<name>`, if `("graphs", name) in FLOAT64_LABEL_KEYS` the leaf is built with `torch.float64` (`_meta_scalar(..., dtype=None)`), regardless of `dtype`. Match the NestedKey `("graphs", name)`, not the leaf `"energy"`: an atoms extra named `energy` is `ATOMIC_ENERGY_KEY` and follows `dtype`. A Python float is already a double; the Frame at rest is fp64; the collate must not call `.to(ftype)` on protected leaves. Integer meta (`task`) stays long. `MoleculeCollator` remains the picklable wrapper with the same `dtype` field. Docstrings: protected graph totals keep IEEE-754 binary64; units are the Frame's — do not pin eV on this generic seam.

**`batch_to`.** Device moves apply to every tensor, including protected keys. When `dtype` is set on a TensorDict, use `TensorDict.named_apply(..., nested_keys=True)` (tensordict ≥0.14; `apply`'s `fn` is tensors only and cannot see nested paths). The callback matches full NestedKeys `ENERGY_KEY` and `E0_KEY` from `FLOAT64_LABEL_KEYS` and calls `tensor.to(device=device)` only for those; every other floating leaf takes `dtype`. Never match the leaf name `"energy"` — that collides with `ATOMIC_ENERGY_KEY`. Bare tensors (no key) keep today's behaviour. Plain-`dict` recurse cannot see a NestedKey: document the dtype skip as **TensorDict-only** (post-collate batches are TensorDict). Do not invent a second apply helper and do not thread a private path prefix through the public signature. The device-only fast path (`dtype is None`) is unchanged. No `.double()` on a column that already lost bits.

**`DataModule._CollateFn` / `_FrameStoreCollateFn`.** They already compose `MoleculeCollator(dtype=ftype)` (Frame route) or `_batch_from_columns` (store route) then `batch_to(..., dtype=ftype)`. After the two generalizations they inherit the policy; they must not add a third cast or a third skip list. Docstrings that currently say every floating leaf is cast to `ftype` are wrong and are updated: Frame collate **emits** binary64 totals; store collate **does not narrow** and **does not upcast**. Store route: `molix.block/1` energy is still written float32 today (`FrameStoreWriter` materialises meta as `torch.float32` at `src/molix/data/frame_store.py`) — this spec does **not** edit the writer and does **not** upcast that leaf. If a future writer stores energy as `<f8>`, `batch_to` will keep it. Writer `<f8>` stays a later spec.

**Autocast.** No new guard type. Protected Frame-route leaves never enter the collate/move as float32, so autocast cannot downcast them. A unit test under `torch.autocast` asserts `graphs.energy` stays float64.

**`TensorFrame`.** Unchanged: `as_tensor_frame` / `from_tensordict` share storage. Mixed dtypes (`pos` float32, `energy` float64) are legal TensorDict leaves; the view does not own a dtype. `PadMolecularBatch` is unmodified and already pads with `t.dtype` / `t.new_zeros`; it does not touch `graphs.energy`.

**`DataMixture._source_atom_counts` / `_source_edge_counts`.** Construction of `DataMixture` is the planning boundary. For each source, read `source.atom_counts` and `source.pair_counts` (FrameDataset, SubsetDataset, FrameStoreDataset, `FrameStoreReader.atom_counts` — pointer arithmetic, no frame unpack). `torch.as_tensor(..., dtype=torch.long).detach().clone()` into an owned CPU tensor; `torch.cat` those parts into `self.atom_counts` / `self.pair_counts`. Those tensors are the immutable snapshot `TokenBudgetBatchSampler` / `TaskMixtureBatchSampler` already consume (`pair_counts` when `max_pairs` is set). They are not live mmap views: mutating the source counts or remapping the store must not change `mix.atom_counts` / `mix.pair_counts`. Missing `atom_counts` or missing `pair_counts` raises `ValueError` naming the source type; the function must not iterate `range(len(source))`, must not build a Python `list[int]` of all rows, and must not zeros-fill a missing vector. Length mismatch with `len(source)` raises. A raising `pair_counts` property (`FrameStoreReader` / `FrameStoreView` — blocks carry no edges) propagates; DataMixture does not catch it. `batch_sampler()` keeps using `self.atom_counts` / `self.pair_counts`; it does not re-read mmap during an epoch. `TokenBudgetBatchSampler` is not edited. A private helper used by both snapshot functions is allowed (second call site); it is not a public type.

**Loss halt (not implemented here).** After this change, Frame + `fp32` is label `float64` vs pred `ftype`, and FrameStore + `fp64` is label `float32` vs pred `float64`. `energy_loss` / `nn.MSELoss` / `view_as` require one dtype. That energy MSE is **unusable** until `cache-layers-05` Residual (float64 subtract, then square in `ftype`). This spec does not implement Residual, does not edit `src/molix/core/losses/molecular.py`, and does not add a mixed-dtype cast in the loss. Silence would be a training-loop break.

### Reuse decision

- reuse `ENERGY_KEY` — member of `FLOAT64_LABEL_KEYS`; raw totals stay this NestedKey.
- new `E0_KEY` / `FLOAT64_LABEL_KEYS` — schema addresses only, next to `ENERGY_KEY`; one skip set imported by collate and `batch_to`. Librarian noted `graphs.e0` had no key yet; this slug names it and does not write it. Not a residual column.
- reuse `PadMolecularBatch` — unmodified; preserves leaf dtypes; does not pad `graphs.energy`.
- reuse `TensorFrame` / `as_tensor_frame` — collate still returns the zero-copy view; no dtype methods on the view.
- reuse `TokenBudgetBatchSampler` — consumes `dataset.atom_counts` / `dataset.pair_counts` as-is.
- reuse `FrameStoreReader.atom_counts` — `_source_atom_counts` reads this property when the source exposes it; no frame scan.
- reuse `molix.units.units` — conversion owner if a caller converts; collate/batch_to docstrings do not pin eV.
- reuse `MSELoss` / `energy_loss` — unchanged; mixed-dtype energy MSE is halted until cache-layers-05 (see Out of scope). Do not generalize the loss here.
- generalize `collate_molecules(dtype=)` / `MoleculeCollator` — NestedKey skip from `FLOAT64_LABEL_KEYS`, same pattern as integer leaves.
- generalize `batch_to` — `named_apply(..., nested_keys=True)` skip on the dtype axis; device axis unchanged; skip is TensorDict-only.
- generalize `DataModule._CollateFn` / `_FrameStoreCollateFn` — inherit the two seams; fix the “every float → ftype” contract text; no extra cast.
- generalize `DataMixture._source_atom_counts` — owned clone at planning boundary; fail closed without `atom_counts`.
- generalize `DataMixture._source_edge_counts` — same snapshot as atoms (clone, length-check, `ValueError`); no zeros-fill.
- out of scope this slug (not reused, not reimplemented): `AtomicReferenceEnergy` / `__init__`, `MACEEncoder.atomic_energies`, `Checkpoint`, `EnergyAggregation`, `GlobalRescale`, `BatchTask` / `TotalCharge` / Residual, `FrameStoreWriter.seal`, `Derivative`, MD `NeighborList`, pipeline `NeighborList`, `CellBound.bind_cell`, `OMol25Lmdb` / `.atom_counts`, `BlobStore`, `PrepareJob`, `DownloadCache` (do not rename), `AtomicDress` (forbidden training path), script `AtomCountCache` / `_fit_e0_subsample` / `ForceAwareModel` / `LazyOMol25`. `PackedCache` stays gone. No `PrecisionPolicy`.

## Files to create or modify

- `src/molix/schema.py`
- `src/molix/data/collate.py`
- `src/molix/data/datamodule.py`
- `src/molix/data/mixture.py`
- `src/molix/core/steps/__init__.py`
- `tests/test_molix/test_schema.py`
- `tests/test_molix/test_data/test_collate.py`
- `tests/test_molix/test_data/test_dtype_propagation.py`
- `tests/test_molix/test_data/test_mixture.py`

## Tasks

- [x] Write failing unit tests for E0_KEY and FLOAT64_LABEL_KEYS (tests/test_molix/test_schema.py)
- [x] Write failing unit tests for collate_molecules column-aware dtype (tests/test_molix/test_data/test_collate.py; rework test_float_leaves_keep_the_frames_precision_unless_a_dtype_is_asked_for so dtype=float32 keeps graphs.energy float64; add e0, Kr ulp goldens, atoms.energy not skipped)
- [x] Write failing unit tests for batch_to and _CollateFn (tests/test_molix/test_data/test_dtype_propagation.py → TestBatchTo, TestPrecisionFlowsToBatch; named_apply NestedKey skip; ATOMIC_ENERGY_KEY still follows dtype; Frame emits float64; store does not upcast; autocast)
- [x] Write failing unit tests for DataMixture count snapshots (tests/test_molix/test_data/test_mixture.py → TestDataMixture; clone atom_counts and pair_counts; missing either raises without __getitem__; source mutation does not change the mix)
- [x] Add E0_KEY and FLOAT64_LABEL_KEYS in src/molix/schema.py
- [x] Generalize collate_molecules(dtype=) and MoleculeCollator in src/molix/data/collate.py (import FLOAT64_LABEL_KEYS; google docstring: binary64, Frame units)
- [x] Generalize batch_to in src/molix/core/steps/__init__.py (TensorDict.named_apply(..., nested_keys=True); skip is TensorDict-only; google docstring)
- [x] Generalize DataModule._CollateFn and _FrameStoreCollateFn in src/molix/data/datamodule.py (inherit collate + batch_to; correct the every-float-cast docstring)
- [x] Generalize DataMixture._source_atom_counts and _source_edge_counts in src/molix/data/mixture.py
- [x] Run full check + test suite

## Testing strategy

Unit-only, path-mirrored. `python -m pytest {path} -v` on each file is green independently. No e2e under `tests/`. This repo has no `regressions/` (deleted 2026-09-07); the public-API regression example is the Kr-scale Frame collate case in `tests/test_molix/test_data/test_collate.py` plus the Frame-route mixed-dtype case in `tests/test_molix/test_data/test_dtype_propagation.py`. Goldens are hard-coded literals.

- `tests/test_molix/test_schema.py`:
  - `E0_KEY == ("graphs", "e0")`
  - `FLOAT64_LABEL_KEYS == frozenset({ENERGY_KEY, E0_KEY})`
  - `ENERGY_KEY` still `("graphs", "energy")`; `ATOMIC_ENERGY_KEY` still `("atoms", "energy")`
  - `ENERGY_RESIDUAL_KEY` is not defined on `molix.schema`

- `tests/test_molix/test_data/test_collate.py` — `collate_molecules` / `MoleculeCollator`:
  - Happy: `dtype=torch.float32` → `atoms.pos` float32, `graphs.energy` float64, `graphs.task` long; wrap with `as_tensor_frame` / `TensorFrame.from_tensordict` shares `data_ptr` on both leaves.
  - Happy: meta `e0` present → `graphs.e0` float64 under the same `dtype=`.
  - Edge: `dtype=None` still fp64 throughout; `dtype=float64` energy stays float64; other meta floats (not in `FLOAT64_LABEL_KEYS`, e.g. `U0`) follow `dtype`.
  - Edge: an atoms extra named `energy` follows `dtype` (not protected; NestedKey is `ATOMIC_ENERGY_KEY`).
  - Domain (OMol-facing, eV): `E_raw = -74800.123456789` eV collated at float32 equals the float64 literal (exact bit match on the energy leaf); `ulp32(74800.0) = 0.0078125` eV hard-coded; `abs(float32(E_raw) - E_raw) > 0.001` eV. Comment: IEEE-754 binary32, Levine 2025 arXiv:2505.08762, FAIR DoubleTensor at `41d36dc6` (offline).
  - Rework the existing `test_float_leaves_keep_the_frames_precision_unless_a_dtype_is_asked_for` assertion that currently requires energy float32.

- `tests/test_molix/test_data/test_dtype_propagation.py` — `TestBatchTo` (owner: `batch_to`) and `TestPrecisionFlowsToBatch` (`_CollateFn` / `_FrameStoreCollateFn`):
  - Happy: nested TensorDict `batch_to(..., dtype=float32)` leaves `ENERGY_KEY` and `E0_KEY` float64, casts `atoms.pos`.
  - Happy: same batch also carrying `ATOMIC_ENERGY_KEY` — `atoms.energy` becomes float32; `graphs.energy` stays float64.
  - Happy: Frame-route `_CollateFn` under `set_precision("fp32")` / mixed modes: `pos` is the mode’s `ftype`, `graphs.energy` is float64.
  - Edge: device-only fast path still returns the same object; long tensors still skip dtype; protected keys still move device.
  - Edge: plain-dict recurse is **not** NestedKey-aware (TensorDict-only skip); a flat dict of floats still follows `dtype`.
  - Edge: FrameStore-route energy that arrives float32 is **not** upcast to float64 (late `.double()` is not the fix). `TestPrecisionFlowsToBatch.test_collate_emits_correct_dtype` currently asserts store energy follows `ftype` — rework: `pos`/`edge_*` follow `ftype`; energy is not narrowed below its incoming dtype and is not repaired to float64 (stays float32 for today's writer).
  - Autocast: after Frame-route collate, `torch.autocast` does not change `graphs.energy` dtype from float64.
  - Rework this module's docstring (“every floating-point tensor … is cast”).
  - Regression example (public collate/device-move contract, hard-coded goldens): `TestPrecisionFlowsToBatch` Frame-route case plus the Kr ulp test in `test_collate.py`. Expected: `pos.dtype is torch.float32`, `energy.dtype is torch.float64`, energy value `-74800.123456789` eV.

- `tests/test_molix/test_data/test_mixture.py` — `TestDataMixture`:
  - Happy: `mix.atom_counts` and `mix.pair_counts` equal the cloned source counts, dtype long.
  - Immutability: `mix.atom_counts.data_ptr() != source.atom_counts.data_ptr()` (same for `pair_counts`); writing `source.atom_counts[0] = 999` (and `pair_counts[0]`) leaves the mixture unchanged.
  - Edge: source without `atom_counts` raises `ValueError`; source without `pair_counts` raises `ValueError`; a `__getitem__` that `pytest.fail`s is never called.
  - Edge: `atom_counts` or `pair_counts` length ≠ `len(source)` raises.
  - `TokenBudgetBatchSampler` is not re-tested here; `mix.atom_counts` / `mix.pair_counts` remain the tensors it already reads.

## Out of scope

- Energy MSE on mixed-dtype batches. After this spec, Frame + fp32 labels are float64 against `ftype` predictions, and FrameStore + fp64 labels stay float32 against float64 predictions. `energy_loss` / `nn.MSELoss` / `view_as` require one dtype. That path is **unusable** until `cache-layers-05` Residual (float64 subtract, then square in `ftype`). This spec does not implement Residual, does not change `energy_loss`, and does not add a mixed-dtype MSE cast. Training on raw `graphs.energy` under mixed dtypes is halted, not papered over.
- Residual target column, `ENERGY_RESIDUAL_KEY`, losses reading it (`cache-layers-05`).
- `AtomicReferenceEnergy` edits (`cache-layers-03-refs` owns coefficients). This spec names `E0_KEY` only; it does not write `graphs.e0`.
- `molhub.cache.Cache`, MolHub writers, scripts.
- `molix.block/1` on-disk dtypes (`FrameStoreWriter` still materialises meta energy as float32 at `src/molix/data/frame_store.py`); no wire-format change, no `<f8>` writer, no upcast of that leaf.
- `NeighborList`, `PadMolecularBatch` behaviour, `TokenBudgetBatchSampler` implementation.
- molpot, molzoo, Class-I kcal defaults, production training GO.
- A `PrecisionPolicy` type, autocast context manager, or `batch_to(..., preserve=)` public knob.
- Catching `FrameStoreReader.pair_counts`'s deliberate `ValueError` (blocks carry no edges) so DataMixture of store views can zeros-fill.
