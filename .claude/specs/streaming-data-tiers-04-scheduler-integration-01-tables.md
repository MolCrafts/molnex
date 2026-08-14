---
title: Scheduler-integration tables — typed block-manifest accessors and a code-form sampler label table
status: approved
slug: streaming-data-tiers-04-scheduler-integration-01-tables
repo: molnex
chain: streaming-data-tiers-04-scheduler-integration (01-tables → 02-view → 03-plan)
created: 2026-08-14
---

# Scheduler-integration tables — typed block-manifest accessors and a code-form sampler label table

## Summary

The READY-tier block store already emits its scheduler metadata in bounded, per-frame integer form (`BlockReader.task_codes` / `group_codes` `(F,) int32` plus the `task_names` dictionary, `src/molix/store/block.py:1516-1537`), but the only consumer that can plan over it — `TaskMixtureBatchSampler` — still declares `task_labels: Sequence[str]` / `group_ids: Sequence[str | int]` and indexes them as Python sequences in parent space (`src/molix/batch/sampler.py:416-417, 484-511, 549-569, 774-807`). Every block consumer therefore has to decode `(F,)` integer columns into per-frame Python string lists before it can plan (exactly what `tests/test_molix/test_store/test_block.py:830-831, 859-860` do today), which is risk **R1** of the streaming-data-tiers strategy: the plan tables are per-frame Python objects and their footprint grows linearly with the frame count the streaming tiers exist to raise. This sub-spec removes that decode requirement by **generalizing the one sampler** to accept either form — never forking it — behind a single small table type, adds the two typed manifest accessors (`units`, `rows`) that the multi-block view (`02`) needs to size a plan without opening a column file, and lands the crossover benchmark that turns "Python lists will not scale" from an assertion into a measured, recorded number. No batching semantics, no plan output, and no on-disk format change: after this step the same fixture plans byte-identically through both representations.

## Domain basis

This sub-spec adds no physics; it preserves, unchanged, the domain semantics the sampler already cites, and adds one **measured** scaling decision.

**Preserved references** (carried verbatim in `TaskMixtureBatchSampler`'s docstring, `src/molix/batch/sampler.py:401-408`; the mixture/apportionment behaviour must remain bit-identical after the generalization):

- Zheng et al. "ByteFF: A Machine Learning Force Field for Drug-like Molecules", arXiv:2408.12817 (2024) — **stage structure only** (the Stage A/B/C mixtures used as fixture data). https://arxiv.org/abs/2408.12817
- Wang, Fass, Chodera. "End-to-End Differentiable Construction of Molecular Mechanics Force Fields", Chem. Sci. 13, 12016 (2022) — **grouping discipline only** (group is the atomic unit of sampling, as it is of splitting). DOI:10.1039/D2SC02739A, https://doi.org/10.1039/D2SC02739A

**Footprint arithmetic (units: bytes per frame, CPython 3.13/3.14, 64-bit).** This is the hand-derivation the bench must confirm or refute; it is *not* a measurement:

| representation | bytes/frame | at F = 1e6 | 1e7 | 1e8 |
|---|---|---|---|---|
| `task_labels` as `list[str]`/`tuple[str, ...]` over a bounded label set (pointers only; the ≤ 10 label objects are shared) | 8 | 8 MB | 80 MB | 800 MB |
| `group_ids` as `list[str]`, one distinct ≤ 8-char id per unit, ~10 frames/unit (8 B pointer/frame + ~56 B per distinct `str`) | ~13.6 | 13.6 MB | 136 MB | 1.36 GB |
| `(F,)` `int32` codes (task + group) + bounded `task_names` tuple | 8 | 8 MB | 80 MB | 800 MB |
| the emitted plan itself, `list[list[int]]` of frame indices (8 B pointer + 28 B `int` object, values > 256) | ~36 | 36 MB | 360 MB | 3.6 GB |

Two consequences the bench must measure rather than assume: (a) the *tables* are only ~2× apart in steady state, so the decisive quantities are **peak** RSS during construction (the string path materializes an intermediate factorization dict and an index list) and **plan wall time** (per-frame `dict` lookups vs one vectorized gather); (b) the **plan** is the larger term at every F, which is why unit-granularity planning is deferred to `03-plan` and why this spec's bench reports plan RSS separately from table RSS.

**Successor note (filled by task 7, binding on `02-view` / `03-plan`).** After `scripts/batch/bench_plan_tables.py` runs on the target filesystem/node, record here and in `.claude/notes/notes.md` (topic `sampler-plan-table-scale`): measured peak RSS and plan wall time for both representations at F ∈ {1e6, 1e7, 1e8}, the smallest F at which the list form costs ≥ 2× the code form in peak RSS (or "no crossover below F_max"), the host/torch version/date, and the **chosen representation** for the successors. Until that record exists, `02` and `03` must treat the table above as an estimate, not a decision.

## Design

### What changes, in one sentence per surface

1. **`BlockManifest` gains two typed accessors** (`units`, `rows`) over the document it already holds. No clone of MolHub's `BlockMetadata`, no digest recomputation.
2. **`FrameLabelTable` (new, in `src/molix/batch/sampler.py`)** owns "who is this frame's task and group" for one parent index space, in code form, constructible from either the legacy sequences or the block-native integer columns.
3. **`TaskMixtureBatchSampler` consumes only that table** — the representation dispatch happens once, inside the table's constructors; every downstream step (`_bucket_units`, `_pack_tasks`, `_interleave`, the audit accessors) sees exactly one shape.

### 1. `BlockManifest.units` / `BlockManifest.rows`

`BlockWriter._write_manifest` (`src/molix/store/block.py:1176-1190`) already stores `"units": self._unit_count` and `"rows": dict(self._rows)` — the per-row-space row counts refreshed by `_recompute_rows` (`:1192-1196`), keyed by the row spaces of the closed column registry (`molecule`, `molecule_atom`, `molecule_term:<family>`, `frame`, `frame_atom`, `bytes`; `_Route`, `:328-341`). Both fields are already covered by `_DIGEST_FIELDS` (`:325`), so exposing them is a read of digest-covered state:

- `units -> int` — committed unit count `M`.
- `rows -> Mapping[str, int]` — row count per row space, returned as a read-only mapping of `int` (the stored JSON values are typed on read, mirroring `schema_version`'s `int(...)` at `:778`).

Two properties, matching the existing accessor block (`:770-813`) in shape and docstring style; no new state, no `__slots__` change.

**Two honest caveats, both to be pinned by tests rather than prose:**

- *Authoritative only when sealed.* `_write_manifest` is called on schema freeze, on task registration and on `seal()` — **not** after every `append_unit` (`:1010-1029`). An unsealed manifest therefore reports the counts as of the last manifest write, i.e. `units` can be `0` while the writer has committed several units. Both docstrings name `sealed` as the authority; `02-view` must only read these from a sealed block.
- *`rows` does not always carry a frame count.* `rows["frame"]` exists only when the block declares a `frame`-space column (`energy` / `torsion_angle` / `hessian_diag`); a block whose only frame payload is `pos` (`frame_atom`) has no `"frame"` key. `F` stays a `BlockReader` fact (`frame_count`, `:1389-1392`, summing the `n_frames` column). The accessor returns the stored map as-is and does not synthesize keys.

Digest verification is untouched: `BlockReader.verify()` (`:1541-1560`) remains the only code path that hashes column bytes, and `BlockManifest._derive_block_digest` (`:815-826`) remains the only path that hashes the manifest core.

### 2. `FrameLabelTable`

One type, one responsibility: the per-frame `(task, group)` identity of a parent index space, plus the bounded dictionaries needed to render a task name (mixture keys, audit dict keys) and a group label (the oversize warning text).

```
FrameLabelTable(*, task_codes, group_codes, task_names, group_names=None)   # canonical, block-native
FrameLabelTable.from_labels(task_labels, group_ids)                         # alternate ctor, legacy sequences
len(table) -> int                    # F
table.task_names -> tuple[str, ...]  # the bounded dictionary
table.task_name(code) -> str
table.group_label(code) -> str       # original string when known, else str(code)
table.codes_at(indices) -> tuple[list[int], list[int]]   # one gather per table, Python ints out
```

- **Canonical constructor** takes the two `(F,)` integral CPU tensors verbatim (`BlockReader.task_codes` and `group_codes` already return materialized tensors — `index_select` and `.to(int32)` copies, not file mappings — so nothing mmap-backed is retained). It validates: 1-D, equal length, CPU, integral dtype, non-negative, and every task code `< len(task_names)`; `group_names` is optional and, when given, must cover `max(group_codes) + 1`.
- **`from_labels`** factorizes both sequences in **first-appearance order** into `int64` code tensors and keeps the two name tuples. First-appearance order (rather than sorted) is a single O(F) pass and is plan-irrelevant: the plan depends on the `(task, group)` *pair* identity and on `sorted(mixture)`, never on code values (`_bucket_units` `:561-569`, `_pack_tasks` `:600`).
- **`codes_at`** is the bucketing primitive: `codes[torch.as_tensor(indices)].tolist()` per table — one vectorized gather instead of `F` per-element tensor reads, which is the whole point of accepting the tensor form.
- No `make_*` / `build_*`; `from_labels` is an alternate constructor with distinct semantics, which the repo's style already permits (`BlockWriter.resume`, `PackedCache`-style classmethods).
- **Placement.** Public in `molix.batch.sampler` (added to that module's `__all__`, `src/molix/batch/sampler.py:39-44`), **not** re-exported from `molix.batch.__init__` in this step: `02-view` is the second consumer and the package façade grows there, per "inline until the second real use".

### 3. Generalized `TaskMixtureBatchSampler`

Signature (keyword-only tail unchanged in order and defaults; two widenings and one new optional argument):

```
task_labels: Sequence[str] | torch.Tensor      # (F,) integral task codes
group_ids:   Sequence[str | int] | torch.Tensor # (F,) integral group codes
task_names:  Sequence[str] | None = None        # required with integer task codes; rejected with string labels
```

`__init__` order becomes: `_validate_sizes` → `_validate_mixture` → **build the one table** → `_validate_tables(len(dataset), indices, table)` → `_compute_batches`. Downstream nothing branches:

- `_bucket_units` (`:549-569`) keys `unit_frames` by the `(task_code, group_code)` pair from `codes_at(indices)`, then maps each unit's codes to `(task_name, group_label)` **once per unit** when building `task_units` — so `_validate_tasks` (`:513-530`), the `sorted(self._mixture)` visit order (`:600`) and the oversize warning text `f"{task}/{units[position][0]}"` (`:611`) are unchanged, including the exact strings asserted by `test_oversize_unit_warns_exactly_once_naming_task_and_group` (`tests/test_molix/test_batch/test_sampler.py:1044-1060`).
- `_validate_tables` (`:484-511`) keeps its dataset-length and index-range checks and its message wording (parameter names `task_labels` / `group_ids`, both lengths, the `indices` range); the table's own constructor rejects a task/group length disagreement naming both arguments and both lengths, so the two existing mismatch tests (`:1395-1413`) stay green whichever check fires first.
- **`_interleave` records the task name of every slot it actually fills** (`:707-716`), and `_compute_batches` returns the plan alongside that parallel `tuple[str, ...]`. `batches_per_task` / `frames_per_task` (`:774-807`) then count from it instead of re-reading `self._task_labels[batch[0]]`. Two consequences: the audit dicts can no longer drift if a caller mutates the tensor it passed in, and **the sampler stops retaining any `(F,)`-length object** — its pickled state is the dataset reference, the budgets, the mixture, the plan and one short task-name tuple. The class docstring's state inventory (`:346-348`) is corrected accordingly; this is the R1 mitigation that matters for a DataModule that reconstructs a sampler per epoch.

**Invariants that must survive (structural, asserted by existing tests):** the single next-fit loop stays inside `TokenBudgetPacker` — `"current.append("` still appears exactly once in the module and the sampler source still shows no `batches.append(current)` / `totals =` markers (`tests/…/test_sampler.py:1107-1127`); no sample is unpacked during construction or iteration (`:1129-1145`); the block path must still map only count columns (`tests/test_molix/test_store/test_block.py:845-868`).

### Reuse decision

No `librarian_report` accompanied this annex; the candidates below were derived by reading the cited code and are resolved here so no successor re-litigates them.

- `BlockReader.task_codes` / `group_codes` / `task_names` (`block.py:1516-1537`) — **reuse**. They already are the code form; this spec exists so a consumer can hand them to the sampler unchanged.
- `BlockReader.verify()` (`block.py:1541`) — **reuse**, and it stays the *only* verifier: the new manifest accessors never hash bytes.
- `BlockManifest` (`block.py:664`) — **generalize**: two accessors added to the existing type over the document it already owns. Explicitly **not** a new metadata type and **not** a port of MolHub's `BlockMetadata`.
- `TokenBudgetPacker` (`sampler.py:47`) — **reuse** untouched; the packer binds cost vectors, not labels.
- `molix.core.seed.make_generator` (`sampler.py:731`) — **reuse**; determinism semantics unchanged.
- `MmapDataset.atom_counts` / `BlockReader.atom_counts` — **reuse**; budget costs stay pointer arithmetic.
- `molix.batch.group_split.group_split_indices` — **reuse by not touching**: splitting stays upstream, this sampler still consumes `indices` as given.
- `scripts/store/bench_block_read.py` — **pattern**: the new bench copies its shape (frozen result dataclasses, `_table` printer, one-line `_verdict`, `argparse` + `main() -> int`, stdlib+torch only, self-cleaning scratch).
- `regressions/universal-class-i-10-sampler.py` — **pattern**: the new regression copies its capture-provenance header and hand-derivation discipline.
- `FrameLabelTable` — **new**, because no in-tree type owns per-frame task/group identity: today it is two loose parallel sequences threaded through four sampler methods, which is precisely what makes the representation un-generalizable. Naming, keyword-only construction and `ValueError`-naming-key-plus-remedy follow `TaskMixtureBatchSampler` itself.

## Files to create or modify

- `src/molix/store/block.py`
- `src/molix/batch/sampler.py`
- `scripts/batch/bench_plan_tables.py` (new)
- `tests/test_molix/test_store/test_block.py`
- `tests/test_molix/test_batch/test_sampler.py`
- `regressions/streaming-data-tiers-04-scheduler-integration-01-tables.py` (new)
- `.claude/notes/notes.md`

## Tasks

- [ ] Write failing unit tests for the manifest row/unit accessors in `tests/test_molix/test_store/test_block.py` (extend `TestBlockManifest`: sealed values vs hand-computed literals, `rows` spaces present/absent, staleness before `seal()`, no sha256 recomputation)
- [ ] Implement `BlockManifest.units` and `BlockManifest.rows` in `src/molix/store/block.py` with Google docstrings naming `sealed` as the authority and the row-space keys
- [ ] Write failing unit tests for the label table and both sampler table forms in `tests/test_molix/test_batch/test_sampler.py` (new `TestFrameLabelTable` + `TestTaskMixtureTableForms`: code/label round-trip, `codes_at`, validation `ValueError`s, plan equality across representations, audit dicts immune to post-construction tensor mutation, no `(F,)` object retained on the instance)
- [ ] Implement `FrameLabelTable` in `src/molix/batch/sampler.py` (canonical tensor constructor + `from_labels` alternate constructor, `task_name` / `group_label` / `codes_at`, module `__all__` entry, Google docstrings with `(F,)` shapes and dtypes)
- [ ] Generalize `TaskMixtureBatchSampler` in `src/molix/batch/sampler.py` onto `FrameLabelTable` (widen `task_labels` / `group_ids`, add `task_names=`, bucket by code pairs, record per-slot task names in `_interleave`, drop the parent-space tables from instance state, refresh the class docstring inventory)
- [ ] Add the R1 crossover bench `scripts/batch/bench_plan_tables.py` (new; list-form vs code-form tables at F ∈ {1e6, 1e7, 1e8}, one subprocess per cell for a clean `VmHWM`, peak RSS + table/plan wall time, plan-equality guard, RSS budget guard, `_verdict` line)
- [ ] Record the measured crossover numbers and the chosen representation in `.claude/notes/notes.md` (new dated topic note `sampler-plan-table-scale`) and in this spec's Domain basis successor note
- [ ] Add regression example `regressions/streaming-data-tiers-04-scheduler-integration-01-tables.py` (public API only: `BlockWriter` → sealed block → `BlockReader`, sampler built twice from the two table forms, hard-coded golden plan / audit dicts, no third-party runtime)
- [ ] Run full check + test suite

## Testing strategy

**Unit tests — `tests/test_molix/test_store/test_block.py`, extending `TestBlockManifest` (`:404`):**

- Happy path: on the sealed fixture block, `manifest.units` equals the number of appended units and `manifest.rows` equals the JSON document's `rows` field entry-for-entry, with hand-computed literals for `molecule` (= `M`), `molecule_atom` (= `ΣN_m`) and `frame_atom` (= `Σ n_m·N_m`) taken from the existing two-unit fixture whose pointer literals are already hand-computed at `:729-752`.
- Edge: `rows` has no `"frame"` key for a block declaring no frame-space column; `rows` carries a `molecule_term:<family>` key exactly for the declared valence families.
- Edge: before `seal()`, `BlockManifest.read(root).units` is the value stored at the last manifest write and is strictly less than `writer.unit_count` after ≥ 2 same-task appends — the staleness contract, tested rather than merely documented.
- Guard: reading both accessors performs no digest work — monkeypatch `molix.store.block._sha256_file` to raise, then read `units` and `rows` successfully.

**Unit tests — `tests/test_molix/test_batch/test_sampler.py`, new classes after the GROUP E block (`:713`), reusing its fixture tables (`TASK_ATOM_COUNTS` / `TASK_LABELS` / `GROUP_IDS` / `STAGE_B` / `TASK_SEED`):**

- `TestFrameLabelTable`, one behaviour per test: `from_labels` factorizes in first-appearance order; `len` is `F`; `task_name` / `group_label` round-trip both forms (`group_label` falls back to `str(code)` without a name dictionary); `codes_at` returns the split's code pairs in the given order.
- `TestFrameLabelTable` validation, one `ValueError` per test, each message naming the offending argument and the remedy: unequal table lengths (names both arguments and both lengths); 2-D or floating or CUDA code tensor; negative code; task code ≥ `len(task_names)`; `task_names` omitted with integer codes; `task_names` passed together with string labels.
- `TestTaskMixtureTableForms`: the Stage B fixture planned from string sequences and from the equivalent code tensors + `task_names` produces equal `list(sampler)`, equal `batches_per_task()` and equal `frames_per_task()`; the oversize warning still names `torsion/tor_a` in the code form when a `group_names` dictionary is supplied and names `torsion/<code>` when it is not; mutating the caller's code tensor after construction changes neither the plan nor the audit dicts; the constructed instance exposes no attribute whose length equals `len(dataset)`.
- Back-compat, unmodified: `test_hard_coded_golden_plan` (`:973`), `test_hard_coded_golden_audit_dicts` (`:987`), the two length-mismatch tests (`:1395`, `:1405`), the packer-loop guards (`:1107`, `:1120`) and the zero-`__getitem__` guard (`:1129`) stay byte-identical and green.

No domain-validation tests apply (no physics is introduced); the mixture/apportionment domain behaviour is protected by the untouched Stage B/C bound tests (`:1151-1242`).

**Regression example — `regressions/streaming-data-tiers-04-scheduler-integration-01-tables.py`** (repo root `regressions/`, never under `tests/`): writes a small block into a `TemporaryDirectory` with `BlockWriter` (a handful of units across three tasks, hand-tabulated atom/frame counts in the module docstring), seals it, opens a `BlockReader`, then builds `TaskMixtureBatchSampler` **twice** over it — once with `task_labels=[reader.task_names[c] for c in reader.task_codes.tolist()]` / `group_ids=[int(g) for g in reader.group_codes.tolist()]` (the legacy decode, exactly as `tests/test_molix/test_store/test_block.py:830-831` does today) and once with `task_labels=reader.task_codes, group_ids=reader.group_codes, task_names=reader.task_names` — and asserts both plans equal one **hard-coded golden** plan plus hard-coded `batches_per_task` / `frames_per_task` dicts, and that `manifest.units` / `manifest.rows` match hard-coded literals. Goldens are captured from this repo's own deterministic plan (comment records commit, torch version, device, date); no third-party oracle is imported, subprocessed or consulted, at capture time or at run time.

**Bench — `scripts/batch/bench_plan_tables.py`** (not a test, never collected by pytest): for each F and each representation it re-invokes itself in a subprocess so `/proc/self/status:VmHWM` is that cell's own peak, builds a synthetic count-only dataset stand-in plus the label tables, constructs the sampler, and reports table-build seconds, plan-build seconds, peak RSS and the resulting batch count; it hashes the two plans and fails loudly if the representations disagree (the correctness guard that makes the timings comparable, mirroring `bench_block_read.py`'s checksum), and skips a cell whose projected footprint exceeds `--rss-budget-gb`, printing the projection instead of dying.

## Out of scope

- The multi-block view over several sealed blocks (`02-view`) — this spec adds only the manifest accessors that view will size itself from.
- Plan/loader wiring, unit-granularity plans, and any change to what the sampler yields (still `list[int]` of parent-space frame indices) — `03-plan`, informed by this spec's recorded bench numbers.
- DDP / rank-sharded planning; edge budgets over blocks (`BlockReader.edge_counts` raises by design, `block.py:1413-1429`) — both unchanged.
- Any on-disk format change: no new column, no registry row, no `schema_version` bump, no `BlockWriter` behaviour change, no digest re-derivation.
- Re-exporting `FrameLabelTable` from `molix.batch.__init__` — deferred to `02-view`, its second consumer.
- Changing `scripts/mm_param_learning/espaloma_mm_protocol_train.py` (its `EnergyTaskSamplerSource` at `:523-568` must keep working verbatim) or MolHub-side block production.
- Alternatives considered and rejected: a second block-native sampler class (rejected — forks the plan algorithm and the golden contract); a `BlockMetadata`-style clone of MolHub's manifest type (rejected — duplicates a digest-covered document); making the sampler always decode codes to strings internally (rejected — keeps R1 intact).
