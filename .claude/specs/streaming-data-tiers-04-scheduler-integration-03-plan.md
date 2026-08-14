---
title: Block-aware plan behind `batch_sampler_source` (streaming-data-tiers -04, link 03)
status: approved
grilled: true
slug: streaming-data-tiers-04-scheduler-integration-03-plan
repo: molnex
chain: streaming-data-tiers-04-scheduler-integration (01-tables → 02-view → 03-plan)
created: 2026-08-14
---

# Block-aware plan behind `batch_sampler_source` (streaming-data-tiers -04, link 03)

## Summary

This is the last link of the `-04-scheduler-integration` chain and the one that
makes the strategy's "plan globally, consume rolling" claim executable: it adds a
`BatchSamplerSource` implementation that plans **once, globally, over every
block's metadata** (via `-02-view`'s multi-block view and `-01-tables`' generalized
`TaskMixtureBatchSampler`), then emits that same plan in **block-windowed
consumption order** so at most `K` blocks need to be resident at a time. The plan
is the authority (I8): its batch order is the consumed order forever, a block that
is not available stalls the consumer at its plan position instead of being skipped
or substituted, and every emitted batch is block-pure so the documented per-block
collate composition — `collate_packed(PackedView(reader.frame_payload(frames)), …)`
— applies unchanged. The plan exports its block visit order as an ordered tuple
plus a stdlib-JSON document that MolHub's conveyor parses into its own
`VisitOrder(order, cursor)` (neither repo imports the other), and reports
blocks-visited coverage (units / frames / batches per block) so a partial pass is
accounted for rather than assumed (I10). One binding gap is closed here:
`BlockReader.frame_payload` deliberately skips `meta.` routes, so the per-frame task
code never reaches a batch, while `TaskRouter` requires a single-task
`("graphs", "task")` label — the block collate path therefore **stamps** that label
from `reader.task_codes` and **asserts** the batch is single-task, with no revision
to the block format.

## Domain basis — sampling discipline (no new physics)

This link introduces no physical quantity and no equation: every physical leaf
(`pos` in Å, `energy` in kcal mol⁻¹, `forces` in kcal mol⁻¹ Å⁻¹) passes through
`BlockReader.frame_payload` (`src/molix/store/block.py:1564`) untouched, and all
quantities this spec computes are dimensionless counts (frames, units, batches,
blocks, bytes). The scientific content is *sampling discipline*, which streaming
must preserve exactly (strategy I7/I8):

- **Molecule-level grouping before configuration sampling** — Wang, Fass, Chodera,
  *End-to-End Differentiable Construction of Molecular Mechanics Force Fields*,
  Chem. Sci. **13**, 12016 (2022). DOI:10.1039/D2SC02739A —
  https://doi.org/10.1039/D2SC02739A. A sampling unit never straddles a block
  (I2, structural in `molix.block/1`), which is exactly why dividing a planned
  batch at a block boundary (§ Design 3) can never cut a unit.
- **Stage / mixture structure the sampler already encodes** — Zheng et al.,
  *ByteFF*, arXiv:2408.12817 (2024) — https://arxiv.org/abs/2408.12817.
- **Why a shuffled visit order over units (not i.i.d. draws) is an acceptable
  optimization regime** — Mishchenko, Khaled, Richtárik, *Random Reshuffling:
  Simple Analysis with Vast Improvements*, NeurIPS 2020, arXiv:2006.05988 —
  https://arxiv.org/abs/2006.05988. This is what licenses reordering the *windows*
  of an epoch's batches while keeping the batch contents and the per-epoch task
  apportionment identical.

The one honest cost of block-windowed consumption is named as risk **R2** in
§ Design 6; it is a statistical-locality cost, not a discipline violation.

## Design

### 1. Entities and ownership

New module `src/molix/batch/block_plan.py` owns four symbols; nothing else in the
tree grows a public symbol.

| Symbol | Kind | Responsibility |
|---|---|---|
| `BlockFrameView` | `runtime_checkable` `Protocol` | The exact `-02-view` surface this link consumes. Declared here so `block_plan.py` binds to a contract, not to a concrete class, and so planning can be unit-tested against a stub view without writing blocks to disk. |
| `BlockCoverage` | frozen dataclass | One block's accounting record: `block` (identity string), `units`, `frames`, `batches`. I10's report row. |
| `BlockPlan` | frozen dataclass | One epoch's block-ordered batch plan. Iterable + `Sized`, so it *is* the object handed to `DataLoader(batch_sampler=…)`. Pure value object: it holds ints and strings, never a view, a reader or a tensor. |
| `BlockPlanSource` | class | The `BatchSamplerSource` implementation. `batch_sampler(dataset, *, seed) -> BlockPlan`. |

`BlockFrameView` pins the members consumed here, reusing `BlockReader`'s vocabulary
verbatim (`src/molix/store/block.py:1404`, `:1414`, `:1522`, `:1531`) so `-02`'s
generalization of that surface satisfies it by construction:

```
__len__() -> int                         # global frame count F
block_ids -> tuple[str, ...]             # block identity strings (directory basenames), ordinal-indexed
max_open -> int                          # K — the view's reader-pool bound, reused as the plan window
split -> str                             # the single split of this view (purity, I7)
atom_counts -> Tensor (F,)               # per-frame atom cost, pointer arithmetic only
task_codes -> Tensor (F,)                # int task code, global dictionary
task_names                               # the global code -> name dictionary, passed through
group_codes -> Tensor (F,)               # int unit code, global
edge_counts                              # raises, by design — never called here
locate(frames) -> (Tensor (B,), Tensor (B,))   # (block ordinal, block-local frame row)
reader(block: int) -> BlockReader        # the sealed block's reader; residency is the view's concern
```

`-02` fixes the concrete class; the two specs were reconciled 2026-08-14 after a
three-way audit (vectorized `locate`, `block_ids` basenames, one `max_open` name —
no `resident_window` alias). Because `runtime_checkable` `isinstance` checks
member *presence* only, the contract-seam test also **calls** the vectorized
`locate([0])` and reads `block_ids` / `max_open` against the real `-02` view — a
behavioural probe, not just an attribute check — so a signature drift fails
loudly at test time rather than on the first production collate.

### 2. Planning is global; only the *order* is windowed

`BlockPlanSource.batch_sampler(view, *, seed)`:

1. Reject a non-`BlockFrameView` argument with an actionable `TypeError`
   (precedent: `EnergyTaskSamplerSource.batch_sampler`,
   `scripts/mm_param_learning/espaloma_mm_protocol_train.py:542-559`).
2. Build **one** `TaskMixtureBatchSampler` (`src/molix/batch/sampler.py:267`) over
   `indices=range(len(view))` with the view's `task_codes` / `task_names` /
   `group_codes` int-code tables (`-01`'s generalized table entry point) and the
   configured mixture, seeded with the per-epoch `seed` the `DataModule` supplies
   (`src/molix/batch/datamodule.py:264-268`). This is the plan authority: the
   per-task unit shuffle spans **all** blocks, so shuffling is plan-level and never
   within-block-only (I8).
3. Divide each planned batch into maximal order-preserving runs of frames sharing a
   block (`view.locate`). Because a unit never straddles a block (I2), a division
   can never cut a unit, never violate a token budget (a part is ≤ the whole), and
   never mix tasks (the planned batch was single-task). The number of planned
   batches that had to be divided is recorded as `boundary_splits`.
4. Order the blocks by **first appearance in the plan** (deterministic, derived from
   the plan itself — no second RNG), partition that order into consecutive windows
   of at most `K = view.max_open` blocks, and emit, window by window, the
   plan's batches restricted to that window in plan order. At `K ≥ n_blocks` the
   windowing is a no-op, so the emitted order equals the plan order whenever
   `boundary_splits == 0` (division still fires independently of `K`); `K = 1`
   makes it strict block-major.
5. Compute per-block `BlockCoverage` (units = distinct `group_codes` among that
   block's planned frames; frames; batches) and construct the `BlockPlan`.

The emitted plan is therefore a **permutation of the planned batches after boundary
division** — nothing is dropped, duplicated, reordered *within* a window, or
substituted. `BlockPlan.__post_init__` validates that invariant locally (coverage
frames sum to the planned frame total, coverage blocks equal the set of per-batch
blocks), so an inconsistent record cannot exist.

Budgets: only `max_atoms` is accepted. `max_edges` is refused eagerly at
`BlockPlanSource.__init__` with a message pointing at `BlockReader.edge_counts`
(`src/molix/store/block.py:1414`) and at building the graph downstream — the block
format stores no edges and this spec does not invent a count.

### 3. Visit-order export — the MolHub handoff

`BlockPlan.visit_order()` returns the ordered tuple of distinct block identity
strings (the order in which the conveyor must make blocks READY).
`BlockPlan.write_visit_order(path)` dumps one stdlib-`json` document:

```json
{"format": "molix.visit-order/1", "split": "train", "seed": 43, "window": 2,
 "order": ["<block id>", "..."],
 "coverage": [{"block": "<id>", "units": 4, "frames": 128, "batches": 6}]}
```

MolHub constructs its own `VisitOrder(order, cursor)` by parsing this file; the
`cursor` is MolHub's progress state and is never written by molnex. `order` is the
contract; `coverage` is advisory sizing/accounting for lease refcounting (I9) and
prefetch depth. **Neither repo imports the other** — the file is the whole
interface, per `tests/test_repository_boundaries.py:72`.

### 4. The block collate route in `DataModule`

`_resolve_collation` (`src/molix/batch/datamodule.py:424`) grows a **third** route,
probed *before* the packed route:

```
block-capable and not DDP   -> (_IndexDataset(len(view)), _BlockCollateFn(view, schema, batch_tasks))
packed-capable and not DDP  -> unchanged (:441-444)
otherwise                   -> unchanged per-sample _CollateFn (:445)
```

`_block_capable(dataset)` mirrors `_packed_capable`'s probe discipline
(`:517-534`): a cheap structural check against `BlockFrameView`, returning `False`
rather than raising for every other dataset. It is probed first because a block view
carries no single packed payload, so the per-sample `_CollateFn` fallback is
**unavailable** for blocks — a block view has no flat-dict `__getitem__` contract to
fall back to, and silently falling back would produce `KeyError`s instead of an
actionable message.

`_IndexDataset` (`:537`) is reused **unchanged**: it is already transparent to any
`batch_sampler`, and here the bare indices are global frame rows in `[0, F)`.

`_BlockCollateFn(view, schema, batch_tasks)` mirrors `_PackedCollateFn` (`:556`)
leaf for leaf in its post-collate contract, and does exactly this per call on
`indices: list[int]`:

1. `blocks, locals_ = view.locate(indices)`; assert a single block ordinal — a mixed
   batch means the plan was bypassed, and raises a `ValueError` naming the block ids
   and pointing at `BlockPlan` (no cross-block payload merge exists on this path,
   by design).
2. `reader = view.reader(int(blocks[0]))`. A block the view cannot supply propagates
   the view's error with the block identity named: the path **never** advances to
   another block's batches, never substitutes, never skips (I8). Whether the view
   waits for the conveyor is `-02`/the conveyor's contract, not this collate's.
3. `codes = reader.task_codes[locals_]` (`src/molix/store/block.py:1522`); assert
   `codes.unique().numel() == 1`, else a `ValueError` naming the block, the codes
   and the single-task requirement of `TaskRouter`
   (`src/molix/core/steps/multitask.py:113`, label at `("graphs", "task")`, `:137`).
4. `payload = reader.frame_payload(locals_.tolist())` and **stamp**
   `payload["graphs"]["targets.task"] = codes.to(torch.int64)`. The stamp rides the
   existing target route (`collate_packed._route_target`,
   `src/molix/batch/collate.py:491-503`) and therefore lands as a `(B,)` int64 leaf
   at `("graphs", "task")` — the same shape the per-sample precedent produces
   (`scripts/mm_param_learning/espaloma_mm_protocol_train.py:247-255`, `:272`). To
   make the route unambiguous, `__init__` refuses eagerly if `"task"` is listed in
   `schema.atom_level`.
5. `collate_packed(PackedView(payload), range(len(indices)), self.schema)`
   (`src/molix/batch/collate.py:421`, `src/molix/store/dataset.py:48`) — the exact
   composition `block.py:225` and `:1572` document; blocks carry no edges, so the
   empty-edge fallback (`collate.py:526-532`) is exercised as designed.
6. `batch_tasks` in order, then `batch_to(dtype=self.ftype)`
   (`src/molix/core/steps/__init__.py:24`) — which casts floating leaves only, so
   the int64 task label survives untouched.

`__getstate__` drops any lazily built per-block state, exactly as
`_PackedCollateFn.__getstate__` does (`:591-595`), so pickling a collate callable
never captures payload tensors.

**No new `DataModule` entry point.** The source enters through the existing
`batch_sampler_source=` kwarg (`:148`), gets `seed + epoch` per epoch (`:264-268`),
and stays hard-refused under DDP by the untouched guard at `:228`.

### 5. Shape discipline

Per strategy § 13 and `CLAUDE.md § Design preferences`: `BlockPlanSource(...)` and
`BlockPlan(...)` are plain constructors (no `make_*` / `create_*`); the plan is a
narrow value object, not an ambient streaming-context blob; there is no
`run_everything()` — composing view + source + `DataModule` + trainer stays the
caller's job (the regression is the in-repo demonstration). Each new method does one
named thing: `visit_order()` reports order, `write_visit_order()` persists it,
`block_of()` answers one lookup.

### 6. Named risk R2 — mixture fidelity is per epoch, not per window

`TaskMixtureBatchSampler` interleaves slots so mixture fidelity holds *per window of
consecutive batches* (`sampler.py:676-716`). Block-windowed consumption preserves the
per-epoch apportionment and every batch's contents exactly, but consecutive batches
now come from at most `K` blocks; if blocks are single-task, task interleaving
degrades toward per-block for `K = 1`. This is recorded, not waived: the levers are
(a) `K > 1`, which restores interleaving across the resident set at bounded
residency, and (b) task-mixed blocks on the producer side. `BlockPlan.window` and
`BlockPlan.coverage` make the realized situation inspectable, and
`batches_per_task()` / `frames_per_task()` (`sampler.py:774`, `:791`) still report
the epoch-level shares unchanged.

### Reuse decision

No `librarian_report` was supplied with this task, so these verdicts come from direct
inspection of the surfaces cited above.

- `TaskMixtureBatchSampler` (`src/molix/batch/sampler.py:267`) — **reuse**, verbatim,
  through `-01`'s generalized int-code tables. No parallel sampler, no reach into
  `_pack_tasks` / `_apportion` / `_interleave`.
- `TokenBudgetPacker` (`sampler.py:47`) — **reuse** transitively through the sampler.
  No block-side packer.
- `BatchSamplerSource` (`sampler.py:811`) + `DataModule(batch_sampler_source=)`
  (`datamodule.py:148`, seeded `:264-268`) — **reuse**. `BlockPlanSource` is a new
  *implementation* of this protocol, not a new seam.
- `_resolve_collation` (`datamodule.py:424`) and `_packed_capable` (`:517`) —
  **generalize**: one more route on the existing dispatcher and one probe helper
  built to the same discipline, rather than a second collation dispatcher.
- `_IndexDataset` (`datamodule.py:537`) — **reuse** unchanged.
- `_PackedCollateFn` (`datamodule.py:556`) — **pattern**, not reuse: `_BlockCollateFn`
  copies its post-collate contract (`batch_tasks` → `batch_to(ftype)`), its lazy-state
  and `__getstate__` discipline; it cannot be generalized into it because payload
  assembly differs structurally (per-batch `frame_payload` vs one cache payload).
- `collate_packed` (`collate.py:421`) + `PackedView` (`store/dataset.py:48`) +
  `BlockReader.frame_payload` (`store/block.py:1564`) — **reuse** as the documented
  adapter composition (`block.py:225`, `:1572`). No second collate.
- `BlockReader.task_codes` / `group_codes` / `atom_counts` / `edge_counts`
  (`block.py:1522`, `:1531`, `:1404`, `:1414`) — **reuse**, including the honest
  `edge_counts` raise: budgets are atoms-only here.
- `TaskRouter` (`core/steps/multitask.py:113`) — **reuse** as the downstream contract;
  untouched. The stamp exists to satisfy it.
- `EnergyTaskSamplerSource` (`scripts/mm_param_learning/espaloma_mm_protocol_train.py:523`)
  — **pattern** for the source shape (script-local, single-task); not importable
  library code, so nothing is lifted.
- MolHub `VisitOrder` — **new and out-of-repo**; molnex writes JSON and never imports
  molhub (`tests/test_repository_boundaries.py:72`).

## Files to create or modify

- `src/molix/batch/block_plan.py` (new) — `BlockFrameView`, `BlockCoverage`,
  `BlockPlan`, `BlockPlanSource`.
- `src/molix/batch/datamodule.py` — `_block_capable`, `_BlockCollateFn`, the third
  route in `_resolve_collation` (`:424`), and the `batch_sampler_source` /
  `_resolve_collation` docstring updates.
- `src/molix/batch/__init__.py` — export the four new symbols; `__all__` stays
  alphabetized (`.claude/notes/notes.md` § "[2026-08-09] `__all__` stays alphabetized").
- `tests/test_molix/test_batch/test_block_plan.py` (new).
- `tests/test_molix/test_batch/test_datamodule.py` — new `TestBlockCollation` class.
- `regressions/streaming-data-tiers-04-scheduler-integration-03-plan.py` (new).

## Tasks

- [ ] Write failing unit tests for `BlockPlan` / `BlockCoverage` (tests/test_molix/test_batch/test_block_plan.py → `TestBlockPlan`, `TestBlockCoverage`): iteration yields the stored `list[int]` batches, `__len__`, `visit_order()` first-appearance order, `block_of()`, `write_visit_order()` document keys parsed back with stdlib `json`, and `__post_init__` rejection of a coverage/batches mismatch
- [ ] Write failing unit tests for `BlockPlanSource` over a stub `BlockFrameView` (tests/test_molix/test_batch/test_block_plan.py → `TestBlockPlanSource`): plan equality with a direct `TaskMixtureBatchSampler` at `K >= n_blocks` and zero boundary splits, block-pure division of a straddling batch with `boundary_splits` counted, strict block-major at `K = 1`, windowed order at `1 < K < n_blocks` (windows of ≤ K distinct blocks, plan order within a window), emitted batches are a permutation of planned ones, coverage totals, eager `max_edges` refusal, non-view `TypeError`
- [ ] Implement `BlockFrameView`, `BlockCoverage` and `BlockPlan` in src/molix/batch/block_plan.py (frozen value object, `__iter__` / `__len__` / `visit_order` / `block_of` / `write_visit_order`, `molix.visit-order/1` document, `__post_init__` consistency validation)
- [ ] Implement `BlockPlanSource.batch_sampler` in src/molix/batch/block_plan.py (one global `TaskMixtureBatchSampler` over the view's int-code tables, block-boundary division, `K`-window ordering, coverage accounting)
- [ ] Write failing unit tests for the block collate route in tests/test_molix/test_batch/test_datamodule.py (`TestBlockCollation`, over two tiny `BlockWriter`-written blocks): `("graphs","task")` stamped int64 from `reader.task_codes`, mixed-task batch raises, mixed-block batch raises, an unavailable block's error propagating from `view.reader(b)` with the block identity named and no substitute batch produced (ac-007's second clause), `_resolve_collation` picks the block route before the packed route and leaves both existing routes untouched, DDP still refused, `batch_tasks` + `batch_to(ftype)` applied and the int label uncast, pickle round-trip carries no payload tensors, `"task"` in `atom_level` refused eagerly
- [ ] Implement `_block_capable` and `_BlockCollateFn` and extend `_resolve_collation` in src/molix/batch/datamodule.py (block route first, `_IndexDataset` reused, `__getstate__` mirroring `_PackedCollateFn:591`)
- [ ] Export `BlockCoverage` / `BlockFrameView` / `BlockPlan` / `BlockPlanSource` from src/molix/batch/__init__.py with an alphabetized `__all__` and add Google-style docstrings with tensor shapes and count units to every new symbol in src/molix/batch/block_plan.py and src/molix/batch/datamodule.py
- [ ] Add regression example regressions/streaming-data-tiers-04-scheduler-integration-03-plan.py (public API only; three `BlockWriter` blocks, plan + `DataModule` iteration offline, hard-coded goldens, no third-party runtime)
- [ ] Run full check + test suite

## Testing strategy

Unit tests only under `tests/`, mirroring source
(`src/molix/batch/block_plan.py` → `tests/test_molix/test_batch/test_block_plan.py`;
the `datamodule.py` extension → the existing
`tests/test_molix/test_batch/test_datamodule.py`), types mirrored
(`BlockPlan` → `TestBlockPlan`, `BlockPlanSource` → `TestBlockPlanSource`,
`BlockCoverage` → `TestBlockCoverage`), one function or method per test, seconds
each. No e2e under `tests/`.

- **Fixtures.** Pure planning tests run against a stub `BlockFrameView` holding
  literal `atom_counts` / `task_codes` / `group_codes` tensors and a `block_ids`
  string tuple — no disk, no fsync. Collate-route tests use real blocks written with `BlockWriter`
  and kept deliberately tiny (≤ 2 units per block, ≤ 4 frames per unit) because a
  unit commit costs ~18 fsyncs (`.claude/notes/notes.md` § "[2026-08-14] Block reads
  stay on mmap").
- **Happy path.** Plan equality against a directly constructed
  `TaskMixtureBatchSampler` over the same metadata and seed at `K >= n_blocks`
  (batch-for-batch list equality); a stamped batch whose `("graphs","task")` equals
  the literal task code of its block; coverage rows whose `frames` sum equals the
  planned frame total and whose `units` equal the distinct planned `group_codes`.
- **Edge cases.** A batch straddling two blocks (divided, counted, never cutting a
  unit); `K = 1` strict block-major; an unavailable block whose `view.reader(b)`
  raises — the error propagates with the block identity named, no substitution; `K` not dividing the block count; a single-block
  view (windowing is the identity); a mixed-task batch handed to `_BlockCollateFn`
  (raises); a mixed-block batch (raises); `max_edges` passed (raises, naming
  `edge_counts`); a non-view dataset (`TypeError`); `"task"` declared `atom_level`
  (raises); DDP + `batch_sampler_source` (existing refusal at `datamodule.py:228`
  still fires); pickle round-trip of `_BlockCollateFn`.
- **Domain validation (hard-coded).** `$META.science.required` is true, but this link
  declares no physics; the domain-side assertions are therefore the discipline
  invariants with literal expected values: unit integrity (every planned unit's
  frames land in one batch), split purity (`view.split` is the only split touched),
  task purity per batch, and exact per-block frame/unit counts.
- **Contract seam.** An `isinstance(view, BlockFrameView)` assertion against the real
  `-02` multi-block view, so a member rename in `-02` fails loudly here.
- **Regression example** — `regressions/streaming-data-tiers-04-scheduler-integration-03-plan.py`
  (repo root `regressions/`, never under `tests/`): writes three small single-task
  blocks with `BlockWriter` (distinct tasks; a train view over the two train blocks
  and a single hot-tier valid block for the val loader — strategy § 6), builds the
  `-02` view, plans with `BlockPlanSource` at a pinned seed, iterates a full
  `DataModule` epoch offline with `num_workers=0`, and asserts against **literals
  embedded in the script**: the emitted batch index lists, the `visit_order()` tuple,
  the coverage rows, the `("graphs","task")` code and dtype of every batch, per-batch
  `graphs.num_atoms` / `graphs.targets.energy` composition goldens, the exported JSON
  document re-parsed with stdlib `json`, and the strategy exit criterion — plan
  equality against a direct `TaskMixtureBatchSampler` over the same metadata. It
  imports `torch` / `tensordict` / `molix` only; no oracle is imported, subprocessed
  or consulted, at runtime or at authoring time, and the docstring carries the
  `capture / verify / commit / torch / device / date` block in the house style of
  `regressions/streaming-data-tiers-02-block-format.py`.

## Out of scope

- **DDP.** `batch_sampler_source` under DDP stays hard-refused at
  `src/molix/batch/datamodule.py:228` — untouched, and the block route inherits the
  refusal. **Cross-rank plan slicing is deferred to the named future spec
  `streaming-data-tiers-06-ddp-plan`** and must not be folded into `-05-multinode`.
- **Edge / neighbour-graph building.** Blocks store no edges by design
  (`src/molix/store/block.py:1414`); budgets here are atoms-only and the graph is
  built downstream by `molix.batch.NeighborList` or the MD bind path.
- **Eviction and deletion.** The visit order is *handed to* MolHub; molnex deletes
  nothing, holds no lease, and implements no refcount (I9, `-03-conveyor`).
- **Prefetch policy, backpressure, waiting.** Whether an unavailable block is waited
  on or errors is the view's / conveyor's contract; this link only guarantees that
  consumption never reorders around it, never skips it and never substitutes.
- **Block format revision.** No new column, no registry row, no `meta.` route change —
  the task label reaches the batch by stamping, not by re-storing.
- **Val-side multi-block planning.** Validation sets are hot-tier and permanently
  resident (strategy § 6), so the fixed-`batch_size` val route assumes a single-block
  val view; a multi-block val view raises the mixed-block `ValueError` rather than
  silently mixing blocks. Adding a val plan is a separate decision, not a silent
  fallback.
- **Sampling policy changes.** Mixtures, token budgets, anti-starvation, stage
  curricula and the `seed + epoch` convention are unchanged; this link alters the
  order in which planned batches are consumed, never what is planned.
- **Any training run.** The Iron law and
  `.claude/notes/production-training-checklist.md` remain fully binding; nothing here
  is calibration evidence.
