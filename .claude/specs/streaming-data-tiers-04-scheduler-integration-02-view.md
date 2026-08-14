---
title: Multi-block dataset view over sealed READY blocks
status: approved
slug: streaming-data-tiers-04-scheduler-integration-02-view
repo: molnex
chain: streaming-data-tiers-04-scheduler-integration (01-tables → 02-view → 03-plan)
created: 2026-08-14
---

# Multi-block dataset view over sealed READY blocks

## Summary

A training epoch consumes many sealed blocks, but `molix.store` today can only read one at a time: `BlockReader` (`src/molix/store/block.py:1269`) is a per-directory reader and `SubsetDataset` (`src/molix/store/dataset.py:399`) wraps exactly one packed payload. This spec adds `BlockDataset`, a read-only `BaseDataset` view over an *ordered* sequence of sealed block directories whose global index is `(block, local frame)` in the given block order. It presents the concatenated metadata surface a batch sampler needs — `atom_counts`, merged `task_codes` + `task_names`, globally unique `group_codes` — and an order-preserving, cross-block `frame_payload` so the existing composition `collate_packed(PackedView(view.frame_payload(frames)), range(len(frames)))` builds a batch whose frames come from several blocks. It refuses, loudly and by named error type, the three ways a multi-block view can lie: mixing manifest splits (I7), a planned block that is missing or unsealed (I8), and blocks that are not concatenable (task dictionary or payload column set disagreeing). Sub-spec 01-tables' `BlockManifest` accessors and generalized sampler tables are consumed as **given contracts** and are not redesigned here; the plan that decides *which* blocks and frames to visit is 03's.

## Domain basis

No physics is introduced — the view moves no numbers, converts no units, and re-exposes block columns verbatim (`pos` Å, `energy` kcal mol⁻¹, `forces` kcal mol⁻¹ Å⁻¹, as recorded in each manifest's `provenance.units`). Its domain content is index algebra plus the molecule-level grouping discipline the block format already encodes.

Let blocks `b = 0 … B-1` be given in caller order, with `F_b = BlockReader.frame_count` (`block.py:1389-1392`) and `M_b = BlockReader.unit_count` (`block.py:1384-1387`). Define the two exclusive cumsums (`_exclusive_cumsum`, `block.py:481-492`):

```
frame_ptr[0] = 0,  frame_ptr[b+1] = frame_ptr[b] + F_b        # (B+1,) long
unit_ptr[0]  = 0,  unit_ptr[b+1]  = unit_ptr[b]  + M_b        # (B+1,) long
```

Then, for a global frame `g ∈ [0, frame_ptr[B])`:

```
b     = searchsorted(frame_ptr, g, right=True) - 1     # owning block
local = g - frame_ptr[b]                               # frame row inside block b
atom_counts[g]  = BlockReader(b).atom_counts[local]                       # (F,) long
group_codes[g]  = BlockReader(b).group_codes[local] + unit_ptr[b]         # (F,) long
task_codes[g]   = global_of[b][ BlockReader(b).task_codes[local] ]        # (F,) long
```

`group_codes` is injective across the whole view because block `b`'s per-block codes are exactly its unit rows `[0, M_b)` (one unit is one group, `block.py:1530-1537`), so the offset images partition `[0, unit_ptr[B])` with no overlap and no gap — the property "a group never straddles a batch" (`molix.batch.sampler.TaskMixtureBatchSampler._bucket_units`, `src/molix/batch/sampler.py:561-569`) therefore survives concatenation. `global_of[b]` is the per-block `local → global` task-code lookup built from the first-appearance merge of the per-block `manifest.task_names` dictionaries (`block.py:790-793`), so equal names map to one global code by construction.

Hand-computed reference fixture used by the tests and the regression (three sealed `split="train"` blocks):

| block | units (`N`, `n_frames`, task) | `F_b` | `M_b` | manifest `task_names` |
|---|---|---|---|---|
| A | `mol-a` (3, 2, equilibrium), `mol-b` (4, 3, relax) | 5 | 2 | `("equilibrium", "relax")` |
| B | `scan-c` (4, 5, torsion) | 5 | 1 | `("torsion",)` |
| C | `mol-d` (3, 1, relax), `scan-e` (4, 4, torsion) | 5 | 2 | `("relax", "torsion")` |

```
frame_ptr   = (0, 5, 10, 15)
unit_ptr    = (0, 2, 3, 5)
len(view)   = 15
atom_counts = (3,3,4,4,4, 4,4,4,4,4, 3,4,4,4,4)
group_codes = (0,0,1,1,1, 2,2,2,2,2, 3,4,4,4,4)
task_names  = ("equilibrium", "relax", "torsion")      # merged, first appearance
task_codes  = (0,0,1,1,1, 2,2,2,2,2, 1,2,2,2,2)        # C remaps local (0,1) -> global (1,2)
locate      : 0 -> (0,0), 4 -> (0,4), 5 -> (1,0), 9 -> (1,4), 10 -> (2,0), 14 -> (2,4)
```

Reference:
    Wang, Fass & Chodera. "End-to-end differentiable construction of molecular
    mechanics force fields." Chem. Sci. 13, 12016 (2022). DOI:10.1039/D2SC02739A
    — molecule-level grouping before configuration sampling; the invariant the
    globally-unique `group_codes` offset preserves across blocks.
    Fey & Lenssen. "Fast Graph Representation Learning with PyTorch Geometric."
    arXiv:1903.02428 — concatenated per-key tensors + slice index, the shape of
    the merged payload.
    Apache Arrow columnar format, variable-size list layout (offsets + values).
    https://arrow.apache.org/docs/format/Columnar.html

## Design

**New module `src/molix/store/block_view.py`**, importing stdlib + torch + `molix.store.block` / `molix.store.cache` / `molix.store.dataset` only. It must not import `tensordict` or any collate symbol — the same layering rule `block.py` states in its module docstring (`block.py:223-225`); batch construction stays the caller's composition.

**`BlockDataset(BaseDataset)`** — one type, primitive methods, OOP default. Named with the module's established `Block*` prefix (`BlockManifest` / `BlockReader` / `BlockWriter`), which also honours the vocabulary guard against a bare `Block` symbol (`block.py:9-12`), and with the `<Kind>Dataset` suffix every `BaseDataset` subclass already uses (`dataset.py:359`, `:377`, `:399`).

```
BlockDataset(roots: Sequence[str | os.PathLike[str]], *, max_open: int = 4)
  __len__() -> int                         # total frames across blocks
  __getitem__(idx) -> dict                 # flat pre-collate sample dict (one frame)
  locate(idx) -> tuple[int, int]           # global -> (block, local frame)
  frame_payload(frames) -> dict[str, Any]  # packed-cache-v1 payload, caller order
  reader(block) -> BlockReader             # bounded-K, LRU
  open_blocks -> tuple[int, ...]           # resident block rows, oldest first
  blocks -> tuple[Path, ...] ; split -> str ; max_atoms -> int
  atom_counts -> Tensor(F,) long ; task_codes -> Tensor(F,) long
  task_names -> tuple[str, ...] ; group_codes -> Tensor(F,) long
  edge_counts -> NoReturn                  # raises, never fabricates
```

*Construction (manifest-only, no column mapped except the per-block counts).* Each root is opened in the given order through `BlockManifest.read` (`block.py:698-727`). Empty `roots` and `max_open < 1` raise `ValueError`. Three named `ValueError` subclasses are defined so 03's scheduler can branch on the failure kind without string matching, while every existing `except ValueError` handler keeps working:

- **`BlockNotReadyError`** — I8. A root with no readable `manifest.json`, an unparsable/wrong-`schema_version` manifest, or `sealed == False` (`block.py:780-783`, `:1308-1312`) raises immediately, naming the block row, its path and its state (`missing manifest` / `unreadable manifest` / `unsealed — the writer has not called seal()`). The block is **never** skipped and **never** substituted by a neighbour: a hole in the plan stalls the view. The same error wraps a mid-epoch failure to (re)open an evicted block, so a block deleted under a running view stalls with the same signal rather than silently shortening the epoch.
- **`BlockSplitMismatchError`** — I7. Distinct `manifest.split` values (`block.py:785-788`) across `roots` raise, listing every root with its split and instructing the caller to construct one view per split. Mixing is never inferred, never silently allowed.
- **`BlockSchemaMismatchError`** — the two "not concatenable" cases: (a) a block's routed (non-`meta.`) column-name set differs from block 0's, checked at construction from the manifests so a cross-block batch cannot fail mid-epoch; (b) a `task` code observed in a block is `>= len(manifest.task_names)`, i.e. the code table and the name table disagree in shape. Both messages name the offending roots and the symmetric difference / the offending code. **(c) named rot, guarded here:** a block declaring the `internal` valence family (`block.py:285-292`, bucket `internals`) is refused, because neither `PackedCache.unpack_sample` (`_VALENCE_PTR_KEY`, `cache.py:402-412`) nor `collate_molecules` has an `internals` route — a view that accepted it would drop that family silently. Refusing is the honest behaviour; wiring `internals` through the collate is a collate-side change (Out of scope, named).

*Bounded-K reader pool.* `reader(block)` keeps at most `max_open` live `BlockReader`s in an `OrderedDict` LRU (hit → `move_to_end`, overflow → `popitem(last=False)`). Dropping a reader releases its `_mapped` storages (`block.py:1350-1365`) and unmaps the columns. Reopening is cheap: it parses `manifest.json` and re-maps only the small count columns (`n_atoms`, `n_frames`, `<family>_n`, `block.py:1315-1327`), and the file pages stay in the OS page cache, so a warm reopen re-reads no bytes from Lustre (warm sequential mmap measured at 1114 MB/s, `.claude/notes/notes.md` block-bench table). Correctness is independent of `max_open` — every tensor the view hands out is produced by advanced indexing inside `BlockReader.frame_payload` (`block.py:1631-1637`, `:1666-1687`), which copies, so an eviction can never invalidate a live batch or a cached table. `max_open` only trades address space / file descriptors against reopen churn; the default 4 covers a batch that touches a few blocks, and `open_blocks` exposes residency for 03's diagnostics.

*Concatenated sampler tables.* `atom_counts`, `task_codes` and `group_codes` are `cached_property` `(F,)` `torch.long` tensors built by walking blocks through the bounded pool and `torch.cat`-ing per-block vectors (copies, so nothing aliases a mapping). They are defined **explicitly**, not attribute-forwarded — the same reason `SubsetDataset` re-defines `atom_counts` / `edge_counts` instead of letting `__getattr__` return an un-remapped parent vector (`dataset.py:483-510`). All three are widened to `torch.long`: `MmapDataset.atom_counts` is already `long` (`dataset.py:314-327`), and the global group id is an offset sum, so one dtype across the three tables removes a cast at every sampler call site (`BlockReader` keeps its `int32` on-disk-facing dtypes unchanged). `task_names` is the merged dictionary; the decode a caller may still want stays the documented one-liner `[view.task_names[c] for c in view.task_codes.tolist()]` (`block.py:1525-1527`).

*Group-code encoding — offset, not `(block_idx, code)`.* The sampler buckets frames by the hashable scalar pair `(task_label[i], group_ids[i])` (`sampler.py:561-569`) and validates the tables as flat per-sample sequences of `len(dataset)` (`sampler.py:484-498`). A flat integer offset therefore keeps the table exactly the 1-D shape 01-tables generalized the samplers to accept; a `(block_idx, code)` pair would force either an `(F, 2)` table (not a table shape the sampler accepts) or a table of Python tuples (un-tensorable, slower to build, and no longer sortable/printable in an audit). Uniqueness is structural, per Domain basis.

*`edge_counts` refuses.* The view delegates to `self.reader(0).edge_counts`, which always raises with the block store's own message (`block.py:1413-1429`) — one copy of the rationale, and the message already points at `molix.batch.NeighborList` as the downstream builder. It must **not** return zeros. The contrast is deliberate: `scripts/mm_param_learning/espaloma_mm_protocol_train.py:492` sets `edge_counts = zeros_like(atom_counts)` because that dataset genuinely has no edges by construction, whereas a block *deliberately excludes* edge columns for a cutoff graph that will exist downstream — zeros there would hand a token-budget sampler a false budget. Downstream samplers over this view are therefore constructed with `max_edges=None`. This propagates correctly today: `TaskMixtureBatchSampler._frame_budgets` catches only `AttributeError` (`sampler.py:532-547`), so the block store's `ValueError` reaches the caller verbatim instead of being reworded into "use MmapDataset".

*`frame_payload(frames)` — order-preserving cross-block gather.* Same name, same contract and same return shape as `BlockReader.frame_payload` (`block.py:1564-1647`), one level up, so the caller composition is byte-identical to the single-block one. Empty / out-of-range `frames` raise `ValueError` mirroring `block.py:1600-1606`. Frames are grouped by owning block (first-appearance order) and each group is materialised by the owning reader; a single-block request returns that payload unchanged (zero merge cost). For a multi-block request the per-block payloads are concatenated in block-major order and re-ordered to caller order in one gather: with `perm` the caller-slot → block-major-row permutation, graph-level leaves become `cat(...)[perm]`, `atom_ptr` is the exclusive cumsum of `counts[perm]`, and atom-level rows are gathered with `bm_atom_ptr[perm][segment] + within` — the identical arithmetic `BlockReader.frame_payload` already uses (`block.py:1608-1617`), applied per valence family with its `<family>_ptr` (`block.py:1666-1670`). Topology atom indices stay molecule-local, so `collate_packed`'s own rebase is unchanged. The merge is a private method with one call site (inline-until-second-use), not a new public type.

*`__getitem__` reuses the packed unpacker.* `PackedCache.unpack_sample(payload, 0)` (`cache.py:271-277`, `_unpack_one` at `cache.py:534-581`) already consumes exactly this payload shape — `atom_ptr` + `atoms`/`graphs` buckets with dotted `targets.*` leaves, `bond_ptr` + `bonds{bond_index (2,T), bond_types}`, and `angle_ptr`/`proper_ptr`/`improper_ptr` bucket names matching `_VALENCE_PTR_KEY` (`cache.py:402-412`) leaf for leaf. The one adapter needed is `"scalars": {}`, because the block column registry has no Python-scalar space (`block.py:60-97`) while `_unpack_one` indexes `payload["scalars"]` unconditionally. So `__getitem__` is `unpack_sample({**self.frame_payload([idx]), "scalars": {}}, 0)` — no second route into the data, and the unit test pins it by collating `view[i]` through `collate_molecules` and comparing against the packed route.

**Reuse decision** (every candidate resolved):

- `PackedView` (`dataset.py:47-97`) — **reuse**, unchanged. `frame_payload` returns precisely what it wraps; no second view type is introduced and `collate_packed` is untouched.
- `SubsetDataset` (`dataset.py:399-546`) — **new**, borrowing only the local→global index-map *idea*. Verified: `SubsetDataset` wraps exactly one payload-backed dataset (`self._dataset`, `dataset.py:412-415`), forwards unknown attributes to it (`:416-435`) and composes a single `PackedView.remap` hop (`:448-460`); nothing in molnex concatenates packed datasets today (repo-grep verified — no `ConcatDataset`-shaped seam exists to generalize). Subclassing it would inherit a one-payload field the multi-block case cannot satisfy, so `BlockDataset` subclasses `BaseDataset` (`dataset.py:105-116`) directly and copies the *shape* of `SubsetDataset`'s explicit, gathered `cached_property` count tables (`dataset.py:483-510`).
- `BlockReader` accessors (`block.py:1269-1688`) — **reuse**: `__len__` / `frame_count` / `unit_count`, `atom_counts`, `unit_frames`, `task_names` / `task_codes` / `group_codes`, `frame_payload`, and the honest `edge_counts` raise. No accessor is re-derived from columns by hand.
- `BlockManifest` (`block.py:664-826`) plus 01-tables' accessors — **reuse as given**: readiness, split, task dictionary and the declared column set are read from the manifest without mapping a column. This spec adds no manifest field and no manifest accessor; where a count is not on the manifest (`frame_count`), the view reads it from `BlockReader` through the bounded pool.
- `PackedCache.unpack_sample` (`cache.py:271-277`) — **reuse** for `__getitem__` (see above).
- `_exclusive_cumsum` (`block.py:481-492`) — **reuse** by intra-package import; re-deriving cumsum semantics is exactly the pointer/payload-disagreement class `block.py`'s docstring exists to prevent.
- `TokenBudgetBatchSampler` / `TaskMixtureBatchSampler` (`sampler.py:142`, `:267`) — **reuse as-is**, with 01-tables' generalized table contract. No sampler file is modified by this spec.
- `BatchTask` (`src/molix/batch/task.py:12`) — **not used here**; named as the extension point where edge building would later attach (Out of scope).

## Files to create or modify

- src/molix/store/block_view.py (new)
- src/molix/store/__init__.py
- tests/test_molix/test_store/test_block_view.py (new)
- regressions/streaming-data-tiers-04-scheduler-integration-02-view.py (new)

## Tasks

- [ ] Write failing unit tests for the BlockDataset index map and the I7/I8 guards (tests/test_molix/test_store/test_block_view.py → TestBlockDataset): `len`/`locate`/`__getitem__` against the hand-computed fixture, missing / unsealed / deleted-mid-epoch → `BlockNotReadyError`, mixed splits → `BlockSplitMismatchError`, mismatched column set or `internal` family → `BlockSchemaMismatchError`, plus the module layering guard (no `tensordict` / collate import)
- [ ] Implement `BlockDataset` global index map, `BaseDataset` surface (`__len__`, `__getitem__` via `PackedCache.unpack_sample`, `locate`, `blocks`, `split`, `max_atoms`) and the three typed guards in src/molix/store/block_view.py (new)
- [ ] Write failing unit tests for the concatenated sampler tables, the bounded-K reader pool and cross-block `frame_payload` (tests/test_molix/test_store/test_block_view.py → TestBlockDataset): `atom_counts` / `task_codes` / `task_names` / `group_codes` literals, dictionary-merge remap and code/name shape mismatch, `edge_counts` refusal incl. propagation through a `max_edges` sampler construction, caller-order parity vs `collate_molecules`, `open_blocks` ≤ `max_open`, payload identity for `max_open=1` vs `8`
- [ ] Implement the concatenated `atom_counts` / `task_codes` + merged `task_names` / offset `group_codes` tables and the delegating `edge_counts` refusal in src/molix/store/block_view.py
- [ ] Implement the bounded-K LRU reader pool (`reader`, `open_blocks`) and the order-preserving cross-block `frame_payload` merge in src/molix/store/block_view.py
- [ ] Export `BlockDataset`, `BlockNotReadyError`, `BlockSchemaMismatchError`, `BlockSplitMismatchError` from src/molix/store/__init__.py (imports, `__all__`, docstring bullet) and add Google-style docstrings with tensor shapes and units to every new public symbol
- [ ] Add regression example regressions/streaming-data-tiers-04-scheduler-integration-02-view.py (public API only; hard-coded goldens, no third-party runtime)
- [ ] Run full check + test suite

## Testing strategy

Unit tests live only in `tests/test_molix/test_store/test_block_view.py`, mirroring `src/molix/store/block_view.py`, one class `TestBlockDataset` mirroring the one public type (plus module-level tests for the layering guard, which targets the module, not a type) — the same layout `tests/test_molix/test_store/test_block.py:1-13` uses. Fixtures are built in-module with the deterministic halves construction (every float a multiple of `0.5`, exactly representable in `float32`, so **no FP tolerance appears anywhere**) writing the three-block A/B/C fixture of § Domain basis through `BlockWriter` into `tmp_path`; each test targets a single method or property.

- **Happy path.** `len(view) == 15`; `locate` returns the six literals; `atom_counts` / `group_codes` / `task_codes` equal the hand-computed vectors and are `torch.long`; `task_names == ("equilibrium", "relax", "torsion")`; block C's local codes `(0, 1)` are remapped to global `(1, 2)` (a non-identity merge, so an identity bug fails); `group_codes` is injective and its set equals `range(5)`; `max_atoms == 4`; `split == "train"`.
- **Cross-block payload.** `frame_payload([12, 1, 7])` (blocks C/A/B, atom counts `4, 3, 4`) collated via `collate_packed(PackedView(...), range(3))` equals `collate_molecules([view[12], view[1], view[7]])` leaf-for-leaf (key sets, `batch_size`, dtype, `torch.equal`), and the reversed frame list produces the reversed batch — pinning order preservation, which equal atom counts would hide. `atoms.batch_size == (11,)`, `graphs.batch_size == (3,)`, `edges.batch_size == (0,)`. Single-block requests return the owning reader's payload unchanged.
- **Edge cases.** Empty `roots`, `max_open < 1`, empty `frames`, out-of-range `frames`, negative/out-of-range `idx`; a one-block view (merge path never entered); a batch of one frame; two blocks whose routed column sets differ; a block whose `task` column carries a code beyond its `task_names` length; a block declaring the `internal` family; an unsealed block; a root with no `manifest.json`; a root deleted after construction and then read.
- **Invariant tests.** I7: a view over one `train` and one `valid` block raises `BlockSplitMismatchError` whose message names both roots and both splits, and the two per-split views each construct fine. I8: the unsealed / missing cases raise `BlockNotReadyError` naming the block row, path and state, and `len(view)` is never computed from the surviving blocks (no partial view is constructible). Bounded-K: with `max_open=1` over three blocks, `len(view.open_blocks) <= 1` after every access during a full sweep, and the payload for a cross-block frame list is `torch.equal` to the same request under `max_open=8`.
- **Refusal of fabricated edges.** `view.edge_counts` raises `ValueError` whose message names `NeighborList`; constructing `TokenBudgetBatchSampler(view, max_edges=...)` propagates that same `ValueError` (not the `AttributeError`-derived "use MmapDataset" message), and constructing with `max_atoms` only succeeds. No test asserts a zero edge count anywhere.
- **Sampler integration (unit-level).** `TaskMixtureBatchSampler` over the view with `max_atoms=8`, `max_edges=None`, mixture `{"equilibrium": 0.0, "relax": 0.5, "torsion": 0.5}`, `seed=42`, using the view's concatenated tables in the form 01-tables generalized the sampler to accept: every yielded batch carries exactly one task, every group present in a batch is present whole (frames of a `group_code` never split across batches), and two identical constructions yield identical plans.
- **Layering guard.** An `ast`-level test asserts `src/molix/store/block_view.py` imports neither `tensordict` nor any `molix.batch` symbol.

**Regression example.** `regressions/streaming-data-tiers-04-scheduler-integration-02-view.py` — public API only (`from molix.store import BlockDataset, BlockWriter, PackedView, SamplingUnit` and `from molix.batch.collate import collate_molecules, collate_packed`, `from molix.batch.sampler import TaskMixtureBatchSampler`), no private attribute touched, run as `PYTHONPATH=src python regressions/streaming-data-tiers-04-scheduler-integration-02-view.py`. It writes the A/B/C fixture plus a `valid`-split block and an unsealed block into a `TemporaryDirectory`, then asserts, against **hard-coded literals** embedded in the script: (1) `frame_ptr`, `unit_ptr`, `len`, the six `locate` pairs, `atom_counts`, `group_codes`, merged `task_names` and `task_codes` exactly as tabulated in § Domain basis; (2) cross-block collate parity for frames `[12, 1, 7]` and its reverse against the `collate_molecules` oracle over `view[i]`, leaf-for-leaf with `torch.equal` and the hand-computed bucket sizes `atoms 11 / graphs 3 / edges 0`; (3) a `TaskMixtureBatchSampler` epoch over the view (`max_atoms=8`, `max_edges=None`, `seed=42`) that is single-task per batch, group-integral, deterministic across two constructions, and covers the hand-computed per-task frame totals `relax = 4`, `torsion = 9`; (4) the three refusals — mixed split, unsealed block, `edge_counts` — each caught by type with its message checked for the offending root / state / the `NeighborList` pointer; (5) bounded-K equivalence, `max_open=1` vs `8`, with `open_blocks` never exceeding the cap. **No third-party oracle is imported, subprocessed or consulted**, here or at authoring time: every golden is closed-form integer arithmetic over a deterministic fixture (`torch` and `tensordict` are the runtime container, not oracles), so the header records `capture: derived by hand from the fixture; no external tool` alongside `torch`, `device`, `commit` and `date`, mirroring `regressions/streaming-data-tiers-02-block-format.py:57-78`.

## Out of scope

- **Edge building.** Blocks store no neighbour graph by design (`block.py:154-163`); this view never builds one and never fabricates a count. The named future extension point is `molix.batch.BatchTask` (`src/molix/batch/task.py:12`), running post-collate inside the DataLoader's `collate_fn`.
- **Routing the `internal` valence family** through `PackedCache.unpack_sample` / `collate_molecules` (neither has an `internals` route, `cache.py:402-412`). This spec **refuses** such a block loudly rather than dropping the family; wiring it is a collate-side change and needs its own spec.
- **The block plan itself** — which blocks are resident, their order, prefetch and pre-warming — is `-03`'s conveyor/scheduler. This view consumes an ordered list of roots as given.
- **Deletion / eviction of block directories** on disk (MolHub's tier policy). The view only refuses to read what is not READY.
- **DDP / sharding.** No rank-aware slicing; a distributed plan is 03's, and the packed fast path is disabled under DDP today anyway.
- **`DataModule` wiring and `TargetSchema` discovery** for block-backed views. Deliberately not named `packed_view()`: `DataModule` detects the packed fast path by `callable(getattr(ds, "packed_view", None))` and then calls it with no arguments, so a same-named per-batch method would route to the fast path and crash at the first batch. Wiring a block-aware collate callable is 03's task.
- **Modifying `molix.batch.sampler`** — 01-tables owns the generalized table contract; this spec adds no sampler parameter and changes no sampler behaviour.
- **Consolidating the sealed-block test fixture builders** shared by `test_block.py` and `test_block_view.py` into `tests/test_molix/test_store/conftest.py`. Named, not silent: the two fixtures differ (two blocks vs three blocks + guard blocks), and the consolidation would edit `test_block.py`, which is outside this spec's file set — route via `/mol:refactor` once a third store test needs the builder.
