---
title: Reusable token packer + DataModule batch-sampler injection seam
status: approved
created: 2026-08-14
grilled: true
---

# Reusable token packer + DataModule batch-sampler injection seam

## Summary

The greedy token-packing loop that turns a shuffled index order into budget-respecting batches is currently trapped inside `TokenBudgetBatchSampler._compute_batches` (`src/molix/data/sampler.py:110-147`), and `DataModule.train_dataloader` hard-codes that one sampler as the only possible `batch_sampler` (`src/molix/data/datamodule.py:234`). The upcoming hierarchical task sampler needs the same packing *within a task draw* — the second real use, so extraction is now permitted under CLAUDE.md "inline until the second real use". This spec does two behavior-preserving things and nothing else: (a) it lifts the packing loop into a small `TokenBudgetPacker` type owned by `sampler.py`, leaving `TokenBudgetBatchSampler` observably identical (same seeded permutation, same batch plan per seed, same eager `ValueError`s, same one-shot `UserWarning`, same picklable instance state); and (b) it opens the batch-sampler branch of `DataModule.train_dataloader` to an injected per-epoch batch-sampler provider, keeping the DDP guard and the `seed + epoch` reseed convention intact. No new sampling policy is introduced here — policy is sibling spec `universal-class-i-10-sampler`. Existing `DataModule` callers are untouched: omit the new keyword and every current code path is byte-for-byte what it is today.

## Domain basis

Not applicable — this spec contains no physical model, no equations, and no units (all quantities are dimensionless integer atom/edge counts), so there is no DOI/arXiv basis and no `scientific` acceptance criterion. The binding basis is algorithmic instead, and is pinned by hard-coded goldens:

- The packing rule is **next-fit** bin packing (exactly one open bin; a closed bin is never reopened), not first-fit over all open bins. The existing docstrings at `sampler.py:6` / `:29` / `:111` say "greedy first-fit", which is imprecise — with a single open bin the two policies differ (counts `[7, 6, 5]`, budget `12`: next-fit gives `[[0], [1, 2]]`, first-fit gives `[[0, 2], [1]]`). This spec keeps the **behavior** and corrects the **wording** to "next-fit (single open bin)" in the new and touched docstrings.
- Determinism basis: the plan is a pure function of `(count vectors, budgets, visit order)`; the order is a pure function of `(n, seed)` via `torch.randperm(n, generator=make_generator(seed))` (`src/molix/core/seed.py:58`).
- Conservation basis: packing is a partition — concatenating the emitted batches reproduces the input order as a multiset, with no sample dropped, including samples that exceed a budget on their own (kept as singletons).

## Design

**Reuse decision.** No `librarian_report` was supplied with this task; I ran a direct scan of `molix.data` and `molix.core` instead and resolve every candidate found:

- `TokenBudgetBatchSampler._compute_batches` (`sampler.py:110-147`) — **generalize** into `TokenBudgetPacker` (this spec's whole purpose). No parallel reimplementation: the sampler becomes a caller of the generalized unit.
- `TokenBudgetBatchSampler._budget_counts` (`sampler.py:93-108`) — **reuse** unchanged. It is dataset-coupled (it owns the "packed-cache-backed dataset required" `ValueError` naming `MmapDataset` / `CachedDataset` / `SubsetDataset`) and stays on the sampler; its existing `list[tuple[int, Tensor]]` return shape becomes the packer's constructor argument, so nothing is reshaped.
- `molix.core.seed.make_generator` (`seed.py:58`) — **reuse** verbatim; the permutation stays in the sampler, never in the packer.
- `torch.utils.data.BatchSampler` — **new** (`TokenBudgetPacker`): the stdlib sampler batches by fixed count with no notion of per-sample size budgets, so it cannot serve the packing concern.
- `torch.utils.data.Sampler` / a pre-built batch-sampler *instance* as the injected object — **new** (`BatchSamplerSource` protocol): an instance cannot receive the per-epoch seed that `train_dataloader` derives as `self.seed + self._epoch` (`datamodule.py:238`), which is the property that makes epoch *k*'s composition re-derivable on resume. The seam therefore injects a per-epoch *provider*, not a sampler instance.
- `DataModuleProtocol` (`datamodule.py:31`) — **reuse** unchanged; the seam adds no lifecycle method.
- `_IndexDataset` / `_PackedCollateFn` / `_resolve_collation` (`datamodule.py:351-372`) — **reuse** untouched; both batch-sampler paths keep routing through `_resolve_collation`, so the packed fast path applies to injected samplers for free (the loader dataset may be an `_IndexDataset`; the batch sampler yields the same index lists either way).

**(a) `TokenBudgetPacker` (`src/molix/data/sampler.py`).** A small type owning one concern — turning a visit order into budget-respecting index batches — with two primitives and no dataset, RNG, or warning coupling:

```python
class TokenBudgetPacker:
    def __init__(self, budgets: Sequence[tuple[int, torch.Tensor]]) -> None: ...
    def oversize_mask(self) -> torch.Tensor:        # bool (n_samples,)
    def pack(self, order: Sequence[int]) -> list[list[int]]: ...
```

- `budgets` is a sequence of `(budget, counts)` pairs — exactly what `_budget_counts()` already returns — where `counts` is a `(n_samples,)` integer vector. Empty `budgets` raises `ValueError` naming the requirement (at least one `(budget, counts)` pair), matching the eager-validation convention at `sampler.py:75-107`. Each count vector is materialised once at construction as `[int(v) for v in counts.tolist()]` — provably identical integers to the current per-index `int(counts[idx].item())`, no float rounding introduced, no plan change.
- `pack(order)` reproduces `sampler.py:121-146` line for line: for each index, compute its per-budget sizes; if the open bin is non-empty and adding the index would exceed **any** budget, close the bin; append the index; if the index exceeds any budget on its own, close the bin immediately (true singleton — an oversize index arriving into a non-empty bin closes that bin first, so it is never merged); flush a non-empty bin at the end. Empty `order` returns `[]`.
- `oversize_mask()` is the elementwise OR over budgets of `counts > budget`, over the **full** count vectors (documented: a caller packing a subset must index the mask by that subset). It carries no state and emits no warning — deciding whether an oversize sample deserves a warning belongs to the caller, which is what keeps the packer reusable by the sibling task sampler.
- The packer never imports or touches `BaseDataset`, `warnings`, or `make_generator`.

**(b) `TokenBudgetBatchSampler` delegation (`src/molix/data/sampler.py`).** `_compute_batches` keeps its signature and becomes: resolve `pairs = self._budget_counts()`; build a **transient** `TokenBudgetPacker(pairs)`; if `packer.oversize_mask().any()` emit the existing `UserWarning` with its message text unchanged (`"TokenBudgetBatchSampler: at least one sample exceeds the per-batch budget on its own; such samples are kept as singleton batches (never dropped). Raise the budget to avoid singletons."`) and `stacklevel=3` from the same `_compute_batches` frame, so the warning still points at the user's construction site and still fires exactly once per instance (`_compute_batches` is called once, from `__init__`); then `return packer.pack(perm)` with `perm` computed exactly as today. The packer is **not** stored on the instance: `vars(sampler)` stays `{_dataset, _max_atoms, _max_edges, _seed, _batches}`, so the pickled payload keeps its current shape and never materialises mmap'd count tensors. Public surface (`__init__` signature, `__iter__`, `__len__`, both `ValueError` classes and messages) is unchanged.

**(c) Injection seam (`src/molix/data/sampler.py`, `src/molix/data/datamodule.py`).** A one-method, `runtime_checkable` protocol declared next to the samplers it describes:

```python
@runtime_checkable
class BatchSamplerSource(Protocol):
    def batch_sampler(self, dataset: BaseDataset, *, seed: int) -> Iterable[list[int]]: ...
```

The returned object is handed straight to `DataLoader(batch_sampler=...)`, so it must also be `Sized` (documented in the protocol docstring; a plain `list[list[int]]` qualifies). `DataModule.__init__` gains one keyword-only parameter `batch_sampler_source: BatchSamplerSource | None = None`, defaulting to `None` = today's behavior. Eager validation in `__init__`, in the style of `sampler.py:75-107`: a non-conforming object (no `batch_sampler` method) raises `ValueError` naming the protocol and its signature; passing `batch_sampler_source` together with `max_atoms_per_batch` / `max_edges_per_batch` raises `ValueError` because they are two competing batch policies, telling the caller to pick one.

`train_dataloader` keeps its current shape; the hard-coded construction at `:234` moves behind one private method:

```python
def _epoch_batch_sampler(self) -> Iterable[list[int]] | None
```

which returns the injected `source.batch_sampler(self.train_dataset, seed=self.seed + self._epoch)` when a source is set, else a `TokenBudgetBatchSampler(...)` seeded `self.seed + self._epoch` when a budget is set, else `None`. Both non-`None` cases first pass the DDP guard, which is factored into `_reject_ddp_batch_sampler(knob: str)` (second real use) and raises the existing `ValueError` text with the offending knob named — `"max_atoms_per_batch / max_edges_per_batch"` for the budget path (substance and both kwarg names preserved from `:224-230`) and `"batch_sampler_source"` for the injected path; rationale sentence ("dynamic batching is incompatible with DistributedSampler's fixed per-rank partitioning… DDP support belongs to a future spec") unchanged. When `_epoch_batch_sampler()` returns non-`None`, `train_dataloader` takes the existing `batch_sampler=` branch verbatim — fresh loader per epoch, `batch_size` / `shuffle` / `sampler` / `drop_last` / `generator` still omitted, `_resolve_collation` still applied, the cross-epoch loader cache at `:258-261` still bypassed (its `_shuffle_generator` reseed path is only for the fixed-`batch_size` route and is untouched). `on_epoch_start` is unchanged: recording `self._epoch` is already the whole per-epoch contract the seam needs.

Both new symbols are re-exported from `src/molix/data/__init__.py`, inserted alphabetically inside the existing `# Batch sampling` group (`BatchSamplerSource`, `TokenBudgetBatchSampler`, `TokenBudgetPacker`), so `regressions/` and user code import from `molix.data`.

## Files to create or modify

- src/molix/data/sampler.py
- src/molix/data/datamodule.py
- src/molix/data/__init__.py
- tests/test_molix/test_data/test_sampler.py
- tests/test_molix/test_data/test_datamodule.py
- regressions/universal-class-i-09-packer.py (new)

## Tasks

- [ ] Write characterization test pinning the pre-refactor batch plans for TokenBudgetBatchSampler (tests/test_molix/test_data/test_sampler.py → TestSamplerGoldenPlan): 16-sample varied cache, three configs (max_atoms=16; max_atoms=16+max_edges=20; max_atoms=8 under pytest.warns) with hard-coded permutation and plan literals captured from the current tree, capture command and torch version recorded in a comment — must pass on the pre-refactor tree
- [ ] Write failing unit tests for TokenBudgetPacker (tests/test_molix/test_data/test_sampler.py → TestTokenBudgetPacker): next-fit closure pinned by counts [7,6,5]/budget 12 → [[0],[1,2]], oversize singleton [3,20,4]/budget 10 → [[0],[1],[2]] with zero warnings emitted, dual-budget closure driven by the edge budget, empty order → [], empty budgets → ValueError, oversize_mask hard-coded values, construction from bare tensors with no dataset
- [ ] Write failing unit tests for the DataModule injection seam (tests/test_molix/test_data/test_datamodule.py → TestBatchSamplerInjection): injected plan is the loader's batch_sampler with loader.batch_size None, collated batches carry exactly the injected index groups, source receives seed == dm.seed + epoch after on_epoch_start, DDP guard raises for batch_sampler_source, eager ValueError for source+budget combination and for a non-conforming object
- [ ] Generalize the `_compute_batches` packing loop into `TokenBudgetPacker` in src/molix/data/sampler.py (pack + oversize_mask primitives, budgets validation, counts materialised once via int(...), no dataset/RNG/warnings coupling)
- [ ] Delegate `TokenBudgetBatchSampler._compute_batches` to a transient `TokenBudgetPacker` in src/molix/data/sampler.py (unchanged randperm/seed, unchanged warning text and stacklevel=3, packer not stored on the instance)
- [ ] Implement `BatchSamplerSource` protocol in src/molix/data/sampler.py plus the `batch_sampler_source` seam in src/molix/data/datamodule.py (keyword-only default None, eager ValueErrors, `_epoch_batch_sampler`, shared `_reject_ddp_batch_sampler`) and export both new symbols from src/molix/data/__init__.py
- [ ] Add Google-style docstrings per doc.style for TokenBudgetPacker, BatchSamplerSource, `_epoch_batch_sampler`, `_reject_ddp_batch_sampler` and the new DataModule kwarg with `(n_samples,)` shape annotations and an explicit dimensionless-counts note, and correct "greedy first-fit" to "next-fit (single open bin)" in the touched sampler.py docstrings
- [ ] Add regression example regressions/universal-class-i-09-packer.py (public API only; hard-coded goldens, no third-party runtime)
- [ ] Verify behavior preservation: the three golden plans reproduce post-refactor, `sorted(vars(sampler))` is unchanged, and every pre-existing test in tests/test_molix/test_data/test_sampler.py (GROUP A/B/C) and test_datamodule.py passes unmodified and unweakened
- [ ] Run full check + test suite

## Testing strategy

Unit tests only under `tests/`, on the mirror paths for the two modules touched: `src/molix/data/sampler.py` → `tests/test_molix/test_data/test_sampler.py`, `src/molix/data/datamodule.py` → `tests/test_molix/test_data/test_datamodule.py`. Type mirror: `TokenBudgetPacker` → `TestTokenBudgetPacker`. Each test targets a single method. No e2e under `tests/`.

- **Characterization (the byte-for-byte gate).** `TestSamplerGoldenPlan` hard-codes, for a 16-sample varied-count cache built with the file's existing `_write_varied_cache(tmp_path, tag, n=16)` helper: the permutation literal for `make_generator(seed)` and the resulting plan literal for `max_atoms=16`, for `max_atoms=16, max_edges=20`, and for `max_atoms=8` (oversize, under `pytest.warns(UserWarning)`). Each config asserts twice — the sampler's plan equals the golden plan, and `TokenBudgetPacker(pairs).pack(golden_perm)` equals the same golden plan. The second assertion isolates the failure mode: if a future torch changes `randperm`, only the permutation assertion breaks and the packing logic is exonerated. Literals are captured from the pre-refactor tree; the capture one-liner, date, and torch version go in a comment above them. Unlike the rest of this spec's tests, this one is green before the refactor by construction — that is its purpose.
- **Packer happy path.** Budget compliance and exact-once coverage of the given order for single and dual budgets; batch boundaries checked against hand-computed sums.
- **Packer edge cases.** Next-fit closure pinned by `[7, 6, 5]` / budget 12 → `[[0], [1, 2]]` (a first-fit-over-open-bins implementation would return `[[0, 2], [1]]` and fail); oversize singleton `[3, 20, 4]` / budget 10 → `[[0], [1], [2]]` asserted under `warnings.simplefilter("error")` to prove the packer itself is silent; an oversize index arriving into a non-empty bin closes that bin first; dual budget where the edge budget closes the batch before the atom budget; empty `order` → `[]`; `budgets=[]` → `ValueError`; construction from bare `torch.tensor` count vectors with no dataset in sight (structural proof of decoupling).
- **Sampler surface preservation.** `sorted(vars(sampler))` equals the hard-coded field list `["_batches", "_dataset", "_max_atoms", "_max_edges", "_seed"]`; the oversize warning message text is asserted verbatim and counted exactly once per instance; existing `TestSamplerPickle`, `TestSamplerNoUnpack`, `TestSamplerValidation`, `TestSamplerDeterminism`, `TestSamplerCoverage`, `TestSamplerOversize` and `TestSamplerBudgetCompliance` classes stay unmodified and green.
- **Seam behavior.** A module-level picklable stub source returning a fixed `list[list[int]]` plan and recording the `seed` it was handed: the loader's `batch_sampler` is that object, `loader.batch_size is None`, iterating the loader yields exactly the injected groups (verified through the `targets.U0` identity values, so the packed collate fast path is proven to compose with injection), and `on_epoch_start(k)` then `train_dataloader()` shows `seed == dm.seed + k`.
- **Seam guards.** `monkeypatch` of `molix.data.datamodule._is_distributed` → `True` with an injected source raises `ValueError` matching `(?i)ddp|distributed`; the same monkeypatch with `max_atoms_per_batch` keeps raising with both kwarg names in the message (existing `TestDataModuleDDP` unmodified); `batch_sampler_source` plus a budget, and an object without a `batch_sampler` method, each raise `ValueError` at `__init__`.
- **Default-path regression.** Existing `TestDataModuleBudgetOptIn`, `TestDataModuleBackwardCompat` and `TestDataModuleEpochRederivation` pass unmodified: no keyword → `TokenBudgetBatchSampler` still drives the budget path, and the non-budget path still shows `RandomSampler` + `BatchSampler` + a seeded generator + `drop_last is False`.
- **Regression example.** `regressions/universal-class-i-09-packer.py`, public API only (`from molix.data import DataModule, TokenBudgetPacker`), RNG-free and hard-coded: part A packs a hard-coded count table under a hard-coded explicit order and asserts the hard-coded expected batches (next-fit closure + one oversize singleton); part B builds a tiny `PackedCache` in a `tempfile.TemporaryDirectory`, injects a module-level stub source into `DataModule(num_workers=0)`, calls `on_epoch_start(2)`, and asserts the stub saw `seed == 44` and that the collated batches carry exactly the injected index groups. No third-party software, no live oracle; header records the capture command and date like `regressions/universal-class-i-04-bundle.py`.

## Out of scope

- **Any new sampling policy** — hierarchical / per-task draws, task weighting, bucketing, curriculum. That is sibling spec `universal-class-i-10-sampler`; this spec ships the seam and the reusable packer only, and deliberately adds no in-tree `BatchSamplerSource` implementation.
- **A `TokenBudgetSource` adapter** expressing the budget path as a `BatchSamplerSource`. The budget branch stays a plain `elif` so the default path is provably unchanged; unify only if a caller needs budget packing *and* an injected policy at once.
- **Performance work.** The refactor is output-identical, not faster by intent; the per-epoch loader rebuild on the batch-sampler path (persistent workers re-spawn) is pre-existing behavior and stays as is.
- **DDP support for dynamic or injected batch samplers** — still guarded off, still a future spec.
- **Val loader changes.** The val path keeps its fixed `batch_size`; `on_epoch_start` and `DataModuleProtocol` are unchanged.
- **Docs.** `docs/api/molix.md` and `docs/molix/user-guide/data-modules.md` currently document neither `TokenBudgetBatchSampler` nor the budget kwargs, so there is no doc surface to keep in sync here; the contract lives in the docstrings this spec writes.
- **Observed debt, named and routed (not fixed here):** (1) the `molix.data` export row in `.claude/notes/architecture.md:102` goes stale once the two new symbols are exported — refresh via `/mol:note` after landing; (2) `src/molix/data/__init__.py`'s `__all__` is grouped by comment rather than globally alphabetized, contradicting the `__all__ stays alphabetized` note (`.claude/notes/notes.md:243`) — a whole-file re-sort is a separate cosmetic diff, route to `/mol:refactor`; this spec only inserts alphabetically within the existing `# Batch sampling` group.
