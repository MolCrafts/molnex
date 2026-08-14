---
title: Hierarchical task-balanced batch sampler (universal-class-i-10-sampler)
status: approved
created: 2026-08-14
grilled: true
---

# Hierarchical task-balanced batch sampler

## Summary

Training a universal Class-I force field on a MolHub collection that mixes
equilibrium snapshots, torsion scans, near-equilibrium relaxations and
electrostatic probes cannot use a flat frame pool: the natural frame counts are
wildly unbalanced, and a torsion scan only carries information about the
torsion profile when its frames are compared *within one batch*. This spec adds
`TaskMixtureBatchSampler` to `src/molix/data/sampler.py`, beside
`TokenBudgetBatchSampler`. It draws one physical **task** per batch from a
caller-supplied mixture (Stage A/B/C mixtures are plain constructor data — no
`Stage` enum, no schedule object), then draws whole **sampling units** inside
that task (one molecule's frames, or one *complete* torsion scan — never a
partial scan), then packs those units under a caller-supplied atom/edge token
budget by delegating, once per task, to the `TokenBudgetPacker` extracted by
`universal-class-i-09-packer`. The whole batch plan is built eagerly in
`__init__` from `seed` alone, so `len()`/iteration are free, the plan is
picklable, `seed + epoch` reproduces epoch *k* on resume, and two read-only
accessors report the realized per-task batch and frame counts as calibration
evidence E2 input (the tolerance comparison stays with the caller — this spec
ships no verdict façade).

## Domain basis

**There is no published precedent for a task → group → configuration
hierarchical batch sampler.** This is stated plainly rather than papered over
with a loose citation: no reference implementation exists to validate against,
so correctness here is *structural* (mixture fidelity, unit integrity,
determinism) and the *scientific* payoff is validated downstream by the sibling
`universal-class-i-13-grad-audit` spec, which measures whether per-task
gradient contributions match the configured mixture.

Two published results are cited for the pieces this sampler borrows, and for
nothing more:

- **Stage structure only** — ByteFF trains a Class-I-style MM parameter
  predictor with a three-stage curriculum over increasingly diverse PES
  regions. Zheng et al., *ByteFF: A Machine Learning Force Field for Drug-like
  Molecules*, arXiv:2408.12817 (2024). The three stage mixtures below are the
  default constructor data derived from that structure; the *mechanism* (per
  batch mixture draw, unit integrity, token packing) is not from ByteFF.
- **Grouping discipline only** — espaloma splits at the molecule level via
  isomeric SMILES so that conformers of one molecule never straddle a split.
  Wang, Fass, Chodera, *End-to-End Differentiable Construction of Molecular
  Mechanics Force Fields*, Chem. Sci. 13, 12016 (2022),
  DOI:10.1039/D2SC02739A. This sampler inherits that discipline one level down:
  a group id is the atomic unit of *sampling*, exactly as it is the atomic unit
  of *splitting* in `molix.data.group_split.group_split_indices`.

Default stage mixtures (constructor data, verified to sum to 1.0):

| Stage | equilibrium | torsion | relax | electrostatic |
|---|---|---|---|---|
| A | 0.70 | — | 0.30 | — |
| B | 0.35 | 0.45 | 0.20 | — |
| C | 0.25 | 0.30 | 0.35 | 0.10 |

**Units.** No physical quantity crosses this API. Task labels and group ids are
opaque strings/ints; budgets are dimensionless atom/edge *counts*; the plan is
a list of dataset indices. Å coordinates, energy units and charges stay inside
the batch `TensorDict` and are never read by the sampler.

## Design

### Dependency contract inherited from 09-packer

`universal-class-i-09-packer` leaves, in `src/molix/data/sampler.py`, the
budget-carrying `TokenBudgetPacker`:

```python
class TokenBudgetPacker:
    def __init__(self, budgets: Sequence[tuple[int, torch.Tensor]]) -> None: ...
    def oversize_mask(self) -> torch.Tensor:        # bool (n_elements,)
    def pack(self, order: Sequence[int]) -> list[list[int]]: ...
```

Budgets **and** the per-element cost vectors bind at *construction*;
`pack(order)` packs **positions** of those vectors in the given visit order
using **next-fit (single open bin)** token packing, keeping an over-budget
position as a singleton (never dropped), and is **silent** — reporting oversize
elements is `oversize_mask()`'s job, warning about them is the caller's.

Two consequences bind this spec's shape:

1. Because the cost vectors bind at construction, a single pre-built packer
   instance **cannot be injected** into this sampler: the cost elements here
   are *units*, whose aggregate costs differ per task. The sampler therefore
   takes the **budgets** themselves (`max_atoms` / `max_edges`, mirroring
   `TokenBudgetBatchSampler` at `sampler.py:26`) and constructs **one transient
   `TokenBudgetPacker` per task** over that task's unit-aggregate cost vectors.
2. Because `budgets` is already an arbitrary sequence of `(budget, counts)`
   pairs, **no change to 09 is required** — aggregate unit costs are just
   another cost vector. This spec **calls** the packer and **never
   re-implements, copies, or inlines a packing loop**; adding a second next-fit
   loop to this module is a spec violation.

A sampling unit is an aggregate of frames, so its cost is the sum of the
pointer-derived `dataset.atom_counts` / `dataset.edge_counts` (already present
on `_CacheBacked` / `SubsetDataset`, `dataset.py:314` / `:329` / `:480` /
`:496`) over the unit's member frames. `pack()` returns groups of *unit
positions*; the sampler expands each group into the concatenated member-frame
indices of those units, which is what makes unit integrity **structural**
rather than merely checked.

### Public surface

```python
class TaskMixtureBatchSampler:
    def __init__(
        self,
        dataset: BaseDataset,
        *,
        indices: Sequence[int],                 # split index list, e.g. group_split_indices()[0]
        task_labels: Sequence[str],             # per-sample, length == len(dataset)
        group_ids: Sequence[str | int],         # per-sample, length == len(dataset)
        mixture: Mapping[str, float],           # task -> probability, sums to 1 within 1e-9
        max_atoms: int | None = None,           # per-batch token budgets; at least one required
        max_edges: int | None = None,
        batches_per_epoch: int | None = None,
        seed: int = 42,
    ) -> None: ...

    def __iter__(self) -> Iterator[list[int]]: ...
    def __len__(self) -> int: ...
    def batches_per_task(self) -> dict[str, int]: ...
    def frames_per_task(self) -> dict[str, int]: ...
```

`task_labels` and `group_ids` are **parent-space** per-sample sequences — the
exact shape already accepted by `group_split_indices` (`group_split.py:11`) and
already produced per bundle by `PesBundle.conformer_group_ids`
(`pes_bundle.py:336`). `indices` is the split index list; the sampler
**consumes** it and never splits, never re-derives groups, and never imports
`PesBundle` or `group_split_indices` (staying dataset-agnostic). Yielded
batches contain parent-space dataset indices, so the object drops straight into
the injected-`batch_sampler` seam opened by 09. Not a `torch.utils.data.Sampler`
subclass — duck-typed, mirroring `TokenBudgetBatchSampler`.

### Sampling unit

**Unit key = `(task_labels[i], group_ids[i])`.** Consequences, all deliberate:

- A torsion scan is one unit: its frames share the `torsion` label and one
  group id, so they land in one batch — scan integrity by construction, no
  special case.
- A molecule that contributes to several tasks (equilibrium snapshots *and* a
  relax trajectory) yields **one unit per task**. Mixed labels under one group
  id are normal, not an error; molecule-level leakage across train/val was
  already prevented upstream by `group_split_indices`.
- Two scans of the *same* molecule under the *same* label and group id merge
  into one unit (coarser, still integral). Pass a finer group id
  (`"<mol>:<dihedral>"`) to separate them — documented in the docstring.

### Eager plan (`__init__`)

1. **Bucket** — walk `indices` in given order, group into units by unit key,
   preserving first-appearance order; collect per-task unit lists.
2. **Cost vectors** — per unit, sum the pointer-derived `atom_counts` /
   `edge_counts` of its frames (pointer arithmetic only; no sample is ever
   unpacked, mirroring `TokenBudgetBatchSampler`), then stack per task into one
   `(n_units_t,)` integer tensor per active budget. A dataset missing those
   accessors raises the same-shaped `ValueError` naming the accessors and the
   supported dataset types as `sampler.py:101-107`.
3. **Shuffle** — one `make_generator(seed)` (`src/molix/core/seed.py:58`), tasks
   visited in `sorted(mixture)` order so the draw is independent of dict
   insertion order; per task, `torch.randperm(n_units, generator=g)`.
4. **Pack** — per task, construct `TokenBudgetPacker([(max_atoms,
   unit_atom_costs), (max_edges, unit_edge_costs)])` over that task's unit-cost
   vectors (only the active budgets appear) and call `.pack(shuffled_unit_order)`;
   expand each returned group of unit positions into its members' frame indices
   → `task_batches[t]: list[list[int]]`. Every batch is single-task by
   construction; the flat frame pool is structurally unreachable.
5. **Epoch length** — `batches_per_epoch` when given; otherwise
   `n = min over {t : p_t > 0} of floor(len(task_batches[t]) / p_t)`, i.e. the
   longest epoch in which **no task is exhausted**. Since every task with
   `p_t > 0` has ≥ 1 batch (validated) and `p_t ≤ 1`, `n ≥ 1` always.
6. **Apportionment (round-off rule)** — deterministic largest-remainder
   (Hamilton): `base_t = floor(p_t · n)`; the `n − Σ base_t` leftover batches go
   to the largest fractional remainders, ties broken by ascending task name.
   **Anti-starvation rule**: after apportionment, every task with `p_t > 0` and
   ≥ 1 available batch receives ≥ 1 batch; each deficit is donated by the task
   with the largest surplus `count_t − p_t·n` (ties: larger count first, then
   ascending name). Bound, documented and asserted:
   `|count_t − p_t·n| ≤ 1` for every task when no override fires, and
   `≤ 1 + K` for donors when `K` rare tasks were rescued.
7. **Interleave** — the task-label sequence for the `n` slots is a
   `randperm`-shuffled multiset `{t repeated count_t}` drawn from the same
   generator, so mixture fidelity holds per window rather than only per epoch.
   Each slot consumes that task's next unused packed batch. If a task is
   demanded beyond its supply (only reachable when the caller forces a larger
   `batches_per_epoch`), it is dropped, the remaining probabilities are
   renormalized, one `UserWarning` fires, and planning continues; when every
   task is exhausted the plan simply ends short and `__len__` reports the
   realized length.

**Stratified, not i.i.d.** Step 6 replaces per-batch multinomial draws. Same
expected mixture, zero sampling variance, exactly reproducible audit numbers,
and no flaky χ² test. This is a deliberate deviation from a naive
"draw a task per batch" reading and is recorded in the docstring.

### Determinism, pickling, audit

The instance holds the dataset reference, the two integer budgets, the two
label sequences (stored as tuples), the mixture dict, `seed`, and the computed
`list[list[int]]` — all picklable; the per-task packers are **transient and not
stored** (mirroring 09's `_compute_batches`), so no mmap-backed count tensor is
ever pickled and `pickle` round-trip yields an identical plan. Same
`(dataset, indices, labels, group_ids, mixture, max_atoms, max_edges, seed)` →
identical plan; the caller reseeds per epoch with `seed + epoch`, the
convention already used at `datamodule.py:238` / `:260`.

`batches_per_task()` / `frames_per_task()` are two primitives (no tuple
unpacking, no combined accessor, no tolerance comparison, no verdict) returning
fresh `dict` copies so caller mutation cannot corrupt the plan. They are free
because the plan is eager. **The configured mixture governs batch shares only** —
frame shares follow from unit sizes and token packing and are reported for
audit, not guaranteed; comparing either against a tolerance is the caller's job
(evidence E2 per `.claude/notes/production-training-checklist.md`).

### Errors (eager, `ValueError` naming arg + value + fix) and warnings

| Condition | Message names |
|---|---|
| neither `max_atoms` nor `max_edges` given, or a given budget `<= 0` (same check shape as `sampler.py:75-85`) | both argument names, the offending value, "pass a positive per-batch budget" |
| `mixture` does not sum to 1 within `1e-9` (same eps as `group_split.py:30`) | the sum, the offending mixture, "rescale so probabilities sum to 1.0" |
| `mixture` empty, or a probability `< 0` / `> 1` | task, value, valid range |
| label present in the split but absent from `mixture` | label, affected sample count, "add it with probability 0.0 to exclude it deliberately" |
| task with `p_t > 0` but zero units in the split | task, its probability, "set its probability to 0.0 or widen the split" |
| `len(task_labels)` / `len(group_ids)` ≠ `len(dataset)` | both lengths, the accessor to fix |
| `indices` empty, or an index out of range | value, valid range |

Warnings — `UserWarning`, degraded-not-fatal, each firing at most once per
instance: (a) task exhausted mid-plan (names the task, the slot, and the
renormalization); (b) oversize unit — a unit whose aggregate cost exceeds a
budget is kept whole as a singleton batch, never split and never dropped.
09's `TokenBudgetPacker` is **silent by design**, so this sampler consults
`TokenBudgetPacker.oversize_mask()` on the task's unit-cost vector and emits its
**own** one-shot warning **naming the unit** (task label + group id) — more
actionable than the frame-level message `TokenBudgetBatchSampler` emits, and the
reason warning ownership sits with the caller in 09's contract.

### Naming decision

`TaskMixtureBatchSampler`, not `CurriculumBatchSampler`: the stage schedule
(which mixture is active at epoch *e*) is caller-owned constructor data, so
"curriculum" in the type name would over-claim. Follows the existing
`TokenBudgetBatchSampler` naming shape.

### Reuse decision

No `librarian_report` accompanied this request (chain sub-spec; the caller
supplied the interaction points inline). Resolved from direct repo reading:

- **reuse** `molix.core.seed.make_generator` (`src/molix/core/seed.py:58`) —
  the only randomness source; no private RNG, no global-RNG consumption.
- **reuse** 09's `TokenBudgetPacker` (`src/molix/data/sampler.py`) — the single
  next-fit (single open bin) token-packing loop and the single owner of budget
  semantics. This sampler constructs **one packer instance per task** over that
  task's aggregate unit-cost vectors and calls `.pack(shuffled_unit_order)`.
  09's constructor already accepts arbitrary `(budget, counts)` cost vectors, so
  **no generalization of 09 is needed** and none is specified here.
- **reuse** `TokenBudgetPacker.oversize_mask()` — consulted by this sampler,
  which emits its own one-shot oversize `UserWarning` naming the unit (09's
  packer is silent by design; warning is explicitly the caller's job).
- **reuse** `dataset.atom_counts` / `dataset.edge_counts`
  (`src/molix/data/dataset.py`) — unit costs by pointer arithmetic; zero
  `__getitem__` calls.
- **reuse (contract, no import)** `PesBundle.conformer_group_ids`
  (`src/molix/data/pes_bundle.py:336`) and `group_split_indices`
  (`src/molix/data/group_split.py:11`) — their output shapes *are* this
  sampler's input shapes; the sampler consumes them and never re-splits or
  re-derives ids.
- **new — `TaskMixtureBatchSampler`**: neither `TokenBudgetBatchSampler`
  (single flat pool, no task axis, no atomic multi-frame units, no
  apportionment) nor `torch.utils.data.WeightedRandomSampler` (per-sample
  weights, no group integrity, no token budget, not eager) covers this. Folding
  a task axis into `TokenBudgetBatchSampler` via a mode flag is forbidden by
  CLAUDE.md (no `Foo(method=…)` multi-behaviour switches), so it is a sibling
  type sharing the packing primitive. Construction, budget knobs, eager-plan
  caching, `__iter__`/`__len__`, error style and one-shot warning style all
  follow `TokenBudgetBatchSampler` so the new code reads like the existing code.

## Files to create or modify

- src/molix/data/sampler.py
- src/molix/data/__init__.py
- tests/test_molix/test_data/test_sampler.py
- regressions/universal-class-i-10-sampler.py (new)

## Tasks

- [ ] Write failing unit tests for TaskMixtureBatchSampler in tests/test_molix/test_data/test_sampler.py (TestTaskMixturePlan, TestTaskMixtureIntegrity, TestTaskMixtureDeterminism, TestTaskMixtureValidation, TestTaskMixtureAudit) covering single-task batches, scan integrity, apportionment bound, anti-starvation, hard-coded golden plan, pickle round-trip, epoch reseed, ValueErrors and the one-shot exhaustion / oversize-unit warnings
- [ ] Implement TaskMixtureBatchSampler unit bucketing, pointer-derived per-unit aggregate cost vectors, per-task shuffle plus per-task `TokenBudgetPacker([(max_atoms, unit_atom_costs), (max_edges, unit_edge_costs)]).pack(order)` delegation with unit-position → frame-index expansion, largest-remainder apportionment with the anti-starvation rule, and the interleaved eager plan in src/molix/data/sampler.py
- [ ] Implement eager validation ValueErrors (missing or non-positive max_atoms/max_edges naming both args, mixture sum/range, unknown label, empty task with p>0, length mismatch, empty/out-of-range indices) plus the one-shot exhaustion UserWarning with renormalization and the one-shot oversize-unit UserWarning driven by TokenBudgetPacker.oversize_mask() in src/molix/data/sampler.py
- [ ] Implement batches_per_task() and frames_per_task() read-only audit accessors returning fresh dicts in src/molix/data/sampler.py
- [ ] Export TaskMixtureBatchSampler from src/molix/data/__init__.py with alphabetized __all__
- [ ] Add Google-style docstrings per doc.style in src/molix/data/sampler.py documenting the Stage A/B/C mixtures, the arXiv:2408.12817 and DOI:10.1039/D2SC02739A citations plus the no-precedent statement, the round-off and anti-starvation rules, batch-vs-frame mixture semantics, and the seed+epoch convention
- [ ] Add regression example regressions/universal-class-i-10-sampler.py (public API only; hard-coded goldens, no third-party runtime)
- [ ] Run full check + test suite

## Testing strategy

Unit tests only, in `tests/test_molix/test_data/test_sampler.py` (mirror of
`src/molix/data/sampler.py`), each test targeting one method/behaviour, reusing
the existing hand-checkable packed-cache fixture style in that file
(`_make_varied_samples`, `MmapDataset`), extended with a task-label and
group-id table so every expected number is derivable by hand.

- **Happy path — no flat pool.** Every yielded batch's frames carry exactly one
  distinct task label; every batch is non-empty; concatenated indices are a
  subset of `indices` with no duplicates within an epoch plan.
- **Scan integrity.** A fixture with multi-frame torsion-scan units: for each
  unit, all its frame indices appear in exactly one batch (never straddling
  two). Includes an oversize unit larger than the budget → kept whole as a
  singleton batch (never split, never dropped), with exactly one `UserWarning`
  naming that unit (the packer itself stays silent).
- **Mixture fidelity (replaces a stochastic χ² test).** Because apportionment
  is deterministic largest-remainder, the bound is hard, not statistical:
  assert `|batches_per_task()[t] − p_t·len(sampler)| ≤ 1` for every task with
  no anti-starvation override, over a large plan (≥ 200 batches) for the Stage
  B and Stage C mixtures, plus the aggregate over many batches. A χ² test would
  only add flakiness for a zero-variance realization; the tolerance form is
  strictly stronger and is documented as such in the test docstring.
- **Anti-starvation.** Stage C with a deliberately scarce `electrostatic` task
  (p=0.10, few units) and a small `n`: the rare task gets ≥ 1 batch, the donor
  task's deviation stays within the documented `1 + K` bound.
- **Budget compliance.** Every batch's summed atom and edge counts stay within
  `max_atoms` / `max_edges` (except documented oversize singletons); a
  source-level assertion that the next-fit (single open bin) packing loop
  appears exactly once in `src/molix/data/sampler.py` — inside
  `TokenBudgetPacker` — and that `TaskMixtureBatchSampler` contains no second
  packing loop.
- **Zero unpack.** `__getitem__` call count is 0 across construction and a full
  epoch (reuse the existing `_CountingDataset` pattern).
- **Determinism / golden plan.** Hard-coded expected plan (`list[list[int]]`)
  and hard-coded `batches_per_task()` / `frames_per_task()` dicts for a fixed
  fixture at `seed=20260814`; same seed → identical plan across instances;
  `seed` vs `seed + 1` differ; pickle round-trip identical; iterating twice
  identical.
- **Resume determinism.** Fresh sampler built with `seed + 1` reproduces the
  plan of the "continuous" epoch-1 sampler, mirroring
  `TestDataModuleEpochRederivation` semantics at `datamodule.py:238`.
- **Errors.** `pytest.raises(ValueError)` for each row of the error table, each
  asserting the message names the offending argument and its value.
- **Warning.** `pytest.warns(UserWarning)` exactly once per instance when a
  forced `batches_per_epoch` exhausts the scarcest task; the plan's realized
  length and post-renormalization composition match hand-computed values.
- **Domain validation.** No physical quantity is computed here, so validation
  is structural (the bounds above) with hard-coded expected integers; the
  scientific claim — that per-task gradient contributions track the configured
  mixture — is deliberately deferred to `universal-class-i-13-grad-audit`.

**Regression example**: `regressions/universal-class-i-10-sampler.py` — one
minimal public-API script (build a small `PackedCache` in a temp dir →
`MmapDataset` → `TaskMixtureBatchSampler` with the Stage B mixture and
`seed=20260814` → assert the hard-coded golden plan, single-task batches, scan
integrity, and the hard-coded `batches_per_task()` / `frames_per_task()`
dicts), printing OK and exiting 0, exiting 1 on drift. Goldens are literals
captured once from this repo's own deterministic plan (header records command,
commit sha, torch version, date, device/precision per `regressions/README.md`);
no third-party oracle is imported or subprocessed at any time.

## Out of scope

- **Stage scheduling.** Choosing which mixture is active at epoch *e* (and any
  `Stage` enum, schedule object, or trainer hook) stays with the caller; this
  spec ships one mixture per sampler instance.
- **09-packer's next-fit packing loop and DataModule's injected-`batch_sampler`
  seam** — both owned by `universal-class-i-09-packer`; not re-specified and
  not modified at all, since `TokenBudgetPacker` already accepts arbitrary
  `(budget, counts)` cost vectors.
- **Task labelling.** MolHub assigns physical task labels; molnex only consumes
  them. No label inference, no PES-region classifier here.
- **Splitting.** `group_split_indices` stays the only splitter; this sampler
  never re-splits and never validates leakage.
- **Tolerance verdicts.** The audit accessors expose counts only; comparing
  them against a calibration tolerance (evidence E2) is the caller's job — no
  `check_mixture()` / `assert_balanced()` façade, per CLAUDE.md.
- **DDP.** Cross-rank partitioning of a task-mixture plan has no protocol yet;
  the existing DDP guard at `datamodule.py:222-230` continues to reject
  budgeted batching under DDP.
- **`.claude/notes/architecture.md` inventory refresh** (line 102 `molix.data`
  export list) — captured via `/mol:note` once the chain lands, not by this
  spec's diff.
- **Frame-level mixture guarantees, replacement sampling within an epoch, and
  per-task loss weighting** — considered and rejected: frame shares follow from
  unit sizes (documented, not guaranteed), within-epoch replacement would break
  the exhaustion-warning contract, and loss weighting belongs to the loss
  layer.
