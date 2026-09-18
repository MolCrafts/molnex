"""Token-budget batch sampling over packed cumsum pointers.

Molecular samples vary widely in size (QM9 spans 3–29 atoms), so a fixed
``batch_size`` makes per-batch compute and memory swing with batch
composition — GNN encoder cost scales with the total edge count O(E), not
the sample count O(B). :class:`TokenBudgetBatchSampler` instead packs each
batch next-fit (single open bin) under per-table budgets (atoms, edges, the
valence families, the graph count), keeping per-batch work approximately
constant. Per-sample sizes come from count columns (``n_atoms``,
``<family>_n``) or a caller-supplied table — no sample is ever unpacked
during sampling. Fed :meth:`~molix.data.pad.MoleculePadder.budgets`, the
plan never exceeds the padder's caps, so one static shape serves every batch.

Four layers, deliberately separated:

* :class:`TokenBudgetPacker` — the packing rule alone (visit order in,
  batch plan out); no dataset, no RNG, no warning policy.
* :class:`TokenBudgetBatchSampler` — seeded shuffle + oversize warning on
  top of the packer, over a count-bearing dataset.
* :class:`FrameLabelTable` — the per-frame ``(task, group)`` identity of one
  parent index space in bounded integer-code form, built either from the
  legacy Python label sequences or from a block's ``(F,)`` code columns.
* :class:`TaskMixtureBatchSampler` — hierarchical task → group →
  configuration sampling: one physical task per batch drawn from a caller
  supplied mixture, whole sampling units inside it, packed by the same
  packer under the same token budgets.
* :class:`BatchSamplerSource` — the protocol a
  :class:`~molix.data.datamodule.DataModule` calls once per epoch to obtain
  a batch sampler, so alternative batching policies can be injected.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Protocol, runtime_checkable

import torch

from molix.core.seed import make_generator
from molix.data.dataset import BaseDataset

__all__ = [
    "BatchSamplerSource",
    "FrameLabelTable",
    "TaskMixtureBatchSampler",
    "TokenBudgetBatchSampler",
    "TokenBudgetPacker",
]


class TokenBudgetPacker:
    """Group a visit order into index batches that stay under budget.

    Next-fit (single open bin) packing: indices are consumed in the given
    order and accumulated into exactly one open batch, which is closed as
    soon as adding the next index would push any budget over its cap. A
    closed batch is never reopened, so the plan is a pure function of
    ``(counts, budgets, visit order)``: counts ``[7, 6, 5]`` under budget
    ``12`` pack as ``[[0], [1, 2]]``, never ``[[0, 2], [1]]`` — the latter
    is what a policy that kept several bins open would return.

    An index whose own count exceeds a budget cannot fit anywhere: it is
    emitted as a singleton batch — never dropped — and closes a non-empty
    open batch first rather than merging into it. The packer stays silent
    about such indices; query :meth:`oversize_mask` and decide (warn,
    drop, route elsewhere) in the caller. Owning no dataset, no RNG and no
    warning policy is what lets one packer serve both
    :class:`TokenBudgetBatchSampler` and any other sampler that already
    knows its own visit order.

    Args:
        budgets: One ``(budget, counts)`` pair per active budget, where
            ``counts`` is an integer ``(n_samples,)`` vector of
            per-sample sizes and ``budget`` caps their per-batch sum. Both
            are dimensionless counts (atoms, edges, …) in the same
            counting unit — no physical units are involved. Every vector
            is materialised once as Python ints, so a later mutation (or a
            closed mmap) cannot change the plan.

    Raises:
        ValueError: *budgets* is empty — at least one ``(budget, counts)``
            pair is required, since packing without a cap is undefined.
    """

    def __init__(self, budgets: Sequence[tuple[int, torch.Tensor]]) -> None:
        if not budgets:
            raise ValueError(
                "TokenBudgetPacker needs at least one (budget, counts) pair, "
                "e.g. [(max_atoms, nrows)] with counts an integer "
                "(n_samples,) vector; got an empty budgets sequence."
            )
        self._budgets = [budget for budget, _ in budgets]
        self._counts = [[int(v) for v in counts.tolist()] for _, counts in budgets]

    def oversize_mask(self) -> torch.Tensor:
        """Flag samples whose own count exceeds a budget.

        Returns:
            Bool tensor ``(n_samples,)`` — the elementwise OR over budgets
            of ``counts > budget``. It covers the **full** count vectors
            given at construction, so a caller that packs only a subset of
            the samples must index this mask by that subset.
        """
        mask = torch.zeros(len(self._counts[0]), dtype=torch.bool)
        for budget, counts in zip(self._budgets, self._counts):
            mask |= torch.tensor(counts, dtype=torch.long) > budget
        return mask

    def pack(self, order: Sequence[int]) -> list[list[int]]:
        """Pack *order* into consecutive batches under every active budget.

        Args:
            order: Sample indices in the order they should be visited
                (e.g. a shuffled permutation). Indices address the
                ``(n_samples,)`` count vectors given at construction.

        Returns:
            The batch plan as ``list[list[int]]``. Concatenating it
            reproduces *order* exactly — packing is a partition, nothing is
            dropped or duplicated, including oversize singletons. An empty
            *order* yields ``[]``.
        """
        batches: list[list[int]] = []
        current: list[int] = []
        totals = [0] * len(self._budgets)

        for idx in order:
            sizes = [counts[idx] for counts in self._counts]
            oversize = any(s > budget for s, budget in zip(sizes, self._budgets))
            fits = all(t + s <= budget for t, s, budget in zip(totals, sizes, self._budgets))
            if current and not fits:
                batches.append(current)
                current = []
                totals = [0] * len(self._budgets)
            current.append(idx)
            totals = [t + s for t, s in zip(totals, sizes)]
            if oversize:
                batches.append(current)
                current = []
                totals = [0] * len(self._budgets)
        if current:
            batches.append(current)
        return batches


class TokenBudgetBatchSampler:
    """Yield index batches whose per-table totals stay under budget.

    Next-fit (single open bin) packing in seeded-shuffle order — or in index
    order under ``shuffle=False`` (validation): samples are visited in a
    ``randperm`` permutation drawn from
    :func:`~molix.core.seed.make_generator` and handed to a
    :class:`TokenBudgetPacker`, which closes the open batch whenever adding
    the next sample would exceed any active budget. A sample whose own count
    exceeds a budget forms a singleton batch — it is never dropped — and one
    :class:`UserWarning` is emitted per sampler instance when the dataset
    contains any such sample. A static-shape loader gets its budgets from
    :meth:`~molix.data.pad.MoleculePadder.budgets`, whose
    :meth:`~molix.data.pad.MoleculePadder.check` refuses such a sample up
    front, so there the singleton never arises.

    The batch list is computed once in ``__init__`` (O(n) pointer
    arithmetic) and cached, so ``__iter__`` / ``__len__`` are free.
    ``len(sampler)`` is the batch count for *this* permutation — it varies
    with ``seed``, so different epochs may have different lengths.
    :meth:`reseed` replans in place for the next epoch (``seed + epoch``),
    which is what lets a :class:`~molix.data.DataModule` keep one
    DataLoader — and its persistent workers — across epochs while epoch
    *k*'s composition stays re-derivable on resume.

    The instance holds only the dataset reference, the budget table, the
    caller's count vectors (if any), the seed, the shuffle flag and the
    computed ``list[list[int]]`` — all picklable. Note that ``DataLoader``
    consumes a ``batch_sampler`` in the main process; workers receive index
    lists, so the sampler itself never crosses a process boundary unless the
    whole DataModule is shipped.

    Args:
        dataset: A dataset of Frames. A budgeted table absent from *counts*
            is read through :meth:`~molix.data.dataset.BaseDataset.nrows_of`
            (frame tables, not a molhub cache).
        budgets: ``{table: per-batch cap}`` — ``atoms``, ``pairs``, a valence
            family (``bonds`` / ``angles`` / ``propers`` / ``impropers``) or
            ``graphs`` (molecules per batch: every sample counts one). Every
            budget applies simultaneously; at least one is required.
        counts: Optional ``{table: (n_samples,) counts}``. When given it
            must carry every budgeted table except ``graphs``: a missing
            entry is refused rather than silently read from the dataset,
            because that fallback walks every sample of a map-style
            dataset. A molhub loader that already scanned its sizes passes
            them here so this sampler never opens a frame.
        seed: Seed for the shuffle permutation.
        shuffle: ``False`` packs the dataset in index order.

    Raises:
        ValueError: No budget given, a budget is ``<= 0``, *counts* names a
            table that is not budgeted or misses one that is, or *dataset*
            declares no route to a table that has to be read from it —
            neither :attr:`~molix.data.dataset.BaseDataset.counted_tables`
            nor a Frame block of that name.
    """

    def __init__(
        self,
        dataset: BaseDataset,
        *,
        budgets: Mapping[str, int],
        counts: Mapping[str, torch.Tensor] | None = None,
        seed: int = 42,
        shuffle: bool = True,
    ) -> None:
        if not budgets:
            raise ValueError(
                "TokenBudgetBatchSampler needs at least one budget: pass "
                "budgets={'atoms': ...} (and/or 'pairs', a valence family, 'graphs') "
                "— per-batch caps on the summed per-sample counts."
            )
        for name, value in budgets.items():
            if value is None or int(value) <= 0:
                raise ValueError(
                    f"budgets[{name!r}] must be > 0, got {value}. Pass a positive "
                    f"per-batch budget or omit {name!r} to disable that cap."
                )
        if counts is not None:
            extra = sorted(set(counts) - set(budgets))
            if extra:
                raise ValueError(
                    f"counts names tables with no budget: {extra}; budgets are {sorted(budgets)}."
                )
            missing = sorted(t for t in budgets if t != "graphs" and t not in counts)
            if missing:
                raise ValueError(
                    f"counts is given but lacks the budgeted tables {missing}; pass "
                    "their per-sample counts too (a partial table is not read from "
                    "the dataset behind your back)."
                )

        self._dataset = dataset
        self._budgets = {str(name): int(value) for name, value in budgets.items()}
        self._counts = (
            None if counts is None else {str(name): value for name, value in counts.items()}
        )
        self._seed = seed
        self._shuffle = bool(shuffle)
        self._batches, oversize = self._plan()
        if oversize:
            warnings.warn(
                "TokenBudgetBatchSampler: at least one sample exceeds "
                "the per-batch budget on its own; such samples are kept "
                "as singleton batches (never dropped). Raise the budget "
                "to avoid singletons.",
                UserWarning,
                stacklevel=2,
            )

    def _budget_counts(self) -> list[tuple[int, torch.Tensor]]:
        """Resolve ``(budget, counts)`` pairs for every active budget."""
        pairs: list[tuple[int, torch.Tensor]] = []
        n = len(self._dataset)
        for table, budget in self._budgets.items():
            if table == "graphs":
                pairs.append((budget, torch.ones(n, dtype=torch.long)))
                continue
            if self._counts is not None:
                pairs.append((budget, self._counts[table]))
                continue
            try:
                pairs.append((budget, BaseDataset.nrows_of(self._dataset, table)))
            except (AttributeError, TypeError) as e:
                raise ValueError(
                    f"TokenBudgetBatchSampler requires frame table nrows ({table}); "
                    f"got {type(self._dataset).__name__}."
                ) from e
        return pairs

    def _plan(self) -> tuple[list[list[int]], bool]:
        """Next-fit (single open bin) packing of the visit order under budget.

        Builds a transient :class:`TokenBudgetPacker` over the active
        ``(budget, counts)`` pairs — transient so the instance state stays
        picklable and never holds a packer — and packs the seeded
        permutation (or the index order under ``shuffle=False``).

        Returns:
            ``(batches, oversize)`` — the plan for this ``seed`` as
            ``list[list[int]]``, and whether any sample is oversize.
        """
        packer = TokenBudgetPacker(self._budget_counts())
        oversize = bool(packer.oversize_mask().any())
        n = len(self._dataset)
        if self._shuffle:
            order = torch.randperm(n, generator=make_generator(self._seed)).tolist()
        else:
            order = list(range(n))
        return packer.pack(order), oversize

    def reseed(self, seed: int) -> None:
        """Replan in place for *seed* (the next epoch's ``seed + epoch``).

        Args:
            seed: The permutation seed of the new plan.
        """
        self._seed = seed
        self._batches, _ = self._plan()

    def __iter__(self) -> Iterator[list[int]]:
        """Iterate the cached batches, each a ``list[int]`` of sample indices."""
        return iter(self._batches)

    def __len__(self) -> int:
        """Number of batches for this permutation.

        Varies with ``seed`` (and therefore epoch when seeded as
        ``seed + epoch``), since next-fit packing of a different shuffle
        order can produce a different batch count.
        """
        return len(self._batches)


class FrameLabelTable:
    """Per-frame ``(task, group)`` identity of one parent index space.

    A scheduler has to answer exactly two questions about frame ``i``:
    which physical task it belongs to, and which sampling group (scan,
    trajectory, molecule) it may not be separated from. This type owns
    that answer in the form the READY-tier block store already emits —
    two bounded ``(F,)`` integer code columns plus the short dictionaries
    that render a code as text — so a consumer never has to decode a
    block's columns into ``F`` Python objects before it can plan.

    Both representations enter through a constructor of their own and
    leave in the same shape: the canonical constructor takes a block's
    code columns verbatim, :meth:`from_labels` factorizes the legacy
    Python label sequences. Downstream code sees codes only.

    Args:
        task_codes: ``(F,)`` integral CPU tensor of task codes; ``code``
            indexes *task_names*. Kept by reference — it is only read,
            never mutated, and never stored beyond this table's lifetime.
        group_codes: ``(F,)`` integral CPU tensor of group codes, same
            length as *task_codes*. Two frames share a sampling unit when
            **both** codes agree.
        task_names: Bounded task dictionary; ``task_names[c]`` is the
            label of task code ``c``. Must cover ``max(task_codes) + 1``.
        group_names: Optional bounded group dictionary used only to render
            a group label (oversize warnings, audit text). When omitted a
            group code renders as ``str(code)``.

    Raises:
        ValueError: A code column is not a 1-D integral CPU tensor, the
            two columns disagree in length, a code is negative, or a code
            reaches outside its dictionary.
    """

    __slots__ = ("_group_codes", "_group_names", "_task_codes", "_task_names")

    def __init__(
        self,
        *,
        task_codes: torch.Tensor,
        group_codes: torch.Tensor,
        task_names: Sequence[str],
        group_names: Sequence[str] | None = None,
    ) -> None:
        self._validate_codes("task_codes", task_codes)
        self._validate_codes("group_codes", group_codes)
        if len(task_codes) != len(group_codes):
            raise ValueError(
                f"len(task_codes) is {len(task_codes)} but len(group_codes) is "
                f"{len(group_codes)}: both are per-frame columns of the same "
                "parent index space, so fix the accessor that produced the "
                "shorter one."
            )
        self._task_codes = task_codes
        self._group_codes = group_codes
        self._task_names = tuple(str(name) for name in task_names)
        self._group_names = None if group_names is None else tuple(str(n) for n in group_names)
        self._validate_dictionary("task_codes", task_codes, "task_names", self._task_names)
        if self._group_names is not None:
            self._validate_dictionary("group_codes", group_codes, "group_names", self._group_names)

    @classmethod
    def from_labels(
        cls, task_labels: Sequence[str], group_ids: Sequence[str | int]
    ) -> FrameLabelTable:
        """Build a table from the legacy per-frame Python label sequences.

        Both sequences are factorized in **first-appearance order** in a
        single ``O(F)`` pass. That order is plan-irrelevant — a plan
        depends on the ``(task, group)`` pair identity and on
        ``sorted(mixture)``, never on code values — and it keeps the
        block-native and label-native tables identical for a block whose
        codes were themselves assigned on first appearance.

        Args:
            task_labels: Per-frame task label, ``F`` long.
            group_ids: Per-frame group identity, ``F`` long. Rendered as
                ``str(id)`` in the group dictionary.

        Returns:
            A table over ``(F,)`` ``int64`` code columns carrying both
            recovered dictionaries.

        Raises:
            ValueError: Either argument is a tensor (its 0-dim elements
                would factorize by identity, one group per frame), or the
                two sequences differ in length.
        """
        for name, labels in (("task_labels", task_labels), ("group_ids", group_ids)):
            if isinstance(labels, torch.Tensor):
                raise ValueError(
                    f"{name} is a tensor, but from_labels factorizes Python "
                    "label sequences (a tensor's 0-dim elements would hash by "
                    "identity, yielding one group per frame). Use "
                    "FrameLabelTable(task_codes=..., group_codes=..., "
                    "task_names=...) for code columns."
                )
        if len(task_labels) != len(group_ids):
            raise ValueError(
                f"len(task_labels) is {len(task_labels)} but len(group_ids) is "
                f"{len(group_ids)}: both are parent-space per-frame sequences, "
                "so fix the accessor that produced the shorter one."
            )
        task_codes, task_names = cls._factorize(task_labels)
        group_codes, group_names = cls._factorize(group_ids)
        return cls(
            task_codes=task_codes,
            group_codes=group_codes,
            task_names=task_names,
            group_names=group_names,
        )

    def __len__(self) -> int:
        """Frame count ``F`` of the parent index space this table covers."""
        return int(self._task_codes.shape[0])

    @property
    def task_names(self) -> tuple[str, ...]:
        """The bounded task dictionary; index ``c`` is the label of task code ``c``."""
        return self._task_names

    @property
    def group_names(self) -> tuple[str, ...] | None:
        """The bounded group dictionary, or ``None`` when groups render as codes."""
        return self._group_names

    def task_name(self, code: int) -> str:
        """Render one task code as its label.

        Args:
            code: Task code, ``0 <= code < len(task_names)``.

        Returns:
            The task label, e.g. ``"torsion"``.
        """
        return self._task_names[code]

    def group_label(self, code: int) -> str:
        """Render one group code as text.

        Args:
            code: Group code, non-negative.

        Returns:
            The original group id when a group dictionary was supplied,
            else ``str(code)`` — the code itself is a stable identity, so
            a missing dictionary degrades the *text* only.
        """
        if self._group_names is None:
            return str(code)
        return self._group_names[code]

    def codes_at(self, indices: Sequence[int]) -> tuple[list[int], list[int]]:
        """Gather the ``(task, group)`` codes of *indices*, in the given order.

        One vectorized gather per column — the reason this type accepts
        the tensor form at all — instead of ``len(indices)`` per-element
        tensor reads.

        Args:
            indices: Parent-space frame indices, each in ``[0, F)``.

        Returns:
            ``(task_codes, group_codes)`` as two Python ``int`` lists,
            parallel to *indices*.
        """
        selector = torch.as_tensor(tuple(indices), dtype=torch.long)
        return self._task_codes[selector].tolist(), self._group_codes[selector].tolist()

    @staticmethod
    def _factorize(labels: Sequence[str | int]) -> tuple[torch.Tensor, tuple[str, ...]]:
        """Map *labels* to ``(F,) int64`` codes plus a first-appearance dictionary."""
        lookup: dict[str | int, int] = {}
        names: list[str] = []
        codes: list[int] = []
        for label in labels:
            code = lookup.get(label)
            if code is None:
                code = len(names)
                lookup[label] = code
                names.append(str(label))
            codes.append(code)
        return torch.tensor(codes, dtype=torch.int64), tuple(names)

    @staticmethod
    def _validate_codes(name: str, codes: torch.Tensor) -> None:
        """Reject a column that is not a non-negative 1-D integral CPU code table."""
        if not isinstance(codes, torch.Tensor):
            raise ValueError(
                f"{name} must be a (F,) integral CPU tensor of codes, got "
                f"{type(codes).__name__}. Pass the block's code column (e.g. "
                f"reader.task_codes), or build the table with "
                "FrameLabelTable.from_labels for Python label sequences."
            )
        if codes.ndim != 1:
            raise ValueError(
                f"{name} has shape {tuple(codes.shape)} but a frame code table is "
                "1-D (F,), one code per frame. Flatten it, or pass the column "
                "the block store emitted."
            )
        if codes.device.type != "cpu":
            raise ValueError(
                f"{name} is on device {codes.device}, but label tables are planned "
                f"on cpu (planning is pointer arithmetic, not compute). Pass "
                f"{name}.cpu()."
            )
        if codes.dtype.is_floating_point or codes.dtype.is_complex:
            raise ValueError(
                f"{name} has dtype {codes.dtype}, but codes index a bounded "
                "dictionary and must be an integral dtype (e.g. int32/int64) — "
                f"a float table is never rounded. Pass {name}.to(torch.int64) "
                "only if the values really are integers."
            )
        if codes.numel() and int(codes.min()) < 0:
            raise ValueError(
                f"{name} contains the negative code {int(codes.min())}; codes "
                "index a dictionary from 0, so a negative entry is a table bug "
                "(never a wrap-around). Fix the producer of that column."
            )

    @staticmethod
    def _validate_dictionary(
        codes_name: str, codes: torch.Tensor, names_name: str, names: tuple[str, ...]
    ) -> None:
        """Reject a code that reaches outside its bounded dictionary."""
        if not codes.numel():
            return
        largest = int(codes.max())
        if largest >= len(names):
            raise ValueError(
                f"{codes_name} contains the code {largest} but {names_name} has "
                f"{len(names)} entries (valid codes are 0..{len(names) - 1}). "
                f"Pass the full dictionary that column was coded against."
            )


class TaskMixtureBatchSampler:
    """Yield single-task index batches drawn from a task mixture.

    A universal Class-I training collection mixes physically different
    data — equilibrium snapshots, torsion scans, near-equilibrium
    relaxations, electrostatic probes — whose natural frame counts are
    wildly unbalanced, and a torsion scan only carries its torsion
    profile when its frames are compared *within one batch*. This sampler
    therefore batches hierarchically: **task → group → configuration**.

    1. Frames of the split are bucketed into **sampling units** keyed
       by the ``(task, group)`` code pair of :class:`FrameLabelTable`, in
       first-appearance order.
    2. Units of one task are shuffled and handed to a transient
       :class:`TokenBudgetPacker` over their *aggregate* atom/edge costs,
       so a unit is packed whole — a scan never straddles two batches, and
       a unit larger than the budget becomes a whole singleton batch
       (warned about once, never split, never dropped).
    3. The epoch's slots are apportioned across tasks by the mixture and
       interleaved, so every batch carries exactly one task label. The
       flat frame pool is structurally unreachable.

    **There is no published precedent for a task → group → configuration
    hierarchical batch sampler**, so there is no reference implementation
    to validate against: correctness here is *structural* (mixture
    fidelity, unit integrity, determinism), while the scientific claim —
    that per-task gradient contributions track the configured mixture —
    is measured downstream by the gradient audit. Two published results
    are cited for the pieces this sampler borrows, and for nothing more:

    * **Stage structure only** — the default Stage A/B/C mixtures below
      are constructor *data* derived from ByteFF's three-stage curriculum
      over increasingly diverse PES regions; the mechanism (per-batch
      mixture draw, unit integrity, token packing) is not from ByteFF.
    * **Grouping discipline only** — espaloma splits at the molecule
      level so conformers never straddle a split; this sampler inherits
      that discipline one level down, making the group id the atomic unit
      of *sampling* exactly as it is the atomic unit of *splitting* in
      :func:`~molix.data.group_split.group_split_indices`.

    Stage mixtures (plain ``dict`` constructor data — no ``Stage`` enum,
    no schedule object; choosing the active stage at epoch *e* stays with
    the caller):

    ===== =========== ======= ===== =============
    Stage equilibrium torsion relax electrostatic
    ===== =========== ======= ===== =============
    A     0.70        —       0.30  —
    B     0.35        0.45    0.20  —
    C     0.25        0.30    0.35  0.10
    ===== =========== ======= ===== =============

    **Stratified, not i.i.d.** Slots are apportioned by deterministic
    largest-remainder (Hamilton) rather than drawn per batch from a
    multinomial: ``base_t = floor(p_t · n)``, and the ``n − Σ base_t``
    leftover slots go to the largest fractional remainders, ties broken by
    ascending task name. An **anti-starvation** rule then guarantees that
    every task with ``p_t > 0`` and at least one available batch receives
    at least one slot; each such deficit is donated by the task with the
    largest surplus ``count_t − p_t·n`` (ties: larger count first, then
    ascending name), taken only from tasks that can spare it (``count ≥
    2``). The bound is therefore hard rather than statistical:
    ``|count_t − p_t·n| ≤ 1`` for every task when no override fires, and
    ``≤ 1 + K`` for donors when ``K`` rare tasks were rescued. Same
    expected mixture as a per-batch draw, zero sampling variance, exactly
    reproducible audit numbers.

    **The mixture governs batch shares only.** Frame shares follow from
    unit sizes and token packing; :meth:`frames_per_task` reports them for
    audit but does not guarantee them. Comparing either share against a
    calibration tolerance is the caller's job — this class ships no
    verdict façade.

    **Determinism.** Every random choice comes from one
    :func:`~molix.core.seed.make_generator` seeded with ``seed``: tasks
    are visited in ``sorted(mixture)`` order (independent of ``dict``
    insertion order), each drawing one unit permutation, and the slot
    order is one final permutation of the apportioned task multiset. The
    whole plan is built eagerly in ``__init__``, so ``len()`` and
    iteration are free and repeatable. The instance holds the dataset
    reference, the budgets, the mixture, the computed ``list[list[int]]``,
    the parallel per-batch task-name tuple, and (for :meth:`reseed`) the
    split indices plus their split-space integer task/group code columns
    and the two short bounded dictionaries — no parent-space ``(F,)``
    label table, no packer, and no mmap-backed count vector. Reseed per
    epoch with
    ``seed + epoch`` (the convention
    :class:`~molix.data.datamodule.DataModule` already uses) to reshuffle
    between epochs while keeping epoch *k* re-derivable on resume.

    Args:
        dataset: A dataset of Frames; packing reads
            :meth:`~molix.data.dataset.BaseDataset.nrows_of` for table
            names. Unit costs are those vectors; no sample is unpacked
            on the frame-table path.
        indices: The split's dataset indices, e.g. the train list from
            :func:`~molix.data.group_split.group_split_indices`. Consumed
            as given — this sampler never splits and never re-derives
            groups. Yielded batches contain these parent-space indices.
        task_labels: Per-sample physical task, ``len(dataset)`` long
            (parent space, the shape
            :func:`~molix.data.group_split.group_split_indices` already
            accepts), in **either** representation: a sequence of opaque
            label strings, or a ``(F,)`` integral CPU tensor of task
            codes (a block's ``task_codes`` column) together with
            *task_names*. molnex never infers a task.
        group_ids: Per-sample group identity, ``len(dataset)`` long, in
            the same representation as *task_labels*: a sequence of
            strings/ints, or a ``(F,)`` integral CPU tensor of group
            codes. Two scans of one molecule sharing a task *and* a group
            merge into one (coarser, still integral) unit — pass a finer
            id such as ``"<mol>:<dihedral>"`` to keep them separate. A
            molecule contributing to several tasks yields one unit per
            task, which is normal: cross-split leakage was already
            prevented upstream.
        task_names: Bounded task dictionary, ``task_names[c]`` naming task
            code ``c``. **Required** with integer *task_labels* (codes are
            undecodable without it) and rejected with string
            *task_labels*, which already carry their names.
        group_names: Optional bounded group dictionary for integer
            *group_ids*; it changes the *text* of the oversize warning
            only (``torsion/tor_a`` instead of ``torsion/6``), never the
            plan. Rejected with string *group_ids*.
        mixture: Task → probability, summing to 1 within ``1e-9``. A label
            present in the split must appear here; give it ``0.0`` to
            exclude it deliberately.
        max_atoms: Maximum total atom count per batch, or ``None``.
        max_pairs: Maximum total edge count per batch, or ``None``. At
            least one budget is required; both apply simultaneously when
            given.
        batches_per_epoch: Force the epoch length. Default ``None`` picks
            the longest epoch in which no task is exhausted,
            ``min over {t : p_t > 0} of floor(len(batches_t) / p_t)``.
            A larger forced value can exhaust a task: it is then dropped,
            the remaining probabilities are renormalized, one
            :class:`UserWarning` fires, and the plan ends at its realized
            length.
        seed: Seed for the unit permutations and the slot order.

    Raises:
        ValueError: A budget is missing or ``<= 0``,
            ``batches_per_epoch <= 0``, *mixture* is empty / does not sum
            to 1 within ``1e-9`` / carries a probability outside
            ``[0, 1]``, a label of the split is absent from *mixture*, a
            task with ``p_t > 0`` has no unit in the split, a label table
            length differs from ``len(dataset)``, *task_names* is missing
            with integer codes or supplied with string labels, a code
            table is not a non-negative 1-D integral CPU tensor,
            *indices* is empty or addresses outside the dataset, or
            *dataset* does not expose the pointer-derived count
            accessors.

    Reference:
        Zheng et al. "ByteFF: A Machine Learning Force Field for
        Drug-like Molecules" arXiv:2408.12817 (2024) — stage structure
        only. https://arxiv.org/abs/2408.12817

        Wang, Fass, Chodera. "End-to-End Differentiable Construction of
        Molecular Mechanics Force Fields" Chem. Sci. 13, 12016 (2022) —
        grouping discipline only. DOI:10.1039/D2SC02739A, https://doi.org/10.1039/D2SC02739A
    """

    def __init__(
        self,
        dataset: BaseDataset,
        *,
        indices: Sequence[int],
        task_labels: Sequence[str] | torch.Tensor,
        group_ids: Sequence[str | int] | torch.Tensor,
        task_names: Sequence[str] | None = None,
        group_names: Sequence[str] | None = None,
        mixture: Mapping[str, float],
        max_atoms: int | None = None,
        max_pairs: int | None = None,
        batches_per_epoch: int | None = None,
        seed: int = 42,
    ) -> None:
        self._validate_sizes(max_atoms, max_pairs, batches_per_epoch)
        self._validate_mixture(mixture)
        # The one representation dispatch: each FrameLabelTable constructor owns
        # one label form, so every step below this line sees codes only.
        if isinstance(task_labels, torch.Tensor):
            if task_names is None:
                raise ValueError(
                    "task_labels is a (F,) integer code column but task_names is "
                    "None: codes cannot be decoded without their bounded "
                    "dictionary. Pass task_names (e.g. reader.task_names), or "
                    "pass task_labels as per-frame label strings."
                )
            if not isinstance(group_ids, torch.Tensor):
                raise ValueError(
                    "task_labels is a (F,) integer code column but group_ids is a "
                    f"{type(group_ids).__name__}: both tables describe the same "
                    "parent index space and must use the same representation. "
                    "Pass group_ids as its (F,) code column too (e.g. "
                    "reader.group_codes)."
                )
            table = FrameLabelTable(
                task_codes=task_labels,
                group_codes=group_ids,
                task_names=task_names,
                group_names=group_names,
            )
        else:
            if isinstance(group_ids, torch.Tensor):
                raise ValueError(
                    "group_ids is a (F,) integer code column but task_labels is a "
                    f"{type(task_labels).__name__} of labels: both tables describe "
                    "the same parent index space and must use the same "
                    "representation. Pass task_labels as its (F,) code column plus "
                    "task_names, or group_ids as per-frame ids."
                )
            if task_names is not None or group_names is not None:
                raise ValueError(
                    "task_names/group_names were given together with string "
                    "task_labels, which already carry their own names. Drop the "
                    "dictionaries, or pass task_labels/group_ids as the (F,) "
                    "integer code columns they belong to."
                )
            table = FrameLabelTable.from_labels(task_labels, group_ids)
        self._validate_tables(len(dataset), indices, table)

        self._dataset = dataset
        self._max_atoms = max_atoms
        self._max_edges = max_pairs
        self._mixture = dict(mixture)
        self._batches_per_epoch = batches_per_epoch
        self._seed = seed
        # Kept so :meth:`reseed` can rebuild the plan in place without the
        # caller re-supplying the split: the split indices, their
        # split-space integer (task, group) codes, and the two short
        # bounded dictionaries — never the parent-space ``(F,)`` label
        # table, which only exists here.
        self._indices = tuple(indices)
        split_task_codes, split_group_codes = table.codes_at(self._indices)
        self._task_codes = tuple(split_task_codes)
        self._group_codes = tuple(split_group_codes)
        self._task_names = table.task_names
        self._group_names = table.group_names
        self._batches, self._batch_tasks = self._compute_batches(
            self._indices,
            self._task_codes,
            self._group_codes,
            self._task_names,
            self._group_names,
        )

    @staticmethod
    def _validate_sizes(
        max_atoms: int | None, max_pairs: int | None, batches_per_epoch: int | None
    ) -> None:
        """Reject missing / non-positive budget and epoch-length knobs."""
        if max_atoms is None and max_pairs is None:
            raise ValueError(
                "TaskMixtureBatchSampler needs at least one budget: pass "
                "max_atoms and/or max_pairs (total per-batch atom/edge caps)."
            )
        for name, value in (("max_atoms", max_atoms), ("max_pairs", max_pairs)):
            if value is not None and value <= 0:
                raise ValueError(
                    f"{name} must be > 0, got {value}. Pass a positive "
                    f"per-batch budget or omit {name} to disable that cap."
                )
        if batches_per_epoch is not None and batches_per_epoch <= 0:
            raise ValueError(
                f"batches_per_epoch must be > 0, got {batches_per_epoch}. "
                "Pass a positive epoch length or omit it to use the "
                "no-exhaustion default."
            )

    @staticmethod
    def _validate_mixture(mixture: Mapping[str, float]) -> None:
        """Reject an empty, out-of-range or non-normalized *mixture*."""
        if not mixture:
            raise ValueError(
                "mixture is empty: pass at least one task -> probability "
                "entry, e.g. {'equilibrium': 0.7, 'relax': 0.3}."
            )
        total = math.fsum(mixture.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"mixture probabilities sum to {total}, not 1.0 (tolerance "
                f"1e-9); got {dict(mixture)}. Rescale so probabilities sum "
                "to 1.0."
            )
        for task, probability in sorted(mixture.items()):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(
                    f"mixture probability for task {task!r} is {probability}, "
                    "outside the valid range [0.0, 1.0]. Pass a probability "
                    "in [0.0, 1.0], using 0.0 to exclude the task."
                )

    @staticmethod
    def _validate_tables(
        n_samples: int,
        indices: Sequence[int],
        table: FrameLabelTable,
    ) -> None:
        """Reject a mis-sized label table and empty / out-of-range *indices*."""
        if len(table) != n_samples:
            raise ValueError(
                f"the label table covers {len(table)} frame(s) but len(dataset) "
                f"is {n_samples}: task_labels and group_ids are parent-space "
                "per-sample tables, so fix the accessor that produced them."
            )
        if len(indices) == 0:
            raise ValueError(
                "indices is empty: pass the split's dataset indices (e.g. the "
                "train list of group_split_indices); no batch can be formed "
                "from an empty split."
            )
        for index in indices:
            if not 0 <= index < n_samples:
                raise ValueError(
                    f"indices contains {index}, outside the valid range "
                    f"[0, {n_samples}) for this dataset. Pass split indices "
                    "addressing the dataset given to this sampler."
                )

    def _validate_tasks(self, task_units: dict[str, list[tuple[str, list[int]]]]) -> None:
        """Reject a split label missing from the mixture, or a supply-less task."""
        for task in sorted(task_units):
            if task not in self._mixture:
                n_frames = sum(len(frames) for _, frames in task_units[task])
                raise ValueError(
                    f"task_labels carries {n_frames} sample(s) labelled "
                    f"{task!r} in the split, but {task!r} is absent from "
                    "mixture. Add it with probability 0.0 to exclude it "
                    "deliberately."
                )
        for task, probability in sorted(self._mixture.items()):
            if probability > 0 and not task_units.get(task):
                raise ValueError(
                    f"mixture task {task!r} has probability {probability} but "
                    "zero sampling units in the split. Set its probability to "
                    "0.0 or widen the split."
                )

    def _frame_budgets(self) -> list[tuple[int, torch.Tensor]]:
        """Resolve ``(budget, per-frame counts)`` pairs for every active budget."""
        pairs: list[tuple[int, torch.Tensor]] = []
        try:
            if self._max_atoms is not None:
                pairs.append((self._max_atoms, BaseDataset.nrows_of(self._dataset, "atoms")))
            if self._max_edges is not None:
                pairs.append((self._max_edges, BaseDataset.nrows_of(self._dataset, "pairs")))
        except (AttributeError, TypeError) as e:
            raise ValueError(
                "TaskMixtureBatchSampler requires frame table nrows "
                f"(atoms / edges); got {type(self._dataset).__name__}."
            ) from e
        return pairs

    @staticmethod
    def _bucket_units(
        indices: tuple[int, ...],
        task_codes: Sequence[int],
        group_codes: Sequence[int],
        task_names: Sequence[str],
        group_names: Sequence[str] | None,
    ) -> dict[str, list[tuple[str, list[int]]]]:
        """Bucket *indices* into sampling units keyed by ``(task, group)`` codes.

        Bucketing is done on the integer code pair — one gather over the
        whole split instead of ``len(indices)`` per-frame table reads —
        and each unit's codes are rendered as text exactly once, when the
        unit is attached to its task.

        Args:
            indices: The split's dataset indices, walked in the given order.
            task_codes: Split-space task codes, parallel to *indices*.
            group_codes: Split-space group codes, parallel to *indices*.
            task_names: Bounded task dictionary; ``task_names[c]`` labels
                task code ``c``.
            group_names: Optional bounded group dictionary; when ``None``
                a group code renders as ``str(code)``.

        Returns:
            Task label → its units as ``(group label, member frame
            indices)`` pairs, both levels in first-appearance order.
        """
        unit_frames: dict[tuple[int, int], list[int]] = {}
        for index, task_code, group_code in zip(indices, task_codes, group_codes, strict=True):
            unit_frames.setdefault((task_code, group_code), []).append(index)

        task_units: dict[str, list[tuple[str, list[int]]]] = {}
        for (task_code, group_code), frames in unit_frames.items():
            task_units.setdefault(task_names[task_code], []).append(
                (str(group_code) if group_names is None else group_names[group_code], frames)
            )
        return task_units

    def _pack_tasks(
        self,
        task_units: dict[str, list[tuple[str, list[int]]]],
        generator: torch.Generator,
    ) -> tuple[dict[str, list[list[int]]], list[str]]:
        """Shuffle and pack each task's units into frame-index batches.

        One transient :class:`TokenBudgetPacker` per task over that task's
        *aggregate* unit costs — transient so no mmap-backed count vector
        reaches the instance state, and per task because the packer binds
        its cost vectors at construction. Packed unit positions are
        expanded into their member frame indices, which is what makes unit
        integrity structural rather than merely checked.

        Args:
            task_units: Output of :meth:`_bucket_units`.
            generator: The single seeded generator; tasks are visited in
                ``sorted(mixture)`` order and a task with no unit in the
                split draws nothing.

        Returns:
            ``(task_batches, oversize)`` — packed frame-index batches per
            task, and the ``"<task>/<group>"`` names of units whose own
            aggregate cost exceeds a budget (the packer is silent about
            them by design; warning is this caller's job).
        """
        frame_budgets = self._frame_budgets()
        task_batches: dict[str, list[list[int]]] = {}
        oversize: list[str] = []
        for task in sorted(self._mixture):
            units = task_units.get(task)
            if not units:
                continue
            packer = TokenBudgetPacker(
                [
                    (budget, torch.tensor([int(counts[frames].sum()) for _, frames in units]))
                    for budget, counts in frame_budgets
                ]
            )
            oversize.extend(
                f"{task}/{units[position][0]}"
                for position, flagged in enumerate(packer.oversize_mask().tolist())
                if flagged
            )
            order = torch.randperm(len(units), generator=generator).tolist()
            task_batches[task] = [
                [index for position in group for index in units[position][1]]
                for group in packer.pack(order)
            ]
        return task_batches, oversize

    def _epoch_length(self, task_batches: dict[str, list[list[int]]]) -> int:
        """Number of slots to plan: the forced length, or the no-exhaustion floor.

        Args:
            task_batches: Packed batches per task, from :meth:`_pack_tasks`.

        Returns:
            ``batches_per_epoch`` when given, else ``min over {t : p_t > 0}
            of floor(len(batches_t) / p_t)`` — the longest epoch in which
            no task runs out. Since every task with ``p_t > 0`` has at
            least one unit (validated) and ``p_t <= 1``, this is ``>= 1``.
        """
        if self._batches_per_epoch is not None:
            return self._batches_per_epoch
        return min(
            math.floor(len(task_batches[task]) / probability)
            for task, probability in self._mixture.items()
            if probability > 0
        )

    def _apportion(self, n: int, task_batches: dict[str, list[list[int]]]) -> dict[str, int]:
        """Split *n* slots across tasks by largest remainder, then rescue rare tasks.

        Args:
            n: Epoch length in batches.
            task_batches: Packed batches per task, used only to tell a
                supply-less task (never rescued) from a starved one.

        Returns:
            Task → slot count, summing to *n*.
        """
        counts = {task: math.floor(probability * n) for task, probability in self._mixture.items()}
        remainders = {
            task: probability * n - counts[task] for task, probability in self._mixture.items()
        }
        leftover = n - sum(counts.values())
        for task in sorted(counts, key=lambda name: (-remainders[name], name))[:leftover]:
            counts[task] += 1

        for task in sorted(counts):
            starved = self._mixture[task] > 0 and counts[task] == 0
            if not starved or not task_batches.get(task):
                continue
            donors = [name for name, count in counts.items() if count >= 2]
            if not donors:
                continue
            donor = min(
                donors,
                key=lambda name: (-(counts[name] - self._mixture[name] * n), -counts[name], name),
            )
            counts[donor] -= 1
            counts[task] += 1
        return counts

    def _interleave(
        self,
        counts: dict[str, int],
        n: int,
        task_batches: dict[str, list[list[int]]],
        generator: torch.Generator,
    ) -> tuple[list[list[int]], tuple[str, ...], tuple[str, int] | None]:
        """Order the apportioned slots and fill each from its task's supply.

        The slot labels are the multiset ``{t repeated counts[t]}`` in
        sorted task order, permuted once by the shared generator, so
        mixture fidelity holds per window rather than only per epoch. The
        task of every slot actually filled is recorded, which is what lets
        the audit accessors count without a parent-space label table.

        Args:
            counts: Apportioned slot count per task.
            n: Epoch length in batches (``sum(counts.values())``).
            task_batches: Packed batches per task, consumed in order.
            generator: The single seeded generator.

        Returns:
            ``(plan, plan_tasks, shortfall)`` — the epoch's frame-index
            batches, the task name of each of them (parallel to *plan*),
            and ``(task, slot)`` of the first slot whose task ran out of
            supply (``None`` when every slot was filled).
        """
        multiset: list[str] = []
        for task in sorted(counts):
            multiset.extend([task] * counts[task])

        cursors = dict.fromkeys(counts, 0)
        plan: list[list[int]] = []
        plan_tasks: list[str] = []
        shortfall: tuple[str, int] | None = None
        for slot, position in enumerate(torch.randperm(n, generator=generator).tolist()):
            task = multiset[position]
            available = task_batches.get(task, [])
            if cursors[task] >= len(available):
                if shortfall is None:
                    shortfall = (task, slot)
                continue
            plan.append(available[cursors[task]])
            plan_tasks.append(task)
            cursors[task] += 1
        return plan, tuple(plan_tasks), shortfall

    def _compute_batches(
        self,
        indices: tuple[int, ...],
        task_codes: Sequence[int],
        group_codes: Sequence[int],
        task_names: Sequence[str],
        group_names: Sequence[str] | None,
    ) -> tuple[list[list[int]], tuple[str, ...]]:
        """Build the whole epoch plan eagerly, and own the warning policy.

        Args:
            indices: The split's dataset indices.
            task_codes: Split-space task codes, parallel to *indices*.
            group_codes: Split-space group codes, parallel to *indices*.
            task_names: Bounded task dictionary; ``task_names[c]`` labels
                task code ``c``.
            group_names: Optional bounded group dictionary, ``None`` to
                render group codes as ``str(code)``.

        Returns:
            ``(plan, plan_tasks)`` — the epoch's batches as
            ``list[list[int]]`` of parent-space dataset indices, every
            batch single-task by construction, and the parallel tuple of
            their task names.
        """
        task_units = self._bucket_units(indices, task_codes, group_codes, task_names, group_names)
        self._validate_tasks(task_units)

        generator = make_generator(self._seed)
        task_batches, oversize = self._pack_tasks(task_units, generator)
        if oversize:
            warnings.warn(
                "TaskMixtureBatchSampler: sampling unit(s) "
                f"{', '.join(oversize)} (task/group) exceed the per-batch "
                "budget on their own; each is kept whole as a singleton "
                "batch — never split, never dropped. Raise max_atoms/"
                "max_pairs to avoid singletons.",
                UserWarning,
                stacklevel=3,
            )

        n = self._epoch_length(task_batches)
        plan, plan_tasks, shortfall = self._interleave(
            self._apportion(n, task_batches), n, task_batches, generator
        )
        if shortfall is not None:
            task, slot = shortfall
            warnings.warn(
                f"TaskMixtureBatchSampler: task {task!r} ran out of packed "
                f"batches at slot {slot} of {n}; its remaining slots are "
                "dropped, the mixture is renormalized over the surviving "
                "tasks, and len(sampler) reports the realized length. Lower "
                "batches_per_epoch (or omit it) to keep every task in supply.",
                UserWarning,
                stacklevel=3,
            )
        return plan, plan_tasks

    def __iter__(self) -> Iterator[list[int]]:
        """Iterate the eager plan, each item a ``list[int]`` of dataset indices."""
        return iter(self._batches)

    def __len__(self) -> int:
        """Realized number of batches in the epoch.

        Equals ``batches_per_epoch`` when that forced length was in
        supply, the no-exhaustion floor when it was not given, and the
        shorter realized length when a forced length exhausted a task.
        """
        return len(self._batches)

    def reseed(self, seed: int) -> None:
        """Recompute the epoch plan in place for a new *seed*.

        Lets a cached DataLoader keep its persistent worker pool alive
        across epochs: the sampler is consumed in the main process, so
        mutating ``_batches`` is visible to the next ``iter(loader)``.
        No-op when *seed* is unchanged (e.g. the first epoch).
        """
        seed = int(seed)
        if seed == self._seed:
            return
        self._seed = seed
        self._batches, self._batch_tasks = self._compute_batches(
            self._indices,
            self._task_codes,
            self._group_codes,
            self._task_names,
            self._group_names,
        )

    def batches_per_task(self) -> dict[str, int]:
        """Count the planned batches carried by each task.

        Free (the plan is eager) and read-only: the returned ``dict`` is a
        fresh copy, so caller mutation cannot corrupt the plan. This is
        the realized batch share — the quantity the mixture governs.

        Counted from the task recorded for each slot when the plan was
        built, so a caller mutating the code tensors it passed in cannot
        make the audit drift from the plan.

        Returns:
            Task label → number of batches in this epoch's plan; tasks
            with no realized batch are absent.
        """
        counts: dict[str, int] = {}
        for task in self._batch_tasks:
            counts[task] = counts.get(task, 0) + 1
        return counts

    def frames_per_task(self) -> dict[str, int]:
        """Count the planned frames carried by each task.

        Frame shares follow from unit sizes and token packing; they are
        reported for audit, **not** guaranteed by the mixture (which
        governs batch shares only). Comparing them against a calibration
        tolerance is the caller's job.

        Counted from the per-slot task record of the eager plan, like
        :meth:`batches_per_task`.

        Returns:
            Task label → number of frames in this epoch's plan; tasks with
            no realized batch are absent.
        """
        counts: dict[str, int] = {}
        for task, batch in zip(self._batch_tasks, self._batches, strict=True):
            counts[task] = counts.get(task, 0) + len(batch)
        return counts


@runtime_checkable
class BatchSamplerSource(Protocol):
    """Per-epoch provider of a train batch sampler.

    Implemented by callers that want to drive batch composition
    themselves; a :class:`~molix.data.datamodule.DataModule` constructed
    with ``batch_sampler_source=`` calls :meth:`batch_sampler` once per
    epoch, immediately before building the train
    :class:`~torch.utils.data.DataLoader`.

    A *provider* rather than a sampler instance, because the seed is only
    known per epoch: the DataModule passes ``seed + epoch``, which keeps
    epoch *k*'s composition re-derivable on resume without checkpointing
    RNG state (the same reseed convention
    :class:`TokenBudgetBatchSampler` gets on the budget path).
    """

    def batch_sampler(self, dataset: BaseDataset, *, seed: int) -> Iterable[list[int]]:
        """Return this epoch's batch sampler over *dataset*.

        The returned object is handed straight to
        :class:`~torch.utils.data.DataLoader` as ``batch_sampler=``, so it
        must be iterable **and** ``Sized`` — each iteration yields one
        ``list[int]`` of sample indices, and ``len(...)`` is the number of
        batches. A plain ``list[list[int]]`` qualifies, as does
        :class:`TokenBudgetBatchSampler`.

        Args:
            dataset: The train dataset the indices address; its
                ``__len__`` bounds the valid index range.
            seed: Per-epoch seed (``DataModule.seed + epoch``). Deriving
                every random choice from it keeps the epoch reproducible.

        Returns:
            An iterable, sized batch plan of sample-index lists.
        """
        ...
