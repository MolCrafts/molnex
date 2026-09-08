"""Block-aware epoch plan behind ``DataModule(batch_sampler_source=...)``.

A streaming training epoch reads from many sealed blocks, but only ``K`` of them
fit in the reader pool at once. This module makes the strategy's "plan globally,
consume rolling" claim executable: :class:`FrameStorePlanSource` plans **once**, over
every block's metadata, with the one
:class:`~molix.batch.sampler.TaskMixtureBatchSampler` the non-streaming path
already uses, and then emits that same plan in **block-windowed** consumption
order.

Plan authority (invariant I8)
-----------------------------

The plan is the authority. Its batch order is the consumed order forever:

* the per-task unit shuffle spans **all** blocks, so shuffling is plan-level and
  never within-block-only;
* the emitted batches are a permutation of the planned ones after block-boundary
  division — nothing is dropped, duplicated, substituted or reordered *within* a
  window;
* a block the view cannot supply stalls the consumer at its plan position
  (:class:`~molix.batch.datamodule._FrameStoreCollateFn` propagates the view's
  refusal); it is never skipped and never replaced by another block's batches.

Division at a block boundary is safe by construction: a sampling unit never
straddles a block (the block format's invariant I2), so cutting a planned batch
into maximal runs of frames sharing a block can never cut a unit, never exceed a
token budget (a part is ≤ the whole) and never mix tasks (the planned batch was
single-task). :attr:`FrameStorePlan.boundary_splits` counts the planned batches that
had to be divided.

Why the collate stamps *global* task codes
------------------------------------------

``FrameStoreReader.frame_payload`` deliberately skips ``meta.`` routes, so the
per-frame task code never reaches a batch, while
:class:`~molix.core.steps.multitask.TaskRouter` requires a single-task
``("graphs", "task")`` label. The block collate route therefore **stamps** that
label — from the **view's** ``task_codes`` (the merged, global dictionary), never
from a block-local ``reader.task_codes``: a block's local ``(0, 1)`` may
legitimately be global ``(1, 2)`` after the view's dictionary merge, and
``TaskRouter`` maps the integer to a :class:`~molix.core.steps.multitask.TaskSpec`,
so a local stamp would route the same physical task to different losses depending
on which block a batch came from. Nothing about the block format changes.

Risk R2 — mixture fidelity is per epoch, not per window
-------------------------------------------------------

:class:`~molix.batch.sampler.TaskMixtureBatchSampler` interleaves slots so mixture
fidelity holds per window of consecutive batches. Block-windowed consumption
preserves the per-epoch apportionment and every batch's contents exactly, but
consecutive batches now come from at most ``K`` blocks; if blocks are single-task,
task interleaving degrades toward per-block at ``K = 1``. The levers are a larger
``K`` (interleaving across the resident set at bounded residency) and task-mixed
blocks on the producer side. :attr:`FrameStorePlan.window` and
:attr:`FrameStorePlan.coverage` make the realized situation inspectable;
``batches_per_task()`` / ``frames_per_task()`` still report the epoch-level shares
unchanged.

Layering: this module binds to the :class:`FrameStoreView` *contract*, not to
:class:`molix.store.frame_store_view.FrameStoreDataset`, so a plan can be built (and unit
tested) over any view exposing that surface. It computes nothing physical — every
quantity here is a dimensionless count of frames, units, batches or blocks.

References:
    Wang, Fass & Chodera. "End-to-End Differentiable Construction of Molecular
    Mechanics Force Fields." Chem. Sci. 13, 12016 (2022). DOI:10.1039/D2SC02739A
    — molecule-level grouping before configuration sampling, the discipline a
    block-windowed order must not disturb.
    Zheng et al. "ByteFF: A Machine Learning Force Field for Drug-like Molecules."
    arXiv:2408.12817 (2024) — the stage / mixture structure the sampler encodes.
    Mishchenko, Khaled & Richtárik. "Random Reshuffling: Simple Analysis with Vast
    Improvements." NeurIPS 2020, arXiv:2006.05988 — why a shuffled visit order over
    units (rather than i.i.d. draws) is an acceptable optimization regime, which is
    what licenses reordering an epoch's *windows*.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

import torch

from molix.batch.sampler import TaskMixtureBatchSampler

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime dependency
    from molix.store.dataset import BaseDataset
    from molix.store.frame_store import FrameStoreReader

__all__ = [
    "FrameStoreCoverage",
    "FrameStoreView",
    "FrameStorePlan",
    "FrameStorePlanSource",
]

#: Version tag of the visit-order document MolHub's conveyor parses.
_VISIT_ORDER_FORMAT = "molix.visit-order/1"


@runtime_checkable
class FrameStoreView(Protocol):
    """The multi-block view surface a block plan consumes.

    Declared here so planning binds to a contract rather than to
    :class:`molix.store.frame_store_view.FrameStoreDataset`, and so it can be exercised
    against an in-memory stub. The vocabulary is
    :class:`~molix.store.frame_store.FrameStoreReader`'s, one level up: ``F`` is the view's
    global frame count, block ordinals index the planned block order.

    Members:
        block_ids: Block identity strings (directory basenames), ordinal-indexed —
            ``n_blocks`` of them.
        max_open: ``K``, the view's reader-pool bound, reused as the plan window
            (a count of blocks).
        split: The single split every planned block carries (invariant I7).
        atom_counts: Atoms per frame ``(F,)``, the only budget this plan spends.
        task_codes: Task code per frame ``(F,)``, in the view's **global** merged
            dictionary.
        task_names: The global code → name dictionary, passed through to the
            sampler.
        group_codes: Sampling-unit code per frame ``(F,)``, globally unique.
        edge_counts: Raises by design — blocks store no neighbour graph, so a plan
            never budgets on edges.

    ``runtime_checkable`` verifies member *presence* only, never a signature, so
    the contract-seam test also calls :meth:`locate` and reads
    :attr:`block_ids` / :attr:`max_open` against the real view. Presence is
    resolved with :func:`inspect.getattr_static` (Python 3.12+, which this project
    requires), so an ``isinstance`` probe reads the *descriptor* and never fires
    ``edge_counts``' deliberate refusal.
    """

    block_ids: tuple[str, ...]
    max_open: int
    split: str
    atom_counts: torch.Tensor
    task_codes: torch.Tensor
    task_names: tuple[str, ...]
    group_codes: torch.Tensor
    edge_counts: torch.Tensor

    def __len__(self) -> int:
        """Global frame count ``F`` across the planned blocks."""
        ...

    def locate(self, frames: Sequence[int] | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map global frame rows to ``(block ordinal, block-local row)``.

        Args:
            frames: Global frame rows, as a sequence or a 1-D tensor, ``B`` long.

        Returns:
            ``(blocks, locals)``, both ``(B,) torch.long``.
        """
        ...

    def reader(self, block: int) -> FrameStoreReader:
        """Return one block's reader; residency is the view's concern.

        Args:
            block: Block ordinal in the planned order.

        Returns:
            The sealed block's :class:`~molix.store.frame_store.FrameStoreReader`.
        """
        ...


@dataclass(frozen=True)
class FrameStoreCoverage:
    """One block's accounting row of a plan (invariant I10).

    Every field is a dimensionless count of the plan, not of the block on disk: a
    partial pass reports what it planned, so coverage is accounted rather than
    assumed.

    Args:
        block: Block identity string (its directory basename).
        units: Distinct sampling units (group codes) among that block's planned
            frames.
        frames: Planned frames of that block.
        batches: Emitted batches owned by that block.
    """

    block: str
    units: int
    frames: int
    batches: int


@dataclass(frozen=True)
class FrameStorePlan:
    """One epoch's block-ordered batch plan — the ``batch_sampler`` object itself.

    Iterable and ``Sized``, so it is handed straight to
    ``DataLoader(batch_sampler=...)``. A pure value object: it holds ints and
    strings only — never a view, a reader or a payload tensor — so it pickles to a
    worker as the plan it is.

    Args:
        batches: Index batches in **emission** order; each is a ``list[int]`` of
            global frame rows, ``B_i`` long.
        blocks: Owning block identity per batch, parallel to *batches*.
        coverage: One :class:`FrameStoreCoverage` row per block the plan touches, in
            visit order.
        split: The view's single split (invariant I7).
        seed: The per-epoch planning seed (``DataModule.seed + epoch``).
        window: ``K`` — the view's ``max_open``, in blocks.
        boundary_splits: Number of *planned* batches that had to be divided at a
            block boundary (a count of batches, not of parts).

    Raises:
        ValueError: *blocks* is not parallel to *batches*, the coverage frame
            total disagrees with the planned frame total, or the coverage rows do
            not name exactly the blocks the batches belong to.
    """

    batches: tuple[list[int], ...]
    blocks: tuple[str, ...]
    coverage: tuple[FrameStoreCoverage, ...]
    split: str
    seed: int
    window: int
    boundary_splits: int

    def __post_init__(self) -> None:
        """Reject a plan whose accounting cannot be true of its own batches."""
        if len(self.blocks) != len(self.batches):
            raise ValueError(
                f"a plan carries one owning block per batch, but it holds "
                f"{len(self.batches)} batch(es) and {len(self.blocks)} block id(s); "
                "block_of(position) could not answer for every emitted batch."
            )
        planned_frames = sum(len(batch) for batch in self.batches)
        covered_frames = sum(row.frames for row in self.coverage)
        if covered_frames != planned_frames:
            raise ValueError(
                f"coverage accounts {covered_frames} frame(s) but the plan's batches "
                f"hold {planned_frames}: a coverage row is a count of the plan, so the "
                "two totals are the same number counted twice."
            )
        covered_blocks = {row.block for row in self.coverage}
        owning_blocks = set(self.blocks)
        if covered_blocks != owning_blocks:
            raise ValueError(
                "coverage must name exactly the blocks the plan's batches belong to, "
                f"but it differs by {sorted(covered_blocks ^ owning_blocks)} "
                f"(coverage: {sorted(covered_blocks)}; batches: {sorted(owning_blocks)})."
            )

    def __iter__(self) -> Iterator[list[int]]:
        """Iterate the emitted batches, each a ``list[int]`` of global frame rows."""
        return iter(self.batches)

    def __len__(self) -> int:
        """Number of emitted batches in this epoch."""
        return len(self.batches)

    def block_of(self, position: int) -> str:
        """Identity of the block the batch at *position* belongs to.

        Args:
            position: Emission position in ``[0, len(self))``.

        Returns:
            The owning block's identity string.
        """
        return self.blocks[position]

    def visit_order(self) -> tuple[str, ...]:
        """Blocks in the order the conveyor must make them READY.

        First appearance in emission order, de-duplicated — a block revisited in a
        later batch of the same window appears once, at its first position.

        Returns:
            The distinct block identity strings, ``n_blocks`` long.
        """
        return tuple(dict.fromkeys(self.blocks))

    def write_visit_order(self, path: str | os.PathLike[str]) -> None:
        """Write the ``molix.visit-order/1`` document MolHub's conveyor parses.

        The file **is** the interface: MolHub builds its own ``VisitOrder(order,
        cursor)`` from it and neither repo imports the other. ``order`` is the
        contract; ``coverage`` is advisory sizing for lease refcounting and
        prefetch depth. The ``cursor`` is MolHub's progress state and is never
        written here.

        Args:
            path: Destination file; its parent must exist.
        """
        document = {
            "format": _VISIT_ORDER_FORMAT,
            "split": self.split,
            "seed": self.seed,
            "window": self.window,
            "order": list(self.visit_order()),
            "coverage": [asdict(row) for row in self.coverage],
        }
        Path(path).write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


class FrameStorePlanSource:
    """Plans one epoch globally over a block view, emits it block-windowed.

    A :class:`~molix.batch.sampler.BatchSamplerSource` implementation:
    :class:`~molix.batch.datamodule.DataModule` calls
    :meth:`batch_sampler` once per epoch with ``seed + epoch``, so epoch *k* stays
    re-derivable on resume. Sampling policy itself is unchanged — mixtures, token
    budgets, anti-starvation and unit integrity are
    :class:`~molix.batch.sampler.TaskMixtureBatchSampler`'s, used verbatim; this
    source only decides the *order* in which planned batches are consumed, and
    divides the few that straddle a block boundary.

    Args:
        mixture: Task name → probability, summing to 1 within ``1e-9`` — passed
            through to the sampler unchanged.
        max_atoms: Per-batch total-atom budget (a count of atoms). Blocks carry no
            edges, so this is the only budget a block plan can spend.
        max_edges: Refused. Accepted in the signature purely so passing it fails
            with an actionable message instead of a bare ``TypeError``.

    Raises:
        ValueError: *max_edges* was given.
    """

    def __init__(
        self,
        *,
        mixture: Mapping[str, float],
        max_atoms: int,
        max_edges: int | None = None,
    ) -> None:
        if max_edges is not None:
            raise ValueError(
                f"max_edges={max_edges} cannot be budgeted over blocks: the block "
                "format stores no neighbour-graph column, so FrameStoreReader.edge_counts "
                "raises by design and this plan will not fabricate a count. The cutoff "
                "graph is built downstream (molix.batch.NeighborList or the MD bind "
                "path) — budget the plan on max_atoms instead."
            )
        self._mixture = dict(mixture)
        self._max_atoms = int(max_atoms)

    def batch_sampler(self, dataset: FrameStoreView, *, seed: int) -> FrameStorePlan:
        """Plan this epoch over *dataset* and return it in consumption order.

        Planning reads metadata only — the ``(F,)`` count and code tables — so no
        block reader is opened here: residency is the view's (and the conveyor's)
        concern, and a plan over an unavailable block is still a valid plan.

        Args:
            dataset: The multi-block view to plan over.
            seed: Per-epoch seed (``DataModule.seed + epoch``); every random choice
                of the plan derives from it.

        Returns:
            The epoch's :class:`FrameStorePlan`: block-pure batches of global frame
            rows, ordered window by window over at most ``dataset.max_open``
            blocks.

        Raises:
            TypeError: *dataset* is not a :class:`FrameStoreView`.
        """
        if not isinstance(dataset, FrameStoreView):
            raise TypeError(
                "FrameStorePlanSource plans over a FrameStoreView (e.g. "
                "molix.store.frame_store_view.FrameStoreDataset), which supplies the block_ids / "
                "max_open / split / atom_counts / task_codes / group_codes / locate "
                f"surface a block plan needs; got {type(dataset).__name__}. Wrap the "
                "blocks in a view, or use TokenBudgetBatchSampler for a packed cache."
            )
        planned = [list(batch) for batch in self._plan(dataset, seed)]
        parts, boundary_splits = self._divide(dataset, planned)
        ordered = self._window(dataset, parts)
        return FrameStorePlan(
            batches=tuple(frames for _, frames in ordered),
            blocks=tuple(dataset.block_ids[block] for block, _ in ordered),
            coverage=self._coverage(dataset, ordered),
            split=dataset.split,
            seed=seed,
            window=dataset.max_open,
            boundary_splits=boundary_splits,
        )

    def _plan(self, view: FrameStoreView, seed: int) -> TaskMixtureBatchSampler:
        """Build the one global sampler — the plan authority (invariant I8).

        One :class:`~molix.batch.sampler.TaskMixtureBatchSampler` over
        ``range(F)`` and the view's own int-code tables, so the per-task unit
        shuffle spans every block instead of being restricted to one.

        Args:
            view: The multi-block view.
            seed: Per-epoch seed.

        Returns:
            The eager, epoch-length sampler over global frame rows.
        """
        # The sampler declares the wider BaseDataset but consumes exactly ``len()``
        # plus the budget count vector — ``atom_counts`` here, since ``max_edges`` is
        # None — both of which FrameStoreView pins. The cast records that, rather
        # than narrowing this module's binding to a concrete view class.
        return TaskMixtureBatchSampler(
            cast("BaseDataset", view),
            indices=range(len(view)),
            task_labels=view.task_codes,
            group_ids=view.group_codes,
            task_names=view.task_names,
            mixture=self._mixture,
            max_atoms=self._max_atoms,
            max_edges=None,
            seed=seed,
        )

    @staticmethod
    def _divide(
        view: FrameStoreView, planned: Sequence[list[int]]
    ) -> tuple[list[tuple[int, list[int]]], int]:
        """Cut each planned batch into maximal runs of frames sharing a block.

        A unit never straddles a block, so a run boundary can never cut a unit,
        never exceed the token budget (a part is ≤ the whole) and never mix tasks.

        Args:
            view: The multi-block view, for :meth:`FrameStoreView.locate`.
            planned: The planned batches, in plan order.

        Returns:
            ``(parts, boundary_splits)`` — the ``(block ordinal, frames)`` parts in
            plan order, and the number of planned *batches* that were divided.
        """
        parts: list[tuple[int, list[int]]] = []
        boundary_splits = 0
        for batch in planned:
            blocks, _ = view.locate(batch)
            before = len(parts)
            for frame, block in zip(batch, blocks.tolist(), strict=True):
                if len(parts) > before and parts[-1][0] == block:
                    parts[-1][1].append(frame)
                else:
                    parts.append((block, [frame]))
            if len(parts) - before > 1:
                boundary_splits += 1
        return parts, boundary_splits

    @staticmethod
    def _window(
        view: FrameStoreView, parts: Sequence[tuple[int, list[int]]]
    ) -> list[tuple[int, list[int]]]:
        """Reorder *parts* so at most ``K`` blocks are resident at a time.

        Blocks are ordered by first appearance in the plan (deterministic, derived
        from the plan itself — no second RNG) and partitioned into consecutive
        windows of at most ``K = view.max_open`` blocks; the sort is stable, so
        plan order is preserved *within* a window. At ``K >= n_blocks`` this is the
        identity; at ``K = 1`` it is strict block-major.

        Args:
            view: The multi-block view, for ``max_open``.
            parts: Block-pure parts in plan order.

        Returns:
            The same parts, in emission order.
        """
        window_of = {
            block: position // view.max_open
            for position, block in enumerate(dict.fromkeys(block for block, _ in parts))
        }
        return sorted(parts, key=lambda part: window_of[part[0]])

    @staticmethod
    def _coverage(
        view: FrameStoreView, ordered: Sequence[tuple[int, list[int]]]
    ) -> tuple[FrameStoreCoverage, ...]:
        """Account the plan per block, in visit order (invariant I10).

        Args:
            view: The multi-block view, for ``block_ids`` and ``group_codes``.
            ordered: The emitted ``(block ordinal, frames)`` parts.

        Returns:
            One :class:`FrameStoreCoverage` row per block the plan touches — units
            (distinct group codes among that block's planned frames), frames and
            batches.
        """
        frames_of: dict[int, list[int]] = {}
        batches_of: dict[int, int] = {}
        for block, frames in ordered:
            frames_of.setdefault(block, []).extend(frames)
            batches_of[block] = batches_of.get(block, 0) + 1
        return tuple(
            FrameStoreCoverage(
                block=view.block_ids[block],
                units=int(
                    view.group_codes[torch.as_tensor(frames, dtype=torch.long)].unique().numel()
                ),
                frames=len(frames),
                batches=batches_of[block],
            )
            for block, frames in frames_of.items()
        )
