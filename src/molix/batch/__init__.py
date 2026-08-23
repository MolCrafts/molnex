"""Batch construction for MolNex: samples ↔ the collated batch a model sees.

Surfaces:

* **Graph construction** — :class:`NeighborList` builds the cutoff graph
  (``edge_index`` / ``edge_diff`` / ``edge_dist``) under the repo Edge
  Convention; :func:`bond_index_from_columns` builds the canonical ``(2, N)``
  covalent ``bond_index``, and :func:`stack_angle_index` /
  :func:`stack_proper_index` / :func:`stack_improper_index` stack post-collate
  valence columns into COO ``[arity, N]`` at kernel call sites.
* **Collation** — :func:`collate_molecules` (and the reusable
  :class:`MoleculeCollator` callable) turn a list of flat sample dicts into the
  nested ``atoms`` / ``edges`` / ``graphs`` / ``bonds`` ``TensorDict``,
  rebasing index columns onto the batch. :class:`TargetSchema` (default
  :data:`DEFAULT_TARGET_SCHEMA`) routes target keys to the graph or atom
  namespace.
* **Post-collate transforms** — :class:`BatchTask` is the extension point for
  transforms that must run on the collated batch; :class:`PadMolecularBatch`
  is the shipped fixed-shape padder for ``torch.compile`` / CUDA graphs.
* **Batch policies** — :class:`TokenBudgetPacker` /
  :class:`TokenBudgetBatchSampler` pack samples under atom / edge budgets,
  :class:`TaskMixtureBatchSampler` interleaves per-task streams, and
  :class:`BatchSamplerSource` is the protocol a :class:`DataModule` calls once
  per epoch. :class:`BlockPlanSource` implements it over a multi-block
  :class:`BlockFrameView`, returning a :class:`BlockPlan` (with its
  :class:`BlockCoverage` rows) that is planned globally and consumed
  block-windowed. :func:`group_split_indices` splits indices without splitting a
  group across sides.
* **Mixture** — :class:`DataMixture` is a Dataset of named sources (the
  key is the task) and the mix policy :class:`DataModule` uses when it
  is the train dataset.
* **Loader** — :class:`DataModule` (protocol: :class:`DataModuleProtocol`)
  wraps prebuilt datasets or a :class:`DataMixture` in DDP-aware train /
  val ``DataLoader`` s.

Storage lives in :mod:`molix.store`; nothing here reads or writes files.
"""

from __future__ import annotations

from molix.batch.block_plan import (
    BlockCoverage,
    BlockFrameView,
    BlockPlan,
    BlockPlanSource,
)
from molix.batch.bond_adapter import bond_index_from_columns
from molix.batch.collate import (
    DEFAULT_TARGET_SCHEMA,
    MoleculeCollator,
    TargetSchema,
    collate_molecules,
)
from molix.batch.datamodule import DataModule, DataModuleProtocol
from molix.batch.group_split import group_split_indices
from molix.batch.mixture import DataMixture
from molix.batch.neighbor import NeighborList
from molix.batch.pad import PadMolecularBatch
from molix.batch.sampler import (
    BatchSamplerSource,
    TaskMixtureBatchSampler,
    TokenBudgetBatchSampler,
    TokenBudgetPacker,
)
from molix.batch.task import BatchTask
from molix.batch.valence_columns import (
    stack_angle_index,
    stack_improper_index,
    stack_proper_index,
)

__all__ = [
    "BatchSamplerSource",
    "BatchTask",
    "BlockCoverage",
    "BlockFrameView",
    "BlockPlan",
    "BlockPlanSource",
    "DEFAULT_TARGET_SCHEMA",
    "DataMixture",
    "DataModule",
    "DataModuleProtocol",
    "MoleculeCollator",
    "NeighborList",
    "PadMolecularBatch",
    "TargetSchema",
    "TaskMixtureBatchSampler",
    "TokenBudgetBatchSampler",
    "TokenBudgetPacker",
    "bond_index_from_columns",
    "collate_molecules",
    "group_split_indices",
    "stack_angle_index",
    "stack_improper_index",
    "stack_proper_index",
]
