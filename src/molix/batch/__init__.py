"""Batch construction for MolNex: samples ↔ the collated batch a model sees.

Surfaces:

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
  per epoch. :func:`group_split_indices` splits indices without splitting a
  group across sides.
* **Loader** — :class:`DataModule` (protocol: :class:`DataModuleProtocol`)
  wraps prebuilt datasets in DDP-aware train / val ``DataLoader`` s.

Storage lives in :mod:`molix.store`; nothing here reads or writes files.
"""

from __future__ import annotations

from molix.batch.collate import (
    DEFAULT_TARGET_SCHEMA,
    MoleculeCollator,
    TargetSchema,
    collate_molecules,
)
from molix.batch.datamodule import DataModule, DataModuleProtocol
from molix.batch.group_split import group_split_indices
from molix.batch.pad import PadMolecularBatch
from molix.batch.sampler import (
    BatchSamplerSource,
    TaskMixtureBatchSampler,
    TokenBudgetBatchSampler,
    TokenBudgetPacker,
)
from molix.batch.task import BatchTask

__all__ = [
    "BatchSamplerSource",
    "BatchTask",
    "DEFAULT_TARGET_SCHEMA",
    "DataModule",
    "DataModuleProtocol",
    "MoleculeCollator",
    "PadMolecularBatch",
    "TargetSchema",
    "TaskMixtureBatchSampler",
    "TokenBudgetBatchSampler",
    "TokenBudgetPacker",
    "collate_molecules",
    "group_split_indices",
]
