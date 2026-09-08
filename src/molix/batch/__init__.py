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
  rebasing index columns onto the batch.
* **Typed view** — :class:`TensorFrame` declares those namespaces as fields
  over the same storage, and :func:`as_tensor_frame` is the one wrap every
  reader (molpot's ``Potentials``, the MD driver, the batch tasks, the
  LAMMPS writer) goes through.
* **Post-collate transforms** — :class:`BatchTask` is the extension point for
  transforms that must run on the collated batch; :class:`PadMolecularBatch`
  is the shipped fixed-shape padder for ``torch.compile`` / CUDA graphs and
  :class:`TotalCharge` owns the per-graph ``graphs.total_charge`` column.
* **Batch policies** — :class:`TokenBudgetPacker` /
  :class:`TokenBudgetBatchSampler` pack samples under atom / edge budgets,
  :class:`TaskMixtureBatchSampler` interleaves per-task streams, and
  :class:`BatchSamplerSource` is the protocol a :class:`DataModule` calls once
  per epoch. :class:`FrameStorePlanSource` implements it over a multi-block
  :class:`FrameStoreView`, returning a :class:`FrameStorePlan` (with its
  :class:`FrameStoreCoverage` rows) that is planned globally and consumed
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

from molix.batch.bond_adapter import bond_index_from_columns
from molix.batch.charge import TotalCharge
from molix.batch.collate import (
    MoleculeCollator,
    collate_molecules,
)
from molix.batch.datamodule import DataModule, DataModuleProtocol
from molix.batch.frame import TensorFrame, as_tensor_frame
from molix.batch.frame_store_plan import (
    FrameStoreCoverage,
    FrameStorePlan,
    FrameStorePlanSource,
    FrameStoreView,
)
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
    "FrameStoreCoverage",
    "FrameStoreView",
    "FrameStorePlan",
    "FrameStorePlanSource",
    "DataMixture",
    "DataModule",
    "DataModuleProtocol",
    "MoleculeCollator",
    "NeighborList",
    "PadMolecularBatch",
    "TaskMixtureBatchSampler",
    "TensorFrame",
    "TokenBudgetBatchSampler",
    "TokenBudgetPacker",
    "TotalCharge",
    "as_tensor_frame",
    "bond_index_from_columns",
    "collate_molecules",
    "group_split_indices",
    "stack_angle_index",
    "stack_improper_index",
    "stack_proper_index",
]
