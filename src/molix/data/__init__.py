"""The data layer: an indexable space of molpy ``Frame``\\ s, and the batch a model reads.

One package for one job: the datasets, the batch policies, the collate and
the loader live here together, and :class:`FrameCollator` owns the padder it
composes with.

Surfaces, in the order a sample travels:

* **Dataset** — :class:`BaseDataset` is the reader protocol, :class:`FrameDataset`
  a map-style view whose record is a ``Sequence[Frame]``, :class:`SubsetDataset`
  an index view over either, and :class:`DataMixture` a Dataset of named sources
  mixed by task. :meth:`BaseDataset.nrows_of` is the per-index table length every
  budget is computed from — resolving a dataset's
  :attr:`~BaseDataset.counted_tables` declaration, then its
  :meth:`~BaseDataset.count_rows` verb, then every sample.
* **Batch policies** — :class:`TokenBudgetPacker` / :class:`TokenBudgetBatchSampler`
  pack under per-table budgets, :class:`TaskMixtureBatchSampler` interleaves
  per-task streams, :class:`BatchPlanShard` strides a plan across DDP ranks,
  :class:`BatchSamplerSource` is the protocol a
  :class:`DataModule` calls once per epoch, and :func:`group_split_indices`
  splits indices without splitting a group across sides.
* **Collate** — :func:`collate_molecules` turns samples into the nested batch,
  :class:`MoleculePadder` gives it a static shape, :class:`BatchTask` is a
  post-collate transform, and :class:`FrameCollator` composes the three.
* **Loader** — :class:`DataModule` wraps it all in a ``torch.utils.data.DataLoader``;
  :class:`JointDataModule` runs several of them in lockstep, one batch of each per step.

Acquiring the data — download, parse, preprocess, cache — is MolHub's, not this
package's. The batch container itself is :class:`molix.frame.TensorFrame`, and
the cutoff graph is :class:`molix.neighbor.NeighborList`: one is what the model
reads and the other is geometry, so neither is loading machinery.
"""

from __future__ import annotations

from molix.data.bond_adapter import bond_index_from_columns
from molix.data.charge import TotalCharge
from molix.data.collate import MoleculeCollator, collate_molecules
from molix.data.collator import FrameCollator
from molix.data.datamodule import DataModule, DataModuleProtocol
from molix.data.dataset import BaseDataset, SubsetDataset
from molix.data.frame_dataset import FrameDataset
from molix.data.group_split import group_split_indices
from molix.data.inlier import ForceInlierWeight
from molix.data.joint import JointDataModule
from molix.data.mixture import DataMixture
from molix.data.pad import MoleculePadder
from molix.data.prepared import PreparedMmap, PreparedRow, PreparedRows
from molix.data.profile import ProfileIndex
from molix.data.relation import PairRelation
from molix.data.resident import ResidentValidation
from molix.data.residual import EnergyResidual
from molix.data.sampler import (
    BatchPlanShard,
    BatchSamplerSource,
    TaskMixtureBatchSampler,
    TokenBudgetBatchSampler,
    TokenBudgetPacker,
)
from molix.data.task import BatchTask
from molix.data.valence_columns import (
    IMPROPER_ROWS_PER_CENTRE,
    cyclic_impropers,
    stack_angle_index,
    stack_improper_index,
    stack_proper_index,
)

__all__ = [
    "BaseDataset",
    "BatchPlanShard",
    "BatchSamplerSource",
    "BatchTask",
    "DataMixture",
    "DataModule",
    "DataModuleProtocol",
    "EnergyResidual",
    "ForceInlierWeight",
    "FrameCollator",
    "FrameDataset",
    "JointDataModule",
    "MoleculeCollator",
    "MoleculePadder",
    "PreparedMmap",
    "PreparedRow",
    "PreparedRows",
    "ResidentValidation",
    "SubsetDataset",
    "TaskMixtureBatchSampler",
    "TokenBudgetBatchSampler",
    "TokenBudgetPacker",
    "PairRelation",
    "ProfileIndex",
    "TotalCharge",
    "bond_index_from_columns",
    "collate_molecules",
    "group_split_indices",
    "IMPROPER_ROWS_PER_CENTRE",
    "cyclic_impropers",
    "stack_angle_index",
    "stack_improper_index",
    "stack_proper_index",
]
