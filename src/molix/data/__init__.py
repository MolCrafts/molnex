"""Data pipeline for molecular ML.

Layering:
    task     — Task / SampleTask / DatasetTask / BatchTask (transform primitives).
    source   — DataSource protocol + in-memory and subset sources.
    pipeline — declarative container: what tasks to run, in what order, with
               what identity; plus ``.run`` / ``.cache`` / ``.cache_key``
               methods for execution and DDP-aware materialisation.

Storage lives in :mod:`molix.store` — ``PipelineSpec.cache`` writes a
:class:`~molix.store.PackedCache` and hands back readers from
:mod:`molix.store.dataset`.

There is no free-function cache / execute / ddp surface — all orchestration
lives on :class:`PipelineSpec` (and on
:class:`~molix.store.PackedCache` for file IO).
"""

# Task hierarchy
# Collation
from molix.data.collate import (
    DEFAULT_TARGET_SCHEMA,
    MoleculeCollator,
    TargetSchema,
    collate_molecules,
)

# DataModule
from molix.data.datamodule import DataModule, DataModuleProtocol
from molix.data.group_split import group_split_indices

# Pipeline DSL
from molix.data.pipeline import DAGCache, Edge, Node, Pipeline, PipelineSpec

# Batch sampling
from molix.data.sampler import (
    BatchSamplerSource,
    TaskMixtureBatchSampler,
    TokenBudgetBatchSampler,
    TokenBudgetPacker,
)

# Data sources
from molix.data.source import DataSource, InMemorySource, SubsetSource
from molix.data.task import (
    BatchTask,
    DatasetTask,
    Runnable,
    SampleTask,
    Task,
)

# Built-in tasks
from molix.data.tasks import (
    AtomicDress,
    ConstantLabel,
    NeighborList,
    PadMolecularBatch,
    UnitConvert,
)

__all__ = [
    # Batch sampling
    "BatchSamplerSource",
    "TaskMixtureBatchSampler",
    "TokenBudgetBatchSampler",
    "TokenBudgetPacker",
    # Task hierarchy
    "Task",
    "SampleTask",
    "DatasetTask",
    "BatchTask",
    "Runnable",
    # Built-in tasks
    "AtomicDress",
    "ConstantLabel",
    "NeighborList",
    "PadMolecularBatch",
    "UnitConvert",
    # Sources
    "DataSource",
    "InMemorySource",
    "SubsetSource",
    # Pipeline
    "DAGCache",
    "Edge",
    "Node",
    "Pipeline",
    "PipelineSpec",
    # DataModule
    "DataModule",
    "DataModuleProtocol",
    # Collation
    "collate_molecules",
    "MoleculeCollator",
    "TargetSchema",
    "DEFAULT_TARGET_SCHEMA",
    # Group splits
    "group_split_indices",
]
