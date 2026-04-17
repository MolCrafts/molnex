"""Data pipeline for molecular ML.

Layering:
    task     — Task / SampleTask / DatasetTask / BatchTask (transform primitives).
    source   — DataSource protocol + in-memory and subset sources.
    pipeline — declarative container: what tasks to run, in what order, with
               what identity.
    execute  — run(), transform(), collect_task_states(): executes a pipeline.
    cache    — cache(), cache_key(), is_ready(), save(), load(): short-lived
               scratch cache of pipeline output.
    ddp      — rank(), wait_for_ready(): opt-in distributed coordination.
    dataset  — MmapDataset, CachedDataset, SubsetDataset: readers over cache
               files.

``execute`` / ``cache`` / ``ddp`` are *sub-module imports* — they belong to
the workflow layer and are kept off the top-level export surface to keep
``from molix.data import *`` focused on the pieces every training script
needs.
"""

# Task hierarchy
from molix.data.task import (
    BatchTask,
    DatasetTask,
    Runnable,
    SampleTask,
    Task,
)

# Built-in tasks
from molix.data.tasks import AtomicDress, NeighborList

# Data sources
from molix.data.source import DataSource, InMemorySource, SubsetSource

# Pipeline DSL
from molix.data.pipeline import Pipeline, PipelineSpec, TaskEntry

# Dataset classes
from molix.data.dataset import (
    BaseDataset,
    CachedDataset,
    MmapDataset,
    SubsetDataset,
)

# DataModule
from molix.data.datamodule import DataModule, DataModuleProtocol

# Collation
from molix.data.collate import (
    DEFAULT_TARGET_SCHEMA,
    TargetSchema,
    collate_molecules,
)

# Types
from molix.data.types import (
    AtomData,
    EdgeData,
    EdgeRepEdges,
    GraphBatch,
    GraphData,
    NodeRepAtoms,
)

__all__ = [
    # Task hierarchy
    "Task",
    "SampleTask",
    "DatasetTask",
    "BatchTask",
    "Runnable",
    # Built-in tasks
    "NeighborList",
    "AtomicDress",
    # Sources
    "DataSource",
    "InMemorySource",
    "SubsetSource",
    # Pipeline
    "Pipeline",
    "PipelineSpec",
    "TaskEntry",
    # Dataset classes
    "BaseDataset",
    "CachedDataset",
    "MmapDataset",
    "SubsetDataset",
    # DataModule
    "DataModule",
    "DataModuleProtocol",
    # Collation
    "collate_molecules",
    "TargetSchema",
    "DEFAULT_TARGET_SCHEMA",
    # Types
    "AtomData",
    "EdgeData",
    "GraphData",
    "GraphBatch",
    "NodeRepAtoms",
    "EdgeRepEdges",
]
