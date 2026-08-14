"""Data pipeline for molecular ML.

Layering:
    task     — Task / SampleTask / DatasetTask (transform primitives).
    source   — DataSource protocol + in-memory and subset sources.
    pipeline — declarative container: what tasks to run, in what order, with
               what identity; plus ``.run`` / ``.cache`` / ``.cache_key``
               methods for execution and DDP-aware materialisation.

Storage lives in :mod:`molix.store` — ``PipelineSpec.cache`` writes a
:class:`~molix.store.PackedCache` and hands back readers from
:mod:`molix.store.dataset`. Batch construction (collation, sampling,
``DataModule``, post-collate ``BatchTask``) lives in :mod:`molix.batch`.

There is no free-function cache / execute / ddp surface — all orchestration
lives on :class:`PipelineSpec` (and on
:class:`~molix.store.PackedCache` for file IO).
"""

# Pipeline DSL
from molix.data.pipeline import DAGCache, Edge, Node, Pipeline, PipelineSpec

# Data sources
from molix.data.source import DataSource, InMemorySource, SubsetSource

# Task hierarchy
from molix.data.task import (
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
    UnitConvert,
)

__all__ = [
    # Task hierarchy
    "Task",
    "SampleTask",
    "DatasetTask",
    "Runnable",
    # Built-in tasks
    "AtomicDress",
    "ConstantLabel",
    "NeighborList",
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
]
