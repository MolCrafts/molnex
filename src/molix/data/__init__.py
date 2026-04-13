"""Data pipeline for molecular ML.

Simple by default. Complex workflows → use molexp.
"""

# Task hierarchy
from molix.data.task import (
    BatchTask,
    DatasetTask,
    Runnable,
    SampleTask,
    Task,
)

# Pipeline DSL
from molix.data.pipeline import PipelineDSL, PipelineSpec, pipeline

# Built-in tasks
from molix.data.tasks import AtomicDress, NeighborList

# Data sources
from molix.data.source import DataSource, InMemorySource, SubsetSource

# Dataset classes
from molix.data.dataset import (
    BaseDataset,
    CacheValidationError,
    CachedDataset,
    MmapDataset,
    SubsetDataset,
    is_cache_ready,
)

# Cache identity helper (workflow-side utility)
from molix.data.identity import compute_cache_identity

# DataModule
from molix.data.datamodule import DataModule, DataModuleProtocol

# Collation
from molix.data.collate import DEFAULT_TARGET_SCHEMA, TargetSchema, collate_molecules

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
    # Pipeline
    "pipeline",
    "PipelineDSL",
    "PipelineSpec",
    # Built-in tasks
    "NeighborList",
    "AtomicDress",
    # Sources
    "DataSource",
    "InMemorySource",
    "SubsetSource",
    # Dataset classes
    "BaseDataset",
    "CachedDataset",
    "MmapDataset",
    "SubsetDataset",
    "CacheValidationError",
    "is_cache_ready",
    # Cache identity
    "compute_cache_identity",
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
