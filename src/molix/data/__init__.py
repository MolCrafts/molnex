"""Data pipeline for molecular ML.

Simple by default. Complex workflows → use molexp.
"""

# Task hierarchy
# Collation
from molix.data.collate import DEFAULT_TARGET_SCHEMA, TargetSchema, collate_molecules
from molix.data.datamodule import DataModule, DataModuleProtocol

# Dataset + DataModule
from molix.data.dataset import CachedDataset

# Pipeline DSL
from molix.data.pipeline import PipelineDSL, PipelineSpec, pipeline

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
from molix.data.tasks import AtomicDress, NeighborList

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
    # Dataset + DataModule
    "CachedDataset",
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
