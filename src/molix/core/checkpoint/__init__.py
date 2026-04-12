"""Checkpoint infrastructure for MolNex training.

Provides :class:`Checkpoint` for unified serialisation of all stateful
training objects, :class:`CheckpointBackend` protocol for pluggable storage,
and RNG state utilities for reproducible resume.
"""

from molix.core.checkpoint.backend import CheckpointBackend, TorchSaveBackend
from molix.core.checkpoint.rng import capture_rng_states, restore_rng_states
from molix.core.checkpoint.state import Checkpoint

__all__ = [
    "CheckpointBackend",
    "Checkpoint",
    "TorchSaveBackend",
    "capture_rng_states",
    "restore_rng_states",
]
