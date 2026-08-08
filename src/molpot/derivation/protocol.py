"""Shared keys and contracts for in-place derivative readouts.

No differentiation logic lives here — only the batch key map, session hooks
(without TensorDict ``set_non_tensor`` — Dynamo-hostile), and how modes invoke
a potential's **energy core** (never the public pipeline).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch
from tensordict import TensorDict

# Nested post-collate keys (in-place writes). The schema is a molix contract;
# the keys live in molix.schema (single owner) and are re-exported here for
# potential-side code.
from molix.schema import (
    ATOMIC_ENERGY_KEY,
    ENERGY_KEY,
    FORCES_KEY,
    POS_KEY,
    has_energy,
    has_forces,
)

__all__ = [
    "ATOMIC_ENERGY_KEY",
    "ENERGY_KEY",
    "FORCES_KEY",
    "POS_KEY",
    "PotentialModule",
    "absorb_model_output",
    "attach_session",
    "call_energy",
    "detach_session",
    "ensure_graphs",
    "get_session",
    "has_energy",
    "has_forces",
    "write_energy",
    "write_forces",
]

# Session side-channel: never batch.set_non_tensor (breaks torch.compile / Dynamo).
_SESSIONS: dict[int, Any] = {}


@runtime_checkable
class PotentialModule(Protocol):
    """Potential that can write energy peers onto a batch."""

    def forward(self, batch: TensorDict) -> TensorDict:
        """Public entry (may compose readouts). Not used by modes for the core."""
        ...


def call_energy(model: Any, batch: TensorDict) -> TensorDict:
    """Run the energy core only.

    Prefer ``model._write_energy(batch)`` when present (full potentials that
    compose readouts in ``forward``). Fall back to ``model(batch)`` for toys
    whose ``forward`` *is* the energy path.
    """
    write = getattr(model, "_write_energy", None)
    if callable(write):
        return write(batch)
    return model(batch)


def ensure_graphs(batch: TensorDict) -> TensorDict:
    """Ensure a ``graphs`` sub-TensorDict exists for writing energy."""
    if "graphs" not in batch.keys():
        batch["graphs"] = TensorDict(batch_size=[])
    return batch


def write_energy(
    batch: TensorDict,
    energy: torch.Tensor,
    *,
    atomic_energy: torch.Tensor | None = None,
) -> TensorDict:
    """Write peer energy keys onto ``batch`` (in-place)."""
    ensure_graphs(batch)
    batch[ENERGY_KEY] = energy
    if atomic_energy is not None:
        batch[ATOMIC_ENERGY_KEY] = atomic_energy
    return batch


def write_forces(batch: TensorDict, forces: torch.Tensor) -> TensorDict:
    """Write peer forces onto ``batch`` (in-place)."""
    batch[FORCES_KEY] = forces
    return batch


def absorb_model_output(batch: TensorDict, out: TensorDict | dict) -> TensorDict:
    """Merge a model return value into ``batch``."""
    if out is batch:
        return batch
    if isinstance(out, TensorDict):
        if has_energy(out):
            batch[ENERGY_KEY] = out[ENERGY_KEY]
        if "atoms" in out.keys() and "energy" in out["atoms"].keys():
            batch[ATOMIC_ENERGY_KEY] = out["atoms", "energy"]
        return batch
    if isinstance(out, dict):
        if "energy" in out:
            write_energy(
                batch,
                out["energy"],
                atomic_energy=out.get("atomic_energy"),
            )
        return batch
    return batch


def attach_session(batch: TensorDict, session: Any) -> None:
    """Store a readout session for ``batch`` (side-channel, not on TensorDict)."""
    _SESSIONS[id(batch)] = session


def get_session(batch: TensorDict) -> Any | None:
    """Return the readout session if EnergyReadout attached one."""
    return _SESSIONS.get(id(batch))


def detach_session(batch: TensorDict) -> None:
    """Drop session for ``batch`` (call after ForceReadout finishes)."""
    _SESSIONS.pop(id(batch), None)
