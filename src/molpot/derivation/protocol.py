"""Shared keys and helpers for writing energies and forces onto a batch.

Every model in this repository communicates through one object: the
*post-collate batch*, a nested :class:`~tensordict.TensorDict` whose ``atoms``
/ ``edges`` / ``graphs`` sub-dictionaries hold per-atom, per-edge and per-graph
tensors. :func:`molix.data.collate.collate_molecules` builds that object and
owns its schema (CLAUDE.md, "Post-collate batch schema"). This module owns only
the handful of keys a *potential* — a model mapping atomic positions to an
energy — writes back into it, plus the helpers that perform those writes in
place.

No differentiation logic lives here: force values arrive already computed.
``F = -dE/dpos`` is taken by the two *modes* (``molpot.derivation.modes.func``
and ``.grad`` — the :mod:`torch.func` and :mod:`torch.autograd` backends). Those
modes call a potential's **energy core**, the private ``_write_energy`` entry
point that computes the energy and nothing else, never the public ``forward``,
which may append further readouts a force pass must not re-run;
:func:`call_energy` is that dispatch.

Readout sessions live in a module-level dictionary keyed by ``id(batch)`` rather
than on the batch itself. TensorDict's ``set_non_tensor`` would place a plain
Python object inside a tensor container, which Dynamo — the Python-bytecode
tracer behind :func:`torch.compile` — cannot trace, so it breaks compilation.
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


def ensure_graphs(batch: TensorDict, num_graphs: int | None = None) -> TensorDict:
    """Ensure a ``graphs`` sub-TensorDict exists for writing energy.

    An existing ``graphs`` namespace is returned untouched — its ``batch_size``
    is never rewritten, even when ``num_graphs`` disagrees with it. The
    canonical producer of that namespace is
    :func:`molix.data.collate.collate_molecules`; this helper only fills the
    gap for batches assembled by hand.

    Args:
        batch: Post-collate root batch, mutated in place.
        num_graphs: Number of graphs ``B``. Creates the namespace with the
            schema-conforming ``batch_size=[B]``. ``None`` (the default)
            creates it with ``batch_size=[]`` instead, which does **not**
            conform to CLAUDE.md's ``"graphs": TensorDict(batch_size=[B])``
            schema: a consumer reading ``batch["graphs"].batch_size[0]``
            (e.g. ``src/molzoo/pinet/potential.py``) raises ``IndexError`` on
            it. The fallback exists only as transitional backward
            compatibility for out-of-tree adapters — every in-tree caller
            passes ``num_graphs``.

    Returns:
        The same ``batch``, with a ``graphs`` namespace guaranteed present.
    """
    if "graphs" not in batch.keys():
        batch["graphs"] = TensorDict(batch_size=[] if num_graphs is None else [num_graphs])
    return batch


def write_energy(
    batch: TensorDict,
    energy: torch.Tensor,
    *,
    atomic_energy: torch.Tensor | None = None,
) -> TensorDict:
    """Write peer energy keys onto ``batch`` (in-place).

    Args:
        batch: Post-collate root batch, mutated in place.
        energy: Per-graph energy ``(B,)`` in eV. A 0-dim tensor is accepted and
            leaves the created ``graphs`` namespace at ``batch_size=[]``.
        atomic_energy: Optional per-atom energy ``(N,)`` in eV, written under
            ``("atoms", "energy")``.

    Returns:
        The same ``batch``.
    """
    # B comes from the static shape of a (B,) energy — no host sync, and the
    # same static-B convention as MACEPotential.energy_core(num_graphs=...).
    ensure_graphs(batch, energy.shape[0] if energy.dim() == 1 else None)
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
