"""Post-collate batch-schema keys shared across the molix stack.

The nested ``atoms / edges / graphs`` TensorDict layout produced by
:func:`molix.data.collate.collate_molecules` is a molix contract (see
CLAUDE.md, "Post-collate batch schema"). The tuple keys under which models
write energy/force outputs live here so every layer — molix execution
utilities and the higher molpot/molzoo packages alike — addresses one schema
without molix ever importing upward. :mod:`molpot.derivation.protocol`
re-exports these names for potential-side code; molix-internal consumers
(:mod:`molix.md`, :mod:`molix.quant`, :mod:`molix.engine`) import them from
here, keeping the one-way package dependency ``molix ← molrep ← molzoo/molpot``
intact.
"""

from __future__ import annotations

from tensordict import TensorDict

#: Graph-level total energy written by a potential (``batch["graphs", "energy"]``).
ENERGY_KEY: tuple[str, str] = ("graphs", "energy")
#: Per-atom energy contributions (``batch["atoms", "energy"]``).
ATOMIC_ENERGY_KEY: tuple[str, str] = ("atoms", "energy")
#: Per-atom forces ``= -∂E/∂pos`` (``batch["atoms", "forces"]``).
FORCES_KEY: tuple[str, str] = ("atoms", "forces")
#: Per-atom positions (``batch["atoms", "pos"]``).
POS_KEY: tuple[str, str] = ("atoms", "pos")


def has_energy(batch: TensorDict) -> bool:
    """Whether ``batch`` carries a graph-level energy at :data:`ENERGY_KEY`."""
    return "graphs" in batch.keys() and "energy" in batch["graphs"].keys()


def has_forces(batch: TensorDict) -> bool:
    """Whether ``batch`` carries per-atom forces at :data:`FORCES_KEY`."""
    return "atoms" in batch.keys() and "forces" in batch["atoms"].keys()
