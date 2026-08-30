"""Post-collate batch-schema keys shared across the molix stack.

The nested ``atoms / edges / graphs`` TensorDict layout produced by
:func:`molix.batch.collate.collate_molecules` is a molix contract (see
CLAUDE.md, "Post-collate batch schema"). The tuple keys under which models
write energy/force outputs live here so every layer — molix execution
utilities and the higher molpot/molzoo packages alike — addresses one schema
without molix ever importing upward. :mod:`molpot.derivation.protocol`
re-exports these names for potential-side code; molix-internal consumers
(:mod:`molix.md`, :mod:`molix.quant`, :mod:`molix.engine`) import them from
here, keeping the one-way package dependency ``molix ← molrep ← molzoo/molpot``
intact.

Alongside the keys, this module owns the numeric constants that two or more
schema consumers must agree on exactly — currently
:data:`EDGE_DIST_CLAMP_MIN`. They live here for the same reason the keys do:
a constant duplicated at each use site is a contract only until the copies
drift.
"""

from __future__ import annotations

from tensordict import TensorDict

#: Graph-level total energy written by a potential (``batch["graphs", "energy"]``).
ENERGY_KEY: tuple[str, str] = ("graphs", "energy")
#: Per-atom energy contributions (``batch["atoms", "energy"]``).
ATOMIC_ENERGY_KEY: tuple[str, str] = ("atoms", "energy")
#: Flat alias of :data:`ATOMIC_ENERGY_KEY` — the *same* quantity addressed
#: without the nesting. The correspondence is exact and one-way::
#:
#:     nested (post-collate TensorDict) : ("atoms", "energy")  → ATOMIC_ENERGY_KEY
#:     flat   (plain dict / export)     : "atomic_energy"      → ATOMIC_ENERGY_FLAT_KEY
#:
#: A potential returning a nested ``TensorDict`` is read at the nested key; one
#: returning a plain dict — and the LAMMPS export's ``outputs`` tensor name — is
#: read at this flat one (``molpot.derivation.protocol.absorb_model_output``,
#: ``molix.engine.adapter``). These two spellings are the complete set: a third
#: (``node_energies`` and friends) must never appear, because a reader looking
#: for the wrong one gets ``None`` and drops the per-atom energy silently.
#: Since spec ``lammps-iface-05-pbc`` that silence would be a silent *zero*
#: partition energy in LAMMPS, the per-atom energy being the only source of
#: ``eng_vdwl`` there.
ATOMIC_ENERGY_FLAT_KEY: str = "atomic_energy"

#: Lower bound for ``edge_dist`` on the flat export paths, in Å.
#:
#: Regularises degenerate near-coincident pairs: ``edge_diff.norm()`` is exactly
#: zero for two coincident atoms (and denormally small just above it), which
#: propagates a ``0/0`` NaN into any radial basis or ``1/r`` term and, through
#: the backward pass, into every force. Clamping the distance keeps such a pair
#: finite and inert instead of poisoning the whole step.
#:
#: Shared — deliberately — by ``molix.engine.adapter.MolnexTensorDictAdapter``
#: (dynamic path) and ``molix.engine.static.StaticForward`` (CUDA-graph path),
#: which are documented as equivalent constructions of the same batch and used
#: to disagree (1e-6 vs 1e-8). One constant is what makes that documented
#: equivalence checkable; the more conservative value wins.
EDGE_DIST_CLAMP_MIN: float = 1e-6
#: Per-atom forces ``= -∂E/∂pos`` (``batch["atoms", "forces"]``).
FORCES_KEY: tuple[str, str] = ("atoms", "forces")
#: Per-atom positions (``batch["atoms", "pos"]``).
POS_KEY: tuple[str, str] = ("atoms", "pos")
#: Per-atom velocities (``batch["atoms", "vel"]``) in Å/fs — MD trajectory.
VEL_KEY: tuple[str, str] = ("atoms", "vel")
#: Per-atom masses (``batch["atoms", "mass"]``) in amu — MD trajectory.
MASS_KEY: tuple[str, str] = ("atoms", "mass")


def has_energy(batch: TensorDict) -> bool:
    """Whether ``batch`` carries a graph-level energy at :data:`ENERGY_KEY`."""
    return "graphs" in batch.keys() and "energy" in batch["graphs"].keys()


def has_forces(batch: TensorDict) -> bool:
    """Whether ``batch`` carries per-atom forces at :data:`FORCES_KEY`."""
    return "atoms" in batch.keys() and "forces" in batch["atoms"].keys()
