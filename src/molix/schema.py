"""Post-collate batch-schema keys, shared constants, and cross-layer protocols.

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

Alongside the keys, this module owns the numeric constants that two or more
schema consumers must agree on exactly — :data:`PAIR_DIST_CLAMP_MIN`, and the
dead-edge convention :func:`dead_pair_length` / :func:`dead_pair_shift` that
every producer of padded edges writes and every consumer relies on. They live
here for the same reason the keys do: a constant duplicated at each use site
is a contract only until the copies drift.

For the same one-way-dependency reason it also owns the **structural protocol
that crosses the package boundary** — :class:`CellBound`. Its implementers
live in molpot/molzoo and its caller in molix, so neither end can name the
other's class: molpot already imports this module and must never import
:mod:`molix.md`, which is why the contract is declared at the base layer
rather than beside the MD driver that invokes it. :mod:`molix.md.forcefield`
re-exports it for MD-side readers; the contract text stays here.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from tensordict import TensorDict

from molix.units import DEAD_PAIR_CUTOFF_FACTOR

#: Graph-level total energy written by a potential (``batch["graphs", "energy"]``).
ENERGY_KEY: tuple[str, str] = ("graphs", "energy")
#: Optional graph-level elemental reference sum (``batch["graphs", "e0"]``).
#: Address only; not a residual column. This module does not write the leaf.
E0_KEY: tuple[str, str] = ("graphs", "e0")
#: In-flight residual energy label (``batch["graphs", "energy_residual"]``).
#: Address only; this module does not write the leaf. Distinct from
#: :data:`E0_KEY`. Not in :data:`FLOAT64_LABEL_KEYS` — the residual follows
#: the batch compute dtype after a float64 subtract.
ENERGY_RESIDUAL_KEY: tuple[str, str] = ("graphs", "energy_residual")
#: Graph totals that skip the collate / ``batch_to`` dtype axis. Matched as
#: full NestedKeys (``ENERGY_KEY``, ``E0_KEY``), never the leaf name ``"energy"``.
FLOAT64_LABEL_KEYS: frozenset[tuple[str, str]] = frozenset({ENERGY_KEY, E0_KEY})
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

#: Lower bound for ``pair_dist`` on the flat export paths, in Å.
#:
#: Regularises degenerate near-coincident pairs: ``pair_diff.norm()`` is exactly
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
PAIR_DIST_CLAMP_MIN: float = 1e-6
#: Per-atom forces ``= -∂E/∂pos`` (``batch["atoms", "forces"]``).
FORCES_KEY: tuple[str, str] = ("atoms", "forces")
#: Graph-level virial ``W = ∂E/∂ε`` (``batch["graphs", "virial"]``), ``(1, 3, 3)``
#: in the energy unit. Cauchy stress is ``σ = W / V``: ``ε`` is the symmetric
#: strain and a shear component is **not** doubled. Producers route their raw
#: strain gradient through ``molix.autograd.symmetrize_strain_grad``.
VIRIAL_KEY: tuple[str, str] = ("graphs", "virial")
#: The symmetric strain ``ε`` (``batch["graphs", "strain"]``), ``(1, 3, 3)``,
#: dimensionless — the variable ``molix.autograd.Derivative`` differentiates
#: against for the virial. It is a zero column at the linearisation point;
#: the op creates it when absent and leaves it on the frame.
STRAIN_KEY: tuple[str, str] = ("graphs", "strain")
#: Per-atom positions (``batch["atoms", "pos"]``).
POS_KEY: tuple[str, str] = ("atoms", "pos")
#: Per-atom velocities (``batch["atoms", "vel"]``) in Å/fs — MD trajectory.
VEL_KEY: tuple[str, str] = ("atoms", "vel")
#: Per-atom masses (``batch["atoms", "mass"]``) in amu — MD trajectory.
MASS_KEY: tuple[str, str] = ("atoms", "mass")


def dead_pair_length(cutoff: float) -> float:
    """Displacement of a dead (padding) edge, in Å: ``DEAD_PAIR_CUTOFF_FACTOR · r_cut``.

    The one dead-edge convention of the batch schema. A padded edge — the tail
    of a fixed-capacity MD neighbour list (:class:`molix.md.NeighborList`), the
    padding rows of a training batch (:class:`molix.data.MoleculePadder`)
    or of an exported fixed-shape forward (:class:`molix.engine.static.StaticForward`)
    — is a self-loop displaced by :func:`dead_pair_shift`, so its ``pair_dist``
    is exactly this length, every cutoff envelope evaluates to zero on it and
    no ``1/r`` term ever sees ``r = 0``. Three producers used to spell it
    separately; a padded batch that one of them wrote and another read is what
    makes one function of the cutoff the contract.

    Args:
        cutoff: Interaction cutoff ``r_cut`` in Å.

    Returns:
        The dead-edge displacement in Å.
    """
    return DEAD_PAIR_CUTOFF_FACTOR * float(cutoff)


def dead_pair_shift(
    cutoff: float, *, dtype: torch.dtype | None = None, device: torch.device | None = None
) -> torch.Tensor:
    """The dead-edge displacement vector ``[dead_pair_length(cutoff), 0, 0]`` ``(3,)``.

    What a producer writes as the ``pair_diff`` (and the periodic ``shifts``)
    of a padding self-loop: with ``pos[t] - pos[s] = 0`` on a self-loop, the
    shift alone carries the displacement, so an op that rebuilds
    ``pair_diff = pos[t] - pos[s] + shifts`` from its own position leaf keeps
    the row dead.

    Args:
        cutoff: Interaction cutoff ``r_cut`` in Å.
        dtype: Tensor dtype (default: torch's default floating dtype).
        device: Tensor device (default: CPU).

    Returns:
        ``(3,)`` tensor in Å.
    """
    return torch.tensor([dead_pair_length(cutoff), 0.0, 0.0], dtype=dtype, device=device)


def has_energy(batch: TensorDict) -> bool:
    """Whether ``batch`` carries a graph-level energy at :data:`ENERGY_KEY`."""
    return "graphs" in batch.keys() and "energy" in batch["graphs"].keys()


def has_forces(batch: TensorDict) -> bool:
    """Whether ``batch`` carries per-atom forces at :data:`FORCES_KEY`."""
    return "atoms" in batch.keys() and "forces" in batch["atoms"].keys()


def has_virial(batch: TensorDict) -> bool:
    """Whether ``batch`` carries a graph-level virial at :data:`VIRIAL_KEY`."""
    return "graphs" in batch.keys() and "virial" in batch["graphs"].keys()


@runtime_checkable
class CellBound(Protocol):
    """A potential that keeps cell-derived state and is told when the cell moves.

    MD hands the live simulation cell to the implementer once when the working
    batch is bound and again after every barostat update — both through
    ``molix.md.driver._TensorDictForce.rebind_cell``, the single owner of cell
    resolution on the path to the potential. A reciprocal-space grid sizing
    itself from the lattice is the motivating implementer.

    Four clauses bind every implementer. This docstring is their only home;
    re-exports must not restate them.

    1. ``bind_cell`` is **idempotent**: calling it twice with the same cell
       leaves the implementer in exactly the state one call produces. MD calls
       it once per barostat update whether or not the lattice actually changed.
    2. It does **O(1)** host work — no allocation, and no host read, that
       scales with the number of atoms or edges. The **first** bind may
       synchronise once to size that state off the lattice (a handful of
       lattice-derived integers — grid counts per axis — read back to Python,
       as ``KSpaceGrid.bind`` and ``CoulombPeriodicPotential.bind_cell`` both
       do); that read is why the shape is fixed here instead of inside the
       traced energy. Every **later**
       bind may read back **at most one scalar** — the single ``bool`` that
       decides clause 4's overflow, computed as a tensor expression so the
       per-axis comparison never leaves the device. Nothing beyond that: a
       later bind runs outside the compiled region, on a path that may fire
       every chunk, and no bind ever happens on an energy call.
    3. It must **re-read** the handed live ``(3, 3)`` tensor on every use and
       **never retain** a copy of its values: molix mutates that exact storage
       in place with ``copy_`` on each barostat update, so a cached clone
       silently freezes the pre-barostat lattice.
    4. On capacity overflow it *may* raise ``RuntimeError`` — the caller lets
       that propagate out of ``MD.run`` — and it must **never truncate**
       silently. A quietly clipped grid or neighbour set is a wrong energy
       reported as a converged one.

    ``bind_cell`` takes the live ``(3, 3)`` cell matrix in Å, rows being the
    lattice vectors. A batched ``(1, 3, 3)`` cell is sliced by the caller, not
    by the implementer.
    """

    def bind_cell(self, cell: torch.Tensor) -> None: ...
