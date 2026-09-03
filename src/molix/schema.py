"""Post-collate batch-schema keys, shared constants, and cross-layer protocols.

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

For the same one-way-dependency reason it also owns the **structural protocols
that cross the package boundary** — :class:`CellBound` and
:class:`CoreCompilable`. Their implementers live in molpot/molzoo and their
callers in molix, so neither end can name the other's class: molpot already
imports this module and must never import :mod:`molix.md`, which is why the
contracts are declared at the base layer rather than beside the MD driver that
invokes them. :mod:`molix.md.forcefield` re-exports both for MD-side readers;
the contract text stays here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import torch
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
#: Graph-level virial ``W = ∂E/∂ε`` (``batch["graphs", "virial"]``), ``(3, 3)``
#: or ``(B, 3, 3)`` in the energy unit. Stress is ``W / V``.
VIRIAL_KEY: tuple[str, str] = ("graphs", "virial")
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
    2. It does **O(1)** host work — no device synchronisation, no allocation
       that scales with the number of atoms or edges. It runs outside the
       compiled region, on a path that may fire every chunk.
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


@runtime_checkable
class CoreCompilable(Protocol):
    """A potential that can install a compiled build of its own hot core.

    ``compiler`` is **structural**, not nominal: any ``callable -> callable``
    of that shape satisfies it, and :meth:`molix.compile.Compiler.__call__`'s
    PEP 695 generic signature does so without this module importing
    :mod:`molix.compile`.

    The implementer decides *what* to compile — typically one flat bound method
    such as an energy core — and *where* to install the result: an attribute
    the per-step evaluation path already calls, so the caller needs no "is it
    compiled?" branch and pays no per-step dispatch. Repeated calls must not
    recompile; a second call is a no-op unless the implementer is genuinely
    handed a different compiler.
    """

    def compile_core(
        self, compiler: Callable[[Callable[..., Any]], Callable[..., Any]]
    ) -> None: ...
