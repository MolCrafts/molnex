"""Interaction-contract vocabulary (spec requirement S1) — torch-free leaf.

A *style* is one named functional form of an interaction, keyed
``"<category>:<name>"`` — ``"bond:harmonic"`` (a spring between two bonded
atoms), ``"pair:lj/cut"`` (Lennard-Jones between every pair inside a cutoff
radius), ``"pair:coulomb"`` (electrostatics between partial charges). Styles are
registered in :class:`molpot.forcefield.PotentialMeta` and are what a
:class:`~molpot.forcefield.ForceField` declares.

Every registered style carries one :class:`InteractionContract` declaring, in
machine-readable form, what the style is and what it may be composed with:

* **inputs** — the batch keys it reads (``"atoms.pos"``, ``"atoms.q"``,
  ``"bonds.bond_index"``, …), so a missing input is a construction-time error
  rather than a ``KeyError`` inside a kernel;
* **transformation law** — :class:`ParamFrame`, carried per parameter on
  ``ParamSpec.frame``, distinguishing a scalar from a lab-frame vector, a
  local-frame vector, and a rank-2 tensor;
* **cutoff or long-range solver** — :class:`CutoffKind`, from an untruncated
  bonded term through a switched pair cutoff to an Ewald or GTO k-space sum;
* **exclusion class** — :class:`ExclusionClass`, keeping permanent-electrostatic
  exclusions separate from induced ones (they are different tables);
* **supported derivatives** — :class:`Derivative`, so a model asked for a
  Hessian it cannot produce is rejected rather than silently approximated;
* **named terms** — :class:`TermSpec`, mapping each ``term_energies()`` key to
  its label in the *S-decomposition* — the split of the total energy into
  physically separate contributions,
  ``E = E_local + E_disp + E_perm + E_pol + E_CT`` (bonded and short-range
  repulsion, dispersion, permanent electrostatics, polarization, charge
  transfer) — so the total partitions exactly and nothing is counted twice.

``InteractionContract.response`` is meaningful **only** for a term whose
contribution is :attr:`Contribution.POL`; it names how the induced response is
obtained — ``"variational"`` (minimise the polarization functional),
``"direct"`` (one-shot ``μ = αE``), or ``"finite-iteration"`` (a fixed,
non-converged number of steps). For every other contribution it stays ``None``.

These are stdlib value objects — frozen, hashable and torch-free — because they
are serialised into ``ForceFieldSchemaView.as_dict()`` (and, once
``composable-a0-04`` lands it, into the versioned model manifest), and are read
by code that must not import torch.

Reference:
    ``.claude/notes/composable-potentials.md`` §"S1 contract"
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "Contribution",
    "CutoffKind",
    "Derivative",
    "ExclusionClass",
    "InteractionContract",
    "ParamFrame",
    "TermSpec",
]


class Contribution(StrEnum):
    """S-decomposition bucket of a named energy term.

    The five members are exhaustive: the total energy partitions as
    ``E = E_local + E_disp + E_perm + E_pol + E_CT``, and every registered
    term maps to exactly one bucket, so no contribution is double counted.

    Attributes:
        LOCAL: Bonded and short-range repulsive energy (the repulsive wall of a
            Lennard-Jones pair included).
        DISP: London dispersion (the attractive ``-C6/r^6`` tail).
        PERM: Electrostatics of the permanent charge distribution.
        POL: Polarization — the energy of the induced response.
        CT: Charge transfer between fragments.
    """

    LOCAL = "local"
    DISP = "disp"
    PERM = "perm"
    POL = "pol"
    CT = "ct"


class CutoffKind(StrEnum):
    """How a style treats interaction range.

    Attributes:
        NONE: Untruncated — a bonded term evaluated on a fixed topology.
        HARD: Truncated at a cutoff with no smoothing (energy is discontinuous
            at the cutoff; forces are not the gradient of the energy there).
        SWITCHED: Truncated through a switching function, continuous in energy
            and force at the cutoff.
        EWALD: Real-space part plus a reciprocal-space lattice sum (Ewald, PME
            or P3M mesh variants).
        GTO_KSPACE: Reciprocal-space sum over a Gaussian-type-orbital density.
    """

    NONE = "none"
    HARD = "hard"
    SWITCHED = "switched"
    EWALD = "ewald"
    GTO_KSPACE = "gto_kspace"


class ExclusionClass(StrEnum):
    """Which exclusion/scaling table a style consumes.

    Permanent and induced exclusions are deliberately distinct: a pair excluded
    from permanent electrostatics is not necessarily excluded from the induced
    response, and conflating them is a physics error, not a bookkeeping detail.

    Attributes:
        NONE: No topological exclusions (a bonded term acts on its own index
            list).
        PERMANENT: Uses the permanent-electrostatics scaling table
            (``ForceField.scaling``).
        INDUCED: Uses the induced-response scaling table
            (``ForceField.induced_scaling``).
    """

    NONE = "none"
    PERMANENT = "permanent"
    INDUCED = "induced"


class Derivative(StrEnum):
    """A derivative of the energy that a style can supply.

    A style declares only what it can actually produce, so a caller asking for
    a derivative it does not support is refused rather than handed a silent
    approximation. Units below are the ``real`` unit system molnex uses for
    classical force fields (energy kcal/mol, length Å).

    Attributes:
        FORCES: Atomic forces ``F = -∂E/∂pos``, shape ``(N, 3)`` over the ``N``
            atoms, in kcal mol⁻¹ Å⁻¹.
        VIRIAL: The response of the energy to a *strain* — a uniform
            deformation ``ε`` of the simulation cell. molnex uses the
            symmetric-strain Cauchy convention throughout: ``W = ∂E/∂ε`` at
            ``ε = 0``, shape ``(3, 3)`` in the energy unit (kcal/mol), from
            which the stress tensor is ``W / V`` with ``V`` the cell volume.
        HESSIAN: Second derivative ``∂²E/∂pos²``, shape ``(3N, 3N)``, in
            kcal mol⁻¹ Å⁻².
    """

    FORCES = "forces"
    VIRIAL = "virial"
    HESSIAN = "hessian"


class ParamFrame(StrEnum):
    """Transformation law of a parameter under rotation of the lab frame.

    Named ``ParamFrame`` rather than ``Frame``: the short name collides with
    ``molpy.Frame`` (a molecular snapshot), which is a wholly unrelated concept
    and is imported alongside this module all over the ForceField layer.

    Attributes:
        SCALAR: Rotation invariant (a force constant, a charge).
        VECTOR_LAB: Rotates with the lab frame (a dipole predicted directly in
            lab coordinates).
        VECTOR_LOCAL: Defined in a molecule-local frame and rotated into the lab
            frame before use.
        RANK2: A rank-2 tensor (a quadrupole, a polarizability tensor).
    """

    SCALAR = "scalar"
    VECTOR_LAB = "vector_lab"
    VECTOR_LOCAL = "vector_local"
    RANK2 = "rank2"


@dataclass(frozen=True)
class TermSpec:
    """One named energy term and the S-decomposition bucket it belongs to.

    Attributes:
        key: The ``Potentials.term_energies()`` key of this term, e.g.
            ``"lj_repulsion"``.
        contribution: The bucket ``key`` sums into.
    """

    key: str
    contribution: Contribution


@dataclass(frozen=True)
class InteractionContract:
    """The S1 declaration a registered style carries.

    Frozen and hashable: contracts are compared and deduplicated when a
    composition is validated, and serialised by
    :meth:`~molpot.forcefield.schema.ForceFieldSchemaView.as_dict` (and into the
    versioned model manifest once ``composable-a0-04`` adds it).

    Attributes:
        inputs: Batch keys the style reads, written as ``"<namespace>.<field>"``
            — ``"atoms.pos"`` names ``batch["atoms", "pos"]`` in the nested
            post-collate ``TensorDict``. Example:
            ``("atoms.pos", "bonds.bond_index")``.
        cutoff: Range treatment or long-range solver.
        exclusion: Which exclusion/scaling table the style consumes.
        derivatives: Derivative orders the style supports; unordered, so two
            otherwise identical contracts compare equal regardless of the order
            they were declared in.
        terms: Named energy terms with their S-decomposition labels.
        response: How the induced response is obtained — ``"variational"``,
            ``"direct"`` or ``"finite-iteration"``. Meaningful only for a style
            with a :attr:`Contribution.POL` term; ``None`` otherwise.
    """

    inputs: tuple[str, ...]
    cutoff: CutoffKind
    exclusion: ExclusionClass
    derivatives: frozenset[Derivative]
    terms: tuple[TermSpec, ...]
    response: str | None = None
