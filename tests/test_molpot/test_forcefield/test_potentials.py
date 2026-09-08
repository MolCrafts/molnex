"""Tests for Potentials force methods (analytic bonded + LJ).

Mirrors ``src/molpot/forcefield/potentials.py``.

``Potentials.calc_energy_forces`` evaluates bonded terms (bond / angle /
proper / improper) and the LJ term with closed-form forces; Coulomb (open and
periodic) falls back to the autograd backend
(:func:`molpot.derivation.autograd_forces_from_energy`), which has no cheap
closed form. No method hand-rolls ``torch.autograd.grad``. Numerics are
pinned here against the analytic harmonic bond so the generalization stays
behaviour-preserving.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Callable

import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from molix.schema import ENERGY_KEY, CellBound
from molpot.forcefield import ForceField, Potentials
from molpot.forcefield.schema import PotentialMeta
from molpot.potentials.elec.prefactors import kcalmol_A
from tests.conftest import (
    BUTANE_EPSILON,
    BUTANE_N_ATOMS,
    BUTANE_PAIR_BOX,
    BUTANE_PAIR_TOPOLOGY_COLUMNS,
    BUTANE_Q,
    BUTANE_SIGMA,
    butane_pair_batch,
    butane_pair_class_i_forcefield,
    butane_pair_positions,
    place_edges,
)

DTYPE = torch.float64

# Analytic two-atom harmonic bond: E = ½k(r−r0)², k = 100 kcal mol⁻¹ Å⁻²,
# r0 = 1.0 Å, r = 1.5 Å along x̂ → E = 12.5 kcal/mol, |F| = k(r−r0) = 50.
BOND_K = 100.0
BOND_R0 = 1.0
BOND_R = 1.5
GOLDEN_ENERGY = 12.5
GOLDEN_FORCE_X = 50.0

FORCE_SEAM = "src/molpot/derivation/force.py"

#: Cubic edge (Å) of the tiny periodic Polar fixture. Its ``capacity_factor =
#: 1.2`` k-grid bins to ``N_c = 4`` per axis (``x = L·k_cutoff/2π = 2.94``), so
#: a 1.5x barostat expansion still fits (``floor(1.5·2.94) = 4``) and a 2.0x one
#: overflows (``floor(2·2.94) = 5``) — the two sides of the capacity contract.
POLAR_CELL_EDGE = 10.0
#: Isotropic expansion that stays inside the k-grid capacity above.
CELL_WITHIN_CAPACITY = 1.5
#: Isotropic expansion that overflows it.
CELL_OVER_CAPACITY = 2.0
#: Displacement of a **single** atom along +x (Å) — a rigid translation of the
#: whole cluster is an exact symmetry of the Polar energy and would prove
#: nothing about the energy being refreshed.
NUDGE = 0.05
#: Energy change (eV) that one ``NUDGE`` must clear to count as "recomputed"
#: rather than floating-point noise on a cached value.
ENERGY_CHANGE_MIN = 1e-6

# ---------------------------------------------------------------------------
# Periodic Coulomb fixture — four ions in a cubic box, one bonded pair
# ---------------------------------------------------------------------------

#: Cubic edge (Å). Every pair below is inside the minimum-image limit ``L/2``.
COULOMB_EDGE = 8.0
#: Real-space cutoff ``r_c`` (Å): past the largest pair (3.32 Å) and inside
#: ``L/2 = 4``, so no pair sits near the cutoff and a 1e-4 Å finite-difference
#: step can never flip a row of the list on or off.
COULOMB_CUTOFF = 3.8
#: Ewald splitting width ``sigma`` (Å) and shortest reciprocal wavelength (Å).
#: The per-axis index ``L/lambda = 10.67`` is well away from an integer, so a
#: strained box asks for the same grid as the unstrained one — ``bind_cell``
#: freezes the shape anyway, and this keeps the fixture honest without it.
COULOMB_SMEARING = 0.8
COULOMB_LR_WAVELENGTH = 0.75
#: Mesh spacing (Å) of the ``"pme"`` route — the value the electrostatics
#: fixtures in ``test_coulomb.py`` use, so both files hand the mesh sizing rule
#: the same grid request for a box of this size.
COULOMB_MESH_SPACING = 1.0
#: Capacity of the padded half list. Six live rows plus two dead ones — the
#: coincident self-loops :class:`molix.md.NeighborList` pads with.
COULOMB_LIST_CAPACITY = 8
#: Wrapped positions (Å). Atom 3 sits at ``z = 5.6``, i.e. ``-2.4`` in the
#: minimum image, so three of the six pairs carry a **non-zero lattice shift**.
#: A fixture whose shifts are all zero cannot see whether the shift strains
#: with the cell, which is exactly what the virial depends on.
COULOMB_POSITIONS = (
    (0.0, 0.0, 0.0),
    (2.2, 0.0, 0.0),
    (0.0, 2.3, 0.0),
    (0.0, 0.0, 5.6),
)
#: Alternating ±1 charges — a neutral cell, so no background correction is
#: doing the work.
COULOMB_CHARGES = (1.0, -1.0, 1.0, -1.0)
#: Central-difference step (Å for positions, dimensionless for strain).
FD_STEP = 1e-4
#: ``max |F_auto - F_fd|`` (kcal mol⁻¹ Å⁻¹). The measured gap is truncation
#: dominated — third derivative times ``dh²``, about 1.4e-7 here.
FD_FORCE_TOL = 1e-6
#: ``max |W_auto - W_fd|`` (kcal/mol) on the virial ``W = dE/deps``.
FD_VIRIAL_TOL = 1e-5


def _two_atom_bond_batch(*, r: float = BOND_R) -> TensorDict:
    pos = torch.tensor([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=DTYPE)
    return TensorDict(
        {
            "atoms": TensorDict(
                {
                    "pos": pos,
                    "Z": torch.tensor([1, 1], dtype=torch.long),
                    "batch": torch.zeros(2, dtype=torch.long),
                },
                batch_size=[2],
            ),
            "bonds": TensorDict(
                {
                    "atomi": torch.tensor([0], dtype=torch.long),
                    "atomj": torch.tensor([1], dtype=torch.long),
                },
                batch_size=[1],
            ),
        },
        batch_size=[],
    )


def _two_molecule_bond_batch() -> TensorDict:
    """Two copies of the two-atom molecule packed into one batch (B = 2)."""
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [BOND_R, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [BOND_R, 0.0, 0.0],
        ],
        dtype=DTYPE,
    )
    return TensorDict(
        {
            "atoms": TensorDict(
                {
                    "pos": pos,
                    "Z": torch.tensor([1, 1, 1, 1], dtype=torch.long),
                    "batch": torch.tensor([0, 0, 1, 1], dtype=torch.long),
                },
                batch_size=[4],
            ),
            "bonds": TensorDict(
                {
                    "atomi": torch.tensor([0, 2], dtype=torch.long),
                    "atomj": torch.tensor([1, 3], dtype=torch.long),
                },
                batch_size=[2],
            ),
        },
        batch_size=[],
    )


def _angle_batch() -> TensorDict:
    """Three atoms carrying one ``angles`` entry — topology present, parameters absent."""
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [BOND_R, 0.0, 0.0], [BOND_R, BOND_R, 0.0]],
        dtype=DTYPE,
    )
    return TensorDict(
        {
            "atoms": TensorDict(
                {
                    "pos": pos,
                    "Z": torch.tensor([1, 1, 1], dtype=torch.long),
                    "batch": torch.zeros(3, dtype=torch.long),
                },
                batch_size=[3],
            ),
            "angles": TensorDict(
                {
                    "atomi": torch.tensor([0], dtype=torch.long),
                    "atomj": torch.tensor([1], dtype=torch.long),
                    "atomk": torch.tensor([2], dtype=torch.long),
                },
                batch_size=[1],
            ),
        },
        batch_size=[],
    )


def _torsion_batch(namespace: str) -> TensorDict:
    """Four atoms carrying one ``propers`` / ``impropers`` entry (improper centre = atomi)."""
    pos = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.5, 0.5],
        ],
        dtype=DTYPE,
    )
    return TensorDict(
        {
            "atoms": TensorDict(
                {
                    "pos": pos,
                    "Z": torch.tensor([1, 1, 1, 1], dtype=torch.long),
                    "batch": torch.zeros(4, dtype=torch.long),
                },
                batch_size=[4],
            ),
            namespace: TensorDict(
                {
                    "atomi": torch.tensor([0], dtype=torch.long),
                    "atomj": torch.tensor([1], dtype=torch.long),
                    "atomk": torch.tensor([2], dtype=torch.long),
                    "atoml": torch.tensor([3], dtype=torch.long),
                },
                batch_size=[1],
            ),
        },
        batch_size=[],
    )


def _bond_potentials(k: torch.Tensor | None = None, *, n_bonds: int = 1) -> Potentials:
    """Harmonic-bond-only :class:`Potentials` with optional differentiable ``k``."""
    force_constant = torch.full((n_bonds,), BOND_K, dtype=DTYPE) if k is None else k
    return (
        ForceField()
        .def_bondstyle("harmonic")
        .with_params(
            bonds={
                "k": force_constant,
                "r0": torch.full((n_bonds,), BOND_R0, dtype=DTYPE),
            }
        )
        .to_potentials()
    )


#: Charges (e), well depth (kcal/mol) and size (Å) of the three-atom Class-I
#: fixture below. Atom 2 sits off the bond axis so the two unexcluded pairs
#: (0–2, 1–2) give LJ and Coulomb a non-zero value while the bonded 1–2 pair
#: stays excluded by the default :class:`NonbondedScaling`.
CHARGES = (-0.5, 0.25, 0.25)
EPSILON = 0.2
SIGMA = 3.0
LJ_CUTOFF = 10.0

#: ``lj/cut`` cutoff (Å) of the empty-topology fixture, whose only pair sits at
#: 10 Å — well outside it, so the term must price that pair at zero.
EMPTY_TOPOLOGY_CUTOFF = 3.0
#: Build radius (Å) of that fixture's placed list. Deliberately *past* the
#: style's cutoff so the list carries the 10 Å pair as a row the liveness rule
#: kills, rather than never producing it: the zero then comes from the mask the
#: pair terms multiply by, which is the thing under test.
EMPTY_TOPOLOGY_LIST_CUTOFF = 12.0

#: Switching window of the LJ style: the taper starts at ``r_s`` and reaches
#: zero at ``LJ_CUTOFF``. ``SWITCH_R`` is the window midpoint, where
#: ``x = (r − r_s)/(r_c − r_s) = ½`` and the quintic switch takes the exact
#: value ``S = 1 − ⅛(10 − 7.5 + 1.5) = ½`` — an analytic golden, not a value
#: read back out of :meth:`LJCutPotential._switch`.
SWITCH_START = 9.0
SWITCH_R = 9.5
SWITCH_AT_MIDPOINT = 0.5


def _lj_pair_forcefield(*, switch_start: float | None = None) -> ForceField:
    """Two-atom ``lj/cut`` field, optionally tapered, declared through ``def_pairstyle``.

    No bond / angle / proper namespace goes with it, so the single pair is in
    no exclusion relation and reads the unscaled ``1.0``: the only thing that
    can change its energy is the switching window.
    """
    options: dict[str, float] = {"cutoff": LJ_CUTOFF}
    if switch_start is not None:
        options["switch_start"] = switch_start
    return (
        ForceField(name="lj-pair")
        .def_pairstyle("lj/cut", **options)
        .with_params(
            atoms={
                "epsilon": torch.full((2,), EPSILON, dtype=DTYPE),
                "sigma": torch.full((2,), SIGMA, dtype=DTYPE),
            }
        )
    )


def _lj_pair_batch(*, r: float) -> TensorDict:
    """Two unbonded atoms ``r`` Å apart along x̂ — one nonbonded pair, nothing else.

    Carries the placed ``edges`` namespace :meth:`Potentials._placed_pairs`
    consumes, built at :data:`LJ_CUTOFF` — the cutoff ``_lj_pair_forcefield``
    declares, so the list holds exactly the row that style may read.
    """
    pos = torch.tensor([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=DTYPE)
    batch = TensorDict(
        {
            "atoms": TensorDict(
                {
                    "pos": pos,
                    "Z": torch.tensor([18, 18], dtype=torch.long),
                    "batch": torch.zeros(2, dtype=torch.long),
                },
                batch_size=[2],
            )
        },
        batch_size=[],
    )
    return place_edges(batch, cutoff=LJ_CUTOFF)


def _class_i_forcefield() -> ForceField:
    """Harmonic bond + ``lj/cut`` + ``coulomb``, every parameter table filled."""
    return (
        ForceField(name="class-i")
        .def_bondstyle("harmonic")
        .def_pairstyle("lj/cut", cutoff=LJ_CUTOFF)
        .def_pairstyle("coulomb")
        .with_params(
            bonds={
                "k": torch.full((1,), BOND_K, dtype=DTYPE),
                "r0": torch.full((1,), BOND_R0, dtype=DTYPE),
            },
            atoms={
                "epsilon": torch.full((3,), EPSILON, dtype=DTYPE),
                "sigma": torch.full((3,), SIGMA, dtype=DTYPE),
                "q": torch.tensor(CHARGES, dtype=DTYPE),
            },
        )
    )


def _class_i_batch() -> TensorDict:
    """Three atoms, one bond — bonded, LJ and Coulomb all contribute.

    The placed list is built at :data:`LJ_CUTOFF`, which is past the largest of
    the three separations (3.35 Å), so it holds **every** pair: ``lj/cut``
    declares that cutoff and the open ``pair:coulomb`` declares none, and both
    read the same rows.
    """
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [BOND_R, 0.0, 0.0], [0.0, 3.0, 0.0]],
        dtype=DTYPE,
    )
    batch = TensorDict(
        {
            "atoms": TensorDict(
                {
                    "pos": pos,
                    "Z": torch.tensor([1, 1, 1], dtype=torch.long),
                    "batch": torch.zeros(3, dtype=torch.long),
                },
                batch_size=[3],
            ),
            "bonds": TensorDict(
                {
                    "atomi": torch.tensor([0], dtype=torch.long),
                    "atomj": torch.tensor([1], dtype=torch.long),
                },
                batch_size=[1],
            ),
        },
        batch_size=[],
    )
    return place_edges(batch, cutoff=LJ_CUTOFF)


#: ``lj/cut`` cutoff (Å) of the open-boundary butane fixture. Its charges /
#: well depths / sizes are :data:`~tests.conftest.BUTANE_Q` and friends, owned
#: by ``tests/conftest.py`` alongside the geometry and topology they belong to.
BUTANE_CUTOFF = 10.0


#: Build radius (Å) of the charged fixture's placed list — past its single
#: 2 Å pair, so the open ``pair:coulomb`` prices the pair it would enumerate.
CHARGED_CUTOFF = 10.0


def _anion_potentials() -> Potentials:
    """Open ``pair:coulomb`` over two atoms of +0.5 e — a graph of net +1 e."""
    return (
        ForceField(name="anion")
        .def_pairstyle("coulomb")
        .with_params(atoms={"q": torch.tensor([0.5, 0.5], dtype=DTYPE)})
        .to_potentials()
    )


def _charged_batch(*, total_charge: float | None) -> TensorDict:
    """Two atoms carrying +0.5 e each; ``graphs.total_charge`` optional.

    Carries a placed list, so the Coulomb term can price it: a batch the charge
    check refuses has to be one the field would otherwise have answered, or
    "refused again" and "could not be evaluated" are the same observation.
    """
    graphs: dict[str, torch.Tensor] = {"num_atoms": torch.tensor([2], dtype=torch.long)}
    if total_charge is not None:
        graphs["total_charge"] = torch.tensor([total_charge], dtype=DTYPE)
    batch = TensorDict(
        {
            "atoms": TensorDict(
                {
                    "pos": torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=DTYPE),
                    "Z": torch.tensor([11, 11], dtype=torch.long),
                    "batch": torch.zeros(2, dtype=torch.long),
                },
                batch_size=[2],
            ),
            "graphs": TensorDict(graphs, batch_size=[1]),
        },
        batch_size=[],
    )
    return place_edges(batch, cutoff=CHARGED_CUTOFF)


def _butane_forcefield() -> ForceField:
    """``lj/cut`` + ``coulomb`` on the butane fixture, Class-I default scaling."""
    return (
        ForceField(name="butane")
        .def_pairstyle("lj/cut", cutoff=BUTANE_CUTOFF)
        .def_pairstyle("coulomb")
        .with_params(
            atoms={
                "q": torch.tensor(BUTANE_Q, dtype=DTYPE),
                "epsilon": torch.tensor(BUTANE_EPSILON, dtype=DTYPE),
                "sigma": torch.tensor(BUTANE_SIGMA, dtype=DTYPE),
            }
        )
    )


# ---------------------------------------------------------------------------
# LJ across a periodic boundary — analytic force vs a finite difference
# ---------------------------------------------------------------------------

#: Cubic edge (Å) of the two-atom periodic LJ fixture.
LJ_BOUNDARY_EDGE = 10.0
#: The two atoms (Å). Their raw separation is 8 Å — **outside** the 4.5 Å
#: cutoff — while their minimum image is 2 Å across the ``x`` boundary and well
#: inside it. The energy therefore comes entirely from a wrapped pair, which is
#: the only configuration in which a force built from the raw ``pos[target] −
#: pos[source]`` and the minimum-image ``r`` can disagree with ``−∂E/∂pos``.
LJ_BOUNDARY_POSITIONS = ((1.0, 5.0, 5.0), (9.0, 5.0, 5.0))
#: ``lj/cut`` cutoff (Å): past the 2 Å minimum image, under the 5 Å
#: minimum-image limit, and far from both, so a 1e-5 Å probe step can never
#: move the pair across it.
LJ_BOUNDARY_CUTOFF = 4.5
#: Well depth (kcal/mol) and size (Å). ``2^(1/6) σ = 1.91 Å`` sits just inside
#: the 2 Å separation, so the pair is on the attractive shoulder with a
#: non-zero gradient rather than at the bottom of the well where every force
#: is zero and any direction would pass.
LJ_BOUNDARY_EPSILON = 0.2
LJ_BOUNDARY_SIGMA = 1.7
#: Central-difference step (Å) and the gap it has to close (kcal mol⁻¹ Å⁻¹).
#: Truncation on this smooth pair energy is ``O(h²) ≈ 1e-11``.
LJ_BOUNDARY_FD_STEP = 1e-5
LJ_BOUNDARY_FORCE_TOL = 1e-6


def _lj_boundary_potentials() -> Potentials:
    """``lj/cut``-only Class-I model of the two-atom periodic fixture."""
    return (
        ForceField(name="lj-boundary", units="real")
        .def_pairstyle("lj/cut", cutoff=LJ_BOUNDARY_CUTOFF)
        .with_params(
            atoms={
                "epsilon": torch.full((2,), LJ_BOUNDARY_EPSILON, dtype=DTYPE),
                "sigma": torch.full((2,), LJ_BOUNDARY_SIGMA, dtype=DTYPE),
            }
        )
        .to_potentials()
    )


def _lj_boundary_batch(positions: torch.Tensor) -> TensorDict:
    """One periodic graph holding the two atoms and a freshly placed list.

    The list is rebuilt for every geometry the finite difference asks about:
    which lattice image of the partner is the nearest one is a function of the
    positions, so a list carried over from the reference geometry would answer
    about a pair that is no longer the pair being differentiated.
    """
    return place_edges(
        TensorDict(
            {
                "atoms": TensorDict(
                    {
                        "pos": positions,
                        "Z": torch.tensor([18, 18], dtype=torch.long),
                        "batch": torch.zeros(2, dtype=torch.long),
                    },
                    batch_size=[2],
                ),
                "graphs": TensorDict(
                    {"cell": torch.eye(3, dtype=DTYPE) * LJ_BOUNDARY_EDGE},
                    batch_size=[],
                ),
            },
            batch_size=[],
        ),
        cutoff=LJ_BOUNDARY_CUTOFF,
    )


# ---------------------------------------------------------------------------
# F29 — a molecule wrapped across a box face is refused at bind
# ---------------------------------------------------------------------------

#: Cubic edge (Å) of the wrapped-bond fixture. Every perpendicular width of a
#: cube is its own edge, so ``bind_topology``'s limit here is ``5``.
WRAP_EDGE = 10.0
#: The wrapped molecule's ``x`` coordinates (Å) — the two bonded atoms either
#: side of the ``x`` face. Their plain displacement is 8 Å, the long way round
#: the box, while their minimum image is the 2 Å the bond really is: exactly
#: what a trajectory writer that wraps coordinates into the cell produces, and
#: exactly the frame whose bonded terms would price a stretched bond.
WRAPPED_X = (1.0, 9.0)
#: The same molecule unwrapped — atom 1 left *outside* the box instead of
#: folded back into it. The bond is the 2 Å above; a coordinate past the cell
#: is not an error, being wrapped is.
UNWRAPPED_X = (9.0, 11.0)
#: Harmonic energy (kcal/mol) of that 2 Å bond: ``½·BOND_K·(2 − BOND_R0)² =
#: 50``, written out here rather than read back off the term.
UNWRAPPED_ENERGY = 50.0
#: Third lattice vector (Å) of the sheared cell: the 10 Å cube's ``c`` tilted
#: in ``x`` so ``|c|`` is still 10 Å while the volume falls to 600 Å³. Its
#: widest face is ``|b × c| = 100 Å²``, so the thinnest slab between two faces
#: — the perpendicular width — is ``600 / 100 = 6 Å`` and the bind limit is
#: 3 Å, while every *edge* is still 10 Å. That gap is the whole point: it
#: tells the width rule and the shortest-edge rule apart.
SHEARED_C = (8.0, 0.0, 6.0)
#: Perpendicular width (Å) of that cell, and the bond length (Å) placed in
#: both cells: inside the cube's 5 Å limit, past the sheared cell's 3 Å one.
SHEARED_WIDTH = 6.0
SHEARED_BOND_R = 4.0


def _periodic_bond_batch(x0: float, x1: float, cell: torch.Tensor | None = None) -> TensorDict:
    """Two bonded atoms on the ``x`` axis, optionally inside a lattice.

    Bond-only, so it carries no placed ``edges`` namespace: no pair style asks
    about this fixture and the bind-time check reads the bond columns and the
    cell, nothing else.

    Args:
        x0: ``x`` of atom 0 in Å.
        x1: ``x`` of atom 1 in Å.
        cell: Lattice vectors ``(3, 3)`` in Å, rows being the vectors, or
            ``None`` for an open boundary.

    Returns:
        The batch, carrying ``atoms`` and ``bonds``, plus ``graphs.cell`` when
        a lattice was given.
    """
    pos = torch.tensor([[x0, 5.0, 5.0], [x1, 5.0, 5.0]], dtype=DTYPE)
    batch = TensorDict(
        {
            "atoms": TensorDict(
                {
                    "pos": pos,
                    "Z": torch.tensor([1, 1], dtype=torch.long),
                    "batch": torch.zeros(2, dtype=torch.long),
                },
                batch_size=[2],
            ),
            "bonds": TensorDict(
                {
                    "atomi": torch.tensor([0], dtype=torch.long),
                    "atomj": torch.tensor([1], dtype=torch.long),
                },
                batch_size=[1],
            ),
        },
        batch_size=[],
    )
    if cell is not None:
        batch["graphs"] = TensorDict({"cell": cell}, batch_size=[])
    return batch


# ---------------------------------------------------------------------------
# S8 symmetry gates on the assembled energy — butane pair, 20 Å Ewald box
# ---------------------------------------------------------------------------

#: Largest energy change (kcal/mol) a symmetry of the periodic system may
#: produce. The assembled energy here is ``54.23`` kcal/mol, so this is
#: ``2e-12`` relative — a few times fp64 round-off on the reduction, and
#: nothing else.
SYMMETRY_TOL = 1e-10
#: Centre of the cubic box (Å); every point operation below is taken about it,
#: because a rotation or a mirror about any other point is a symmetry only when
#: composed with a translation the lattice may not have.
BUTANE_PAIR_CENTRE = 0.5 * BUTANE_PAIR_BOX
#: Seeds of the two random transformations. Fixed, so a gate that fires names
#: one reproducible configuration rather than "some run last week".
TRANSLATION_SEED = 20260908
PERMUTATION_SEED = 7

#: Harmonic improper of the chiral negative control: ``E = ½ k (χ − χ₀)²`` with
#: ``χ`` the *signed* dihedral of ``(atomj, atomi, atomk, atoml)``.
CHIRAL_IMPROPER_K = 10.0
CHIRAL_IMPROPER_CHI0 = 0.3
#: Smallest energy change (kcal/mol) that counts as "this term told the two
#: mirror images apart" rather than round-off. The measured gap is ``13.1``.
CHIRAL_TOL = 1e-6


def _central_difference_forces(
    energy_at: Callable[[torch.Tensor], float],
    positions: torch.Tensor,
    step: float,
) -> torch.Tensor:
    """``F = −dE/dpos`` by a central difference on every atom and axis.

    The reference an analytic or autograd force is checked against: it asks the
    model for nothing but energies, so it cannot inherit the gradient path's
    own mistake. Truncation is ``O(step²)`` on a smooth energy, which is what
    sets the tolerance each caller declares.

    Args:
        energy_at: Total energy (kcal/mol) of a geometry. Must rebuild whatever
            the geometry implies — a placed neighbour list above all — rather
            than reuse the reference one, or the difference measures the wrong
            function.
        positions: Reference geometry ``(N, 3)`` in Å.
        step: Half-width ``h`` of the difference in Å.

    Returns:
        Forces ``(N, 3)`` in kcal mol⁻¹ Å⁻¹.
    """
    forces = torch.zeros_like(positions)
    for atom in range(positions.shape[0]):
        for axis in range(3):
            forward, backward = positions.clone(), positions.clone()
            forward[atom, axis] += step
            backward[atom, axis] -= step
            forces[atom, axis] = -(energy_at(forward) - energy_at(backward)) / (2.0 * step)
    return forces


def _leaf(batch: TensorDict, namespace: str, name: str) -> torch.Tensor:
    """``batch[namespace, name]`` narrowed to a tensor.

    ``TensorDict.__getitem__`` is typed as "a tensor **or** a sub-collection",
    so reading a leaf needs the narrowing for the result to be usable as a
    tensor without an escape hatch.
    """
    value = batch[namespace, name]
    assert isinstance(value, torch.Tensor)
    return value


def _butane_pair_energy(positions: torch.Tensor) -> float:
    """Assembled energy (kcal/mol) of the butane-pair box at ``positions``.

    A fresh :class:`Potentials` and a freshly placed neighbour list per call:
    the list is a function of the geometry, and the periodic Coulomb term
    freezes its reciprocal shape on the first cell it sees, so reusing either
    across a transformed geometry would answer about the *previous* one.
    """
    potentials = butane_pair_class_i_forcefield().to_potentials()
    return float(potentials.calc_energy(butane_pair_batch(positions)))


def _translated_whole_molecules(positions: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """Translate by ``shift`` and wrap each molecule back into the box as a unit.

    Wrapping per **atom** would split a molecule across the boundary, and the
    bonded terms read plain displacements rather than minimum-image ones — a
    split molecule would report 20 Å bonds and the test would be measuring that
    instead of the symmetry. Wrapping per molecule moves each one by a whole
    lattice vector, which leaves every intramolecular displacement and every
    minimum-image separation exactly where it was.

    Args:
        positions: Atom positions ``(2 * BUTANE_N_ATOMS, 3)`` in Å, molecule 0
            first.
        shift: Translation ``(3,)`` in Å.

    Returns:
        The translated, per-molecule-wrapped positions, every coordinate inside
        ``[0, BUTANE_PAIR_BOX)``.
    """
    moved = positions + shift
    for molecule in range(moved.shape[0] // BUTANE_N_ATOMS):
        rows = slice(molecule * BUTANE_N_ATOMS, (molecule + 1) * BUTANE_N_ATOMS)
        centroid = moved[rows].mean(dim=0)
        moved[rows] -= torch.floor(centroid / BUTANE_PAIR_BOX) * BUTANE_PAIR_BOX
    return moved


def _coulomb_forcefield(*, calculator: str = "ewald") -> ForceField:
    """Coulomb-only Class-I ForceField over the four fixture ions.

    ``calculator="ewald"`` by default on purpose: the explicit k-vector sum is
    the small-box reference, and a finite difference has to differentiate the
    same function twice — an FFT mesh adds interpolation noise at the level the
    tolerances here are written for. Whether a route can be *traced*, though,
    is a property of that route's own arithmetic, so the tracing test asks for
    both.

    Both routes' sizing parameters are always supplied, so the named algorithm
    is the only thing that differs between the assemblies.

    Args:
        calculator: Reciprocal-space algorithm — ``"ewald"`` or ``"pme"``.

    Returns:
        The assembled, unbound ForceField.
    """
    return (
        ForceField(name="periodic-es")
        .def_pairstyle(
            "coulomb",
            calculator=calculator,
            smearing=COULOMB_SMEARING,
            real_space_cutoff=COULOMB_CUTOFF,
            lr_wavelength=COULOMB_LR_WAVELENGTH,
            mesh_spacing=COULOMB_MESH_SPACING,
        )
        .with_params(atoms={"q": torch.tensor(COULOMB_CHARGES, dtype=DTYPE)})
    )


def _coulomb_edges() -> tuple[torch.Tensor, torch.Tensor]:
    """The fixed half list of the periodic fixture, padded to capacity.

    Enumerated once from the reference geometry, the way MD enumerates once per
    rebuild and then reuses the rows: ``edge_index`` and the lattice ``shifts``
    are *connectivity*, and every later energy recomputes only
    ``edge_diff = pos[target] - pos[source] + shift`` from them. Handing the
    batch a placed list — rather than letting the builder re-enumerate — is
    also what keeps the distances on the position autograd graph, which the
    enumerated path (numpy, via molpy) cannot be.

    Returns:
        ``(edge_index, shifts)`` — ``(C, 2)`` int64 and ``(C, 3)`` Å at the
        capacity :data:`COULOMB_LIST_CAPACITY`. Live rows come first; padding
        rows are coincident self-loops ``(0, 0)`` with a zero shift, which the
        list's own ``source < target`` liveness rule kills.
    """
    positions = torch.tensor(COULOMB_POSITIONS, dtype=DTYPE)
    cell = torch.eye(3, dtype=DTYPE) * COULOMB_EDGE
    lattice = [
        torch.tensor(image, dtype=DTYPE) @ cell
        for image in itertools.product((-1.0, 0.0, 1.0), repeat=3)
    ]
    edge_index = torch.zeros(COULOMB_LIST_CAPACITY, 2, dtype=torch.long)
    shifts = torch.zeros(COULOMB_LIST_CAPACITY, 3, dtype=DTYPE)
    row = 0
    for source in range(positions.shape[0]):
        for target in range(source + 1, positions.shape[0]):
            for shift in lattice:
                if float((positions[target] - positions[source] + shift).norm()) < COULOMB_CUTOFF:
                    edge_index[row] = torch.tensor([source, target], dtype=torch.long)
                    shifts[row] = shift
                    row += 1
    assert row < COULOMB_LIST_CAPACITY, "the fixture must keep at least one dead padding row"
    return edge_index, shifts


def _coulomb_batch(
    positions: torch.Tensor,
    cell: torch.Tensor,
    edge_index: torch.Tensor,
    shifts: torch.Tensor,
) -> TensorDict:
    """A one-molecule periodic batch carrying the placed list and one bond.

    The bond makes atoms 0 and 1 a 1-2 pair, so the exclusion correction is
    live: it removes a term whose size depends on the positions *and* on the
    lattice, and therefore contributes to both the force and the virial. A
    fixture without a bond would leave that whole branch underivatived.

    Args:
        positions: Atom positions ``(N, 3)`` in Å.
        cell: Lattice vectors ``(3, 3)`` in Å, rows being the vectors.
        edge_index: Placed half list ``(C, 2)``.
        shifts: Lattice remainder of each row ``(C, 3)`` in Å.

    Returns:
        The batch, with ``edges.edge_diff`` rebuilt from ``positions`` and
        ``shifts`` so it is a function of the arguments, never a stale buffer.
    """
    n_atoms = positions.shape[0]
    edge_diff = positions[edge_index[:, 1]] - positions[edge_index[:, 0]] + shifts
    return TensorDict(
        {
            "atoms": TensorDict(
                {"pos": positions, "batch": torch.zeros(n_atoms, dtype=torch.long)},
                batch_size=[n_atoms],
            ),
            "edges": TensorDict(
                {
                    "edge_index": edge_index,
                    "edge_diff": edge_diff,
                    "edge_dist": edge_diff.norm(dim=-1),
                    "shifts": shifts,
                },
                batch_size=[edge_index.shape[0]],
            ),
            "graphs": TensorDict({"cell": cell.unsqueeze(0)}, batch_size=[1]),
            "bonds": TensorDict(
                {
                    "atomi": torch.tensor([0], dtype=torch.long),
                    "atomj": torch.tensor([1], dtype=torch.long),
                },
                batch_size=[1],
            ),
        },
        batch_size=[],
    )


class _EdgeSpring(nn.Module):
    """The minimal mlip shape ``def_pairstyle`` accepts: ``_write_energy`` only.

    Neither :class:`molix.schema.CellBound` nor
    :class:`~molix.schema.CoreCompilable` — which is the point: it is the
    in-tree proof that both are optional on an mlip, so ``Potentials`` must
    guard its delegation instead of calling through unconditionally.
    """

    def _write_energy(self, system: TensorDict) -> TensorDict:
        pos = system["atoms", "pos"]
        index = system["edges", "edge_index"]
        imaged = system["edges", "edge_diff"]
        raw = pos[index[:, 1]] - pos[index[:, 0]]
        # Same STE Polar/MACE use: frozen buffer supplies the MIC value, live
        # pos supplies ∂/∂pos. A requires_grad edge_diff is trusted as the
        # live geometry.
        vectors = imaged if imaged.requires_grad else imaged + (raw - raw.detach())
        energy = 0.5 * vectors.pow(2).sum()
        system[ENERGY_KEY] = energy.reshape(1)
        return system


def _edge_batch(*, r: float = 1.5) -> TensorDict:
    """Two atoms, both edges, a unit cell — the ``_EdgeSpring`` system."""
    pos = torch.tensor([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=DTYPE)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    shifts = torch.zeros(2, 3, dtype=DTYPE)
    edge_diff = pos[edge_index[:, 1]] - pos[edge_index[:, 0]] + shifts
    return TensorDict(
        {
            "atoms": TensorDict(
                {
                    "pos": pos.clone(),
                    "Z": torch.tensor([1, 1], dtype=torch.long),
                    "batch": torch.zeros(2, dtype=torch.long),
                },
                batch_size=[2],
            ),
            "edges": TensorDict(
                {
                    "edge_index": edge_index,
                    "edge_diff": edge_diff.clone(),
                    "edge_dist": edge_diff.norm(dim=-1),
                    "shifts": shifts.clone(),
                },
                batch_size=[2],
            ),
            "graphs": TensorDict({"cell": torch.eye(3, dtype=DTYPE).unsqueeze(0)}, batch_size=[1]),
        },
        batch_size=[],
    )


class TestPotentials:
    def test_calc_forces_analytic_golden(self) -> None:
        """F = −∂E/∂x = ∓k(r−r0) x̂ for the stretched two-atom harmonic bond."""
        forces = _bond_potentials().calc_forces(_two_atom_bond_batch())
        expected = torch.tensor(
            [[GOLDEN_FORCE_X, 0.0, 0.0], [-GOLDEN_FORCE_X, 0.0, 0.0]],
            dtype=DTYPE,
        )
        assert forces.shape == (2, 3)
        assert torch.allclose(forces.detach(), expected, atol=1e-10, rtol=0.0)

    def test_calc_energy_forces_matches_separate_calls(self) -> None:
        pots = _bond_potentials()
        batch = _two_atom_bond_batch()
        energy, forces = pots.calc_energy_forces(batch)
        assert float(energy.detach()) == pytest.approx(GOLDEN_ENERGY, abs=1e-10)
        assert torch.allclose(forces.detach(), pots.calc_forces(batch).detach(), atol=1e-10)

    def test_calc_energy_forces_uses_the_mlip_branch(self) -> None:
        """An mlip ForceField declares no classical style, so the term loop is empty.

        Falling through it returns ``(0, zeros)`` — a silent wrong answer, not a
        missing feature. ``calc_energy`` and ``calc_forces`` both branch on the
        mlip; this third entry must agree with them.
        """
        batch = _edge_batch()
        pots = ForceField().def_pairstyle(_EdgeSpring()).to_potentials()
        energy, forces = pots.calc_energy_forces(batch)
        assert float(energy.detach().reshape(())) == pytest.approx(
            float(pots.calc_energy(batch).detach().reshape(())), abs=1e-10
        )
        assert torch.allclose(forces.detach(), pots.calc_forces(batch).detach(), atol=1e-10)
        assert float(forces.detach().abs().max()) > 0.0

    def test_calc_forces_double_backward_reaches_parameters(self) -> None:
        """``create_graph`` keeps ∂²E/∂x∂θ alive so a force loss reaches θ."""
        parameter = nn.Parameter(torch.tensor([BOND_K], dtype=DTYPE))
        pots = _bond_potentials(parameter * 1.0)
        forces = pots.calc_forces(_two_atom_bond_batch())
        torch.mean(forces**2).backward()
        assert parameter.grad is not None
        # d/dk mean(F²) with F = ±k(r−r0): mean over 6 components = k²·0.5²·2/6
        assert float(parameter.grad) == pytest.approx(2.0 * BOND_K * 0.25 * 2.0 / 6.0, abs=1e-10)

    def test_virial_forces_match_calc_forces_when_energy_uses_edge_diff(self) -> None:
        """NPT strain must keep ∂E/∂pos through imaged edge geometry.

        Polar / MACE read ``edges.edge_diff``. If that tensor is strained as an
        independent cartesian, local graph energy drops off ``pos`` and MD
        integrates electrostatics-only forces.
        """
        batch = _edge_batch()
        pots = ForceField().def_pairstyle(_EdgeSpring()).to_potentials()
        reference = pots.calc_forces(batch)
        _, virial_forces, virial = pots.calc_energy_forces_virial(batch)
        assert torch.allclose(virial_forces, reference, atol=1e-10, rtol=0.0)
        assert virial.shape == (3, 3)
        assert virial.abs().sum().item() > 0.0

    def test_calc_forces_on_multi_graph_batch(self) -> None:
        """Per-graph energies ``(B,)`` reduce to one scalar before differentiating."""
        forces = _bond_potentials(n_bonds=2).calc_forces(_two_molecule_bond_batch())
        expected = torch.tensor(
            [
                [GOLDEN_FORCE_X, 0.0, 0.0],
                [-GOLDEN_FORCE_X, 0.0, 0.0],
                [GOLDEN_FORCE_X, 0.0, 0.0],
                [-GOLDEN_FORCE_X, 0.0, 0.0],
            ],
            dtype=DTYPE,
        )
        assert forces.shape == (4, 3)
        assert torch.allclose(forces.detach(), expected, atol=1e-10, rtol=0.0)

    def test_term_energies_splits_lj_into_repulsion_and_dispersion(self) -> None:
        """S3: one LJ evaluation, two labelled halves — the ``lj`` key is gone."""
        terms = _class_i_forcefield().to_potentials().term_energies(_class_i_batch())
        assert "lj" not in terms
        assert float(terms["lj_repulsion"]) > 0.0
        assert float(terms["lj_dispersion"]) < 0.0

    def test_term_energies_keys_are_exactly_the_registered_term_keys(self) -> None:
        """Every emitted key is a ``TermSpec`` of a declared style, and vice versa."""
        forcefield = _class_i_forcefield()
        terms = forcefield.to_potentials().term_energies(_class_i_batch())
        registered = {
            spec.key
            for style in forcefield.style_names()
            for spec in PotentialMeta.contract_for(style).terms
        }
        assert set(terms) == registered

    def test_contribution_energies_partition_the_total_energy(self) -> None:
        """The S-decomposition is a grouping of ``term_energies``, never a second path."""
        potentials = _class_i_forcefield().to_potentials()
        batch = _class_i_batch()
        total = potentials.calc_energy(batch)
        assert abs(float(total)) > 1e-6
        grouped = sum(potentials.contribution_energies(batch).values())
        assert float(grouped) == pytest.approx(float(total), abs=1e-12)

    def test_contribution_energies_are_keyed_by_s_decomposition_label(self) -> None:
        from molpot.contract import Contribution

        grouped = _class_i_forcefield().to_potentials().contribution_energies(_class_i_batch())
        assert set(grouped) == {Contribution.LOCAL, Contribution.DISP, Contribution.PERM}

    def test_declared_lj_style_without_its_parameter_table_raises(self) -> None:
        """F2: a missing table silently contributed zero energy before."""
        potentials = ForceField(name="lj").def_pairstyle("lj/cut", cutoff=LJ_CUTOFF).to_potentials()
        with pytest.raises(ValueError) as excinfo:
            potentials.calc_energy(_class_i_batch())
        message = str(excinfo.value)
        assert "pair:lj/cut" in message
        assert "epsilon" in message or "sigma" in message

    def test_declared_coulomb_style_without_charges_raises(self) -> None:
        potentials = ForceField(name="es").def_pairstyle("coulomb").to_potentials()
        with pytest.raises(ValueError) as excinfo:
            potentials.calc_energy(_class_i_batch())
        message = str(excinfo.value)
        assert "pair:coulomb" in message
        assert re.search(r"\bq\b", message) is not None

    def test_declared_bond_style_without_its_parameter_table_raises(self) -> None:
        """S1 on the bonded branch: no ``bonds`` table is a missing parameter, not a zero.

        The nonbonded guards above already refuse this. The bonded branches read
        ``k`` / ``r0`` off a table that was never filled and contributed an
        explicit zero instead, which is indistinguishable from a molecule whose
        bonds genuinely cost nothing.
        """
        potentials = ForceField(name="bonded").def_bondstyle("harmonic").to_potentials()
        with pytest.raises(ValueError) as excinfo:
            potentials.calc_energy(_two_atom_bond_batch())
        message = str(excinfo.value)
        assert "bond:harmonic" in message
        assert "bonds" in message

    def test_declared_angle_style_without_its_parameter_table_raises(self) -> None:
        """Same rule for ``angle:harmonic`` and its ``angles`` table (``k`` / ``theta0``)."""
        potentials = ForceField(name="angular").def_anglestyle("harmonic").to_potentials()
        with pytest.raises(ValueError) as excinfo:
            potentials.calc_energy(_angle_batch())
        message = str(excinfo.value)
        assert "angle:harmonic" in message
        assert "angles" in message

    def test_declared_dihedral_style_without_its_parameter_table_raises(self) -> None:
        """Same rule for ``dihedral:periodic`` and its ``propers`` table."""
        potentials = ForceField(name="torsional").def_dihedralstyle("periodic").to_potentials()
        with pytest.raises(ValueError) as excinfo:
            potentials.calc_energy(_torsion_batch("propers"))
        message = str(excinfo.value)
        assert "dihedral:periodic" in message
        assert "propers" in message

    def test_declared_improper_style_without_its_parameter_table_raises(self) -> None:
        """Same rule for ``improper:harmonic`` and its ``impropers`` table."""
        potentials = ForceField(name="improper").def_improperstyle("harmonic").to_potentials()
        with pytest.raises(ValueError) as excinfo:
            potentials.calc_energy(_torsion_batch("impropers"))
        message = str(excinfo.value)
        assert "improper:harmonic" in message
        assert "impropers" in message

    def test_declared_styles_with_tables_but_empty_topology_are_zero(self) -> None:
        """Absent *topology* is not absent *parameters* — it stays an explicit zero."""
        forcefield = (
            ForceField(name="empty")
            .def_bondstyle("harmonic")
            .def_pairstyle("lj/cut", cutoff=EMPTY_TOPOLOGY_CUTOFF)
            .with_params(
                bonds={
                    "k": torch.full((1,), BOND_K, dtype=DTYPE),
                    "r0": torch.full((1,), BOND_R0, dtype=DTYPE),
                },
                atoms={
                    "epsilon": torch.full((2,), EPSILON, dtype=DTYPE),
                    "sigma": torch.full((2,), SIGMA, dtype=DTYPE),
                },
            )
        )
        far_apart = TensorDict(
            {
                "atoms": TensorDict(
                    {
                        "pos": torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=DTYPE),
                        "Z": torch.tensor([1, 1], dtype=torch.long),
                        "batch": torch.zeros(2, dtype=torch.long),
                    },
                    batch_size=[2],
                )
            },
            batch_size=[],
        )
        # The list reaches past the style's cutoff, so the 10 Å pair *is* a row
        # of it — one the ``0 < r < cutoff`` liveness rule marks dead. The
        # zero below is therefore the mask doing its job, not an empty list.
        place_edges(far_apart, cutoff=EMPTY_TOPOLOGY_LIST_CUTOFF)
        energy = forcefield.to_potentials().calc_energy(far_apart)
        assert float(energy) == pytest.approx(0.0, abs=1e-12)

    def test_the_exclusion_table_reaches_both_pair_terms(
        self,
        butane_batch: TensorDict,
        butane_scales: torch.Tensor,
    ) -> None:
        """Hard-coded golden: the butane scale vector, read through ``term_energies``.

        Not a second copy of the ``ExclusionTable`` unit test — what is pinned
        here is the *wiring*: an unbound ``Potentials`` builds the table lazily
        on the first energy call, hands ``"q"`` to Coulomb and ``"lj"`` to LJ,
        and its neighbour list enumerates the 91 half-pairs in the same
        ``triu`` order the golden was captured in (every butane pair is inside
        the 10 Å cutoff). ``BUTANE_OPLS_SCALES`` holds the 1-4 entries at 0.5;
        under the Class-I defaults used here Coulomb rescales those to 5/6
        while LJ keeps 0.5, so a term reading the wrong kind is visible.
        """
        pairs = torch.triu_indices(14, 14, offset=1).t()
        positions = butane_batch["atoms", "pos"]
        distances = (positions[pairs[:, 1]] - positions[pairs[:, 0]]).norm(dim=-1)
        lj_scales = butane_scales
        q_scales = torch.where(
            butane_scales == 0.5,
            torch.full_like(butane_scales, 5.0 / 6.0),
            butane_scales,
        )

        charges = torch.tensor(BUTANE_Q, dtype=DTYPE)
        epsilon = torch.tensor(BUTANE_EPSILON, dtype=DTYPE)
        sigma = torch.tensor(BUTANE_SIGMA, dtype=DTYPE)
        epsilon_ij = (epsilon[pairs[:, 0]] * epsilon[pairs[:, 1]]).sqrt()
        sigma_ij = 0.5 * (sigma[pairs[:, 0]] + sigma[pairs[:, 1]])
        sr6 = (sigma_ij / distances).pow(6)
        expected_repulsion = float((lj_scales * 4.0 * epsilon_ij * sr6.pow(2)).sum())
        expected_dispersion = float((-lj_scales * 4.0 * epsilon_ij * sr6).sum())
        expected_coulomb = float(
            (q_scales * kcalmol_A * charges[pairs[:, 0]] * charges[pairs[:, 1]] / distances).sum()
        )

        terms = _butane_forcefield().to_potentials().term_energies(butane_batch)
        assert float(terms["coulomb"]) == pytest.approx(expected_coulomb, rel=1e-12)
        assert float(terms["lj_repulsion"]) == pytest.approx(expected_repulsion, rel=1e-12)
        assert float(terms["lj_dispersion"]) == pytest.approx(expected_dispersion, rel=1e-12)

    # ------------------------------------------------------------------
    # composable-a0-03b — enumeration is molix's; Potentials consumes a
    # placed list through ``_placed_pairs`` and fails closed without one.
    # ------------------------------------------------------------------

    def test_a_pair_style_on_a_batch_without_edges_names_both_molix_owners(self) -> None:
        """No ``edges`` namespace is a missing neighbour list, not an invitation to build one.

        ``molpot`` is downstream of ``molix`` in the dependency graph and does
        not own pair enumeration: the cutoff graph is built by
        :class:`molix.batch.NeighborList` on the training path and by
        :class:`molix.md.NeighborList` on the MD path. A ``Potentials`` that
        silently enumerated its own all-pairs list when the batch arrived
        without one answered with a *different* neighbour set than the model
        was trained or integrated against — same call, two physics — and hid
        the ownership of the Edge Convention while doing it.

        The message has to name **both** owners, because which one the caller
        wants depends on which path they are on and neither is guessable from
        the batch.
        """
        potentials = _lj_pair_forcefield().to_potentials()
        bare = _lj_pair_batch(r=SWITCH_R)
        del bare["edges"]

        with pytest.raises(ValueError) as excinfo:
            potentials.calc_energy(bare)
        message = str(excinfo.value)
        assert "molix.batch.NeighborList" in message
        assert "molix.md.NeighborList" in message

    def test_placed_pairs_returns_the_list_at_capacity_with_a_live_mask(self) -> None:
        """Padding survives in place; only the mask says which rows count.

        Moved from the deleted ``test_neighbor_list.py`` (``composable-a0-02``
        contract, now owned by :meth:`Potentials._placed_pairs`). Live is
        exactly ``source < target`` (the half list), ``0 < r < cutoff`` and
        inside ``num_edges``. Row 1 is the reversed duplicate of row 0, rows
        2-3 are outside the cutoff, rows 4-5 are dead capacity — none may be
        dropped, because dropping them makes the output shape depend on the
        data, which is a graph break on the compiled force path.

        The list carries ``edge_diff`` beside ``edge_dist`` because that is
        what :class:`molix.batch.NeighborList` writes and because an
        ``edge_dist``-only list is refused
        (:meth:`test_placed_pairs_refuses_an_edge_dist_only_list`); the
        capacity / mask contract asserted here is independent of which geometry
        branch prices the rows.
        """
        potentials = _lj_pair_forcefield().to_potentials()
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=DTYPE)
        edge_index = torch.tensor(
            [[0, 1], [1, 0], [0, 2], [2, 0], [0, 0], [0, 0]], dtype=torch.long
        )
        system = {
            "atoms": {"pos": pos, "batch": torch.zeros(3, dtype=torch.long)},
            "edges": {
                "edge_index": edge_index,
                "edge_diff": pos[edge_index[:, 1]] - pos[edge_index[:, 0]],
                "edge_dist": torch.tensor([1.0, 1.0, 3.0, 3.0, 0.0, 0.0], dtype=DTYPE),
                "num_edges": 4,
            },
        }

        returned_index, _vectors, dist, live_mask = potentials._placed_pairs(
            system, pos, cutoff=2.5
        )

        assert returned_index.shape == (6, 2)
        assert dist.shape == (6,)
        assert live_mask.dtype == torch.bool
        assert torch.equal(returned_index, edge_index)
        assert torch.equal(live_mask, torch.tensor([True, False, False, False, False, False]))

    def test_placed_pairs_refuses_an_edge_dist_only_list(self) -> None:
        """A stored scalar distance is not geometry, so the list is refused.

        Forwarding ``edge_dist`` as the answer reads a number that is not a
        function of the ``pos`` leaf the energy is differentiated against, so
        ``dE/dpos`` is structurally zero: such a batch used to price a finite
        energy and hand back **zero forces**, silently, which is the worst
        possible failure on an MD path. Nothing in the tree produces this list
        either — :class:`molix.batch.NeighborList` writes ``edge_diff`` beside
        ``edge_dist`` and :class:`molix.md.NeighborList` writes ``shifts`` as
        well — so the only reachable caller was a hand-built dict.

        The fixture is deliberately the one that used to pass: the stored 2 Å
        matches neither the raw separation (9 Å) nor its minimum image in the
        10 Å cell (1 Å). A list that disagrees with the geometry it was placed
        on has to be refused rather than believed, and the message has to name
        the geometry a list must carry.
        """
        potentials = _lj_pair_forcefield().to_potentials()
        pos = torch.tensor([[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]], dtype=DTYPE)
        system = {
            "atoms": {"pos": pos, "batch": torch.zeros(2, dtype=torch.long)},
            "graphs": {"cell": torch.eye(3, dtype=DTYPE) * 10.0},
            "edges": {
                "edge_index": torch.tensor([[0, 1]], dtype=torch.long),
                "edge_dist": torch.tensor([2.0], dtype=DTYPE),
            },
        }

        with pytest.raises(ValueError) as excinfo:
            potentials._placed_pairs(system, pos, cutoff=5.0)
        message = str(excinfo.value)
        assert "edge_diff" in message or "shifts" in message

    def test_placed_pairs_prefers_shifts_over_a_stored_edge_diff(self) -> None:
        """F13: ``shifts`` is the branch, ``edge_diff`` only the fallback.

        Both keys are present and they disagree — ``edge_diff`` claims a 4 Å
        separation the geometry plus ``shifts`` does not support. The
        ``shifts`` branch rebuilds the pair vector as
        ``pos[t] - pos[s] + shifts`` and must win, because it is the only one
        that is a function of the *live* lattice; the ``edge_diff``
        reconstruction can recover the lattice remainder only as a detached
        constant.
        """
        potentials = _lj_pair_forcefield().to_potentials()
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=DTYPE)
        edge_index = torch.tensor([[0, 1]], dtype=torch.long)
        shifts = torch.tensor([[10.0, 0.0, 0.0]], dtype=DTYPE)
        system = {
            "atoms": {"pos": pos, "batch": torch.zeros(2, dtype=torch.long)},
            "edges": {
                "edge_index": edge_index,
                "shifts": shifts,
                "edge_diff": torch.tensor([[4.0, 0.0, 0.0]], dtype=DTYPE),
                "edge_dist": torch.tensor([4.0], dtype=DTYPE),
            },
        }

        _, _vectors, dist, _live = potentials._placed_pairs(system, pos, cutoff=20.0)

        # pos[1] - pos[0] + shift = 1 + 10 = 11, not the stored 4.
        assert float(dist[0]) == pytest.approx(11.0, abs=1e-12)

    def test_placed_pairs_keeps_the_shift_on_the_autograd_graph(self) -> None:
        """F13: the ``shifts`` branch is not detached, so ``dE/deps`` sees the lattice.

        A pair that reaches its partner through a lattice vector contributes a
        shift term to the strain derivative. The ``edge_diff`` fallback can only
        recover that remainder from detached tensors — a constant — so its
        virial is exact only while every pair sits in the same image. The
        ``shifts`` branch has to keep the remainder differentiable, which is
        observable here as a non-zero gradient of the returned distance with
        respect to ``shifts``.
        """
        potentials = _lj_pair_forcefield().to_potentials()
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=DTYPE)
        shifts = torch.tensor([[10.0, 0.0, 0.0]], dtype=DTYPE, requires_grad=True)
        system = {
            "atoms": {"pos": pos, "batch": torch.zeros(2, dtype=torch.long)},
            "edges": {"edge_index": torch.tensor([[0, 1]], dtype=torch.long), "shifts": shifts},
        }

        _, _vectors, dist, _live = potentials._placed_pairs(system, pos, cutoff=20.0)
        (grad,) = torch.autograd.grad(dist.sum(), shifts)

        # d|d|/d(shift_x) = (pos[t] - pos[s] + shift)_x / |...| = 11 / 11 = 1.
        assert float(grad[0, 0]) == pytest.approx(1.0, abs=1e-12)

    def test_placed_pairs_on_an_empty_list_is_empty_not_absent(self) -> None:
        """A list with no rows means "no pair in range", never "no list".

        :class:`molix.batch.NeighborList` returns zero rows for a system whose
        atoms are all outside the cutoff, and that is a perfectly ordinary
        batch — the pair terms must price it at zero. Only a *missing* ``edges``
        namespace is the fail-closed case
        (:meth:`test_a_pair_style_on_a_batch_without_edges_names_both_molix_owners`),
        and conflating the two either resurrects the enumerator or refuses a
        dilute molecule.
        """
        potentials = _lj_pair_forcefield().to_potentials()
        pos = torch.tensor([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]], dtype=DTYPE)
        system = {
            "atoms": {"pos": pos, "batch": torch.zeros(2, dtype=torch.long)},
            "edges": {
                "edge_index": torch.zeros(0, 2, dtype=torch.long),
                "edge_diff": torch.zeros(0, 3, dtype=DTYPE),
                "edge_dist": torch.zeros(0, dtype=DTYPE),
            },
        }

        edge_index, _vectors, dist, live = potentials._placed_pairs(system, pos, cutoff=LJ_CUTOFF)

        assert edge_index.shape == (0, 2)
        assert dist.shape == (0,)
        assert live.shape == (0,)

    def test_switch_start_declared_on_the_style_reaches_the_pair_term(self) -> None:
        """``def_pairstyle`` is the only route to the LJ taper, so it must carry it.

        :meth:`Potentials._assemble_pair_potentials` reads ``cutoff`` off the
        style options and builds the :class:`LJCutPotential` with it;
        ``switch_start`` is the second half of the same option. Dropped there,
        the quintic switch a0-02 put on the term is unreachable from any
        ForceField — the S8 cutoff-continuity gate would then be closed only in
        the term's own unit test and open on every assembled Class-I model.

        The un-tapered field is checked in the same test because "reaches the
        term" and "is not invented when absent" are one contract: the default
        stays a hard truncation (``switch_start is None``).
        """
        tapered = _lj_pair_forcefield(switch_start=SWITCH_START).to_potentials()
        assert tapered.lj is not None
        assert tapered.lj.cutoff == LJ_CUTOFF
        assert tapered.lj.switch_start == SWITCH_START

        truncated = _lj_pair_forcefield().to_potentials()
        assert truncated.lj is not None
        assert truncated.lj.cutoff == LJ_CUTOFF
        assert truncated.lj.switch_start is None

    def test_assembled_pair_terms_and_the_recorded_method_agree_field_by_field(self) -> None:
        """F7: one resolver for the pair-style options, read from both ends.

        The declared options are turned into numbers twice — by
        :meth:`Potentials._assemble_pair_potentials` into the terms that
        actually run, and by ``ForceField._nonbonded_method_record`` into the
        record ``to_ir`` and the manifest carry. Two independent readers of one
        options dict is one edit away from a manifest that describes a model
        which was never assembled, so every field is asserted against the
        *term*, never against a literal: the test cannot pass with two readers
        that agree only by coincidence.

        ``mixing`` rides here for the same reason the radii do: the
        Lennard-Jones combining rule is model identity (ledger finding
        **F24**), so the rule the assembled term mixes with and the rule the
        manifest reports are one resolved value, not two readings that happen
        to agree while both default to Lorentz–Berthelot.

        ``real_space_cutoff`` is deliberately left undeclared, because it is
        the one option whose fallback is *derived* rather than absolute — the
        periodic Coulomb term runs at ``3 × smearing = 12 Å``, and a record
        that reports ``None`` for it names no radius at all.
        """
        ff = (
            ForceField(name="pair-style-resolver")
            .def_pairstyle("lj/cut", cutoff=8.5, switch_start=7.0, mixing="geometric")
            .def_pairstyle(
                "coulomb",
                calculator="ewald",
                smearing=4.0,
                lr_wavelength=0.5,
                capacity_factor=1.3,
            )
        )
        pots = ff.to_potentials()
        record = ff.to_ir().nonbonded_method
        assert pots.lj is not None
        assert pots.coulomb_open is not None
        assert pots.coulomb_periodic is not None

        assert pots.lj.cutoff == record.lj_cutoff
        assert pots.lj.switch_start == record.lj_switch_start
        assert pots.lj.mixing == record.lj_mixing
        assert pots.coulomb_open.cutoff == record.coulomb_cutoff
        assert pots.coulomb_periodic.calculator_name == record.coulomb_calculator
        assert pots.coulomb_periodic.smearing == record.smearing
        assert pots.coulomb_periodic.lr_wavelength == record.lr_wavelength
        assert pots.coulomb_periodic.mesh_spacing == record.mesh_spacing
        assert pots.coulomb_periodic.capacity_factor == record.capacity_factor
        assert pots.coulomb_periodic.real_space_cutoff == record.real_space_cutoff

    def test_switched_lj_energy_is_the_analytic_taper_inside_the_window(self) -> None:
        """Domain golden: ``E = S(r)·E_LJ(r)`` at the midpoint of ``[r_s, r_c]``.

        Two argon-sized atoms sit at ``r = 9.5 Å`` — inside the ``9–10 Å``
        window, so the taper is neither ``1`` nor ``0`` and a field that ignored
        ``switch_start`` is visible as a *value*, not just as a dropped
        attribute. With ``x = (r − r_s)/(r_c − r_s) = ½`` the quintic switch
        ``S(x) = 1 − x³(10 − 15x + 6x²)`` is exactly ``½``, so the tapered
        energy must be half the truncated one, both computed here from
        ``4ε[(σ/r)¹² − (σ/r)⁶]`` in fp64 rather than read back out of the term.

        Reference:
            OpenMM ``useSwitchingFunction`` taper (algebraically the LAMMPS
            ``lj/mdf`` form), spec ``composable-a0-02`` §Domain basis;
            Lennard-Jones, J. E. *Proc. Phys. Soc.* **43**, 461 (1931).
        """
        batch = _lj_pair_batch(r=SWITCH_R)
        sr6 = (SIGMA / SWITCH_R) ** 6
        lj = 4.0 * EPSILON * (sr6 * sr6 - sr6)

        truncated = float(_lj_pair_forcefield().to_potentials().calc_energy(batch))
        tapered = float(
            _lj_pair_forcefield(switch_start=SWITCH_START).to_potentials().calc_energy(batch)
        )
        assert truncated == pytest.approx(lj, abs=1e-10)
        assert tapered == pytest.approx(SWITCH_AT_MIDPOINT * lj, abs=1e-10)
        # The whole point of the window: inside it the two fields disagree, by
        # exactly the part the switch removed.
        assert tapered - truncated == pytest.approx((SWITCH_AT_MIDPOINT - 1.0) * lj, abs=1e-10)

    def test_a_bound_topology_is_refused_for_a_batch_of_another_shape(self) -> None:
        """Rebinding is the caller's job, so a stale table must be caught, not used.

        :meth:`bind_topology` builds the exclusion table once per topology and
        every later energy call reuses it. Handed a batch whose topology has a
        *different shape*, the table answers with the previous molecule's
        relations: here the extra ``angles`` column makes ``(0, 2)`` a 1-3 pair,
        which the stale table still reads as unexcluded, and the total energy
        comes out 4.65 instead of 18.49 kcal/mol — silently, with no shape
        error anywhere. Fail closed instead, naming ``bind_topology`` so the
        message says what to call.

        The fingerprint is deliberately **shape-level only**: positions change
        every MD step and must stay free, and a fresh batch of the same
        topology (new tensors, same shapes — the ordinary DataLoader case) must
        keep reusing the bound table. That is the second leg below, pinned
        against a freshly assembled ``Potentials`` so "does not raise" also
        means "still right".
        """
        potentials = _class_i_forcefield().to_potentials()
        bound = _class_i_batch()
        potentials.bind_topology(bound)

        restated = _class_i_batch()
        restated["atoms", "pos"] = restated["atoms", "pos"] + torch.tensor(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, -0.2, 0.0]], dtype=DTYPE
        )
        # The placed list is connectivity *plus* the geometry it was built at:
        # ``_placed_pairs`` recovers each row's lattice remainder as
        # ``edge_diff - (pos[t] - pos[s])``, so a list left at the old
        # positions would hand back the old distances. Rebuild it, exactly as
        # molix does after atoms move.
        place_edges(restated, cutoff=LJ_CUTOFF)
        reference = _class_i_forcefield().to_potentials()
        assert float(potentials.calc_energy(restated)) == pytest.approx(
            float(reference.calc_energy(restated)), abs=1e-10
        )

        with_angle = _class_i_batch()
        with_angle["angles"] = TensorDict(
            {
                "atomi": torch.tensor([0], dtype=torch.long),
                "atomj": torch.tensor([1], dtype=torch.long),
                "atomk": torch.tensor([2], dtype=torch.long),
            },
            batch_size=[1],
        )
        with pytest.raises(ValueError, match="bind_topology"):
            potentials.calc_energy(with_angle)

    def test_bind_topology_rejects_charges_that_miss_the_declared_total(self) -> None:
        """S5 fail-closed: ``Σ_g q`` must equal ``graphs.total_charge`` to 1e-6 e.

        A learned charge head that drifts off the constraint changes the
        molecule's net charge, and under PBC that is an infinite-energy system
        silently regularised by the Ewald background term. The check is eager
        and per graph, so it fires at bind time rather than as a wrong number.
        """
        potentials = _anion_potentials()
        with pytest.raises(ValueError, match="total_charge"):
            potentials.bind_topology(_charged_batch(total_charge=0.0))

        potentials.bind_topology(_charged_batch(total_charge=1.0))
        potentials.bind_topology(_charged_batch(total_charge=None))

    def test_bind_topology_refuses_a_bond_wrapped_across_a_box_face(self) -> None:
        """F29 fail-closed: under a cell, a bond over half the box is a folded molecule.

        The two halves of an assembled energy read geometry differently. The
        bonded terms take the plain ``pos_j − pos_i`` off their index columns;
        the pair terms take minimum-image separations. Wrapping a periodic
        frame into the box — what a trajectory writer does by default — is
        therefore invisible to the pair half and fatal to the bonded one: the
        two atoms below are 2 Å apart through the ``x`` face and 8 Å apart in
        the coordinates as given, and every bond, angle and torsion built on
        them would be priced at the 8 Å. The number that comes back is finite,
        differentiable and wrong, so the frame is refused at the bind instead,
        where one host sync is affordable and the force path stays free of it.

        Three legs, because the rule is only as good as its boundaries: the
        wrapped frame raises and the message names the bond, its length, the
        limit and the finding; the *unwrapped* copy of the same molecule —
        a coordinate outside the box, which is legal — binds and prices the
        2 Å bond; and the same 8 Å displacement with **no** lattice binds,
        since an open boundary has nothing to be wrapped into.
        """
        potentials = _bond_potentials()
        cell = torch.eye(3, dtype=DTYPE) * WRAP_EDGE

        with pytest.raises(ValueError, match="F29") as excinfo:
            potentials.bind_topology(_periodic_bond_batch(*WRAPPED_X, cell=cell))
        message = str(excinfo.value)
        # The message has to identify *which* bond and *by how much*: the
        # two atom indices, the 8 Å it spans and the 5 Å half-width it broke.
        # Only the numbers are pinned — the phrasing around them is not.
        for token in ("0", "1", "8", "5"):
            assert re.search(rf"\b{token}\b", message), message

        unwrapped = _periodic_bond_batch(*UNWRAPPED_X, cell=cell)
        potentials.bind_topology(unwrapped)
        assert float(potentials.calc_energy(unwrapped)) == pytest.approx(
            UNWRAPPED_ENERGY, abs=1e-10
        )

        potentials.bind_topology(_periodic_bond_batch(*WRAPPED_X))

    def test_the_bind_limit_is_the_perpendicular_width_not_the_shortest_edge(self) -> None:
        """A flat triclinic cell fails a bond a cube of the same edges accepts.

        How far a molecule can reach before its own image is nearer than itself
        is set by the thinnest slab between two faces, ``V / max |a_i × a_j|``,
        not by the shortest lattice *vector*. Shear the 10 Å cube's ``c`` in
        ``x`` and every edge stays 10 Å long while the volume — and with it the
        slab between the ``b``/``c`` faces — falls to 6 Å. A 4 Å bond is
        therefore fine in the cube (limit 5 Å) and wrapped in the sheared cell
        (limit 3 Å); an implementation that measured against the edge length
        would accept both and answer the folded frame.
        """
        cube = torch.eye(3, dtype=DTYPE) * WRAP_EDGE
        sheared = torch.tensor(
            [[WRAP_EDGE, 0.0, 0.0], [0.0, WRAP_EDGE, 0.0], SHEARED_C], dtype=DTYPE
        )
        # The fixture's own claim, checked rather than asserted in a comment:
        # same three edge lengths, a thinner slab.
        assert [float(row.norm()) for row in sheared] == pytest.approx([WRAP_EDGE] * 3)
        volume = float(torch.linalg.det(sheared).abs())
        widest_face = float(torch.linalg.cross(sheared[1], sheared[2]).norm())
        assert volume / widest_face == pytest.approx(SHEARED_WIDTH, abs=1e-12)

        potentials = _bond_potentials()
        potentials.bind_topology(_periodic_bond_batch(0.0, SHEARED_BOND_R, cell=cube))

        with pytest.raises(ValueError, match="F29") as excinfo:
            potentials.bind_topology(_periodic_bond_batch(0.0, SHEARED_BOND_R, cell=sheared))
        message = str(excinfo.value)
        # 4 Å of bond against a 3 Å limit — the half-width, not the 5 Å the
        # 10 Å edges would have allowed.
        assert re.search(r"\b4\b", message), message
        assert re.search(r"\b3\b", message), message

    def test_a_wrapped_bond_is_refused_through_calc_energy_without_an_explicit_bind(
        self,
    ) -> None:
        """F29 has to hold at the entries a caller uses, not only at the bind.

        :meth:`bind_topology` refuses a folded frame, but nothing makes a
        caller call it. For a **bonded-only** field there is no nonbonded term,
        so the lazy bind inside ``_scaled_pairs`` never runs either, and the
        wrapped molecule of :data:`WRAPPED_X` — 2 Å apart through the ``x``
        face, 8 Å apart in the coordinates as given — used to be priced at
        ``½·BOND_K·(8 − BOND_R0)² = 2450`` kcal/mol straight through the public
        API: finite, differentiable and wrong, which is exactly what F29
        exists to refuse. A guard that only fires on an explicit bind is a
        guard on the path that was never the problem.

        Both public energy entries are checked, each on a **fresh** container,
        because a bind performed by the first call would otherwise be what
        protected the second.
        """
        cell = torch.eye(3, dtype=DTYPE) * WRAP_EDGE

        with pytest.raises(ValueError, match="F29"):
            _bond_potentials().calc_energy(_periodic_bond_batch(*WRAPPED_X, cell=cell))

        with pytest.raises(ValueError, match="F29"):
            _bond_potentials().calc_energy_forces(_periodic_bond_batch(*WRAPPED_X, cell=cell))

    def test_a_second_frame_of_the_same_shape_is_checked_afresh(self) -> None:
        """The F29 memo may not vouch for a frame it never read.

        The gate is remembered per *topology* — a shape-level fingerprint of
        the atom count and the bonded column counts — so a later call on a
        batch of the same shape is passed straight through. That positions are
        free afterwards is the point: an MD rollout cannot pay a host sync per
        step, and an integrator never wraps in place, so a rollout whose first
        frame is whole stays whole.

        A **frame** is not a topology, though. The same two bonded atoms arrive
        whole or folded, in one cell or another, at the same shape either way —
        which is precisely the training path, where one container prices batch
        after batch. Priced one whole frame first, the container then answered
        the folded one behind it with ``½·BOND_K·(8 − BOND_R0)² = 2450``
        kcal/mol: the guard reduced to a check on whichever frame happened to
        arrive first.
        """
        cell = torch.eye(3, dtype=DTYPE) * WRAP_EDGE
        potentials = _bond_potentials()

        # A whole frame, priced — this is what arms the memo.
        potentials.calc_energy(_periodic_bond_batch(*UNWRAPPED_X, cell=cell))

        with pytest.raises(ValueError, match="F29"):
            potentials.calc_energy(_periodic_bond_batch(*WRAPPED_X, cell=cell))

    @pytest.mark.slow
    def test_a_wrapped_frame_is_refused_even_under_torch_compile(self) -> None:
        """Compiling the caller's own entry point may not switch the F29 gate off.

        The check is host-side, so it cannot be traced — and
        :meth:`_check_geometry` therefore returns early under
        ``torch.compiler.is_compiling()``. That is right for the *compiled
        core*, which :meth:`calc_energy_forces` calls the gate before entering.
        It is wrong for a caller who wraps a whole public entry themselves,
        which is the ordinary way to compile a single-point energy: the guard
        then evaporates exactly where nothing else is watching, and the folded
        frame comes back as ``½·BOND_K·(8 − BOND_R0)² = 2450`` kcal/mol —
        measured, under ``torch.compile`` — while the same call eagerly raises.
        A precondition that holds only in eager mode is not a precondition; it
        is a lint.

        The narrow opt-out the compiled region actually needs is the memo:
        a frame an eager :meth:`calc_energy` or :meth:`bind_topology` has
        already vouched for stays free, and anything else is refused with a
        message naming that as the way through. Both halves are pinned here —
        the unvouched frame raises, and the vouched one runs.

        Marked ``slow``: the first ``torch.compile`` of this field costs 47 s
        on the CI box (0.1 s for a second one in the same process), which is
        the compile tier by any reading of the rule.
        """
        cell = torch.eye(3, dtype=DTYPE) * WRAP_EDGE

        refusing = _bond_potentials()
        with pytest.raises(ValueError, match="F29"):
            torch.compile(lambda batch: refusing.calc_energy(batch))(
                _periodic_bond_batch(*WRAPPED_X, cell=cell)
            )

        # Vouched for eagerly first, then compiled: the same object, the same
        # topology, so the memo carries it and the compiled call goes through.
        vouched = _bond_potentials()
        whole = _periodic_bond_batch(*UNWRAPPED_X, cell=cell)
        vouched.calc_energy(whole)
        torch.compile(lambda batch: vouched.calc_energy(batch))(whole)

    def test_a_refused_bind_leaves_nothing_armed(self) -> None:
        """A bind that raised must not have installed half of itself.

        :meth:`bind_topology` builds the exclusion table and records the
        topology fingerprint *before* it checks the declared net charge, so a
        refusal left the container looking bound: the exclusion table is not
        ``None`` any more, the lazy bind inside the next evaluation sees
        nothing to do, and the very batch that was just refused is priced —
        41.51 kcal/mol here, from charges the field itself says are wrong. One
        caught exception in a fitting loop (or a notebook cell re-run) is all
        it takes, and every number after it is quietly outside the S5
        constraint.

        Refusing repeatedly is the only safe state: the container is either
        bound to a frame it accepted or bound to nothing. Both entry orders are
        checked, because the arming happens on the bind and the *lazy* bind is
        the one no caller can see.
        """
        batch = _charged_batch(total_charge=0.0)

        explicit = _anion_potentials()
        with pytest.raises(ValueError, match="total_charge"):
            explicit.bind_topology(batch)
        with pytest.raises(ValueError, match="total_charge"):
            explicit.calc_energy(batch)

        lazy = _anion_potentials()
        with pytest.raises(ValueError, match="total_charge"):
            lazy.calc_energy(batch)
        with pytest.raises(ValueError, match="total_charge"):
            lazy.calc_energy(batch)

    def test_forward_writes_a_per_graph_energy_column_on_a_one_graph_batch(self) -> None:
        """``graphs`` sized ``[1]`` takes a ``(1,)`` energy, not a bare scalar.

        A post-collate batch carries one row per graph
        (``graphs.batch_size == [B]``), and ``B = 1`` — one molecule per
        forward — is the shape the composer, the loss and the MD adapter all
        meet first. The classical sum, though, is a 0-dim tensor whenever the
        batch holds a single graph, so :meth:`Potentials.forward` writing it
        under a ``[1]`` namespace hands TensorDict a value of the wrong rank:
        the batch dimension the namespace declares is not there. The energy
        column has to arrive shaped like the namespace that holds it, and the
        forces like the atoms.
        """
        batch = _two_atom_bond_batch()
        batch["graphs"] = TensorDict(
            {"num_atoms": torch.tensor([2], dtype=torch.long)}, batch_size=[1]
        )

        out = _bond_potentials()(batch)

        assert out is batch
        assert out["graphs", "energy"].shape == (1,)
        assert out["atoms", "forces"].shape == (2, 3)

    def test_two_polarization_styles_are_refused_at_construction(
        self, fake_pol_styles: tuple[str, str]
    ) -> None:
        """S1: unsupported combinations are rejected, not warned about at forward."""
        forcefield = ForceField(name="double-pol").def_pairstyle("coulomb")
        for key in fake_pol_styles:
            forcefield = forcefield.def_pairstyle(key.split(":", 1)[1])
        with pytest.raises(ValueError):
            forcefield.to_potentials()

    def test_unregistered_style_key_is_refused_at_construction(self) -> None:
        """F6: a typo'd style used to assemble into a silent no-op."""
        forcefield = ForceField(name="typo").def_pairstyle("lj/cutt", cutoff=LJ_CUTOFF)
        with pytest.raises(ValueError):
            forcefield.to_potentials()

    def test_periodic_coulomb_forces_match_a_central_finite_difference(self) -> None:
        """``F = -dE/dpos`` of the assembled periodic model, against 24 differences.

        The autograd force has to be the derivative of the energy the same
        object reports — every part of it: the real-space ``erfc`` sum over the
        placed list, the reciprocal sum over the frozen grid, and the exclusion
        correction that removes the bonded pair. The last one is the easy piece
        to get wrong, because it is subtracted *outside* the neighbour list: a
        correction evaluated on detached positions leaves the energy right and
        the force wrong by exactly the excluded pair's gradient.

        The cell is bound first so the reciprocal **shape** cannot change
        between the differenced evaluations; positions alone would not move it,
        but the strain test next door needs the same guarantee and the two must
        differentiate the same function.
        """
        potentials = _coulomb_forcefield().to_potentials()
        cell = torch.eye(3, dtype=DTYPE) * COULOMB_EDGE
        potentials.bind_cell(cell)
        edge_index, shifts = _coulomb_edges()
        positions = torch.tensor(COULOMB_POSITIONS, dtype=DTYPE)

        _, forces, _ = potentials.calc_energy_forces_virial(
            _coulomb_batch(positions, cell, edge_index, shifts)
        )
        forces = forces.detach()

        def energy_at(stepped: torch.Tensor) -> float:
            with torch.no_grad():
                return float(
                    potentials.calc_energy(_coulomb_batch(stepped, cell, edge_index, shifts))
                )

        differences = _central_difference_forces(energy_at, positions, FD_STEP)

        gap = float((forces - differences).abs().max())
        assert gap < FD_FORCE_TOL, f"max |F_auto - F_fd| = {gap:.3e} kcal/mol/A"

    def test_periodic_coulomb_virial_matches_a_central_strain_difference(self) -> None:
        """``W = dE/deps`` against six symmetric-strain differences.

        The deformation is the one
        :func:`~molpot.derivation.force.autograd_forces_and_virial` applies:
        positions, lattice vectors **and** the per-edge lattice shifts are all
        mapped by ``v @ (I + eps)^T``. Straining the shifts is not decoration —
        atom 3 of the fixture is wrapped, so three of its six pairs reach their
        partner through a lattice vector, and a virial that holds those vectors
        fixed while the box deforms is missing a term of order the energy
        itself.

        ``eps`` is the **symmetric** strain tensor and ``W`` its matrix
        gradient, so a shear moves ``eps_ab`` and ``eps_ba`` together and the
        difference quotient counts the pair twice: the off-diagonals are halved
        to land on the Cauchy convention
        (:func:`~molpot.derivation.stress.symmetrize_strain_grad`).
        """
        potentials = _coulomb_forcefield().to_potentials()
        cell = torch.eye(3, dtype=DTYPE) * COULOMB_EDGE
        potentials.bind_cell(cell)
        edge_index, shifts = _coulomb_edges()
        positions = torch.tensor(COULOMB_POSITIONS, dtype=DTYPE)

        _, _, virial = potentials.calc_energy_forces_virial(
            _coulomb_batch(positions, cell, edge_index, shifts)
        )
        virial = virial.detach()

        def deformed(deformation: torch.Tensor) -> torch.Tensor:
            transform = deformation.T
            with torch.no_grad():
                return potentials.calc_energy(
                    _coulomb_batch(
                        positions @ transform,
                        cell @ transform,
                        edge_index,
                        shifts @ transform,
                    )
                )

        identity = torch.eye(3, dtype=DTYPE)
        differences = torch.zeros(3, 3, dtype=DTYPE)
        for row in range(3):
            for column in range(row, 3):
                strain = torch.zeros(3, 3, dtype=DTYPE)
                strain[row, column] = FD_STEP
                strain[column, row] = FD_STEP
                derivative = (deformed(identity + strain) - deformed(identity - strain)) / (
                    2.0 * FD_STEP
                )
                counted_twice = 1.0 if row == column else 2.0
                differences[row, column] = derivative / counted_twice
                differences[column, row] = differences[row, column]

        gap = float((virial - differences).abs().max())
        assert gap < FD_VIRIAL_TOL, f"max |W_auto - W_fd| = {gap:.3e} kcal/mol"

    def test_periodic_coulomb_on_a_multi_graph_batch_is_refused(self) -> None:
        """Two graphs under a cell fail closed instead of sharing one box.

        The periodic Coulomb term takes exactly **one** lattice —
        ``Potentials._cell`` reads ``graphs.cell[0]`` — and returns one scalar
        for the whole batch, which :meth:`Potentials.calc_energy` then adds
        into its ``(B,)`` accumulator. Every graph is therefore handed the
        *whole* box's electrostatic energy, evaluated in graph 0's cell, and no
        other graph's cell is ever read: on the fixture below the two entries
        of the returned ``(2,)`` come out bit-identical, each counting all
        eight ions. A packed periodic training batch is exactly that shape, so
        the wrongness is silent and per-sample.

        Per-graph periodic electrostatics needs a per-graph reciprocal sum and
        a neighbour list that never bridges two graphs — finding F4 in
        ``.claude/notes/composable-potentials.md``, which owns its own spec.
        Until that lands the batch is refused rather than answered wrongly.
        The refusal is scoped to the periodic branch: open-boundary Coulomb
        already reduces per graph through ``edge_batch`` / ``num_graphs``.

        Fixture: two copies of the four-ion box concatenated (``atoms.batch =
        [0,0,0,0,1,1,1,1]``), the second copy translated by half the cell edge
        along every axis so no two ions coincide — a coincident pair would make
        the reciprocal sum singular and hide the broadcast behind a NaN — with
        the placed half list, its lattice shifts and its 1-2 bond duplicated
        and offset by four, and ``graphs`` at ``batch_size=[2]`` carrying a
        ``(2, 3, 3)`` cell.
        """
        n_atoms = len(COULOMB_POSITIONS)
        single = torch.tensor(COULOMB_POSITIONS, dtype=DTYPE)
        positions = torch.cat([single, single + COULOMB_EDGE / 2.0], dim=0)
        cell = torch.eye(3, dtype=DTYPE) * COULOMB_EDGE
        half_list, half_shifts = _coulomb_edges()
        edge_index = torch.cat([half_list, half_list + n_atoms], dim=0)
        shifts = torch.cat([half_shifts, half_shifts], dim=0)
        edge_diff = positions[edge_index[:, 1]] - positions[edge_index[:, 0]] + shifts
        batch = TensorDict(
            {
                "atoms": TensorDict(
                    {
                        "pos": positions,
                        "batch": torch.repeat_interleave(
                            torch.tensor([0, 1], dtype=torch.long), n_atoms
                        ),
                    },
                    batch_size=[2 * n_atoms],
                ),
                "edges": TensorDict(
                    {
                        "edge_index": edge_index,
                        "edge_diff": edge_diff,
                        "edge_dist": edge_diff.norm(dim=-1),
                        "shifts": shifts,
                    },
                    batch_size=[edge_index.shape[0]],
                ),
                "graphs": TensorDict(
                    {"cell": torch.stack([cell, cell])},
                    batch_size=[2],
                ),
                "bonds": TensorDict(
                    {
                        "atomi": torch.tensor([0, n_atoms], dtype=torch.long),
                        "atomj": torch.tensor([1, n_atoms + 1], dtype=torch.long),
                    },
                    batch_size=[2],
                ),
            },
            batch_size=[],
        )
        potentials = (
            _coulomb_forcefield()
            .with_params(atoms={"q": torch.tensor(COULOMB_CHARGES * 2, dtype=DTYPE)})
            .to_potentials()
        )

        with pytest.raises(ValueError, match=r"per-graph|num_graphs|one graph"):
            potentials.calc_energy(batch)

    def test_periodic_virial_without_stored_shifts_is_refused(self) -> None:
        """A placed list with no ``edges.shifts`` prices E and F, never W.

        ``Potentials._placed_pairs`` recovers the lattice remainder of
        a shift-less list as ``(edge_diff - (pos[t] - pos[s])).detach()``. That
        constant carries the right *value* — energy and forces below match the
        stored-``shifts`` batch to the bit — but it is not a function of the
        cell, so the strain applied by
        :meth:`Potentials.calc_energy_forces_virial` never reaches the pairs
        that reach their partner through a lattice vector. On this wrapped
        four-ion fixture that silently drops ~13.5 kcal/mol from ``tr W`` (the
        F13 boundary in ``.claude/notes/composable-potentials.md``): the
        virial is wrong, and nothing in the returned value says so.

        So the boundary is pinned as a boundary. The ``edge_diff`` fallback
        stays complete for energy and forces; the virial refuses a periodic
        batch whose list does not carry ``shifts``, and the message names the
        key that is missing.
        """
        potentials = _coulomb_forcefield().to_potentials()
        positions = torch.tensor(COULOMB_POSITIONS, dtype=DTYPE)
        cell = torch.eye(3, dtype=DTYPE) * COULOMB_EDGE
        edge_index, shifts = _coulomb_edges()
        stored = _coulomb_batch(positions, cell, edge_index, shifts)
        implied = _coulomb_batch(positions, cell, edge_index, shifts)
        del implied["edges", "shifts"]

        assert torch.allclose(
            potentials.calc_energy(implied).detach(),
            potentials.calc_energy(stored).detach(),
            atol=1e-10,
            rtol=0.0,
        )
        assert torch.allclose(
            potentials.calc_forces(implied).detach(),
            potentials.calc_forces(stored).detach(),
            atol=1e-8,
            rtol=0.0,
        )
        with pytest.raises(ValueError, match="shifts"):
            potentials.calc_energy_forces_virial(implied)

    @pytest.mark.slow
    @pytest.mark.parametrize("calculator", ("ewald", "pme"))
    def test_periodic_coulomb_traces_clean_and_the_step_breaks_only_in_force_py(
        self, calculator: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The bound term is one graph; the step around it breaks only at the seam.

        A cell-bound periodic Coulomb term is a fixed-shape computation: the
        real-space sum runs over the list's capacity with a mask, the
        reciprocal sum over the frozen grid, and the exclusion correction over
        the topology's pair table — no boolean selection, no host read, no
        Python branch on a tensor. It must therefore trace as **one** dynamo
        graph with zero breaks.

        **Every reciprocal route, not just the reference one.** ``bind_cell``
        freezes an integer shape for whichever algorithm the style names, and
        ``pair:coulomb`` picks ``"pme"`` by default under a cell — so a rule
        proved only on ``"ewald"`` says nothing about what a user actually
        runs. The mesh route has to hold the same line: the mesh interpolator
        and the k-space filter must take their grid counts from the frozen
        shape rather than reading a ``ns_mesh`` tensor back element by element.

        The term's arguments are captured from a real
        :meth:`~molpot.forcefield.Potentials.calc_energy` call rather than
        spelled out here: what the assembly hands the term is the assembly's
        business, and a test that re-spells it pins plumbing instead of
        behaviour.

        Around that graph the whole step still breaks, and every break must sit
        in ``molpot/derivation/force.py`` — the leaf creation and
        ``autograd.grad`` that dynamo cannot trace and that molpot deliberately
        keeps outside the compiled region. A break anywhere else (a boolean
        index in the term, a ``.item()`` on the k-grid, a style-registry lookup
        on the hot path) is a regression.
        """
        from molix.compile import Compiler

        potentials = _coulomb_forcefield(calculator=calculator).to_potentials()
        cell = torch.eye(3, dtype=DTYPE) * COULOMB_EDGE
        potentials.bind_cell(cell)
        edge_index, shifts = _coulomb_edges()
        positions = torch.tensor(COULOMB_POSITIONS, dtype=DTYPE)

        term = potentials.coulomb_periodic
        assert term is not None
        energy_of = term.energy
        captured: dict[str, object] = {}

        def capturing(**kwargs: object) -> torch.Tensor:
            captured.update(kwargs)
            return energy_of(**kwargs)

        monkeypatch.setattr(term, "energy", capturing)
        potentials.calc_energy(_coulomb_batch(positions, cell, edge_index, shifts))
        monkeypatch.undo()
        assert captured, "the periodic term was never reached"

        flat = Compiler.explain(energy_of, **captured)
        assert (flat.graph_count, flat.graph_break_count) == (1, 0), (
            f"the bound {calculator!r} term must trace as one graph; "
            f"break sites: {flat.break_sites}"
        )

        # Host-side preconditions are stated *before* the compiled region —
        # the same rule that puts ``bind_cell`` at the top of this test. The
        # F29 geometry gate (:meth:`Potentials._check_geometry`) is one host
        # sync per **frame**, memoized on the batch object, so a frame nothing
        # has vouched for runs that sync wherever it is first evaluated —
        # inside this trace, if that is where it happens, for four breaks at
        # ``potentials.py:952-953``. Every production entry vouches outside the
        # compiled region already: ``calc_energy_forces`` gates before entering
        # ``_classical_core``, MD binds one work batch for the whole rollout,
        # and a training step binds at its first evaluation. The bind here is
        # that step, not a concession.
        step_batch = _coulomb_batch(positions, cell, edge_index, shifts)
        potentials.bind_topology(step_batch)
        step = Compiler.explain(potentials.calc_energy_forces_virial, step_batch)
        assert step.graph_break_count > 0
        strays = _strays_outside_the_force_seam(step.break_sites)
        assert strays == [], (
            f"the periodic Coulomb step ({calculator}) may break only at the named autograd "
            f"seam in {FORCE_SEAM} (owner: molpot derivation); stray break sites: {strays}"
        )

    def test_lj_analytic_forces_match_a_finite_difference_across_the_boundary(self) -> None:
        """The LJ closed form must differentiate the **minimum-image** pair.

        ``lj/cut`` is the one pair style with an analytic force branch: the
        energy path reads the neighbour list's minimum-image ``edge_dist``,
        while the force path builds the pair direction itself. The two have to
        be the same pair. For a pair whose nearest image is across a box face
        the raw displacement ``pos[target] − pos[source]`` is the *long* way
        round — here 8 Å against a 2 Å minimum image — so a direction taken
        from it is both reversed and four times too long, and the analytic
        force stops being the gradient of the energy that is reported. In MD
        that is not a small error: a non-conservative force does work around
        every cycle and pumps energy into an NVE trajectory.

        The reference is a central difference of ``calc_energy`` — the same
        function whose gradient the closed form claims to be — taken on both
        atoms and all three axes, with the neighbour list rebuilt at each
        displaced geometry.
        """
        positions = torch.tensor(LJ_BOUNDARY_POSITIONS, dtype=DTYPE)
        potentials = _lj_boundary_potentials()
        _, forces = potentials.calc_energy_forces(_lj_boundary_batch(positions))

        def energy_at(stepped: torch.Tensor) -> float:
            return float(_lj_boundary_potentials().calc_energy(_lj_boundary_batch(stepped)))

        finite = _central_difference_forces(energy_at, positions, LJ_BOUNDARY_FD_STEP)

        gap = float((forces - finite).abs().max())
        assert gap < LJ_BOUNDARY_FORCE_TOL, (
            f"analytic LJ force is not -dE/dpos across the boundary: max gap {gap:.6g} "
            f"kcal/mol/A; analytic {forces.tolist()} vs finite difference {finite.tolist()}"
        )

    # -- S8 gate 1: symmetries of the assembled Class-I energy ---------------

    def test_assembled_energy_is_translation_invariant(self) -> None:
        """A rigid translation of the contents of a periodic box costs nothing.

        Every term of the assembled model is a function of separations:
        bonded terms of intramolecular displacements, the pair terms of
        minimum-image distances, and the Ewald reciprocal sum of ``|S(k)|²``,
        whose phase factor ``exp(i k · δ)`` cancels against its conjugate. None
        of them can see where the box's origin is, so a translation followed by
        a whole-molecule wrap must return the same number.

        The wrap is what makes this a *periodic* statement rather than a
        trivially exact one: the shift is drawn over the whole box, so both
        molecules leave it and come back on the far side, and the energy is
        then re-derived from a neighbour list rebuilt under the minimum-image
        convention.
        """
        positions = butane_pair_positions()
        generator = torch.Generator().manual_seed(TRANSLATION_SEED)
        shift = torch.rand(3, generator=generator, dtype=DTYPE) * BUTANE_PAIR_BOX
        moved = _translated_whole_molecules(positions.clone(), shift)

        assert float(moved.min()) >= 0.0 and float(moved.max()) < BUTANE_PAIR_BOX
        assert _butane_pair_energy(moved) == pytest.approx(
            _butane_pair_energy(positions), abs=SYMMETRY_TOL, rel=0.0
        )

    def test_assembled_energy_is_rotation_invariant(self) -> None:
        """A 90° rotation about z through the box centre costs nothing.

        **Why not a random rotation.** The energy is invariant under an
        arbitrary rotation only for an isolated cluster. Here the molecules sit
        in a lattice, and rotating the contents while leaving the box alone is
        a symmetry of the *system* only when the rotation maps the lattice onto
        itself — i.e. for an element of the cubic box's point group ``O_h``. A
        random ``SO(3)`` element would move the periodic images and change the
        minimum-image distances, so it would fail here for a reason that has
        nothing to do with the force field.

        A quarter turn about ``ẑ``, ``(x, y, z) → (L − y, x, z)``, is such an
        element: it permutes the lattice vectors and leaves the neighbour list
        the same set of separations.
        """
        positions = butane_pair_positions()
        centred = positions - BUTANE_PAIR_CENTRE
        rotated = (
            torch.stack([-centred[:, 1], centred[:, 0], centred[:, 2]], dim=1) + BUTANE_PAIR_CENTRE
        )

        assert _butane_pair_energy(rotated) == pytest.approx(
            _butane_pair_energy(positions), abs=SYMMETRY_TOL, rel=0.0
        )

    def test_assembled_energy_is_permutation_invariant(self) -> None:
        """Relabelling the atoms costs nothing when every table is relabelled with them.

        Permutation invariance is a statement about *bookkeeping*, not geometry:
        the same physical configuration written down in another atom order must
        price the same. That takes the permutation through all five places an
        atom index lives — the positions, the per-atom parameter tables
        (``q`` / ``epsilon`` / ``sigma``) and ``Z`` that travel with them, and
        the ``bonds`` / ``angles`` / ``propers`` columns, which are remapped by
        the *inverse* permutation because they name where an atom went rather
        than which atom a row came from. The bonded parameter rows are not
        touched: rows keep their identity, only the indices in them move.
        """
        reference = butane_pair_batch()
        positions = butane_pair_positions()
        n_atoms = int(positions.shape[0])
        generator = torch.Generator().manual_seed(PERMUTATION_SEED)
        permutation = torch.randperm(n_atoms, generator=generator)
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(n_atoms)

        permuted = butane_pair_batch(positions[permutation])
        for field in ("Z", "mass"):
            permuted["atoms", field] = _leaf(reference, "atoms", field)[permutation]
        for namespace, columns in BUTANE_PAIR_TOPOLOGY_COLUMNS.items():
            for column in columns:
                permuted[namespace, column] = inverse[_leaf(reference, namespace, column)]
        forcefield = butane_pair_class_i_forcefield()
        per_atom = forcefield.params("atoms") or {}
        forcefield = forcefield.with_params(
            atoms={name: table[permutation] for name, table in per_atom.items()}
        )

        assert float(forcefield.to_potentials().calc_energy(permuted)) == pytest.approx(
            float(butane_pair_class_i_forcefield().to_potentials().calc_energy(reference)),
            abs=SYMMETRY_TOL,
            rel=0.0,
        )

    def test_assembled_energy_is_reflection_invariant(self) -> None:
        """The Class-I template is achiral: inversion and a mirror both cost nothing.

        A Class-I template whose torsional and improper phases are ``0`` or
        ``π`` is achiral. Improper rotations flip the sign of every signed
        dihedral, ``φ → −φ``, and leave every distance and every bond angle
        alone; the only ``φ``-dependent term here is the proper torsion
        ``Σ_m k_m [1 + cos(n_m φ − γ_m)]``, and ``cos(nφ − γ)`` is even in
        ``φ`` for exactly those two phases
        (:data:`~tests.conftest.BUTANE_PAIR_TORSION_PHASE`), so the whole
        assembled energy is invariant. A phase anywhere else — or a harmonic
        improper with ``χ₀ ≠ 0``, which
        :meth:`test_harmonic_improper_is_chiral` shows — would not be, so this
        gate is scoped to the template's phases rather than to the machinery.

        Both improper rotations of the cube are taken about the box centre:
        the inversion ``pos → −pos`` and the one-axis mirror ``x → −x``. The
        cubic lattice is invariant under both.
        """
        positions = butane_pair_positions()
        reference = _butane_pair_energy(positions)
        inverted = 2.0 * BUTANE_PAIR_CENTRE - positions
        mirrored = positions.clone()
        mirrored[:, 0] = 2.0 * BUTANE_PAIR_CENTRE - mirrored[:, 0]

        assert _butane_pair_energy(inverted) == pytest.approx(reference, abs=SYMMETRY_TOL, rel=0.0)
        assert _butane_pair_energy(mirrored) == pytest.approx(reference, abs=SYMMETRY_TOL, rel=0.0)

    def test_harmonic_improper_is_chiral(self) -> None:
        """``improper:harmonic`` with ``χ₀ ≠ 0`` tells the two mirror images apart.

        The negative control for the reflection gate above. ``χ`` is a
        **signed** dihedral, so an improper rotation sends ``χ → −χ`` and
        ``E = ½ k (χ − χ₀)²`` to ``½ k (χ + χ₀)²``: the two agree only at
        ``χ₀ = 0``. A force field that declares a non-zero equilibrium improper
        angle is therefore chiral by construction, and the reflection gate must
        be read as a property of this template's phases — not as something the
        assembled-energy machinery enforces on every model.

        The smallest fixture that can show it: one non-planar quartet with the
        improper centre at ``atomi`` (the molnex column convention).
        """
        forcefield = (
            ForceField(name="chiral-improper", units="real")
            .def_improperstyle("harmonic")
            .with_params(
                impropers={
                    "k": torch.full((1,), CHIRAL_IMPROPER_K, dtype=DTYPE),
                    "chi0": torch.full((1,), CHIRAL_IMPROPER_CHI0, dtype=DTYPE),
                }
            )
        )
        upright = _torsion_batch("impropers")
        reflected = _torsion_batch("impropers")
        reflected["atoms", "pos"] = -reflected["atoms", "pos"]

        energy = float(forcefield.to_potentials().calc_energy(upright))
        mirror_energy = float(forcefield.to_potentials().calc_energy(reflected))
        assert abs(mirror_energy - energy) > CHIRAL_TOL


def _strays_outside_the_force_seam(sites: tuple[str, ...]) -> list[str]:
    """Break sites that are not the retained autograd seam of ``force.py``.

    ``calc_energy_forces_virial`` builds a ``requires_grad`` leaf and calls
    ``torch.autograd.grad``; dynamo traces neither, so the whole step is
    *expected* to break there — and nowhere else. One definition, because the
    mlip route and the classical route assert the same rule about the same
    seam.

    Args:
        sites: ``ExplainSummary.break_sites``, one ``file:line`` per break.

    Returns:
        The sites outside :data:`FORCE_SEAM`, in the order reported — empty
        when every break is a sanctioned one.
    """
    return [site for site in sites if not re.fullmatch(rf".*{re.escape(FORCE_SEAM)}:\d+", site)]


#: Type of the classical ``(system, pos) -> (energy, forces)`` core that
#: :meth:`~molpot.forcefield.Potentials.compile_core` hands a compiler.
ClassicalCore = Callable[..., tuple[torch.Tensor, torch.Tensor]]

#: Separation (Å) of the two-atom ``lj/cut`` fixture the classical core is
#: compiled on. Past the ``2^(1/6) σ = 3.37 Å`` minimum, so the pair sits on
#: the attractive shoulder with a non-zero force — at the well bottom every
#: build agrees on a zero and the parity leg would prove nothing.
COMPILED_CORE_R = 3.5


def _polar_forcefield() -> tuple[ForceField, nn.Module]:
    """A tiny periodic Polar mlip behind a ``metal``-unit ForceField.

    The only in-tree potential that is both :class:`~molix.schema.CellBound`
    and :class:`~molix.schema.CoreCompilable`, so the compiled-route criteria
    need it rather than a stub. Its hyper-parameters
    come from the mace test package's conftest — one definition of "tiny
    Polar", not a second copy that drifts.

    The caller must already have raised the global precision to fp64 (the
    ``fp64`` fixture): cuEquivariance freezes its working precision at
    construction, so a model built here at fp32 stays fp32.
    """
    from molzoo.mace.polar import MacePolar
    from molzoo.mace.spec import MACEPolarSpec
    from tests.test_molzoo.test_mace.conftest import (
        ATOMIC_ENERGIES,
        ATOMIC_NUMBERS,
        TINY_POLAR_KWARGS,
        polar_kwargs,
    )

    spec = MACEPolarSpec(
        atomic_numbers=ATOMIC_NUMBERS,
        atomic_energies=ATOMIC_ENERGIES,
        **TINY_POLAR_KWARGS,
    )
    torch.manual_seed(0)
    model = MacePolar(**polar_kwargs(spec, compute_forces=False)).eval()
    forcefield = ForceField("mace-polar", units="metal")
    forcefield.def_pairstyle(model)
    return forcefield, model


def _polar_batch(pos: torch.Tensor, cell: torch.Tensor) -> TensorDict:
    """A one-graph periodic batch on the mace-package cluster geometry."""
    from tests.test_molzoo.test_mace.conftest import CLUSTER_Z, SINGLE_GRAPH, full_edge_index

    edge_index = full_edge_index(SINGLE_GRAPH)
    diff = pos[edge_index[:, 1]] - pos[edge_index[:, 0]]
    n_atoms = len(CLUSTER_Z)
    return TensorDict(
        {
            "atoms": TensorDict(
                {
                    "Z": torch.tensor(CLUSTER_Z, dtype=torch.long),
                    "pos": pos.clone(),
                    "batch": torch.zeros(n_atoms, dtype=torch.long),
                },
                batch_size=[n_atoms],
            ),
            "edges": TensorDict(
                {
                    "edge_index": edge_index,
                    "edge_diff": diff.clone(),
                    "edge_dist": diff.norm(dim=-1),
                },
                batch_size=[edge_index.shape[0]],
            ),
            "graphs": TensorDict({"cell": cell.unsqueeze(0)}, batch_size=[1]),
        },
        batch_size=[],
    )


class TestPotentialsCompiledCore:
    """``bind_cell`` / ``compile_core`` delegation and the compiled mlip route.

    ``Potentials`` owns one line of each: a protocol-guarded hand-over to the
    mlip. It keeps no compiled state of its own — that lives on the mlip, which
    is what makes it survive ``to_potentials()`` rebuilds — and its ``mlip``
    branch of ``calc_energy`` has exactly one policy, ``protocol.call_energy``.
    """

    @pytest.fixture
    def fp64(self):
        """Raise the global precision for the Polar build, then put it back.

        ``molix.config`` is a process-wide singleton and cuEquivariance freezes
        its working precision at **construction**, so the default has to be fp64
        before the model exists. Leaving it raised would be a cross-file side
        effect — later modules would be built fp64 while their own fixtures feed
        fp32 tensors. Same save/restore as
        ``tests/test_molzoo/test_mace/conftest.py``.
        """
        from molix import config

        previous = config["ftype"]
        config.set_precision("fp64")
        yield
        config.set_precision("fp64" if previous == torch.float64 else "fp32")

    def test_bind_cell_and_compile_core_delegate_behind_the_protocol_guards(self) -> None:
        """Guarded, one-layer, side-effect-free — and molpot stays downstream.

        ``def_pairstyle`` accepts any ``nn.Module``, so both protocols are
        optional on an mlip: ``bind_cell`` must silently do nothing (and change
        no result) for one that lacks it, while ``compile_core`` must refuse
        loudly rather than report success having compiled nothing — an mlip
        owns the whole energy, so a container that cannot reach its core has
        nothing left to compile. A **classical** field is the other case
        entirely: it owns its core and compiles it
        (:meth:`test_a_classical_field_compiles_its_own_core_and_routes_through_it`),
        so only its ``bind_cell`` is a no-op here. When the mlip does satisfy
        both protocols, exactly one call is forwarded, with the caller's own
        storage. The import guards keep the package direction
        ``molix <- molrep <- molzoo/molpot`` intact: the protocols come from
        ``molix.schema``, never from ``molix.md`` or ``molix.compile``.
        """
        batch = _edge_batch()
        toy = ForceField().def_pairstyle(_EdgeSpring()).to_potentials()
        before = toy.calc_forces(batch).detach().clone()
        toy.bind_cell(torch.eye(3, dtype=DTYPE))
        assert torch.equal(toy.calc_forces(batch).detach(), before)
        with pytest.raises(RuntimeError, match="compile_core"):
            toy.compile_core(lambda core: core)

        # A bonded-only field holds nothing sized by the lattice: the
        # documented no-op, which is a raise-free call and nothing else.
        classical = _bond_potentials()
        classical.bind_cell(torch.eye(3, dtype=DTYPE))

        bound: list[int] = []
        compilers: list[object] = []

        class _CellBoundCompilable(nn.Module):
            def bind_cell(self, cell: torch.Tensor) -> None:
                bound.append(cell.data_ptr())

            def compile_core(self, compiler: object) -> None:
                compilers.append(compiler)

        mlip = _CellBoundCompilable()
        pots = ForceField().def_pairstyle(mlip).to_potentials()
        cell = torch.eye(3, dtype=DTYPE) * 10.0
        pots.bind_cell(cell)
        compiler = object()
        pots.compile_core(compiler)
        assert bound == [cell.data_ptr()]
        assert compilers == [compiler]
        assert not hasattr(pots, "_compiled_core")
        assert isinstance(pots, CellBound)

    def test_calc_energy_routes_through_call_energy_and_the_pos_override_lands_first(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One policy for the mlip branch, and ``pos=`` reaches the batch first.

        Which entry of an mlip evaluates the energy is ``protocol.call_energy``'s
        decision alone; re-implementing that probe here is how the two copies
        drift. The override has to be written onto ``atoms.pos`` *before* the
        call, because the mlip reads its positions off the batch — writing
        afterwards would be silently ignored and the energy would be the one at
        the old geometry.
        """
        import molpot.forcefield.potentials as module

        calls: list[object] = []
        real = module.call_energy

        def counting(model: object, system: TensorDict) -> TensorDict:
            calls.append(model)
            return real(model, system)

        class _PosSpring(nn.Module):
            """Energy read off ``atoms.pos`` only — so a stale override shows up."""

            def _write_energy(self, system: TensorDict) -> TensorDict:
                system[ENERGY_KEY] = 0.5 * system["atoms", "pos"].pow(2).sum().reshape(1)
                return system

        monkeypatch.setattr(module, "call_energy", counting)
        mlip = _PosSpring()
        pots = ForceField().def_pairstyle(mlip).to_potentials()
        batch = _edge_batch()

        energy = pots.calc_energy(batch)
        assert calls == [mlip]
        assert energy is batch[ENERGY_KEY]

        shifted = batch["atoms", "pos"] + torch.tensor(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=DTYPE
        )
        moved = pots.calc_energy(batch, pos=shifted)
        assert len(calls) == 2
        assert batch["atoms", "pos"] is shifted
        assert not torch.allclose(moved.detach(), energy.detach())

    def test_a_classical_field_compiles_its_own_core_and_routes_through_it(self) -> None:
        """A field with no mlip hands out ``_classical_core`` and then calls it.

        The classical analytic step is a hot path, so ``compile_core`` must have
        something to install for a ForceField that declares only styles: the
        whole ``(system, pos) -> (energy, forces)`` evaluation, as one bound
        method of the container. Two things are pinned, because either one alone
        can be true while the seam is useless. The compiler is applied **once**,
        to a bound method of *this* ``Potentials`` — not to a lambda closing
        over a rebuilt one, which would be a second evaluation path. And
        ``calc_energy_forces`` then goes through the build the compiler returned
        — a container that installed it and kept calling the eager method would
        report a compiled run and never have one.

        The compiler here is the identity, so the values must be untouched: one
        function, two builds, and the only licensed difference between them is
        reduction order. ``1e-12`` on a two-atom LJ pair is well under that.
        """
        installed: list[ClassicalCore] = []
        routed: list[int] = []

        def recording(core: ClassicalCore) -> ClassicalCore:
            installed.append(core)

            def through(*args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
                routed.append(1)
                return core(*args, **kwargs)

            return through

        potentials = _lj_pair_forcefield().to_potentials()
        batch = _lj_pair_batch(r=COMPILED_CORE_R)
        eager_energy, eager_forces = potentials.calc_energy_forces(batch)
        eager = (eager_energy.detach().clone(), eager_forces.detach().clone())

        potentials.compile_core(recording)
        assert len(installed) == 1
        assert getattr(installed[0], "__self__", None) is potentials

        energy, forces = potentials.calc_energy_forces(batch)
        assert routed == [1]
        assert torch.allclose(energy.detach(), eager[0], atol=1e-12, rtol=0.0)
        assert torch.allclose(forces.detach(), eager[1], atol=1e-12, rtol=0.0)

    @pytest.mark.slow
    def test_the_installed_classical_core_traces_with_no_break_outside_force_py(self) -> None:
        """What ``compile_core`` hands a compiler has to be compilable.

        The iron law is per hot path, and the classical analytic step is one: a
        ``lj/cut`` field is fixed-shape arithmetic over the list's capacity with
        a liveness mask — no boolean selection, no host read, no Python branch
        on a tensor value — and must trace as **one** graph with zero breaks. A
        break there is not a slow path, it is the compiled region collapsing to
        the eager one.

        A declared ``pair:coulomb`` has no closed-form force, so the core builds
        a position leaf and calls ``autograd.grad`` through
        ``molpot/derivation/force.py``: dynamo cannot trace either, that break
        is deliberate and named, and it is the *only* one a classical field may
        show. Owner: molpot derivation. The core is taken from the compiler
        itself rather than off the container, so this asks about the function
        the seam actually installs.

        ``bind_topology`` first, which is the composition
        :meth:`~molpot.forcefield.Potentials.bind_topology` documents for a
        compiled path: the exclusion table and the orientation check are
        topology work, they are host-side by design, and left to the lazy call
        inside the first evaluation they would be traced as breaks of the core.
        Everything that survives that bind is per-step work and has to compile.
        """
        from molix.compile import Compiler

        captured: list[ClassicalCore] = []

        def capture(core: ClassicalCore) -> ClassicalCore:
            captured.append(core)
            return core

        lj = _lj_pair_forcefield().to_potentials()
        lj_batch = _lj_pair_batch(r=COMPILED_CORE_R)
        lj.bind_topology(lj_batch)
        lj.compile_core(capture)
        flat = Compiler.explain(captured[-1], lj_batch, _leaf(lj_batch, "atoms", "pos"))
        assert (flat.graph_count, flat.graph_break_count) == (1, 0), (
            "a bonded + lj/cut core has no autograd seam and must trace as one graph; "
            f"break sites: {flat.break_sites}"
        )

        coulomb = _coulomb_forcefield().to_potentials()
        cell = torch.eye(3, dtype=DTYPE) * COULOMB_EDGE
        coulomb.bind_cell(cell)
        edge_index, shifts = _coulomb_edges()
        periodic = _coulomb_batch(
            torch.tensor(COULOMB_POSITIONS, dtype=DTYPE), cell, edge_index, shifts
        )
        coulomb.bind_topology(periodic)
        coulomb.compile_core(capture)
        core = Compiler.explain(captured[-1], periodic, _leaf(periodic, "atoms", "pos"))
        assert core.graph_break_count > 0
        strays = _strays_outside_the_force_seam(core.break_sites)
        assert strays == [], (
            f"the classical core may break only at the named autograd seam in {FORCE_SEAM} "
            f"(owner: molpot derivation); stray break sites: {strays}"
        )

    @pytest.mark.slow
    def test_the_compiled_route_matches_eager_and_breaks_only_in_force_py(self, fp64) -> None:
        """The prescribed sequence: parity, the break split, and fail-closed.

        ``bind_cell -> one eager call -> compile_core`` is the caller's
        composition. After it, the cell-bound Polar core is one dynamo graph
        with no break, while the whole step still breaks — and every break must
        be the autograd seam in ``molpot/derivation/force.py`` (leaf creation
        and ``autograd.grad``), which Q3 deliberately keeps outside the compiled
        region. Owner: molpot derivation. A break anywhere else is a regression.

        Compiled and eager agree to 1e-10 in fp64 (reduction order only). The
        installed core lives on the mlip, so a ``Potentials`` rebuilt by
        ``to_potentials()`` still dispatches through it. And an over-capacity
        cell fails closed twice over: loudly at the eager rebind, as an
        in-graph NaN if the rebind is skipped — never as a truncated finite
        energy.

        The MD route is first-order **inference**: those evaluations run under
        ``torch.no_grad()``. Outside it the ambient grad mode asks for
        ``create_graph=True``, which AOTAutograd serves only while buffer
        donation is off — ``register_inductor_fallbacks`` pins
        ``torch._functorch.config.donated_buffer = False`` for exactly that
        reason. Pinned here rather than left as folklore: it is the difference
        between "MD" and "force-loss training" reaching the same compiled core
        (spec ``cuet-force-doublebackward``).
        """
        from molix.compile import Compiler
        from tests.test_molzoo.test_mace.conftest import CLUSTER_POS

        forcefield, mlip = _polar_forcefield()
        pots = forcefield.to_potentials()
        pos = torch.tensor(CLUSTER_POS, dtype=DTYPE)
        cell = torch.eye(3, dtype=DTYPE) * POLAR_CELL_EDGE

        pots.bind_cell(cell)
        pots.calc_energy(_polar_batch(pos, cell))
        # MD inference is measured under no_grad; the grad-enabled route is
        # asserted after this block, once buffer donation is known to be off.
        with torch.no_grad():
            eager = tuple(
                tensor.detach().clone()
                for tensor in pots.calc_energy_forces_virial(_polar_batch(pos, cell))
            )

            pots.compile_core(Compiler(backend="aot_eager", dynamic=True))
            compiled = pots.calc_energy_forces_virial(_polar_batch(pos, cell))
            for got, want in zip(compiled, eager, strict=True):
                assert torch.allclose(got.detach(), want, atol=1e-10, rtol=0.0)

            work = _polar_batch(pos, cell)
            first = pots.calc_energy(work).detach().clone()
            assert torch.equal(work[ENERGY_KEY].detach(), first)
            # ONE atom moves. A rigid translation is *exactly* a symmetry of
            # this energy — the k-space |S(k)|² phases cancel and the local term
            # reads the frozen edge_diff — so shifting every atom asserts
            # nothing: the two values are bitwise equal on aarch64 and differ
            # only by last-bit round-off on x86.
            moved = pos.clone()
            moved[1, 0] += NUDGE
            second = pots.calc_energy(work, pos=moved).detach().clone()
            assert float((second - first).abs().max()) > ENERGY_CHANGE_MIN
            assert torch.equal(work[ENERGY_KEY].detach(), second)

            hits: list[int] = []
            installed = mlip._core

            def counting(*args: object, **kwargs: object) -> torch.Tensor:
                hits.append(1)
                return installed(*args, **kwargs)

            mlip._core = counting
            rebuilt = forcefield.to_potentials()
            assert rebuilt is not pots
            assert rebuilt.mlip is pots.mlip
            rebuilt.calc_energy(_polar_batch(pos, cell))
            assert hits == [1]

            over = cell * CELL_OVER_CAPACITY
            with pytest.raises(RuntimeError) as excinfo:
                pots.bind_cell(over)
            assert "KSpaceGrid" in str(excinfo.value)
            assert "capacity_factor" in str(excinfo.value)
            assert bool(torch.isnan(pots.calc_energy(_polar_batch(pos, over))).all())
            within = cell * CELL_WITHIN_CAPACITY
            pots.bind_cell(within)
            assert bool(torch.isfinite(pots.calc_energy(_polar_batch(pos, within))).all())

        # Grad enabled: the same compiled core now answers a create_graph=True
        # backward instead of refusing on donated buffers, and returns the
        # inference values.
        assert torch._functorch.config.donated_buffer is False
        with_grad = pots.calc_energy_forces_virial(_polar_batch(pos, cell))
        for got, want in zip(with_grad, eager, strict=True):
            assert torch.allclose(got.detach(), want, atol=1e-10, rtol=0.0)

        probe = _polar_batch(pos, cell)
        flat = Compiler.explain(
            mlip.energy_core,
            pos,
            probe["atoms", "Z"],
            probe["edges", "edge_index"],
            probe["atoms", "batch"],
            1,
            edge_diff=probe["edges", "edge_diff"],
            cell=cell.unsqueeze(0),
        )
        assert (flat.graph_count, flat.graph_break_count) == (1, 0)

        step = Compiler.explain(pots.calc_energy_forces_virial, _polar_batch(pos, cell))
        assert step.graph_break_count > 0
        strays = _strays_outside_the_force_seam(step.break_sites)
        assert strays == [], (
            f"the whole step may break only at the named autograd seam in {FORCE_SEAM} "
            f"(owner: molpot derivation, retained by Q3); stray break sites: {strays}"
        )
