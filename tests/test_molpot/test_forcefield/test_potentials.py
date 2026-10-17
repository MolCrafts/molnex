"""Tests for Potentials — the assembled Class-I energy and the op over it.

Mirrors ``src/molpot/forcefield/potentials.py``.

``Potentials.energy`` writes the assembled energy; every derivative —
forces for NVE, forces and virial for NPT — is ``molix.autograd.Derivative``
over that one energy, built in by ``wrt=`` at construction or composed
outside. No method hand-rolls ``torch.autograd.grad``. Numerics are pinned
here against the analytic harmonic bond, a periodic LJ pair and a periodic
Coulomb box.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Callable

import mollog
import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from molix.autograd import Derivative
from molix.frame import TensorFrame, as_tensor_frame
from molix.schema import ENERGY_KEY, FORCES_KEY, POS_KEY, VIRIAL_KEY, CellBound
from molpot.forcefield import ForceField, Potentials
from molpot.forcefield.partial_hessian import PartialHessianPairs
from molpot.forcefield.potentials import logger as potentials_logger
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
from tests.test_molpot.test_forcefield.conftest import with_graphs

DTYPE = torch.float64

# Analytic two-atom harmonic bond: E = ½k(r−r0)², k = 100 kcal mol⁻¹ Å⁻²,
# r0 = 1.0 Å, r = 1.5 Å along x̂ → E = 12.5 kcal/mol, |F| = k(r−r0) = 50.
BOND_K = 100.0
BOND_R0 = 1.0
BOND_R = 1.5
GOLDEN_ENERGY = 12.5
GOLDEN_FORCE_X = 50.0

FORCE_SEAM = "src/molix/autograd/variables.py"
FORCES = ("atoms.pos",)
FORCES_AND_VIRIAL = ("atoms.pos", "graphs.strain")

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
    return with_graphs(
        TensorDict(
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
    return with_graphs(
        TensorDict(
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
    )


def _angle_batch() -> TensorDict:
    """Three atoms carrying one ``angles`` entry — topology present, parameters absent."""
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [BOND_R, 0.0, 0.0], [BOND_R, BOND_R, 0.0]],
        dtype=DTYPE,
    )
    return with_graphs(
        TensorDict(
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
    )


def _opls_proper_forcefield(n: int = 1) -> ForceField:
    """``dihedral:opls`` with ``k1..k4`` of length *n* — the learned-head spelling."""
    ones = torch.ones(n, dtype=DTYPE)
    zeros = torch.zeros(n, dtype=DTYPE)
    return (
        ForceField(name="opls-proper")
        .def_dihedralstyle("opls")
        .with_params(propers={"k1": ones, "k2": zeros, "k3": zeros, "k4": zeros})
    )


def _two_proper_batch() -> TensorDict:
    """Five-atom chain, two overlapping proper quartets (n_propers=2)."""
    pos = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.5, 0.5],
            [2.0, 0.5, 0.0],
        ],
        dtype=DTYPE,
    )
    return with_graphs(
        TensorDict(
            {
                "atoms": TensorDict(
                    {
                        "pos": pos,
                        "Z": torch.ones(5, dtype=torch.long),
                        "batch": torch.zeros(5, dtype=torch.long),
                    },
                    batch_size=[5],
                ),
                "propers": TensorDict(
                    {
                        "atomi": torch.tensor([0, 1], dtype=torch.long),
                        "atomj": torch.tensor([1, 2], dtype=torch.long),
                        "atomk": torch.tensor([2, 3], dtype=torch.long),
                        "atoml": torch.tensor([3, 4], dtype=torch.long),
                    },
                    batch_size=[2],
                ),
            },
            batch_size=[],
        )
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
    return with_graphs(
        TensorDict(
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
    )


def _bond_potentials(
    k: torch.Tensor | None = None, *, n_bonds: int = 1, wrt: tuple[str, ...] = ()
) -> Potentials:
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
        .to_potentials(wrt=wrt)
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

    Carries the placed ``pairs`` namespace :meth:`Potentials._placed_pairs`
    consumes, built at :data:`LJ_CUTOFF` — the cutoff ``_lj_pair_forcefield``
    declares, so the list holds exactly the row that style may read.
    """
    pos = torch.tensor([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=DTYPE)
    batch = with_graphs(
        TensorDict(
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
                "charge": torch.tensor(CHARGES, dtype=DTYPE),
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
    batch = with_graphs(
        TensorDict(
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
        .with_params(atoms={"charge": torch.tensor([0.5, 0.5], dtype=DTYPE)})
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


def _reversed_edge_batch() -> TensorDict:
    """A placed list carrying only ``source > target`` rows.

    :meth:`Potentials._check_pair_orientation` refuses it: ``_placed_pairs``
    prices the ``source < target`` half, so every pair term on this batch would
    cost exactly zero. Used as the vehicle for "a refusal arms nothing".
    """
    batch = _charged_batch(total_charge=1.0)
    pairs = batch["pairs"]
    index = pairs["pair_index"]
    reversed_rows = index[:, 0] > index[:, 1]
    if not bool(reversed_rows.any()):
        index = index.flip(-1)
        reversed_rows = index[:, 0] > index[:, 1]
    keep = torch.nonzero(reversed_rows, as_tuple=True)[0]
    batch["pairs"] = TensorDict(
        {name: pairs[name][keep] for name in pairs.keys()},
        batch_size=[int(keep.numel())],
    )
    return batch


def _butane_forcefield() -> ForceField:
    """``lj/cut`` + ``coulomb`` on the butane fixture, Class-I default scaling."""
    return (
        ForceField(name="butane")
        .def_pairstyle("lj/cut", cutoff=BUTANE_CUTOFF)
        .def_pairstyle("coulomb")
        .with_params(
            atoms={
                "charge": torch.tensor(BUTANE_Q, dtype=DTYPE),
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
        with_graphs(
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
                },
                batch_size=[],
            ),
            cell=torch.eye(3, dtype=DTYPE) * LJ_BOUNDARY_EDGE,
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

    Bond-only, so it carries no placed ``pairs`` namespace: no pair style asks
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
    return with_graphs(
        TensorDict(
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
        ),
        cell=cell,
    )


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
        .with_params(atoms={"charge": torch.tensor(COULOMB_CHARGES, dtype=DTYPE)})
    )


def _coulomb_edges() -> tuple[torch.Tensor, torch.Tensor]:
    """The fixed half list of the periodic fixture, padded to capacity.

    Enumerated once from the reference geometry, the way MD enumerates once per
    rebuild and then reuses the rows: ``pair_index`` and the lattice ``shifts``
    are *connectivity*, and every later energy recomputes only
    ``pair_diff = pos[target] - pos[source] + shift`` from them. Handing the
    batch a placed list — rather than letting the builder re-enumerate — is
    also what keeps the distances on the position autograd graph, which the
    enumerated path (numpy, via molpy) cannot be.

    Returns:
        ``(pair_index, shifts)`` — ``(C, 2)`` int64 and ``(C, 3)`` Å at the
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
    pair_index = torch.zeros(COULOMB_LIST_CAPACITY, 2, dtype=torch.long)
    shifts = torch.zeros(COULOMB_LIST_CAPACITY, 3, dtype=DTYPE)
    row = 0
    for source in range(positions.shape[0]):
        for target in range(source + 1, positions.shape[0]):
            for shift in lattice:
                if float((positions[target] - positions[source] + shift).norm()) < COULOMB_CUTOFF:
                    pair_index[row] = torch.tensor([source, target], dtype=torch.long)
                    shifts[row] = shift
                    row += 1
    assert row < COULOMB_LIST_CAPACITY, "the fixture must keep at least one dead padding row"
    return pair_index, shifts


def _coulomb_batch(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pair_index: torch.Tensor,
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
        pair_index: Placed half list ``(C, 2)``.
        shifts: Lattice remainder of each row ``(C, 3)`` in Å.

    Returns:
        The batch, with ``edges.pair_diff`` rebuilt from ``positions`` and
        ``shifts`` so it is a function of the arguments, never a stale buffer.
    """
    n_atoms = positions.shape[0]
    pair_diff = positions[pair_index[:, 1]] - positions[pair_index[:, 0]] + shifts
    return TensorDict(
        {
            "atoms": TensorDict(
                {"pos": positions, "batch": torch.zeros(n_atoms, dtype=torch.long)},
                batch_size=[n_atoms],
            ),
            "pairs": TensorDict(
                {
                    "pair_index": pair_index,
                    "pair_diff": pair_diff,
                    "pair_dist": pair_diff.norm(dim=-1),
                    "shifts": shifts,
                },
                batch_size=[pair_index.shape[0]],
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
    """The minimal mlip shape ``def_pairstyle`` accepts: ``energy`` only.

    Not :class:`molix.schema.CellBound` — which is the point: it is the
    in-tree proof that the protocol is optional on an mlip, so ``Potentials``
    must guard its delegation instead of calling through unconditionally.
    It reads ``edges.pair_diff`` as handed — the derivative op rebuilds that
    from its own leaf, so ``∂/∂pos`` flows without any trick here.
    """

    def energy(self, system: TensorDict) -> TensorDict:
        energy = 0.5 * system["pairs", "pair_diff"].pow(2).sum()
        system[ENERGY_KEY] = energy.reshape(1)
        return system


def _edge_batch(*, r: float = 1.5) -> TensorDict:
    """Two atoms, both edges, a unit cell — the ``_EdgeSpring`` system."""
    pos = torch.tensor([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=DTYPE)
    pair_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    shifts = torch.zeros(2, 3, dtype=DTYPE)
    pair_diff = pos[pair_index[:, 1]] - pos[pair_index[:, 0]] + shifts
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
            "pairs": TensorDict(
                {
                    "pair_index": pair_index,
                    "pair_diff": pair_diff.clone(),
                    "pair_dist": pair_diff.norm(dim=-1),
                    "shifts": shifts.clone(),
                },
                batch_size=[2],
            ),
            "graphs": TensorDict({"cell": torch.eye(3, dtype=DTYPE).unsqueeze(0)}, batch_size=[1]),
        },
        batch_size=[],
    )


class TestPotentials:
    def test_forces_match_the_analytic_golden(self) -> None:
        """F = −∂E/∂x = ∓k(r−r0) x̂ for the stretched two-atom harmonic bond."""
        forces = _bond_potentials(wrt=FORCES)(_two_atom_bond_batch())[FORCES_KEY]
        expected = torch.tensor(
            [[GOLDEN_FORCE_X, 0.0, 0.0], [-GOLDEN_FORCE_X, 0.0, 0.0]],
            dtype=DTYPE,
        )
        assert forces.shape == (2, 3)
        assert torch.allclose(forces.detach(), expected, atol=1e-10, rtol=0.0)

    def test_forward_is_the_same_op_composed_outside(self) -> None:
        """``wrt`` at construction and ``Derivative(pots.energy, wrt=...)`` are one code path."""
        batch = _two_atom_bond_batch()
        inside = _bond_potentials(wrt=FORCES)(batch.clone())
        outside = Derivative(_bond_potentials().energy, wrt=FORCES)(batch.clone())
        assert torch.equal(inside[FORCES_KEY], outside[FORCES_KEY])
        assert torch.equal(inside[ENERGY_KEY], outside[ENERGY_KEY])

    def test_derivative_exposes_the_init_fixed_op(self) -> None:
        """The training step marks the leaves through ``derivative`` — the op ``forward`` runs."""
        assert _bond_potentials(wrt=FORCES).derivative.wrt == (POS_KEY,)

    def test_the_mlip_branch_feeds_the_op(self) -> None:
        """An mlip ForceField declares no classical style, so the term loop is empty.

        Falling through it would return ``(0, zeros)`` — a silent wrong answer,
        not a missing feature. ``energy`` must hand the batch to the mlip and
        the op over it must derive non-zero forces from that energy.
        """
        batch = _edge_batch()
        pots = ForceField().def_pairstyle(_EdgeSpring()).to_potentials(wrt=FORCES)
        out = pots(batch)
        assert float(out[ENERGY_KEY].detach().reshape(())) == pytest.approx(
            float(pots.calc_energy(batch).detach().reshape(())), abs=1e-10
        )
        assert float(out[FORCES_KEY].detach().abs().max()) > 0.0

    def test_force_loss_double_backward_reaches_parameters(self) -> None:
        """``create_graph`` keeps ∂²E/∂x∂θ alive so a force loss reaches θ."""
        parameter = nn.Parameter(torch.tensor([BOND_K], dtype=DTYPE))
        pots = _bond_potentials(parameter * 1.0, wrt=FORCES)
        forces = pots(_two_atom_bond_batch())[FORCES_KEY]
        torch.mean(forces**2).backward()
        assert parameter.grad is not None
        # d/dk mean(F²) with F = ±k(r−r0): mean over 6 components = k²·0.5²·2/6
        assert float(parameter.grad) == pytest.approx(2.0 * BOND_K * 0.25 * 2.0 / 6.0, abs=1e-10)

    def test_virial_forces_match_the_force_op_when_energy_uses_edge_diff(self) -> None:
        """NPT strain must keep ∂E/∂pos through imaged edge geometry.

        Polar / MACE read ``edges.pair_diff``. If that tensor is strained as an
        independent cartesian, local graph energy drops off ``pos`` and MD
        integrates electrostatics-only forces.
        """
        pots = ForceField().def_pairstyle(_EdgeSpring()).to_potentials()
        reference = Derivative(pots.energy, wrt=FORCES)(_edge_batch())[FORCES_KEY]
        npt = Derivative(pots.energy, wrt=FORCES_AND_VIRIAL)(_edge_batch())
        assert torch.allclose(npt[FORCES_KEY], reference, atol=1e-10, rtol=0.0)
        assert npt[VIRIAL_KEY].shape == (1, 3, 3)
        assert npt[VIRIAL_KEY].abs().sum().item() > 0.0

    def test_forces_on_a_multi_graph_batch(self) -> None:
        """Per-graph energies ``(B,)`` reduce to one scalar before differentiating."""
        forces = _bond_potentials(n_bonds=2, wrt=FORCES)(_two_molecule_bond_batch())[FORCES_KEY]
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
        assert re.search(r"\bcharge\b", message) is not None

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
        far_apart = with_graphs(
            TensorDict(
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
        on the first energy call, hands ``"charge"`` to Coulomb and ``"lj"`` to LJ,
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
        """No ``pairs`` namespace is a missing neighbour list, not an invitation to build one.

        ``molpot`` is downstream of ``molix`` in the dependency graph and does
        not own pair enumeration: the cutoff graph is built by
        :class:`molix.data.NeighborList` on the training path and by
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
        del bare["pairs"]

        with pytest.raises(ValueError) as excinfo:
            potentials.calc_energy(bare)
        message = str(excinfo.value)
        assert "molix.data.NeighborList" in message
        assert "molix.md.NeighborList" in message

    def test_placed_pairs_returns_the_list_at_capacity_with_a_live_mask(self) -> None:
        """Padding survives in place; only the mask says which rows count.

        Moved from the deleted ``test_neighbor_list.py`` (``composable-a0-02``
        contract, now owned by :meth:`Potentials._placed_pairs`). Live is
        exactly ``source < target`` (the half list), ``0 < r < cutoff`` and
        inside ``num_pairs``. Row 1 is the reversed duplicate of row 0, rows
        2-3 are outside the cutoff, rows 4-5 are dead capacity — none may be
        dropped, because dropping them makes the output shape depend on the
        data, which is a graph break on the compiled force path.

        The list carries ``pair_diff`` beside ``pair_dist`` because that is
        what :class:`molix.data.NeighborList` writes and because an
        ``pair_dist``-only list is refused
        (:meth:`test_placed_pairs_refuses_an_edge_dist_only_list`); the
        capacity / mask contract asserted here is independent of which geometry
        branch prices the rows.
        """
        potentials = _lj_pair_forcefield().to_potentials()
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=DTYPE)
        pair_index = torch.tensor(
            [[0, 1], [1, 0], [0, 2], [2, 0], [0, 0], [0, 0]], dtype=torch.long
        )
        system = {
            "atoms": {"pos": pos, "batch": torch.zeros(3, dtype=torch.long)},
            "pairs": {
                "pair_index": pair_index,
                "pair_diff": pos[pair_index[:, 1]] - pos[pair_index[:, 0]],
                "pair_dist": torch.tensor([1.0, 1.0, 3.0, 3.0, 0.0, 0.0], dtype=DTYPE),
                "num_pairs": 4,
            },
        }

        returned_index, _vectors, dist, live_mask = potentials._placed_pairs(
            as_tensor_frame(system), pos, cutoff=2.5
        )

        assert returned_index.shape == (6, 2)
        assert dist.shape == (6,)
        assert live_mask.dtype == torch.bool
        assert torch.equal(returned_index, pair_index)
        assert torch.equal(live_mask, torch.tensor([True, False, False, False, False, False]))

    def test_placed_pairs_refuses_an_edge_dist_only_list(self) -> None:
        """A stored scalar distance is not geometry, so the list is refused.

        Forwarding ``pair_dist`` as the answer reads a number that is not a
        function of the ``pos`` leaf the energy is differentiated against, so
        ``dE/dpos`` is structurally zero: such a batch used to price a finite
        energy and hand back **zero forces**, silently, which is the worst
        possible failure on an MD path. Nothing in the tree produces this list
        either — :class:`molix.data.NeighborList` writes ``pair_diff`` beside
        ``pair_dist`` and :class:`molix.md.NeighborList` writes ``shifts`` as
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
            "pairs": {
                "pair_index": torch.tensor([[0, 1]], dtype=torch.long),
                "pair_dist": torch.tensor([2.0], dtype=DTYPE),
            },
        }

        with pytest.raises(ValueError) as excinfo:
            potentials._placed_pairs(as_tensor_frame(system), pos, cutoff=5.0)
        message = str(excinfo.value)
        assert "pair_diff" in message or "shifts" in message

    def test_placed_pairs_prefers_shifts_over_a_stored_edge_diff(self) -> None:
        """F13: ``shifts`` is the branch, ``pair_diff`` only the fallback.

        Both keys are present and they disagree — ``pair_diff`` claims a 4 Å
        separation the geometry plus ``shifts`` does not support. The
        ``shifts`` branch rebuilds the pair vector as
        ``pos[t] - pos[s] + shifts`` and must win, because it is the only one
        that is a function of the *live* lattice; the ``pair_diff``
        reconstruction can recover the lattice remainder only as a detached
        constant.
        """
        potentials = _lj_pair_forcefield().to_potentials()
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=DTYPE)
        pair_index = torch.tensor([[0, 1]], dtype=torch.long)
        shifts = torch.tensor([[10.0, 0.0, 0.0]], dtype=DTYPE)
        system = {
            "atoms": {"pos": pos, "batch": torch.zeros(2, dtype=torch.long)},
            "pairs": {
                "pair_index": pair_index,
                "shifts": shifts,
                "pair_diff": torch.tensor([[4.0, 0.0, 0.0]], dtype=DTYPE),
                "pair_dist": torch.tensor([4.0], dtype=DTYPE),
            },
        }

        _, _vectors, dist, _live = potentials._placed_pairs(
            as_tensor_frame(system), pos, cutoff=20.0
        )

        # pos[1] - pos[0] + shift = 1 + 10 = 11, not the stored 4.
        assert float(dist[0]) == pytest.approx(11.0, abs=1e-12)

    def test_placed_pairs_keeps_the_shift_on_the_autograd_graph(self) -> None:
        """F13: the ``shifts`` branch is not detached, so ``dE/deps`` sees the lattice.

        A pair that reaches its partner through a lattice vector contributes a
        shift term to the strain derivative. The ``pair_diff`` fallback can only
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
            "pairs": {"pair_index": torch.tensor([[0, 1]], dtype=torch.long), "shifts": shifts},
        }

        _, _vectors, dist, _live = potentials._placed_pairs(
            as_tensor_frame(system), pos, cutoff=20.0
        )
        (grad,) = torch.autograd.grad(dist.sum(), shifts)

        # d|d|/d(shift_x) = (pos[t] - pos[s] + shift)_x / |...| = 11 / 11 = 1.
        assert float(grad[0, 0]) == pytest.approx(1.0, abs=1e-12)

    def test_placed_pairs_on_an_empty_list_is_empty_not_absent(self) -> None:
        """A list with no rows means "no pair in range", never "no list".

        :class:`molix.data.NeighborList` returns zero rows for a system whose
        atoms are all outside the cutoff, and that is a perfectly ordinary
        batch — the pair terms must price it at zero. Only a *missing* ``pairs``
        namespace is the fail-closed case
        (:meth:`test_a_pair_style_on_a_batch_without_edges_names_both_molix_owners`),
        and conflating the two either resurrects the enumerator or refuses a
        dilute molecule.
        """
        potentials = _lj_pair_forcefield().to_potentials()
        pos = torch.tensor([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]], dtype=DTYPE)
        system = {
            "atoms": {"pos": pos, "batch": torch.zeros(2, dtype=torch.long)},
            "pairs": {
                "pair_index": torch.zeros(0, 2, dtype=torch.long),
                "pair_diff": torch.zeros(0, 3, dtype=DTYPE),
                "pair_dist": torch.zeros(0, dtype=DTYPE),
            },
        }

        pair_index, _vectors, dist, live = potentials._placed_pairs(
            as_tensor_frame(system), pos, cutoff=LJ_CUTOFF
        )

        assert pair_index.shape == (0, 2)
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
        # ``pair_diff - (pos[t] - pos[s])``, so a list left at the old
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

    def test_bind_topology_does_not_sum_seed_q(self) -> None:
        """Seed ForceField q is a previous topology; pad made N static (GH200 2390967).

        EspalomaCharge / ByteFF conserve ``sum q = Q_tot`` on the charges just
        assigned, not a leftover table. Bind must stay silent even when seed
        ``charge`` length matches this batch and the declared total does not.
        """
        captured: list[str] = []

        class _Capture(mollog.Handler):
            def emit(self, record: mollog.LogRecord) -> None:
                captured.append(record.message)

        capture = _Capture()
        capture.set_level(mollog.WARNING)
        potentials_logger.add_handler(capture)
        try:
            potentials = _anion_potentials()
            potentials.bind_topology(_charged_batch(total_charge=0.0))
            assert captured == []
        finally:
            potentials_logger.remove_handler(capture)

    def test_coulomb_warns_when_live_q_misses_the_declared_total(self) -> None:
        """``Σ_g q`` off ``graphs.total_charge`` is reported on the priced charges."""
        captured: list[str] = []

        class _Capture(mollog.Handler):
            def emit(self, record: mollog.LogRecord) -> None:
                captured.append(record.message)

        capture = _Capture()
        capture.set_level(mollog.WARNING)
        potentials_logger.add_handler(capture)
        try:
            potentials = _anion_potentials()
            potentials.energy(_charged_batch(total_charge=0.0))
            assert any("total_charge" in message for message in captured)
            assert any("drift 1 e" in message for message in captured)

            captured.clear()
            potentials.energy(_charged_batch(total_charge=1.0))
            potentials.energy(_charged_batch(total_charge=None))
            assert captured == []
        finally:
            potentials_logger.remove_handler(capture)

    def test_coulomb_masks_ghost_q_out_of_the_net_charge(self) -> None:
        """Padding ghosts (``atoms.mask`` False) must not enter ``Σ q``."""
        captured: list[str] = []

        class _Capture(mollog.Handler):
            def emit(self, record: mollog.LogRecord) -> None:
                captured.append(record.message)

        capture = _Capture()
        capture.set_level(mollog.WARNING)
        potentials_logger.add_handler(capture)
        try:
            live = _charged_batch(total_charge=1.0)
            pos = torch.zeros(4, 3, dtype=DTYPE)
            pos[:2] = live["atoms", "pos"]
            batch = TensorDict(
                {
                    "atoms": TensorDict(
                        {
                            "pos": pos,
                            "Z": torch.tensor([11, 11, 1, 1], dtype=torch.long),
                            "batch": torch.zeros(4, dtype=torch.long),
                            "mask": torch.tensor([True, True, False, False]),
                        },
                        batch_size=[4],
                    ),
                    "pairs": live["pairs"],
                    "graphs": live["graphs"],
                },
                batch_size=[],
            )
            pots = (
                ForceField(name="anion")
                .def_pairstyle("coulomb")
                .with_params(atoms={"charge": torch.tensor([0.5, 0.5, 9.0, -9.0], dtype=DTYPE)})
                .to_potentials()
            )
            pots.energy(batch)
            assert captured == []
        finally:
            potentials_logger.remove_handler(capture)

    def test_bind_topology_skips_charge_sum_when_q_is_a_different_n(self) -> None:
        """Reused container: seed q is N=2, next batch is N=3 — no index_add crash."""
        potentials = _anion_potentials()
        potentials.bind_topology(_charged_batch(total_charge=1.0))
        bigger = TensorDict(
            {
                "atoms": TensorDict(
                    {
                        "pos": torch.tensor(
                            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
                            dtype=DTYPE,
                        ),
                        "Z": torch.tensor([11, 11, 11], dtype=torch.long),
                        "batch": torch.zeros(3, dtype=torch.long),
                    },
                    batch_size=[3],
                ),
                "graphs": TensorDict(
                    {
                        "num_atoms": torch.tensor([3], dtype=torch.long),
                        "total_charge": torch.zeros(1, dtype=DTYPE),
                    },
                    batch_size=[1],
                ),
            },
            batch_size=[],
        )
        potentials.bind_topology(place_edges(bigger, cutoff=CHARGED_CUTOFF))

    def test_atom_params_for_n_drops_a_seed_table_of_the_wrong_length(self) -> None:
        """Predicted q of N=3 must not keep seed epsilon of N=2."""
        potentials = (
            ForceField(name="anion")
            .def_pairstyle("coulomb")
            .with_params(
                atoms={
                    "charge": torch.tensor([0.5, 0.5], dtype=DTYPE),
                    "epsilon": torch.tensor([1.0, 1.0], dtype=DTYPE),
                    "sigma": torch.tensor([3.0, 3.0], dtype=DTYPE),
                }
            )
            .to_potentials()
        )
        merged = potentials._atom_params_for_n(
            {"charge": torch.tensor([0.1, 0.2, 0.3], dtype=DTYPE)},
            n_atoms=3,
        )
        assert merged["charge"].tolist() == pytest.approx([0.1, 0.2, 0.3])
        assert "epsilon" not in merged
        assert "sigma" not in merged

    def test_called_proper_tables_maps_opls_k1_of_this_n_onto_k(self) -> None:
        """Live OPLS tables must not fall back to the seed cosine ``k``."""
        k1 = torch.ones(3, dtype=DTYPE)
        called = {"k1": k1, "k2": k1, "k3": torch.zeros(3, dtype=DTYPE), "k4": k1 * 0}
        out = Potentials._opls_to_cosine(called)
        assert out["k"].shape == (3, 4)
        torch.testing.assert_close(out["k"][:, 0], 0.5 * k1)
        assert "k1" not in out
        mapped = Potentials._called_proper_tables(called)
        assert mapped["k"].shape[0] == 3

    def test_opls_proper_energy_on_a_bigger_batch_uses_called_k1k4(self) -> None:
        """Reused OPLS container: seed 1 proper, next topology 2 with this-n k1..k4."""
        potentials = _opls_proper_forcefield(n=1).to_potentials()
        potentials.bind_topology(_torsion_batch("propers"))
        bigger = _two_proper_batch()
        potentials.bind_topology(bigger)
        k1 = torch.ones(2, dtype=DTYPE, requires_grad=True)
        tables = {
            "propers": {
                "k1": k1,
                "k2": torch.zeros(2, dtype=DTYPE),
                "k3": torch.zeros(2, dtype=DTYPE),
                "k4": torch.zeros(2, dtype=DTYPE),
            }
        }
        energy = potentials.energy(bigger, tables)[ENERGY_KEY]
        assert torch.isfinite(energy).all()
        energy.sum().backward()
        assert k1.grad is not None
        fresh = _opls_proper_forcefield(n=2).to_potentials().energy(bigger)[ENERGY_KEY]
        torch.testing.assert_close(energy.detach(), fresh)

    def test_net_charge_tolerance_scales_with_the_sum_it_checks(self) -> None:
        """A fixed 1e-6 e cannot separate fp32 noise from a head off its constraint.

        On a 350-atom molecule the charges themselves sum to ~100 e, where
        float32 accumulation alone costs a few 1e-6 e. The fixed threshold
        fired 33 times on one production-shaped profile batch, every one of
        them arithmetic; the 0.01 e of a real modelling error must still be
        reported.
        """
        captured: list[str] = []

        class _Capture(mollog.Handler):
            def emit(self, record: mollog.LogRecord) -> None:
                captured.append(record.message)

        capture = _Capture()
        capture.set_level(mollog.WARNING)
        potentials_logger.add_handler(capture)
        try:
            n_atoms = 350
            potentials = (
                ForceField(name="big-neutral")
                .def_pairstyle("coulomb")
                .with_params(atoms={"charge": torch.zeros(n_atoms, dtype=DTYPE)})
                .to_potentials()
            )
            batch = TensorDict(
                {
                    "atoms": TensorDict(
                        {
                            "pos": torch.zeros((n_atoms, 3), dtype=DTYPE),
                            "Z": torch.ones(n_atoms, dtype=torch.long),
                            "batch": torch.zeros(n_atoms, dtype=torch.long),
                        },
                        batch_size=[n_atoms],
                    ),
                    "graphs": TensorDict(
                        {
                            "num_atoms": torch.tensor([n_atoms], dtype=torch.long),
                            "total_charge": torch.zeros(1, dtype=DTYPE),
                        },
                        batch_size=[1],
                    ),
                },
                batch_size=[],
            )
            frame = potentials._frame(batch)
            # float32 because that is what production trains in, and the
            # tolerance tracks the dtype: the same 5e-6 e in float64 is a real
            # error and must still be reported there.
            charge = torch.full((n_atoms,), 0.3, dtype=torch.float32)
            charge[n_atoms // 2 :] = -0.3

            arithmetic = charge.clone()
            arithmetic[0] += 5e-6
            potentials._warn_net_charge_drift(frame, arithmetic)
            assert captured == [], f"arithmetic-scale drift must not warn: {captured}"

            # A small molecule has almost no size-scaled headroom, and the fp32
            # EEM solve still returns its constraint only to ~1e-5 e; the floor
            # is what covers it.
            small = torch.zeros(4, dtype=torch.float32)
            small[0] = 3.8e-6
            small_frame = potentials._frame(
                TensorDict(
                    {
                        "atoms": TensorDict(
                            {
                                "pos": torch.zeros((4, 3), dtype=DTYPE),
                                "Z": torch.ones(4, dtype=torch.long),
                                "batch": torch.zeros(4, dtype=torch.long),
                            },
                            batch_size=[4],
                        ),
                        "graphs": TensorDict(
                            {
                                "num_atoms": torch.tensor([4], dtype=torch.long),
                                "total_charge": torch.zeros(1, dtype=DTYPE),
                            },
                            batch_size=[1],
                        ),
                    },
                    batch_size=[],
                )
            )
            potentials._warn_net_charge_drift(small_frame, small)
            assert captured == [], f"fp32 solve residual must not warn: {captured}"

            off_constraint = charge.clone()
            off_constraint[0] += 0.01
            potentials._warn_net_charge_drift(frame, off_constraint)
            assert len(captured) == 1, "a head off its constraint must still be reported"
            assert "total_charge" in captured[0]

            captured.clear()
            exact_in_double = charge.to(torch.float64)
            exact_in_double[0] += 5e-6
            potentials._warn_net_charge_drift(frame, exact_in_double)
            assert len(captured) == 1, "float64 has no excuse for a 5e-6 e miss"
        finally:
            potentials_logger.remove_handler(capture)

    def test_cartesian_proper_tables_price_a_torsion_without_the_seed_phase(self) -> None:
        """``ProperTorsionParamHead`` speaks ``(a, b)``; the seed speaks ``(k, phase)``."""
        seed = (
            ForceField(name="periodic-proper")
            .def_dihedralstyle("periodic")
            .with_params(
                propers={
                    "k": torch.ones(1, 2, dtype=DTYPE),
                    "periodicity": torch.tensor([[1, 2]], dtype=torch.long),
                    "phase": torch.zeros(1, 2, dtype=DTYPE),
                }
            )
            .to_potentials()
        )
        batch = _torsion_batch("propers")
        a = torch.ones(1, 2, dtype=DTYPE, requires_grad=True)
        tables = {
            "propers": {
                "a": a,
                "b": torch.zeros(1, 2, dtype=DTYPE),
                "periodicity": torch.tensor([[1, 2]], dtype=torch.long),
            }
        }
        energy = seed.energy(batch, tables)[ENERGY_KEY]
        assert torch.isfinite(energy).all()
        energy.sum().backward()
        assert a.grad is not None

    def test_opls_proper_energy_refuses_a_bigger_batch_without_this_n_tables(self) -> None:
        """Seed cosine k of 1 proper must not be gathered with 2 identity types."""
        potentials = _opls_proper_forcefield(n=1).to_potentials()
        potentials.bind_topology(_torsion_batch("propers"))
        bigger = _two_proper_batch()
        potentials.bind_topology(bigger)
        with pytest.raises(ValueError, match="k"):
            potentials.energy(bigger)

    def test_lj_energy_on_a_bigger_batch_uses_called_tables(self) -> None:
        """Reused lj/cut container: seed N=2, next batch N=3 with this-N tables."""
        potentials = _lj_pair_forcefield().to_potentials()
        potentials.bind_topology(_lj_pair_batch(r=2.0))
        bigger = with_graphs(
            TensorDict(
                {
                    "atoms": TensorDict(
                        {
                            "pos": torch.tensor(
                                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
                                dtype=DTYPE,
                            ),
                            "Z": torch.tensor([18, 18, 18], dtype=torch.long),
                            "batch": torch.zeros(3, dtype=torch.long),
                        },
                        batch_size=[3],
                    ),
                },
                batch_size=[],
            )
        )
        bigger = place_edges(bigger, cutoff=LJ_CUTOFF)
        potentials.bind_topology(bigger)
        tables = {
            "atoms": {
                "epsilon": torch.full((3,), EPSILON, dtype=DTYPE),
                "sigma": torch.full((3,), SIGMA, dtype=DTYPE),
            }
        }
        energy = potentials.energy(bigger, tables)[ENERGY_KEY]
        assert torch.isfinite(energy).all()

    def test_lj_energy_refuses_a_bigger_batch_without_this_n_tables(self) -> None:
        """Seed epsilon of N=2 must not be gathered with a 3-atom edge list."""
        potentials = _lj_pair_forcefield().to_potentials()
        potentials.bind_topology(_lj_pair_batch(r=2.0))
        bigger = with_graphs(
            TensorDict(
                {
                    "atoms": TensorDict(
                        {
                            "pos": torch.tensor(
                                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
                                dtype=DTYPE,
                            ),
                            "Z": torch.tensor([18, 18, 18], dtype=torch.long),
                            "batch": torch.zeros(3, dtype=torch.long),
                        },
                        batch_size=[3],
                    ),
                },
                batch_size=[],
            )
        )
        bigger = place_edges(bigger, cutoff=LJ_CUTOFF)
        potentials.bind_topology(bigger)
        with pytest.raises(ValueError, match="epsilon"):
            potentials.energy(bigger)

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
        """F29 on public ``calc_energy``, even without an explicit bind.

        :meth:`bind_topology` refuses a folded frame, but nothing makes a
        caller call it. For a **bonded-only** field there is no nonbonded term,
        so the lazy bind inside ``_scaled_pairs`` never runs either, and the
        wrapped molecule of :data:`WRAPPED_X` — 2 Å apart through the ``x``
        face, 8 Å apart in the coordinates as given — would otherwise be a
        finite, differentiable, wrong bonded energy. The public energy entry
        that must still refuse is :meth:`calc_energy`. :meth:`energy` /
        ``Potentials.__call__`` is the :class:`~molix.autograd.Derivative`
        seam and does not run that host check.

        Checked on a **fresh** container: a bind performed by an earlier call
        would otherwise be what protected this one.
        """
        cell = torch.eye(3, dtype=DTYPE) * WRAP_EDGE

        with pytest.raises(ValueError, match="F29"):
            _bond_potentials().calc_energy(_periodic_bond_batch(*WRAPPED_X, cell=cell))

    def test_a_second_frame_of_the_same_shape_is_checked_afresh(self) -> None:
        """A second frame of the same topology shape is still a new geometry.

        Public :meth:`calc_energy` re-runs F29 on the positions and cell it is
        handed. Two TensorDicts of the same bonded-column counts can still be
        a whole molecule and a folded one; a previous Python object is not a
        physical invariant. MD does not skip this check because an identity
        memo keyed the first frame: an integrator never wraps in place, and
        bonded-only ``MD.run`` is the compiled :meth:`energy` residual (the
        caller unwraps before ``run``). Training hands a new container every
        step, so the folded frame behind a whole one must still raise.
        """
        cell = torch.eye(3, dtype=DTYPE) * WRAP_EDGE
        potentials = _bond_potentials()

        potentials.calc_energy(_periodic_bond_batch(*UNWRAPPED_X, cell=cell))

        with pytest.raises(ValueError, match="F29"):
            potentials.calc_energy(_periodic_bond_batch(*WRAPPED_X, cell=cell))

    def test_an_in_place_wrap_of_the_same_object_is_refused(self) -> None:
        """Rewriting ``atoms.pos`` in place is a new geometry, not a voucher.

        Object identity of a TensorDict is not a physical invariant: the same
        storage that priced the unwrapped 2 Å bond at 50.0 kcal/mol can hold
        the folded 8 Å span a moment later. Public :meth:`calc_energy` and
        :meth:`bind_topology` re-read that storage.
        """
        cell = torch.eye(3, dtype=DTYPE) * WRAP_EDGE
        potentials = _bond_potentials()
        batch = _periodic_bond_batch(*UNWRAPPED_X, cell=cell)
        pos = _leaf(batch, "atoms", "pos")

        assert float(potentials.calc_energy(batch)) == pytest.approx(UNWRAPPED_ENERGY, abs=1e-10)

        pos[0, 0] = WRAPPED_X[0]
        pos[1, 0] = WRAPPED_X[1]
        assert _leaf(batch, "atoms", "pos") is pos

        with pytest.raises(ValueError, match="F29") as excinfo:
            potentials.calc_energy(batch)
        message = str(excinfo.value)
        for token in ("8", "5"):
            assert re.search(rf"\b{token}\b", message), message

        with pytest.raises(ValueError, match="F29") as excinfo:
            potentials.bind_topology(batch)
        message = str(excinfo.value)
        for token in ("8", "5"):
            assert re.search(rf"\b{token}\b", message), message

    def test_an_in_place_shear_of_the_same_cell_is_refused(self) -> None:
        """Rewriting ``graphs.cell`` in place is a new geometry, not a voucher.

        A 4 Å bond is legal in the 10 Å cube (limit 5 Å) and wrapped in the
        sheared cell whose perpendicular width is 6 Å (limit 3 Å). Writing
        that lattice into the same ``graphs.cell`` storage must refuse.
        """
        cube = torch.eye(3, dtype=DTYPE) * WRAP_EDGE
        sheared = torch.tensor(
            [[WRAP_EDGE, 0.0, 0.0], [0.0, WRAP_EDGE, 0.0], SHEARED_C], dtype=DTYPE
        )
        potentials = _bond_potentials()
        batch = _periodic_bond_batch(0.0, SHEARED_BOND_R, cell=cube)
        cell_storage = _leaf(batch, "graphs", "cell")

        potentials.calc_energy(batch)

        cell_storage.copy_(sheared)
        assert _leaf(batch, "graphs", "cell") is cell_storage

        with pytest.raises(ValueError, match="F29") as excinfo:
            potentials.calc_energy(batch)
        message = str(excinfo.value)
        assert re.search(r"\b4\b", message), message
        assert re.search(r"\b3\b", message), message

        with pytest.raises(ValueError, match="F29") as excinfo:
            potentials.bind_topology(batch)
        message = str(excinfo.value)
        assert re.search(r"\b4\b", message), message
        assert re.search(r"\b3\b", message), message

    def test_warmup_does_not_voucher_a_fresh_wrapped_batch(self) -> None:
        """Warming one unwrapped frame does not validate the next one.

        :meth:`warmup` binds and F29-checks *that* batch. A fresh wrapped
        TensorDict of the same topology shape is a new geometry: public
        :meth:`calc_energy` must still raise F29 (8 Å vs 5 Å). The warmed
        batch itself still prices at 50.0 kcal/mol.
        """
        cell = torch.eye(3, dtype=DTYPE) * WRAP_EDGE
        potentials = _bond_potentials()
        unwrapped = _periodic_bond_batch(*UNWRAPPED_X, cell=cell)
        potentials.warmup(unwrapped)

        assert float(potentials.calc_energy(unwrapped)) == pytest.approx(
            UNWRAPPED_ENERGY, abs=1e-10
        )

        with pytest.raises(ValueError, match="F29") as excinfo:
            potentials.calc_energy(_periodic_bond_batch(*WRAPPED_X, cell=cell))
        message = str(excinfo.value)
        for token in ("8", "5"):
            assert re.search(rf"\b{token}\b", message), message

    def test_warmup_does_not_voucher_an_in_place_wrap_of_the_warmed_object(self) -> None:
        """In-place wrap of the warmed TensorDict is still F29.

        Validating the warmup batch must not validate a later geometry written
        into the same object. Public :meth:`calc_energy` re-reads ``atoms.pos``.
        """
        cell = torch.eye(3, dtype=DTYPE) * WRAP_EDGE
        potentials = _bond_potentials()
        batch = _periodic_bond_batch(*UNWRAPPED_X, cell=cell)
        pos = _leaf(batch, "atoms", "pos")
        potentials.warmup(batch)

        assert float(potentials.calc_energy(batch)) == pytest.approx(UNWRAPPED_ENERGY, abs=1e-10)

        pos[0, 0] = WRAPPED_X[0]
        pos[1, 0] = WRAPPED_X[1]
        assert _leaf(batch, "atoms", "pos") is pos

        with pytest.raises(ValueError, match="F29") as excinfo:
            potentials.calc_energy(batch)
        message = str(excinfo.value)
        for token in ("8", "5"):
            assert re.search(rf"\b{token}\b", message), message

    def test_bind_topology_does_not_voucher_an_in_place_wrap(self) -> None:
        """An explicit bind of the unwrapped frame is not a later-geometry voucher.

        :meth:`bind_topology` F29-checks the coordinates it is handed. Writing
        :data:`WRAPPED_X` into the same ``atoms.pos`` storage afterwards is a
        new geometry: public :meth:`calc_energy` must still raise F29.
        """
        cell = torch.eye(3, dtype=DTYPE) * WRAP_EDGE
        potentials = _bond_potentials()
        batch = _periodic_bond_batch(*UNWRAPPED_X, cell=cell)
        pos = _leaf(batch, "atoms", "pos")
        potentials.bind_topology(batch)

        assert float(potentials.calc_energy(batch)) == pytest.approx(UNWRAPPED_ENERGY, abs=1e-10)

        pos[0, 0] = WRAPPED_X[0]
        pos[1, 0] = WRAPPED_X[1]
        assert _leaf(batch, "atoms", "pos") is pos

        with pytest.raises(ValueError, match="F29") as excinfo:
            potentials.calc_energy(batch)
        message = str(excinfo.value)
        for token in ("8", "5"):
            assert re.search(rf"\b{token}\b", message), message

    @pytest.mark.slow
    def test_a_wrapped_frame_is_refused_even_under_torch_compile(self) -> None:
        """Compiling public ``calc_energy`` may not switch the F29 gate off.

        The check is host-side, so dynamo cannot trace the ``argmax`` / Python
        raise: an unvouched compiled public :meth:`calc_energy` graph-breaks
        on that host sync and the eager re-run still runs the check
        (fail-closed). Handing a :class:`~molix.data.TensorFrame` so
        ``_frame`` does not wrap is not a substitute for F29 — a wrap-hit
        plus ``is_compiling()`` skip is the fail-open this pins closed.
        Scoring is not the MD kernel: :meth:`energy` is the Derivative seam
        and is not compiled here.

        Both unvouched containers raise, TensorDict and TensorFrame: a wrap
        break on the former is not what refuses the bond. An unwrapped
        TensorFrame under ``torch.compile(calc_energy)`` may still price at
        50.0 kcal/mol because the host check runs and *passes*, not because
        a previous eager call skipped F29.

        Marked ``slow``: the first ``torch.compile`` of this field costs 47 s
        on the CI box (0.1 s for a second one in the same process), which is
        the compile tier by any reading of the rule.
        """
        cell = torch.eye(3, dtype=DTYPE) * WRAP_EDGE

        refusing = _bond_potentials()
        with pytest.raises(ValueError, match="F29") as excinfo:
            torch.compile(lambda batch: refusing.calc_energy(batch))(
                _periodic_bond_batch(*WRAPPED_X, cell=cell)
            )
        message = str(excinfo.value)
        for token in ("8", "5"):
            assert re.search(rf"\b{token}\b", message), message

        framed = _bond_potentials()
        with pytest.raises(ValueError, match="F29") as excinfo:
            torch.compile(lambda batch: framed.calc_energy(batch))(
                as_tensor_frame(_periodic_bond_batch(*WRAPPED_X, cell=cell))
            )
        message = str(excinfo.value)
        for token in ("8", "5"):
            assert re.search(rf"\b{token}\b", message), message

        passing = _bond_potentials()
        whole = as_tensor_frame(_periodic_bond_batch(*UNWRAPPED_X, cell=cell))
        energy = torch.compile(lambda batch: passing.calc_energy(batch))(whole)
        assert float(energy) == pytest.approx(UNWRAPPED_ENERGY, abs=1e-10)

    def test_a_refused_bind_leaves_nothing_armed(self) -> None:
        """A bind that raised must not have installed half of itself.

        :meth:`bind_topology` used to build the exclusion table and record the
        topology fingerprint *before* its refusals ran, so a refusal left the
        container looking bound: the exclusion table is not ``None`` any more,
        the lazy bind inside the next evaluation sees nothing to do, and the
        very batch that was just refused is priced. One caught exception in a
        fitting loop (or a notebook cell re-run) is all it takes, and every
        number after it is quietly outside the constraint it was refused for.

        Refusing repeatedly is the only safe state: the container is either
        bound to a frame it accepted or bound to nothing. Both entry orders are
        checked, because the arming happens on the bind and the *lazy* bind is
        the one no caller can see.

        Driven by the edge-orientation refusal. The net-charge check that used
        to drive it warns rather than raises (operator ruling 2026-09-11), and
        a warning arms nothing — but the ordering rule it exposed is the
        bind's, not that one check's.
        """
        batch = _reversed_edge_batch()

        explicit = _anion_potentials()
        with pytest.raises(ValueError, match="source > target"):
            explicit.bind_topology(batch)
        with pytest.raises(ValueError, match="source > target"):
            explicit.calc_energy(batch)

        lazy = _anion_potentials()
        with pytest.raises(ValueError, match="source > target"):
            lazy.calc_energy(batch)
        with pytest.raises(ValueError, match="source > target"):
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

        out = _bond_potentials(wrt=FORCES)(batch)

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
        pair_index, shifts = _coulomb_edges()
        positions = torch.tensor(COULOMB_POSITIONS, dtype=DTYPE)

        forces = Derivative(potentials.energy, wrt=FORCES_AND_VIRIAL)(
            _coulomb_batch(positions, cell, pair_index, shifts)
        )[FORCES_KEY].detach()

        def energy_at(stepped: torch.Tensor) -> float:
            with torch.no_grad():
                return float(
                    potentials.calc_energy(_coulomb_batch(stepped, cell, pair_index, shifts))
                )

        differences = _central_difference_forces(energy_at, positions, FD_STEP)

        gap = float((forces - differences).abs().max())
        assert gap < FD_FORCE_TOL, f"max |F_auto - F_fd| = {gap:.3e} kcal/mol/A"

    def test_periodic_coulomb_virial_matches_a_central_strain_difference(self) -> None:
        """``W = dE/deps`` against six symmetric-strain differences.

        The deformation is the one :class:`molix.autograd.Derivative`
        applies: positions, lattice vectors **and** the per-edge lattice shifts
        are all mapped by ``v @ (I + eps)^T``. Straining the shifts is not decoration —
        atom 3 of the fixture is wrapped, so three of its six pairs reach their
        partner through a lattice vector, and a virial that holds those vectors
        fixed while the box deforms is missing a term of order the energy
        itself.

        ``eps`` is the **symmetric** strain tensor and ``W`` its matrix
        gradient, so a shear moves ``eps_ab`` and ``eps_ba`` together and the
        difference quotient counts the pair twice: the off-diagonals are halved
        to land on the Cauchy convention
        (:func:`molix.autograd.symmetrize_strain_grad`).
        """
        potentials = _coulomb_forcefield().to_potentials()
        cell = torch.eye(3, dtype=DTYPE) * COULOMB_EDGE
        potentials.bind_cell(cell)
        pair_index, shifts = _coulomb_edges()
        positions = torch.tensor(COULOMB_POSITIONS, dtype=DTYPE)

        virial = Derivative(potentials.energy, wrt=FORCES_AND_VIRIAL)(
            _coulomb_batch(positions, cell, pair_index, shifts)
        )[VIRIAL_KEY][0].detach()

        def deformed(deformation: torch.Tensor) -> torch.Tensor:
            transform = deformation.T
            with torch.no_grad():
                return potentials.calc_energy(
                    _coulomb_batch(
                        positions @ transform,
                        cell @ transform,
                        pair_index,
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
        already reduces per graph through ``pair_batch`` / ``num_graphs``.

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
        pair_index = torch.cat([half_list, half_list + n_atoms], dim=0)
        shifts = torch.cat([half_shifts, half_shifts], dim=0)
        pair_diff = positions[pair_index[:, 1]] - positions[pair_index[:, 0]] + shifts
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
                "pairs": TensorDict(
                    {
                        "pair_index": pair_index,
                        "pair_diff": pair_diff,
                        "pair_dist": pair_diff.norm(dim=-1),
                        "shifts": shifts,
                    },
                    batch_size=[pair_index.shape[0]],
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
            .with_params(atoms={"charge": torch.tensor(COULOMB_CHARGES * 2, dtype=DTYPE)})
            .to_potentials()
        )

        with pytest.raises(ValueError, match=r"per-graph|num_graphs|one graph"):
            potentials.calc_energy(batch)

    def test_periodic_virial_without_stored_shifts_is_refused(self) -> None:
        """A placed list with no ``edges.shifts`` prices E and F, never W.

        ``Potentials._placed_pairs`` recovers the lattice remainder of
        a shift-less list as ``(pair_diff - (pos[t] - pos[s])).detach()``. That
        constant carries the right *value* — energy and forces below match the
        stored-``shifts`` batch to the bit — but it is not a function of the
        cell, so the strain the derivative op applies never reaches the pairs
        that reach their partner through a lattice vector. On this wrapped
        four-ion fixture that silently drops ~13.5 kcal/mol from ``tr W`` (the
        F13 boundary in ``.claude/notes/composable-potentials.md``): the
        virial is wrong, and nothing in the returned value says so.

        So the boundary is pinned as a boundary. The ``pair_diff`` fallback
        stays complete for energy and forces; the virial refuses a periodic
        batch whose list does not carry ``shifts``, and the message names the
        key that is missing.
        """
        potentials = _coulomb_forcefield().to_potentials()
        positions = torch.tensor(COULOMB_POSITIONS, dtype=DTYPE)
        cell = torch.eye(3, dtype=DTYPE) * COULOMB_EDGE
        pair_index, shifts = _coulomb_edges()
        stored = _coulomb_batch(positions, cell, pair_index, shifts)
        implied = _coulomb_batch(positions, cell, pair_index, shifts)
        del implied["pairs", "shifts"]

        assert torch.allclose(
            potentials.calc_energy(implied).detach(),
            potentials.calc_energy(stored).detach(),
            atol=1e-10,
            rtol=0.0,
        )
        forces = Derivative(potentials.energy, wrt=FORCES)
        assert torch.allclose(
            forces(implied)[FORCES_KEY].detach(),
            forces(stored)[FORCES_KEY].detach(),
            atol=1e-8,
            rtol=0.0,
        )
        with pytest.raises(ValueError, match="shifts"):
            Derivative(potentials.energy, wrt=FORCES_AND_VIRIAL)(implied)

    @pytest.mark.slow
    def test_cartesian_proper_tables_trace_as_one_graph(self) -> None:
        """The ``(a, b)`` → ``(k, γ)`` map runs inside the compiled region.

        Converting through :meth:`ProperTorsionBag.from_cartesian` would drag
        the bag's eager positivity check (a ``bool()`` on a tensor) onto the
        hot path, where it is a host sync and a data-dependent guard.
        """
        from molix.compile import Compiler

        potentials = (
            ForceField(name="periodic-proper")
            .def_dihedralstyle("periodic")
            .with_params(
                propers={
                    "k": torch.ones(1, 2, dtype=DTYPE),
                    "periodicity": torch.tensor([[1, 2]], dtype=torch.long),
                    "phase": torch.zeros(1, 2, dtype=DTYPE),
                }
            )
            .to_potentials()
        )
        batch = _torsion_batch("propers")
        # The ``graphs`` namespace is required: ``_num_graphs`` is its batch size.
        batch["graphs"] = TensorDict(
            {"num_atoms": torch.tensor([4], dtype=torch.long)}, batch_size=[1]
        )
        tables = {
            "propers": {
                "a": torch.ones(1, 2, dtype=DTYPE),
                "b": torch.zeros(1, 2, dtype=DTYPE),
                "periodicity": torch.tensor([[1, 2]], dtype=torch.long),
            }
        }
        explanation = Compiler.explain(potentials.energy, batch, tables)
        assert (explanation.graph_count, explanation.graph_break_count) == (1, 0), (
            f"break sites: {explanation.break_sites}"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("calculator", ("ewald", "pme"))
    def test_periodic_coulomb_traces_clean_and_the_step_breaks_only_at_the_leaf(
        self, calculator: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The bound term is one graph; the training step around it breaks only at the leaf.

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

        Around that graph the whole NPT step is one graph in inference, and
        traced with grad enabled it breaks only where the attached outputs
        leak the leaves — ``molix/autograd/variables.py``. A break anywhere
        else (a boolean index in the term, a ``.item()`` on the k-grid, a
        style-registry lookup on the hot path) is a regression.
        """
        from molix.compile import Compiler

        potentials = _coulomb_forcefield(calculator=calculator).to_potentials()
        cell = torch.eye(3, dtype=DTYPE) * COULOMB_EDGE
        potentials.bind_cell(cell)
        pair_index, shifts = _coulomb_edges()
        positions = torch.tensor(COULOMB_POSITIONS, dtype=DTYPE)

        term = potentials.coulomb_periodic
        assert term is not None
        energy_of = term.energy
        captured: dict[str, object] = {}

        def capturing(**kwargs: object) -> torch.Tensor:
            captured.update(kwargs)
            return energy_of(**kwargs)

        monkeypatch.setattr(term, "energy", capturing)
        potentials.calc_energy(_coulomb_batch(positions, cell, pair_index, shifts))
        monkeypatch.undo()
        assert captured, "the periodic term was never reached"

        flat = Compiler.explain(energy_of, **captured)
        assert (flat.graph_count, flat.graph_break_count) == (1, 0), (
            f"the bound {calculator!r} term must trace as one graph; "
            f"break sites: {flat.break_sites}"
        )

        # Host-side preconditions sit *before* the compiled region — the same
        # rule that puts ``bind_cell`` at the top of this test. ``_coulomb_batch``
        # HAS bonds (atoms 0–1, 2.2 Å vs 4 Å half-width of the 8 Å cube); F29
        # runs at eager ``bind_topology`` / ``calc_energy``. The explained graph
        # is ``energy()`` after that. Wrap identity (``_frame_view``) keeps
        # ``as_tensor_frame`` out of the compiled MD ``op(work)`` region; it is
        # not an F29 voucher, and MD ``bind_work`` does not eager-vouch F29
        # (``Derivative`` differentiates ``energy()``). The bind here is
        # exclusion-table work, not a concession.
        step_batch = _coulomb_batch(positions, cell, pair_index, shifts)
        potentials.bind_topology(step_batch)
        step = Compiler.explain(Derivative(potentials.energy, wrt=FORCES_AND_VIRIAL), step_batch)
        assert step.graph_break_count > 0
        strays = _strays_outside_the_force_seam(step.break_sites)
        assert strays == [], (
            f"the periodic Coulomb step ({calculator}) may break only at the leaf in "
            f"{FORCE_SEAM} (owner: molix autograd); stray break sites: {strays}"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("calculator", ("ewald", "pme"))
    def test_the_periodic_coulomb_op_traces_to_one_graph_in_inference(
        self, calculator: str
    ) -> None:
        """The iron law on the periodic NPT step: one graph, zero breaks, under ``no_grad``.

        ``_coulomb_batch`` HAS bonds; F29 runs at the eager ``bind_topology`` /
        ``calc_energy`` below. The explained graph is ``energy()`` after that.
        ``(1, 0)`` is wrap identity (``_frame_view``) for compiled MD
        ``op(work)``, not an F29-skip green and not ``bind_work`` vouching F29.
        """
        from molix.compile import Compiler

        potentials = _coulomb_forcefield(calculator=calculator).to_potentials()
        cell = torch.eye(3, dtype=DTYPE) * COULOMB_EDGE
        potentials.bind_cell(cell)
        pair_index, shifts = _coulomb_edges()
        positions = torch.tensor(COULOMB_POSITIONS, dtype=DTYPE)
        step_batch = _coulomb_batch(positions, cell, pair_index, shifts)
        potentials.bind_topology(step_batch)
        potentials.calc_energy(step_batch)
        with torch.no_grad():
            inference = Compiler.explain(
                Derivative(potentials.energy, wrt=FORCES_AND_VIRIAL), step_batch
            )
        assert (inference.graph_count, inference.graph_break_count) == (1, 0), (
            f"the periodic Coulomb NPT step ({calculator}) must be one graph in inference; "
            f"break sites: {inference.break_sites}"
        )

    def test_lj_forces_match_a_finite_difference_across_the_boundary(self) -> None:
        """The LJ force must differentiate the **minimum-image** pair.

        For a pair whose nearest image is across a box face the raw
        displacement ``pos[target] − pos[source]`` is the *long* way round —
        here 8 Å against a 2 Å minimum image — so an energy priced on the
        minimum image but differentiated through the raw displacement would
        report a force that is not the gradient of the energy. The pair
        vectors :meth:`Potentials._placed_pairs` builds from the leaf and the
        lattice remainder are what keeps the two one geometry. In MD that is
        not a small error: a non-conservative force does work around every
        cycle and pumps energy into an NVE trajectory.

        The reference is a central difference of ``calc_energy`` — the same
        function whose gradient the op takes — on both atoms and all three
        axes, with the neighbour list rebuilt at each displaced geometry.
        """
        positions = torch.tensor(LJ_BOUNDARY_POSITIONS, dtype=DTYPE)
        potentials = _lj_boundary_potentials()
        forces = Derivative(potentials.energy, wrt=FORCES)(_lj_boundary_batch(positions))[
            FORCES_KEY
        ]

        def energy_at(stepped: torch.Tensor) -> float:
            return float(_lj_boundary_potentials().calc_energy(_lj_boundary_batch(stepped)))

        finite = _central_difference_forces(energy_at, positions, LJ_BOUNDARY_FD_STEP)

        gap = float((forces.detach() - finite).abs().max())
        assert gap < LJ_BOUNDARY_FORCE_TOL, (
            f"LJ force is not -dE/dpos across the boundary: max gap {gap:.6g} "
            f"kcal/mol/A; op {forces.tolist()} vs finite difference {finite.tolist()}"
        )

    # -- S8 gate 1: symmetries of the assembled Class-I energy ---------------

    # -- the typed view of the batch: TensorFrame in, same answer out -------

    def test_a_typed_view_prices_the_same_energy_as_the_batch_it_wraps(self) -> None:
        """``TensorFrame.from_tensordict`` is a view, so the energy cannot move.

        molnex speaks two shapes — a molpy ``Frame`` at rest, the post-collate
        batch in the model — and ``TensorFrame`` is the *typed read* of the
        second (CLAUDE.md § TensorDict Contract): the same leaves, declared, no
        storage of its own. The pair of assertions is the whole contract: the
        wrapped ``pos`` is the same tensor, and the assembled energy of the
        butane-pair box is the same number to the last bit — not to a tolerance,
        because nothing here is recomputed.
        """
        batch = butane_pair_batch()
        view = TensorFrame.from_tensordict(batch)

        viewed = butane_pair_class_i_forcefield().to_potentials().calc_energy(view)
        plain = butane_pair_class_i_forcefield().to_potentials().calc_energy(batch)

        assert float(viewed) == float(plain)

    def test_bind_topology_reads_the_topology_through_a_typed_view(self) -> None:
        """A bind taken on the view is the bind the plain batch would have taken.

        ``bind_topology`` is where the exclusion table, the graph count and the
        Σq check are read — the one host-side pass over the batch's topology.
        Binding on the view and then pricing the batch has to answer exactly as
        pricing it alone does: if the view hid a namespace, the table would be
        built from a topology the energy is not evaluated on, and the 1-2 / 1-3
        pairs would come back into the nonbonded sum with nothing raised.
        """
        batch = butane_pair_batch()
        potentials = butane_pair_class_i_forcefield().to_potentials()
        potentials.bind_topology(TensorFrame.from_tensordict(batch))

        assert float(potentials.calc_energy(batch)) == float(
            butane_pair_class_i_forcefield().to_potentials().calc_energy(batch)
        )

    def test_forward_on_a_typed_view_writes_into_the_batch_it_wraps(self) -> None:
        """``forward`` fills ``graphs.energy`` / ``atoms.forces`` on a view too.

        The MD adapter and the composer both call ``forward`` for its *write*:
        a batch goes in and comes back carrying the energy and the forces. The
        view is not a second container — it wraps the caller's leaves — so a
        ``forward`` handed one has to leave the caller's batch filled, and with
        the same numbers. Answering with a bare energy tensor instead is the
        silent version of this failing: nothing raises, and the caller's batch
        simply has no forces in it.
        """
        batch = butane_pair_batch()
        reference = butane_pair_class_i_forcefield().to_potentials(wrt=FORCES)(butane_pair_batch())
        view = TensorFrame.from_tensordict(batch)

        butane_pair_class_i_forcefield().to_potentials(wrt=FORCES)(view)

        assert ENERGY_KEY in batch.keys(include_nested=True)
        assert FORCES_KEY in batch.keys(include_nested=True)
        assert torch.equal(batch[ENERGY_KEY], reference[ENERGY_KEY])
        assert torch.equal(batch[FORCES_KEY], reference[FORCES_KEY])

    def test_the_strained_op_accepts_a_typed_view(self) -> None:
        """The strained path reads the view like any other batch.

        The typed view — a ``TensorDictBase`` that is deliberately *not* a
        ``TensorDict`` — must be read by what it describes, not refused by an
        ``isinstance`` check. The periodic Coulomb fixture is the one with
        stored ``edges.shifts``, which a virial requires (F13).
        """
        potentials = _coulomb_forcefield().to_potentials()
        cell = torch.eye(3, dtype=DTYPE) * COULOMB_EDGE
        potentials.bind_cell(cell)
        pair_index, shifts = _coulomb_edges()
        positions = torch.tensor(COULOMB_POSITIONS, dtype=DTYPE)
        batch = _coulomb_batch(positions, cell, pair_index, shifts)
        npt = Derivative(potentials.energy, wrt=FORCES_AND_VIRIAL)

        viewed = npt(TensorFrame.from_tensordict(batch))
        expected = npt(_coulomb_batch(positions, cell, pair_index, shifts))

        assert torch.equal(viewed[ENERGY_KEY].detach(), expected[ENERGY_KEY].detach())
        assert torch.equal(viewed[FORCES_KEY].detach(), expected[FORCES_KEY].detach())
        assert torch.equal(viewed[VIRIAL_KEY].detach(), expected[VIRIAL_KEY].detach())

    def test_a_plain_mapping_is_refused(self) -> None:
        """The container evaluates a batch or its view; a bare mapping is not one."""
        batch = butane_pair_batch()
        mapping = {name: batch[name] for name in batch.keys()}
        with pytest.raises(TypeError, match="TensorDict"):
            butane_pair_class_i_forcefield().to_potentials().calc_energy(mapping)  # type: ignore[arg-type]

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
        (``charge`` / ``epsilon`` / ``sigma``) and ``Z`` that travel with them, and
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
        # ``bonds`` is the ``(2, N)`` COO block ``collate_molecules`` writes, so
        # it is remapped as one tensor; the valence namespaces are 1-D columns.
        permuted["bonds", "bond_index"] = inverse[_leaf(reference, "bonds", "bond_index")]
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
    """Break sites that are not the leaf in ``molix/autograd/variables.py``.

    With grad enabled, the derivative op returns forces (and a virial) that
    still require grad and derive from the leaves made there; stock dynamo
    breaks where such a leaf leaks (pytorch/pytorch#196653 lifts it), so a
    training-mode step is *expected* to break there — and nowhere else. One
    definition, because the mlip route and the classical route assert the
    same rule about the same seam.

    Args:
        sites: ``ExplainSummary.break_sites``, one ``file:line`` per break.

    Returns:
        The sites outside :data:`FORCE_SEAM`, in the order reported — empty
        when every break is a sanctioned one.
    """
    return [site for site in sites if not re.fullmatch(rf".*{re.escape(FORCE_SEAM)}:\d+", site)]


#: Separation (Å) of the two-atom ``lj/cut`` fixture the classical op is
#: traced on. Past the ``2^(1/6) σ = 3.37 Å`` minimum, so the pair sits on
#: the attractive shoulder with a non-zero force — at the well bottom every
#: build agrees on a zero and the parity leg would prove nothing.
COMPILED_CORE_R = 3.5


def _polar_forcefield() -> tuple[ForceField, nn.Module]:
    """A tiny periodic Polar mlip behind a ``metal``-unit ForceField.

    The only in-tree potential that is :class:`~molix.schema.CellBound` with
    a cell-sized reciprocal grid, so the compiled-route criteria need it
    rather than a stub. Its hyper-parameters
    come from the mace test package's conftest — one definition of "tiny
    Polar", not a second copy that drifts.

    Built at :data:`DTYPE` (fp64): cuEquivariance freezes its working
    precision at construction, so a model built at fp32 would stay fp32.
    """
    from molzoo.mace.polar import MacePolar
    from molzoo.mace.spec import MACEPolarSpec
    from tests.test_molzoo.test_mace.conftest import (
        ATOMIC_ENERGIES,
        ATOMIC_NUMBERS,
        TINY_POLAR_KWARGS,
    )

    spec = MACEPolarSpec(
        atomic_numbers=ATOMIC_NUMBERS,
        atomic_energies=ATOMIC_ENERGIES,
        **TINY_POLAR_KWARGS,
    )
    torch.manual_seed(0)
    model = MacePolar(spec, dtype=DTYPE).eval()
    forcefield = ForceField("mace-polar", units="metal")
    forcefield.def_pairstyle(model)
    return forcefield, model


def _polar_batch(pos: torch.Tensor, cell: torch.Tensor) -> TensorDict:
    """A one-graph periodic batch on the mace-package cluster geometry."""
    from tests.test_molzoo.test_mace.conftest import CLUSTER_Z, SINGLE_GRAPH, full_edge_index

    pair_index = full_edge_index(SINGLE_GRAPH)
    diff = pos[pair_index[:, 1]] - pos[pair_index[:, 0]]
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
            "pairs": TensorDict(
                {
                    "pair_index": pair_index,
                    "pair_diff": diff.clone(),
                    "pair_dist": diff.norm(dim=-1),
                },
                batch_size=[pair_index.shape[0]],
            ),
            "graphs": TensorDict({"cell": cell.unsqueeze(0)}, batch_size=[1]),
        },
        batch_size=[],
    )


class TestPotentialsMlip:
    """``bind_cell`` delegation, the mlip ``energy`` branch and the compiled routes.

    ``Potentials`` owns one line of ``bind_cell``: a protocol-guarded hand-over
    to the mlip. It keeps no compiled state — the composed op is what a
    caller compiles — and its ``mlip`` branch of ``calc_energy`` calls the
    mlip's ``energy`` seam, never its ``forward``.
    """

    def test_bind_cell_delegates_behind_the_protocol_guard(self) -> None:
        """Guarded, one-layer, side-effect-free — and molpot stays downstream.

        ``def_pairstyle`` accepts any ``nn.Module``, so the protocol is optional
        on an mlip: ``bind_cell`` must silently do nothing (and change no
        result) for one that lacks it, and a **classical** field holds nothing
        sized by the lattice, so its ``bind_cell`` is a documented no-op too.
        When the mlip does satisfy ``CellBound``, exactly one call is forwarded,
        with the caller's own storage. The import guard keeps the package
        direction ``molix <- molrep <- molzoo/molpot`` intact: the protocol
        comes from ``molix.schema``, never from ``molix.md``.
        """
        toy = ForceField().def_pairstyle(_EdgeSpring()).to_potentials(wrt=FORCES)
        before = toy(_edge_batch())[FORCES_KEY].detach().clone()
        toy.bind_cell(torch.eye(3, dtype=DTYPE))
        assert torch.equal(toy(_edge_batch())[FORCES_KEY].detach(), before)

        classical = _bond_potentials()
        classical.bind_cell(torch.eye(3, dtype=DTYPE))

        bound: list[int] = []

        class _CellBoundMlip(nn.Module):
            def bind_cell(self, cell: torch.Tensor) -> None:
                bound.append(cell.data_ptr())

        pots = ForceField().def_pairstyle(_CellBoundMlip()).to_potentials()
        cell = torch.eye(3, dtype=DTYPE) * 10.0
        pots.bind_cell(cell)
        assert bound == [cell.data_ptr()]
        assert isinstance(pots, CellBound)

    def test_calc_energy_calls_the_mlip_energy_seam(self) -> None:
        """The mlip branch hands the batch to ``mlip.energy``, never to ``mlip.forward``.

        ``forward`` of a potential is an op of its own; reaching it from the
        container would derive twice.
        """

        class _PosSpring(nn.Module):
            """Energy read off ``atoms.pos`` only; ``forward`` must stay unreached."""

            def energy(self, system: TensorDict) -> TensorDict:
                system[ENERGY_KEY] = 0.5 * system["atoms", "pos"].pow(2).sum().reshape(1)
                return system

            def forward(self, system: TensorDict) -> TensorDict:
                raise AssertionError("the container must call energy, not forward")

        pots = ForceField().def_pairstyle(_PosSpring()).to_potentials()
        batch = _edge_batch()

        energy = pots.calc_energy(batch)
        assert energy is batch[ENERGY_KEY]

    @pytest.mark.slow
    def test_the_classical_op_traces_to_one_graph_in_inference(self) -> None:
        """The iron law on the classical hot path: NVE and NPT, one graph, zero breaks.

        A ``lj/cut`` field is fixed-shape arithmetic over the list's capacity
        with a liveness mask — no boolean selection, no host read, no Python
        branch on a tensor value — and the op over it must trace as **one**
        graph with zero breaks, forces alone and forces with the virial. A
        declared ``pair:coulomb`` has no closed-form force either way; with
        ``trace_autograd_ops`` on (the flag ``Compiler`` sets and ``explain``
        patches in) the whole op — leaves, geometry, energy, ``autograd.grad``
        — is one graph too.

        ``bind_topology`` first, which is the composition
        :meth:`~molpot.forcefield.Potentials.bind_topology` documents for a
        compiled path: the exclusion table and the orientation check are
        topology work, they are host-side by design, and left to the lazy call
        inside the first evaluation they would be traced as breaks.

        The ``(1, 0)`` asserts pin wrap identity (``_frame_view``) for compiled
        MD ``op(work)``, not an F29 skip: LJ (``_lj_pair_batch``) has no
        ``bonds``, so ``_check_bonds_unwrapped`` returns before ``argmax``;
        Coulomb (``_coulomb_batch``) HAS bonds and F29 runs at the eager
        ``bind_topology`` / ``calc_energy`` below, after which the explained
        graph is ``energy()``. MD ``bind_work`` does not voucher F29.
        """
        from molix.compile import Compiler

        lj = _lj_pair_forcefield().to_potentials()
        lj_batch = _lj_pair_batch(r=COMPILED_CORE_R)
        # The strain variable needs a lattice to deform and a stored lattice
        # remainder to strain (F13); a 10 Å box with zero shifts leaves the
        # placed pair and its LJ energy exactly where they were.
        lj_batch["graphs"] = TensorDict(
            {"cell": torch.eye(3, dtype=DTYPE).unsqueeze(0) * 10.0}, batch_size=[1]
        )
        lj_batch["pairs", "shifts"] = torch.zeros_like(lj_batch["pairs", "pair_diff"])
        # No bonds on this fixture: F29 returns before argmax. (1, 0) is wrap
        # identity (``_frame_view``), not an F29-skip green.
        lj.bind_topology(lj_batch)
        lj.calc_energy(lj_batch)
        for wrt in (FORCES, FORCES_AND_VIRIAL):
            with torch.no_grad():
                summary = Compiler.explain(Derivative(lj.energy, wrt=wrt), lj_batch)
            assert (summary.graph_count, summary.graph_break_count) == (1, 0), (
                f"the lj/cut op wrt={wrt} must trace as one graph; "
                f"break sites: {summary.break_sites}"
            )

        coulomb = _coulomb_forcefield().to_potentials()
        cell = torch.eye(3, dtype=DTYPE) * COULOMB_EDGE
        coulomb.bind_cell(cell)
        pair_index, shifts = _coulomb_edges()
        periodic = _coulomb_batch(
            torch.tensor(COULOMB_POSITIONS, dtype=DTYPE), cell, pair_index, shifts
        )
        # Bonds are present; F29 runs at this eager bind / calc_energy. The
        # explained graph is energy() after that — wrap identity, not F29 skip.
        coulomb.bind_topology(periodic)
        coulomb.calc_energy(periodic)
        for wrt in (FORCES, FORCES_AND_VIRIAL):
            with torch.no_grad():
                summary = Compiler.explain(Derivative(coulomb.energy, wrt=wrt), periodic)
            assert (summary.graph_count, summary.graph_break_count) == (1, 0), (
                f"the periodic Coulomb op wrt={wrt} must trace as one graph; "
                f"break sites: {summary.break_sites}"
            )

    @pytest.mark.slow
    def test_the_compiled_route_matches_eager_and_breaks_only_at_the_leaf(self) -> None:
        """The prescribed sequence: parity, the break split, and fail-closed.

        ``bind_cell -> one eager call -> Compiler()(op)`` is the caller's
        composition. After it, the compiled NPT op over the cell-bound Polar
        mlip matches eager to 1e-10 in fp64 (reduction order only), the
        mlip's own op and the whole NPT op are one dynamo graph each in
        inference, and the op traced with grad enabled still breaks —
        attached outputs leak the leaves until pytorch/pytorch#196653 — and
        every break must be that leaf in ``molix/autograd/variables.py``.
        Owner: molix autograd. A break anywhere else is a regression.

        An over-capacity cell fails closed twice over: loudly at the eager
        rebind, as an in-graph NaN if the rebind is skipped — never as a
        truncated finite energy.

        The MD route is first-order **inference**: those evaluations run under
        ``torch.no_grad()``. Outside it the ambient grad mode asks for
        ``create_graph=True``, which AOTAutograd serves only while buffer
        donation is off — ``register_inductor_fallbacks`` pins
        ``torch._functorch.config.donated_buffer = False`` for exactly that
        reason. Pinned here rather than left as folklore: it is the difference
        between "MD" and "force-loss training" reaching the same compiled op
        (spec ``cuet-force-doublebackward``).
        """
        from molix.compile import Compiler
        from tests.test_molzoo.test_mace.conftest import CLUSTER_POS

        forcefield, mlip = _polar_forcefield()
        pots = forcefield.to_potentials()
        pos = torch.tensor(CLUSTER_POS, dtype=torch.float64)
        cell = torch.eye(3, dtype=torch.float64) * POLAR_CELL_EDGE

        pots.bind_cell(cell)
        pots.calc_energy(_polar_batch(pos, cell))
        npt = Derivative(pots.energy, wrt=FORCES_AND_VIRIAL)
        keys = (ENERGY_KEY, FORCES_KEY, VIRIAL_KEY)
        # MD inference is measured under no_grad; the grad-enabled route is
        # asserted after this block, once buffer donation is known to be off.
        with torch.no_grad():
            eager_out = npt(_polar_batch(pos, cell))
            eager = tuple(eager_out[key].detach().clone() for key in keys)

            compiled = Compiler(backend="aot_eager", dynamic=True)(npt)
            out = compiled(_polar_batch(pos, cell))
            for key, want in zip(keys, eager, strict=True):
                assert torch.allclose(out[key].detach(), want, atol=1e-10, rtol=0.0)

            work = _polar_batch(pos, cell)
            first = pots.calc_energy(work).detach().clone()
            assert torch.equal(work[ENERGY_KEY].detach(), first)
            # ONE atom moves. A rigid translation is *exactly* a symmetry of
            # this energy — the k-space |S(k)|² phases cancel and the local term
            # reads the frozen pair_diff — so shifting every atom asserts
            # nothing: the two values are bitwise equal on aarch64 and differ
            # only by last-bit round-off on x86.
            moved = pos.clone()
            moved[1, 0] += NUDGE
            work[POS_KEY] = moved
            second = pots.calc_energy(work).detach().clone()
            assert float((second - first).abs().max()) > ENERGY_CHANGE_MIN
            assert torch.equal(work[ENERGY_KEY].detach(), second)

            over = cell * CELL_OVER_CAPACITY
            with pytest.raises(RuntimeError) as excinfo:
                pots.bind_cell(over)
            assert "KSpaceGrid" in str(excinfo.value)
            assert "capacity_factor" in str(excinfo.value)
            # The compiled op never re-checks the capacity: an over-capacity
            # lattice comes back as an in-graph NaN, never a truncated number.
            assert bool(torch.isnan(compiled(_polar_batch(pos, over))[ENERGY_KEY]).all())
            within = cell * CELL_WITHIN_CAPACITY
            pots.bind_cell(within)
            assert bool(torch.isfinite(pots.calc_energy(_polar_batch(pos, within))).all())

        # The mlip's own op (built with ``wrt=()``: the energy) is one graph too.
        with torch.no_grad():
            own = Compiler.explain(mlip, _polar_batch(pos, cell))
        assert (own.graph_count, own.graph_break_count) == (1, 0), own.break_sites

        # Polar is the mlip ``energy`` branch and never reaches F29. The
        # (1, 0) is wrap identity (``_frame_view``) of the reused work TensorDict
        # for compiled MD ``op(work)``: a fresh wrap would break at
        # ``TypedTensorDict.__setattr__``. bind/cell already sit outside the
        # explained ``energy()`` graph. Not an F29-skip green, and not MD
        # ``bind_work`` vouching F29.
        step_pots = forcefield.to_potentials()
        step = Derivative(step_pots.energy, wrt=FORCES_AND_VIRIAL)
        step_batch = _polar_batch(pos, cell)
        with torch.no_grad():
            step(step_batch)
            inference = Compiler.explain(step, step_batch)
        assert (inference.graph_count, inference.graph_break_count) == (1, 0), (
            f"the NPT step must be one graph in inference; break sites: {inference.break_sites}"
        )

    @pytest.mark.slow
    def test_the_grad_enabled_compiled_route_is_one_graph_after_mark_leaves(self) -> None:
        """With grad enabled the compiled NPT op answers a create_graph=True call and,
        on marked leaves, traces as one graph with zero breaks on stock torch.

        Two preconditions, each owned by someone else than the op. The leaf:
        unmarked, stock torch 2.15 breaks where the in-region leaves leak
        through the attached outputs (``molix/autograd/variables.py``), so the
        columns are marked before the region (``mark_leaves``). The frame:
        ``Potentials._frame_view`` is wrap identity for compiled MD
        ``op(work)`` — a reused TensorDict does not wrap inside the graph.
        Polar is the mlip ``energy`` branch and never reaches F29. A batch
        it has never wrapped pays that host work inside the region: the
        typed-view wrap breaks at tensordict's ``TypedTensorDict.__setattr__``
        ("mapping proxy affected by dictionary mutation"), the resumed trace
        then refuses the in-graph ``autograd.grad`` ("external grad_fn"), and
        AOTAutograd's backward prologue fails on a tangent leaf. Measured
        2026-09-11 (``.scratch/compile-seam/repro/diag_polar_marked.py``, job
        2306962): fresh batch unmarked 11 graphs, fresh batch marked 7 graphs
        (the wrap and the grad, no leaf), a batch wrapped once before —
        marked, grad enabled — **1 graph, 0 breaks**. That ``(1, 0)`` is wrap
        identity, not an F29-skip green, and not MD ``bind_work`` vouching F29
        (``Derivative`` differentiates ``energy()``). The classical route's
        documented precondition is therefore one eager wrap of the very frame
        the compiled op is handed, as the inference test above applies.
        """
        from molix.compile import Compiler
        from tests.test_molzoo.test_mace.conftest import CLUSTER_POS

        forcefield, _mlip = _polar_forcefield()
        pots = forcefield.to_potentials()
        pos = torch.tensor(CLUSTER_POS, dtype=torch.float64)
        cell = torch.eye(3, dtype=torch.float64) * POLAR_CELL_EDGE
        pots.bind_cell(cell)
        pots.calc_energy(_polar_batch(pos, cell))
        npt = Derivative(pots.energy, wrt=FORCES_AND_VIRIAL)
        keys = (ENERGY_KEY, FORCES_KEY, VIRIAL_KEY)
        with torch.no_grad():
            eager_out = npt(_polar_batch(pos, cell))
            eager = tuple(eager_out[key].detach().clone() for key in keys)
        compiled = Compiler(backend="aot_eager", dynamic=True)(npt)
        # Buffer donation is pinned off by the compile itself (the fallback
        # registration); a create_graph=True call needs it off.
        assert torch._functorch.config.donated_buffer is False
        # The same work object: ``_frame_view`` is wrap identity, so the
        # explained ``energy()`` graph does not wrap inside the region. Polar
        # never reaches F29 (mlip ``energy``).
        batch = _polar_batch(pos, cell)
        with torch.no_grad():
            npt(batch)
        with_grad = compiled(npt.mark_leaves(batch))
        for key, want in zip(keys, eager, strict=True):
            assert torch.allclose(with_grad[key].detach(), want, atol=1e-10, rtol=0.0)
        training = Compiler.explain(npt, npt.mark_leaves(batch))
        assert (training.graph_count, training.graph_break_count) == (1, 0), (
            f"the marked NPT step must be one graph with grad enabled; "
            f"break sites: {training.break_sites}"
        )


class TestPotentialsPaddedBatch:
    """``Potentials.energy`` on a ``MoleculePadder`` batch: ghosts enter no term.

    The classical container has no per-atom energy term — every declared style
    is a bonded term over the topology namespaces or a pair term over the placed
    list — so an ``atoms.mask`` has nothing to multiply: a ghost atom enters no
    bond / angle / torsion column, and a padding edge is a ``(ghost, ghost)``
    self-loop that the ``source < target`` liveness rule already kills. The
    padded energy is therefore the unpadded one without any mask read, the real
    atoms' forces are unchanged and the ghosts' are zero. A *pair* style prices
    a padded batch once its per-atom ``epsilon`` / ``sigma`` / ``charge`` tables are
    handed in per batch at the padded atom count (the Class-I training path,
    where the heads predict them); the seed tables a container carries at the
    real atom count cannot index a ghost — see ``Potentials.energy``.
    """

    def test_ghost_atoms_and_dead_edges_price_nothing(self) -> None:
        from molix.data import MoleculePadder

        batch = _two_atom_bond_batch()
        batch["pairs"] = TensorDict(
            {
                "pair_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                "pair_diff": torch.tensor([[BOND_R, 0.0, 0.0], [-BOND_R, 0.0, 0.0]], dtype=DTYPE),
                "pair_dist": torch.tensor([BOND_R, BOND_R], dtype=DTYPE),
            },
            batch_size=[2],
        )
        batch["graphs"] = TensorDict({}, batch_size=[1])
        reference = _bond_potentials(wrt=FORCES)(batch.clone())
        padded = MoleculePadder(
            max_atoms=4,
            max_pairs=5,
            cutoff=BOND_R0,
            max_bonds=1,
            max_angles=0,
            max_propers=0,
            max_impropers=0,
        )(batch.clone())
        out = _bond_potentials(wrt=FORCES)(padded)
        assert torch.equal(out[ENERGY_KEY], reference[ENERGY_KEY])
        assert torch.equal(out[FORCES_KEY][:2], reference[FORCES_KEY])
        assert torch.equal(out[FORCES_KEY][2:], torch.zeros(2, 3, dtype=DTYPE))
        assert float(out[ENERGY_KEY].reshape(())) == pytest.approx(GOLDEN_ENERGY, abs=1e-10)

    def test_dummy_bonds_with_mask_price_nothing(self) -> None:
        from molix.data import MoleculePadder

        batch = _two_atom_bond_batch()
        batch["pairs"] = TensorDict(
            {
                "pair_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                "pair_diff": torch.tensor([[BOND_R, 0.0, 0.0], [-BOND_R, 0.0, 0.0]], dtype=DTYPE),
                "pair_dist": torch.tensor([BOND_R, BOND_R], dtype=DTYPE),
            },
            batch_size=[2],
        )
        batch["graphs"] = TensorDict({}, batch_size=[1])
        reference = _bond_potentials(wrt=FORCES)(batch.clone())
        padded = MoleculePadder(
            max_atoms=8,
            max_pairs=8,
            cutoff=BOND_R0,
            max_bonds=4,
            max_angles=0,
            max_propers=0,
            max_impropers=0,
        )(batch.clone())
        out = _bond_potentials(n_bonds=4, wrt=FORCES)(padded)
        assert torch.allclose(out[ENERGY_KEY], reference[ENERGY_KEY], atol=1e-10)
        assert torch.allclose(out[FORCES_KEY][:2], reference[FORCES_KEY], atol=1e-10)
        assert torch.equal(out[FORCES_KEY][2:], torch.zeros(6, 3, dtype=DTYPE))

    def test_pair_style_prices_a_padded_batch_with_per_batch_tables(self) -> None:
        """LJ over a padded batch with graph slots: live graph exact, dead slots zero.

        The per-atom tables arrive per call at the padded atom count (as the
        Class-I heads predict them); padding edges are ``(ghost0, ghost0)``
        self-loops the ``source < target`` rule kills, ghosts sit on no live
        edge, and ghost atoms scatter into the sink slot.
        """
        from functools import partial

        from molix.data import MoleculePadder

        potentials = _lj_pair_forcefield().to_potentials()
        batch = _lj_pair_batch(r=3.5)
        batch["graphs"] = TensorDict({"num_atoms": torch.tensor([2])}, batch_size=[1])
        reference = potentials.calc_energy(batch.clone())
        forces_ref = Derivative(potentials.energy, wrt=FORCES)(batch.clone())[FORCES_KEY]
        padded = MoleculePadder(
            max_atoms=6,
            max_pairs=8,
            cutoff=LJ_CUTOFF,
            max_bonds=0,
            max_angles=0,
            max_propers=0,
            max_impropers=0,
            max_graphs=3,
        )(batch.clone())
        tables = {
            "atoms": {
                "epsilon": torch.full((6,), EPSILON, dtype=DTYPE),
                "sigma": torch.full((6,), SIGMA, dtype=DTYPE),
            }
        }
        # A new topology shape is a new bind; the exclusion table is never
        # rebuilt implicitly (bind_topology's contract).
        potentials.bind_topology(padded)
        out = potentials.calc_energy(padded.clone(), tables)
        assert out.shape == (3,)
        assert torch.allclose(out[0], reference.reshape(-1)[0], atol=1e-12)
        assert torch.equal(out[1:], torch.zeros(2, dtype=DTYPE))

        op = Derivative(partial(potentials.energy, tables=tables), wrt=FORCES)
        forces = op(padded)[FORCES_KEY]
        assert torch.allclose(forces[:2], forces_ref, atol=1e-12)
        assert torch.equal(forces[2:], torch.zeros(4, 3, dtype=DTYPE))


class TestPartialHessian:
    """``Potentials.partial_hessian`` — stage 1's MM target.

    The pairs, the accumulation over terms and the ``3x3`` sub-block indices are
    all asserted through one property: the blocks must equal the corresponding
    blocks of the full ``(3N, 3N)`` Hessian of the very same
    :meth:`Potentials.energy`. Nothing weaker distinguishes a correct scatter
    from a plausible one.
    """

    #: A five-atom chain with a branch: four bonds, four angles, one proper and
    #: one improper, so every term type contributes and pairs are shared.
    POSITIONS = torch.tensor(
        [
            [0.00, 0.00, 0.00],
            [1.52, 0.10, 0.05],
            [2.10, 1.42, -0.20],
            [3.55, 1.50, 0.35],
            [1.95, -0.85, 0.90],
        ],
        dtype=torch.float64,
    )
    BONDS = torch.tensor([[0, 1, 2, 1], [1, 2, 3, 4]])
    ANGLES = torch.tensor([[0, 1, 0, 2], [1, 2, 1, 1], [2, 3, 4, 4]])
    PROPERS = torch.tensor([[0], [1], [2], [3]])
    #: molrs center-first: the improper's central atom is row 0.
    IMPROPERS = torch.tensor([[1], [0], [2], [4]])

    @staticmethod
    def _forcefield() -> ForceField:
        """Bonded-only Class-I field, one parameter row per term."""
        return (
            ForceField(name="partial-hessian", units="real")
            .def_atomstyle("full")
            .def_bondstyle("harmonic")
            .def_anglestyle("harmonic")
            .def_dihedralstyle("periodic")
            .def_improperstyle("periodic")
            .with_params(
                bonds={
                    "k": torch.tensor([310.0, 300.0, 320.0, 340.0], dtype=torch.float64),
                    "r0": torch.tensor([1.526, 1.510, 1.500, 1.090], dtype=torch.float64),
                },
                angles={
                    "k": torch.tensor([80.0, 75.0, 70.0, 65.0], dtype=torch.float64),
                    "theta0": torch.full((4,), 1.9670, dtype=torch.float64),
                },
                propers={
                    "k": torch.tensor([[1.3, 0.4, 0.25]], dtype=torch.float64),
                    "periodicity": torch.tensor([[1, 2, 3]]),
                    "phase": torch.tensor([[0.0, torch.pi, 0.0]], dtype=torch.float64),
                },
                impropers={
                    "k": torch.tensor([[2.1, 1.1]], dtype=torch.float64),
                    "periodicity": torch.tensor([[1, 2]]),
                    "phase": torch.tensor([[0.0, torch.pi]], dtype=torch.float64),
                },
            )
        )

    @classmethod
    def _layout(cls) -> PartialHessianPairs:
        return PartialHessianPairs(
            {
                "bonds": cls.BONDS,
                "angles": cls.ANGLES,
                "propers": cls.PROPERS,
                "impropers": cls.IMPROPERS,
            }
        )

    @classmethod
    def _batch(cls, positions: torch.Tensor) -> TensorDict:
        layout = cls._layout()
        n_atoms = int(positions.shape[0])
        batch = TensorDict(
            {
                "atoms": TensorDict(
                    {
                        "pos": positions,
                        "Z": torch.tensor([6, 6, 6, 6, 1]),
                        "batch": torch.zeros(n_atoms, dtype=torch.long),
                    },
                    batch_size=[n_atoms],
                ),
                "bonds": TensorDict(
                    {"atomi": cls.BONDS[0], "atomj": cls.BONDS[1]},
                    batch_size=[cls.BONDS.shape[1]],
                ),
                "angles": TensorDict(
                    {"atomi": cls.ANGLES[0], "atomj": cls.ANGLES[1], "atomk": cls.ANGLES[2]},
                    batch_size=[cls.ANGLES.shape[1]],
                ),
                "propers": TensorDict(
                    {
                        "atomi": cls.PROPERS[0],
                        "atomj": cls.PROPERS[1],
                        "atomk": cls.PROPERS[2],
                        "atoml": cls.PROPERS[3],
                    },
                    batch_size=[1],
                ),
                "impropers": TensorDict(
                    {
                        "atomi": cls.IMPROPERS[0],
                        "atomj": cls.IMPROPERS[1],
                        "atomk": cls.IMPROPERS[2],
                        "atoml": cls.IMPROPERS[3],
                    },
                    batch_size=[1],
                ),
                "graphs": TensorDict({"num_atoms": torch.tensor([n_atoms])}, batch_size=[1]),
            },
            batch_size=[],
        )
        for namespace, slots in layout.slots.items():
            for (i, j), record in slots.items():
                batch[namespace, f"hpair_{i}_{j}"] = record
        batch["hessian_pairs"] = TensorDict(
            {"atomi": layout.pairs[:, 0], "atomj": layout.pairs[:, 1]},
            batch_size=[len(layout)],
        )
        return batch

    def test_the_blocks_are_the_full_hessians_blocks_on_those_pairs(self) -> None:
        """Every accumulated pair block equals the full Hessian's, to fp64.

        This is the whole contract: the deduplicated pairs, the sum over the
        bonds / angles / propers / impropers that share each pair, the ``(i, j)``
        sub-block picks, and the improper's center-first atom order.
        """
        potentials = Potentials(self._forcefield())
        layout = self._layout()

        def total(flat: torch.Tensor) -> torch.Tensor:
            return potentials.energy(self._batch(flat.reshape(-1, 3)))[ENERGY_KEY].sum()

        full = torch.autograd.functional.hessian(total, self.POSITIONS.reshape(-1), vectorize=True)
        # Marked as the host prologue marks it: the curvature is taken in the
        # positions, and `blocks` refuses to mark them itself because creating a
        # leaf inside a compiled region does not trace. A clone, so the reference
        # above differentiates an untouched tensor.
        marked = self.POSITIONS.clone().requires_grad_(True)
        got = potentials.partial_hessian(self._batch(marked))
        assert got.shape == (len(layout), 9)
        torch.testing.assert_close(got, layout.gather(full))

    def test_a_populated_namespace_without_slot_columns_is_refused(self) -> None:
        """A term with rows and no slots contributes nothing, and used to say nothing.

        ``ClassIPrepare.label`` re-derived the valence families and replaced the
        blocks that ``HessianTargets`` had written the slot columns onto, so a
        production batch reached this method with slot columns on ``bonds`` and
        none on ``angles`` / ``propers`` / ``impropers``. The loop skipped them,
        and stage 2 of the curriculum supervised the bond term alone for four
        hours — its angle term was still at 0.004 kcal/mol afterwards. Dropping a
        whole family from a target is not a thing to infer from a missing column.
        """
        batch = self._batch(self.POSITIONS.clone().requires_grad_(True))
        for key in [k for k in batch["angles"].keys() if k.startswith("hpair_")]:
            del batch["angles", key]
        with pytest.raises(ValueError, match="no hpair_<i>_<j> slot column"):
            Potentials(self._forcefield()).partial_hessian(batch)

    def test_an_empty_namespace_without_slot_columns_is_skipped(self) -> None:
        """A molecule with no propers has nothing to contribute and no columns."""
        batch = self._batch(self.POSITIONS.clone().requires_grad_(True))
        for key in [k for k in batch["propers"].keys() if k.startswith("hpair_")]:
            del batch["propers", key]
        batch["propers"] = batch["propers"][:0]
        Potentials(self._forcefield()).partial_hessian(batch)

    def test_a_batch_without_the_namespace_is_refused(self) -> None:
        """The pair list is the target's own; there is nothing to answer without it."""
        batch = self._batch(self.POSITIONS)
        del batch["hessian_pairs"]
        with pytest.raises(ValueError, match="no hessian_pairs namespace"):
            Potentials(self._forcefield()).partial_hessian(batch)
