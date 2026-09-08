"""Coulomb term on ForceField.to_potentials() reading atoms.q.

Open boundaries truncate ``1/r`` at a cutoff; a ``graphs.cell`` switches the
same style to a range-separated sum — Ewald (explicit k-vectors) or PME (the
same sum on an FFT mesh). The split ``1/r = erfc(r/σ√2)/r + erf(r/σ√2)/r`` puts
the second half in the reciprocal sum, which runs over **every** pair of the
cell: an atom's Gaussian charge cloud is felt by its bonded partner exactly as
by any other neighbour. Removing a 1-2 / 1-3 pair therefore costs the *whole*
``(1 − s) k_e q_i q_j / r``, not only the real-space ``erfc`` share — LAMMPS
``special_bonds`` with ``coul/long`` and OpenMM ``NonbondedForce`` exceptions
both subtract the full term. That identity is what
:class:`TestCoulombPeriodicPotential` pins, together with the frozen
reciprocal-space shape that :meth:`molix.schema.CellBound.bind_cell` fixes and
its fail-closed capacity.

The Madelung energy of a rocksalt lattice is the one absolute number here: a
literature constant of the lattice, hard-coded, never re-derived from a
third-party code at test time.
"""

from __future__ import annotations

import itertools
import math

import pytest
import torch
from tensordict import TensorDict

from molpot.charge import (
    AtomicChargeModel,
    ChiEtaHead,
    DiagonalHardnessKernel,
    ElectronegativityEqualization,
)
from molpot.forcefield import ForceField, Potentials
from molpot.ir.scaling import ExclusionScales, NonbondedScaling
from molpot.potentials.elec.prefactors import kcalmol_A
from tests.conftest import place_edges

#: Build radius (Å) of the placed list on the **open**-boundary fixtures of
#: :class:`TestCoulombEnergy`. Open ``pair:coulomb`` declares no cutoff there
#: (``CoulombOpenPotential.cutoff is None``, i.e. every pair counts), so the
#: list has to reach past the largest separation any of those fixtures uses
#: (2.83 Å) for its pair set to be the one the deleted molpot enumerator
#: produced. ``composable-a0-03b``: molpot consumes a placed list, molix builds
#: it.
OPEN_LIST_CUTOFF = 10.0

#: Cubic edge (Å) of the small periodic box every exclusion / capacity test
#: below runs in. Large enough that the 5 Å probe pair is still its own
#: minimum image (``r < L/2``), small enough that a converged k-sum is cheap.
BOX = 12.0
#: Ewald / PME splitting width ``σ`` (Å). With ``r_c = 4 Å`` the neglected
#: real-space tail is ``erfc(r_c/σ√2) ≈ 6e-7``, and with the wavelength below
#: the neglected reciprocal tail is smaller still — both sums are converged
#: well past every tolerance asserted here.
SMEARING = 0.8
#: Real-space cutoff ``r_c`` (Å), inside the minimum-image limit ``L/2 = 6``.
REAL_SPACE_CUTOFF = 4.0
#: Shortest reciprocal wavelength (Å) of the ``"ewald"`` sum. Deliberately not
#: an exact divisor of :data:`BOX`: the per-axis index ``L/λ = 38.7`` is then
#: never within round-off of an integer, so the grid the box asks for is
#: unambiguous (39 here, 37 after :data:`SHRINK`).
LR_WAVELENGTH = 0.31
#: Mesh spacing (Å) of the ``"pme"`` sum — 32³ on :data:`BOX`.
MESH_SPACING = 1.0
#: Isotropic contraction used to prove the k-space **shape** is frozen: the
#: contracted box needs a strictly smaller grid than the bound one.
SHRINK = 0.95
#: Separation (Å) of the bonded probe pair inside :data:`REAL_SPACE_CUTOFF`.
BONDED_INSIDE = 1.5
#: Separation (Å) of the bonded probe pair beyond it, still inside ``L/2``.
BONDED_BEYOND = 5.0
#: Build radius (Å) of the placed list for that fixture: past
#: :data:`BONDED_BEYOND` and still inside the minimum-image limit ``L/2 = 6``,
#: so the list *carries* the 5 Å pair as a row the ``0 < r < r_c`` liveness
#: rule kills. Building it at :data:`REAL_SPACE_CUTOFF` would produce no row at
#: all, and then "the real-space sum did not add this pair" would be a property
#: of the list's emptiness rather than of the mask under test.
BEYOND_LIST_CUTOFF = 5.5
#: Energies here are ``O(200)`` kcal/mol, so an exact fp64 identity lands
#: ~1e-13; 1e-8 kcal/mol is the contract with room for the reduction order.
EXCLUSION_TOL = 1e-8
#: Two energies that must agree because one was computed at a *frozen* k-space
#: shape and the other at the live one. Both sums are converged far below this.
FROZEN_SHAPE_TOL = 1e-10

#: Madelung constant of the rocksalt (NaCl, B1) lattice — the dimensionless
#: value of the conditionally convergent lattice sum ``Σ' ±1/(r/d)`` taken in
#: the neutral-shell order. **Hard-coded golden**: a mathematical constant of
#: the lattice, quoted from the standard tabulation (Kittel, *Introduction to
#: Solid State Physics*, Table 3.6 lists ``α_NaCl = 1.747565``); the extra
#: digits below are the same constant to fp64-usable precision. Not captured
#: from any third-party code, and nothing here recomputes it.
MADELUNG_ROCKSALT = 1.74756459463
#: Nearest-neighbour separation ``d`` (Å) of the rocksalt fixture — the NaCl
#: experimental value, chosen only so the golden reads in a familiar scale.
ROCKSALT_SPACING = 2.82
#: ``2 x 2 x 2`` conventional cells → 64 ions, edge ``4 d = 11.28`` Å. The
#: single conventional cell (8 ions, edge ``2 d``) is **not** usable: its
#: nearest neighbours sit at exactly ``L/2``, where the minimum-image
#: convention keeps one of the two equidistant images and the real-space sum
#: silently halves. Doubling the cell moves every neighbour strictly inside
#: ``L/2``.
ROCKSALT_CELLS = 2
#: Ewald / PME parameters for the rocksalt fixture (edge 11.28 Å, ``r_c`` just
#: inside ``L/2 = 5.64``). Both sums converge to ~6e-10 relative there.
ROCKSALT_SMEARING = 0.9
ROCKSALT_CUTOFF = 5.6
ROCKSALT_LR_WAVELENGTH = 0.5
ROCKSALT_MESH_SPACING = 1.0


class TestCoulombEnergy:
    def test_two_charge_analytic(self):
        """q=±1 at r=1 Å → E = −ke."""
        ff = (
            ForceField(name="es")
            .def_pairstyle("coulomb")
            .with_params(
                atoms={
                    "q": torch.tensor([1.0, -1.0], dtype=torch.float64),
                }
            )
        )
        batch = TensorDict(
            {
                "atoms": TensorDict(
                    {
                        "pos": torch.tensor(
                            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64
                        ),
                        "batch": torch.zeros(2, dtype=torch.long),
                    },
                    batch_size=[2],
                )
            },
            batch_size=[],
        )
        place_edges(batch, cutoff=OPEN_LIST_CUTOFF)
        e = ff.to_potentials().calc_energy(batch)
        assert math.isclose(float(e), -kcalmol_A, rel_tol=1e-10)

    def test_bond_exclusion_scale_zero(self):
        """1–2 pair with the Class-I default ``q.s12 = 0`` contributes nothing."""
        ff = (
            ForceField(
                name="es",
                nonbonded_scaling=NonbondedScaling(),
            )
            .def_pairstyle("coulomb")
            .with_params(atoms={"q": torch.tensor([1.0, -1.0], dtype=torch.float64)})
        )
        batch = TensorDict(
            {
                "atoms": TensorDict(
                    {
                        "pos": torch.tensor(
                            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64
                        ),
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
        place_edges(batch, cutoff=OPEN_LIST_CUTOFF)
        e = ff.to_potentials().calc_energy(batch)
        assert math.isclose(float(e), 0.0, abs_tol=1e-12)

    def test_scale_14(self):
        """Four atoms; only i–l is 1–4 with scale 0.5."""
        scaling = NonbondedScaling(q=ExclusionScales(0.0, 0.0, 0.5))
        q = torch.tensor([1.0, 0.0, 0.0, -1.0], dtype=torch.float64)
        # place 0 and 3 at r=2 so E_full = ke*(1)*(-1)/2 = -ke/2; with s14=0.5 → -ke/4
        pos = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],  # will move
            ],
            dtype=torch.float64,
        )
        pos[3] = torch.tensor([0.0, 0.0, 2.0], dtype=torch.float64)
        ff = (
            ForceField(name="es", nonbonded_scaling=scaling)
            .def_pairstyle("coulomb")
            .with_params(atoms={"q": q})
        )
        batch = TensorDict(
            {
                "atoms": TensorDict(
                    {"pos": pos, "batch": torch.zeros(4, dtype=torch.long)},
                    batch_size=[4],
                ),
                "bonds": TensorDict(
                    {
                        "atomi": torch.tensor([0, 1, 2], dtype=torch.long),
                        "atomj": torch.tensor([1, 2, 3], dtype=torch.long),
                    },
                    batch_size=[3],
                ),
                "propers": TensorDict(
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
        place_edges(batch, cutoff=OPEN_LIST_CUTOFF)
        e = ff.to_potentials().calc_energy(batch)
        # only non-excluded pairs among (0,2)=1-3 scale0, (0,3)=1-4 scale0.5, (1,3)=1-3 scale0
        # r_03 = 2 → contribution 0.5 * ke * (1)*(-1)/2 = -0.25 ke
        expected = -0.25 * kcalmol_A
        assert math.isclose(float(e), expected, rel_tol=1e-8)

    def test_charge_model_into_coulomb(self):
        """EEM q written then Coulomb energy finite and neutral-consistent."""
        charge = AtomicChargeModel(
            ChiEtaHead(feature_dim=4, hidden_dim=8),
            ElectronegativityEqualization(DiagonalHardnessKernel()),
        )
        ff = (
            ForceField(name="mm")
            .def_pairstyle("coulomb")
            .with_params(atoms={"q": torch.tensor([0.3, -0.3], dtype=torch.float64)})
        )
        batch = TensorDict(
            {
                "atoms": TensorDict(
                    {
                        "pos": torch.tensor(
                            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float64
                        ),
                        "batch": torch.zeros(2, dtype=torch.long),
                    },
                    batch_size=[2],
                )
            },
            batch_size=[],
        )
        place_edges(batch, cutoff=OPEN_LIST_CUTOFF)
        e = ff.to_potentials().calc_energy(batch)
        expected = kcalmol_A * (0.3 * -0.3) / 1.5
        assert math.isclose(float(e), expected, rel_tol=1e-10)
        ff2 = charge.write_forcefield(
            ForceField(name="mm").def_pairstyle("coulomb"),
            torch.zeros(2, 4),
            batch,
        )
        e2 = ff2.to_potentials().calc_energy(batch)
        assert torch.isfinite(e2)

    def test_uses_coulomb_potential_kernel(self):
        """Open-boundary path must match CoulombPotential.from_dist."""
        from molpot.potentials.elec.potentials import CoulombPotential

        q = torch.tensor([0.5, -0.5], dtype=torch.float64)
        r = 2.0
        pot = CoulombPotential(prefactor=kcalmol_A)
        e_ref = (q[0] * q[1] * pot.from_dist(torch.tensor([r], dtype=torch.float64))).sum()
        ff = ForceField(name="es").def_pairstyle("coulomb").with_params(atoms={"q": q})
        batch = TensorDict(
            {
                "atoms": TensorDict(
                    {
                        "pos": torch.tensor([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=torch.float64),
                        "batch": torch.zeros(2, dtype=torch.long),
                    },
                    batch_size=[2],
                )
            },
            batch_size=[],
        )
        place_edges(batch, cutoff=OPEN_LIST_CUTOFF)
        e = ff.to_potentials().calc_energy(batch)
        assert math.isclose(float(e), float(e_ref), rel_tol=1e-12)

    def test_ewald_path_with_cell(self):
        """graphs.cell present → EwaldCalculator path (finite energy)."""
        a = 10.0
        q = torch.tensor([1.0, -1.0], dtype=torch.float64)
        ff = (
            ForceField(name="es")
            .def_pairstyle(
                "coulomb",
                calculator="ewald",
                smearing=4.0,
                lr_wavelength=1.5,
                real_space_cutoff=8.0,
            )
            .with_params(atoms={"q": q})
        )
        batch = TensorDict(
            {
                "atoms": TensorDict(
                    {
                        "pos": torch.tensor(
                            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64
                        ),
                        "batch": torch.zeros(2, dtype=torch.long),
                    },
                    batch_size=[2],
                ),
                "graphs": TensorDict(
                    {
                        "cell": torch.eye(3, dtype=torch.float64) * a,
                    },
                    batch_size=[],
                ),
            },
            batch_size=[],
        )
        # The style's own ``real_space_cutoff``: the list holds exactly the
        # rows the real-space sum is allowed to read.
        place_edges(batch, cutoff=8.0)
        e = ff.to_potentials().calc_energy(batch)
        assert torch.isfinite(e)
        # opposite charges should give attractive (negative) total energy
        assert float(e) < 0.0

    def test_cutoff_lives_on_lj_potential_not_potentials(self):
        ff = ForceField(name="lj").def_pairstyle("lj/cut", cutoff=12.5)
        pots = ff.to_potentials()
        assert pots.lj is not None
        assert pots.lj.cutoff == 12.5
        assert not hasattr(pots, "lj_cutoff") or getattr(pots, "lj_cutoff", None) is None


#: Growth factor whose k-space demand exceeds a ``capacity_factor = 1`` bind.
OVER_CAPACITY = 2.0
#: Growth factor used for the in-graph half of the same guard — a cell this
#: much larger cannot be served by the frozen grid under any head-room here.
FAR_OVER_CAPACITY = 3.0
#: Head-room asked for explicitly, and a growth that stays inside it.
CAPACITY_FACTOR = 1.5
GROWTH_WITHIN_CAPACITY = 1.4


def _periodic_potentials(
    charges: torch.Tensor,
    *,
    calculator: str = "ewald",
    smearing: float = SMEARING,
    real_space_cutoff: float = REAL_SPACE_CUTOFF,
    lr_wavelength: float = LR_WAVELENGTH,
    mesh_spacing: float = MESH_SPACING,
    **options: float,
) -> Potentials:
    """Assemble a Coulomb-only ForceField over ``charges`` and hand back its terms.

    Args:
        charges: Per-atom partial charges ``(N,)`` in ``e``.
        calculator: ``"ewald"`` or ``"pme"`` — the reciprocal-space algorithm.
        smearing: Gaussian splitting width ``σ`` in Å.
        real_space_cutoff: Real-space cutoff ``r_c`` in Å.
        lr_wavelength: Shortest reciprocal wavelength resolved by ``"ewald"``.
        mesh_spacing: Grid spacing of the ``"pme"`` mesh in Å.
        **options: Extra ``pair:coulomb`` style options (``capacity_factor``).

    Returns:
        The assembled :class:`~molpot.forcefield.Potentials`.
    """
    forcefield = (
        ForceField(name="es")
        .def_pairstyle(
            "coulomb",
            calculator=calculator,
            smearing=smearing,
            real_space_cutoff=real_space_cutoff,
            lr_wavelength=lr_wavelength,
            mesh_spacing=mesh_spacing,
            **options,
        )
        .with_params(atoms={"q": charges})
    )
    return forcefield.to_potentials()


def _periodic_batch(
    positions: torch.Tensor,
    *,
    edge: float = BOX,
    list_cutoff: float = REAL_SPACE_CUTOFF,
    bonds: tuple[tuple[int, int], ...] = (),
    propers: tuple[tuple[int, int, int, int], ...] = (),
) -> TensorDict:
    """One periodic molecule: ``atoms`` + ``graphs.cell`` + ``edges`` (+ topology).

    The ``edges`` namespace is built by :func:`place_edges`, which routes the
    cell into :class:`molix.batch.NeighborList` under ``pbc=True`` — the same
    minimum-image half list the deleted molpot enumerator produced from molpy's
    ``NeighborQuery``, verified pair-for-pair on every geometry in this file
    (``composable-a0-03b``).

    Args:
        positions: Atom positions ``(N, 3)`` in Å.
        edge: Cubic lattice edge in Å.
        list_cutoff: Build radius of the placed list in Å. Defaults to the
            real-space cutoff the periodic term declares, so the list holds
            exactly the rows that term is allowed to read — a separation past
            it yields an **empty** list, which is what "the pair is not in the
            real-space sum" means here.
        bonds: ``(i, j)`` covalent bonds — the 1-2 relations of the topology.
        propers: ``(i, j, k, l)`` proper torsions — the 1-4 relations.

    Returns:
        A post-collate batch carrying a placed minimum-image ``edges``
        namespace, the lattice and any topology asked for.
    """
    n_atoms = positions.shape[0]
    namespaces: dict[str, TensorDict] = {
        "atoms": TensorDict(
            {"pos": positions, "batch": torch.zeros(n_atoms, dtype=torch.long)},
            batch_size=[n_atoms],
        ),
        "graphs": TensorDict({"cell": torch.eye(3, dtype=positions.dtype) * edge}, batch_size=[]),
    }
    if bonds:
        namespaces["bonds"] = TensorDict(
            {
                "atomi": torch.tensor([i for i, _ in bonds], dtype=torch.long),
                "atomj": torch.tensor([j for _, j in bonds], dtype=torch.long),
            },
            batch_size=[len(bonds)],
        )
    if propers:
        namespaces["propers"] = TensorDict(
            {
                key: torch.tensor([q[axis] for q in propers], dtype=torch.long)
                for axis, key in enumerate(("atomi", "atomj", "atomk", "atoml"))
            },
            batch_size=[len(propers)],
        )
    return place_edges(TensorDict(namespaces, batch_size=[]), cutoff=list_cutoff)


def _ion_pair(separation: float) -> tuple[torch.Tensor, torch.Tensor]:
    """A ``q = ±1`` pair ``separation`` Å apart along x̂ — positions and charges."""
    positions = torch.tensor([[0.0, 0.0, 0.0], [separation, 0.0, 0.0]], dtype=torch.float64)
    return positions, torch.tensor([1.0, -1.0], dtype=torch.float64)


def _rocksalt(cells: int, spacing: float) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Alternating ±1 ions on a simple-cubic grid — the rocksalt sublattice pair.

    Args:
        cells: Conventional cells per axis. ``2 cells`` ions sit on each axis.
        spacing: Nearest-neighbour separation ``d`` in Å.

    Returns:
        ``(positions, charges, edge)`` — ``(N, 3)`` Å, ``(N,)`` in ``e`` and the
        cubic lattice edge ``2 cells d`` in Å, with ``N = (2 cells)³`` ions,
        half of each sign, so the cell is neutral.
    """
    sites = range(2 * cells)
    grid = list(itertools.product(sites, sites, sites))
    positions = torch.tensor(
        [[i * spacing, j * spacing, k * spacing] for i, j, k in grid], dtype=torch.float64
    )
    charges = torch.tensor([(-1.0) ** (i + j + k) for i, j, k in grid], dtype=torch.float64)
    return positions, charges, 2 * cells * spacing


def _record_kspace_sizes(potentials: Potentials, monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Reciprocal-vector count of every later energy call, in call order.

    The one observable that separates a **frozen** k-space shape from one sized
    off the live cell without naming private state: the range-separated kernel
    ``lr_from_k_sq`` is evaluated exactly once per energy call, on exactly the
    grid the calculator sums over, so its input size *is* the shape in use.

    Args:
        potentials: The assembled terms whose periodic Coulomb term to watch.
        monkeypatch: pytest's patcher — the wrapper is removed with the test.

    Returns:
        A list the wrapper appends to; empty until the first energy call.
    """
    term = potentials.coulomb_periodic
    assert term is not None
    kernel = term.potential.lr_from_k_sq
    sizes: list[int] = []

    def counting(k_sq: torch.Tensor) -> torch.Tensor:
        sizes.append(int(k_sq.numel()))
        return kernel(k_sq)

    monkeypatch.setattr(term.potential, "lr_from_k_sq", counting)
    return sizes


class TestCoulombPeriodicPotential:
    """The periodic branch of ``pair:coulomb``: exclusions, goldens, k-space shape.

    Everything here goes through the assembled
    :meth:`~molpot.forcefield.Potentials.calc_energy`, because the exclusion
    correction is a collaboration between the topology table
    (:class:`~molpot.forcefield.exclusions.ExclusionTable`) and the term — the
    term alone cannot state the identity.
    """

    @pytest.mark.parametrize("calculator", ["ewald", "pme"])
    def test_an_excluded_pair_inside_the_cutoff_costs_its_whole_coulomb_term(
        self, calculator: str
    ) -> None:
        """A 1-2 pair within ``r_c`` must lose ``k_e q_i q_j / r``, erf share included.

        The reciprocal sum carries ``erf(r/σ√2)/r`` for *every* pair of the
        cell, bonded ones included, so an exclusion that only cancels the
        real-space ``erfc`` share leaves that half standing at full strength.
        At ``r = 1.5 Å`` with ``σ = 0.8 Å`` the surviving half is
        ``k_e erf(1.326)/1.5 ≈ 208 kcal/mol`` — two thirds of the interaction
        the force field believes it removed.
        """
        positions, charges = _ion_pair(BONDED_INSIDE)
        free = _periodic_potentials(charges, calculator=calculator).calc_energy(
            _periodic_batch(positions)
        )
        excluded = _periodic_potentials(charges, calculator=calculator).calc_energy(
            _periodic_batch(positions, bonds=((0, 1),))
        )
        removed = kcalmol_A * float(charges[0] * charges[1]) / BONDED_INSIDE
        assert math.isclose(
            float(excluded), float(free) - removed, abs_tol=EXCLUSION_TOL, rel_tol=0.0
        )

    def test_an_excluded_pair_beyond_the_cutoff_costs_only_its_reciprocal_share(self) -> None:
        """Past ``r_c`` the pair is not in the real-space list — but is in the k-sum.

        The correction is therefore ``k_e q_i q_j erf(r/σ√2)/r`` and nothing
        more: subtracting the full ``1/r`` here would over-correct by the
        ``erfc`` share the real-space sum never added. The placed list is built
        past this separation (:data:`BEYOND_LIST_CUTOFF`) and so *does* carry
        the pair — as a row the ``r < r_c`` liveness rule kills. That is the
        point: the correction is a function of the **topology**, not of what
        the list happened to hold.
        """
        positions, charges = _ion_pair(BONDED_BEYOND)
        free = _periodic_potentials(charges).calc_energy(
            _periodic_batch(positions, list_cutoff=BEYOND_LIST_CUTOFF)
        )
        excluded = _periodic_potentials(charges).calc_energy(
            _periodic_batch(positions, bonds=((0, 1),), list_cutoff=BEYOND_LIST_CUTOFF)
        )
        reciprocal_share = math.erf(BONDED_BEYOND / (SMEARING * math.sqrt(2.0))) / BONDED_BEYOND
        removed = kcalmol_A * float(charges[0] * charges[1]) * reciprocal_share
        assert math.isclose(
            float(excluded), float(free) - removed, abs_tol=EXCLUSION_TOL, rel_tol=0.0
        )

    def test_a_one_four_pair_keeps_the_fraction_its_scale_declares(self) -> None:
        """``s = 5/6`` removes ``(1 − s) k_e q_i q_j / r``, again including the erf share.

        Atoms 1 and 2 carry no charge, so the only surviving interaction is the
        torsion-end pair 0-3 at 2 Å, whose Class-I electrostatic scale is the
        AMBER ``5/6``.
        """
        charges = torch.tensor([1.0, 0.0, 0.0, -1.0], dtype=torch.float64)
        separation = 2.0
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 0.0, separation],
            ],
            dtype=torch.float64,
        )
        free = _periodic_potentials(charges).calc_energy(_periodic_batch(positions))
        excluded = _periodic_potentials(charges).calc_energy(
            _periodic_batch(
                positions,
                bonds=((0, 1), (1, 2), (2, 3)),
                propers=((0, 1, 2, 3),),
            )
        )
        scale_14 = NonbondedScaling().triple("q")[2]
        removed = (1.0 - scale_14) * kcalmol_A * float(charges[0] * charges[3]) / separation
        assert math.isclose(
            float(excluded), float(free) - removed, abs_tol=EXCLUSION_TOL, rel_tol=0.0
        )

    @pytest.mark.parametrize("calculator", ["ewald", "pme"])
    def test_rocksalt_lattice_energy_matches_the_madelung_constant(self, calculator: str) -> None:
        """``E = −(N/2) M k_e q²/d`` for a neutral ±1 rocksalt cell (hard-coded golden).

        The absolute scale of the periodic sum, and the only test here that a
        wrong prefactor, a dropped self-energy or a mis-signed background term
        cannot pass. ``M = 1.74756459463`` is the literature Madelung constant
        of the B1 lattice (see :data:`MADELUNG_ROCKSALT`); ``k_e`` is the field's
        own ``kcalmol_A``, never a restated float. With ``d = 2.82 Å`` and 64
        ions the closed form is ``−6584.9961818`` kcal/mol.

        Ewald and PME must both land there: they are two evaluations of the same
        sum, and a mesh that silently changes the answer is a bug, not a
        trade-off.
        """
        positions, charges, edge = _rocksalt(ROCKSALT_CELLS, ROCKSALT_SPACING)
        potentials = _periodic_potentials(
            charges,
            calculator=calculator,
            smearing=ROCKSALT_SMEARING,
            real_space_cutoff=ROCKSALT_CUTOFF,
            lr_wavelength=ROCKSALT_LR_WAVELENGTH,
            mesh_spacing=ROCKSALT_MESH_SPACING,
        )
        energy = potentials.calc_energy(
            _periodic_batch(positions, edge=edge, list_cutoff=ROCKSALT_CUTOFF)
        )
        formula_units = charges.shape[0] / 2.0
        expected = -formula_units * MADELUNG_ROCKSALT * kcalmol_A / ROCKSALT_SPACING
        assert math.isclose(float(energy), expected, rel_tol=1e-6, abs_tol=0.0)

    def test_binding_the_cell_freezes_the_reciprocal_shape_not_the_energy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After ``bind_cell`` the k-vectors are regenerated from the live cell.

        The **shape** is frozen — that is what makes the sum traceable at a
        fixed size and its strain derivative continuous — but the vectors
        themselves must follow the lattice, or a barostat step would evaluate
        the reciprocal sum of the previous box. So a bound term contracted by
        5 % must reproduce, to fp64, what a term that had never been bound
        computes on the contracted box, while still summing the *number* of
        k-vectors the original box asked for.

        ``bind_cell`` is called twice on purpose: MD calls it once per barostat
        update whether or not the lattice moved, and the second call must leave
        the state the first produced.
        """
        positions, charges = _ion_pair(BONDED_INSIDE)
        cell = torch.eye(3, dtype=torch.float64) * BOX
        bound = _periodic_potentials(charges)
        bound.bind_cell(cell)
        bound.bind_cell(cell)
        free_full = _periodic_potentials(charges)
        free_shrunk = _periodic_potentials(charges)

        bound_sizes = _record_kspace_sizes(bound, monkeypatch)
        full_sizes = _record_kspace_sizes(free_full, monkeypatch)
        shrunk_sizes = _record_kspace_sizes(free_shrunk, monkeypatch)

        batch_full = _periodic_batch(positions)
        batch_shrunk = _periodic_batch(positions, edge=BOX * SHRINK)
        assert math.isclose(
            float(bound.calc_energy(batch_full)),
            float(free_full.calc_energy(batch_full)),
            abs_tol=FROZEN_SHAPE_TOL,
            rel_tol=0.0,
        )
        assert math.isclose(
            float(bound.calc_energy(batch_shrunk)),
            float(free_shrunk.calc_energy(batch_shrunk)),
            abs_tol=FROZEN_SHAPE_TOL,
            rel_tol=0.0,
        )
        assert shrunk_sizes[0] < full_sizes[0], (
            "the contracted box must need strictly fewer k-vectors, or this test "
            "cannot tell a frozen shape from a live one"
        )
        assert bound_sizes == [full_sizes[0], full_sizes[0]]

    def test_a_cell_beyond_the_bound_reciprocal_capacity_fails_closed(self) -> None:
        """Over-capacity raises on the eager rebind and poisons the in-graph energy.

        Both halves of the same guard, because MD reaches the term through two
        doors: the barostat's ``bind_cell``, which is eager and may raise, and
        the compiled energy, which may not — there the only fail-closed answer
        is ``NaN``. What neither may do is truncate the k-sum and report a
        finite, converged-looking energy for a box the frozen grid cannot
        resolve.
        """
        positions, charges = _ion_pair(BONDED_INSIDE)
        cell = torch.eye(3, dtype=torch.float64) * BOX
        potentials = _periodic_potentials(charges)
        potentials.bind_cell(cell)

        with pytest.raises(RuntimeError, match="capacity_factor"):
            potentials.bind_cell(cell * OVER_CAPACITY)

        energy = potentials.calc_energy(_periodic_batch(positions, edge=BOX * FAR_OVER_CAPACITY))
        assert bool(torch.isnan(energy).all())

    def test_capacity_factor_buys_head_room_for_a_growing_cell(self) -> None:
        """``capacity_factor`` is the declared expansion the frozen grid still serves.

        An NPT run that expands by up to the factor it asked for must not have
        to rebuild anything; one that expands past it must be told. Without the
        head-room the default (``1.0``, sized for the bound box exactly) is the
        NVT contract.
        """
        positions, charges = _ion_pair(BONDED_INSIDE)
        cell = torch.eye(3, dtype=torch.float64) * BOX
        potentials = _periodic_potentials(charges, capacity_factor=CAPACITY_FACTOR)
        potentials.bind_cell(cell)

        potentials.bind_cell(cell * GROWTH_WITHIN_CAPACITY)
        grown = potentials.calc_energy(
            _periodic_batch(positions, edge=BOX * GROWTH_WITHIN_CAPACITY)
        )
        assert bool(torch.isfinite(grown).all())

        with pytest.raises(RuntimeError, match="capacity_factor"):
            potentials.bind_cell(cell * OVER_CAPACITY)

    def test_a_cell_selects_the_mesh_calculator_by_default(self) -> None:
        """``pair:coulomb`` under PBC defaults to PME; ``"ewald"`` stays selectable.

        The explicit ``O(N²)`` k-vector sum is the small-box reference — the
        default a user gets for a real periodic system has to be the mesh one.
        """
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64)
        defaulted = (
            ForceField(name="es")
            .def_pairstyle("coulomb")
            .with_params(atoms={"q": charges})
            .to_potentials()
        )
        assert defaulted.coulomb_periodic is not None
        assert defaulted.coulomb_periodic.calculator_name == "pme"
        assert _periodic_potentials(charges).coulomb_periodic.calculator_name == "ewald"
