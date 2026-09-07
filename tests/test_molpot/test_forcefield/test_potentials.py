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

import re
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from molix.schema import ENERGY_KEY, CellBound
from molpot.forcefield import ForceField, Potentials
from molpot.forcefield.schema import PotentialMeta
from molpot.potentials.elec.prefactors import kcalmol_A

DTYPE = torch.float64

# Analytic two-atom harmonic bond: E = ½k(r−r0)², k = 100 kcal mol⁻¹ Å⁻²,
# r0 = 1.0 Å, r = 1.5 Å along x̂ → E = 12.5 kcal/mol, |F| = k(r−r0) = 50.
BOND_K = 100.0
BOND_R0 = 1.0
BOND_R = 1.5
GOLDEN_ENERGY = 12.5
GOLDEN_FORCE_X = 50.0

SOURCE = Path(__file__).resolve().parents[3] / "src/molpot/forcefield/potentials.py"
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
    """Two unbonded atoms ``r`` Å apart along x̂ — one nonbonded pair, nothing else."""
    pos = torch.tensor([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=DTYPE)
    return TensorDict(
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
    """Three atoms, one bond — bonded, LJ and Coulomb all contribute."""
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [BOND_R, 0.0, 0.0], [0.0, 3.0, 0.0]],
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


#: OPLS-AA all-atom alkane charges (e), well depths (kcal/mol) and sizes (Å) for
#: the butane fixture in ``conftest.py``: CH3 carbon −0.18, CH2 carbon −0.12,
#: every hydrogen +0.06 — a neutral molecule.
BUTANE_Q = (-0.18, -0.12, -0.12, -0.18) + (0.06,) * 10
BUTANE_EPSILON = (0.066,) * 4 + (0.030,) * 10
BUTANE_SIGMA = (3.50,) * 4 + (2.50,) * 10
BUTANE_CUTOFF = 10.0


def _charged_batch(*, total_charge: float | None) -> TensorDict:
    """Two atoms carrying +0.5 e each; ``graphs.total_charge`` optional."""
    graphs: dict[str, torch.Tensor] = {"num_atoms": torch.tensor([2], dtype=torch.long)}
    if total_charge is not None:
        graphs["total_charge"] = torch.tensor([total_charge], dtype=DTYPE)
    return TensorDict(
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
            .def_pairstyle("lj/cut", cutoff=3.0)
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
        potentials = (
            ForceField(name="anion")
            .def_pairstyle("coulomb")
            .with_params(atoms={"q": torch.tensor([0.5, 0.5], dtype=DTYPE)})
            .to_potentials()
        )
        with pytest.raises(ValueError, match="total_charge"):
            potentials.bind_topology(_charged_batch(total_charge=0.0))

        potentials.bind_topology(_charged_batch(total_charge=1.0))
        potentials.bind_topology(_charged_batch(total_charge=None))

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
    from tests.test_molzoo.test_mace.conftest import (
        ATOMIC_ENERGIES,
        ATOMIC_NUMBERS,
        TINY_POLAR_KWARGS,
        polar_kwargs,
    )

    from molzoo.mace.polar import MacePolar
    from molzoo.mace.spec import MACEPolarSpec

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
        loudly rather than report success having compiled nothing. When the
        mlip does satisfy them, exactly one call is forwarded, with the
        caller's own storage. The import guards keep the package direction
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

        classical = _bond_potentials()
        classical.bind_cell(torch.eye(3, dtype=DTYPE))
        with pytest.raises(RuntimeError, match="compile_core"):
            classical.compile_core(lambda core: core)

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
        from tests.test_molzoo.test_mace.conftest import CLUSTER_POS

        from molix.compile import Compiler

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
        strays = [
            site
            for site in step.break_sites
            if not re.fullmatch(rf".*{re.escape(FORCE_SEAM)}:\d+", site)
        ]
        assert strays == [], (
            f"the whole step may break only at the named autograd seam in {FORCE_SEAM} "
            f"(owner: molpot derivation, retained by Q3); stray break sites: {strays}"
        )
