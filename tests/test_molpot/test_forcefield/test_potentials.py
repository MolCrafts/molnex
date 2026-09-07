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
