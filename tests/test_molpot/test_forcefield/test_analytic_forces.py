"""Analytic bonded forces parity against the autograd backend.

``calc_energy_forces`` now evaluates bond / angle / proper / improper forces in
closed form instead of through ``autograd_forces_from_energy``. These tests pin
that the closed forms are bit-compatible with autograd (energy) and within
float64 round-off (forces), and that the force still back-propagates into the
parameters.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from molpot.forcefield import ForceField
from molpot.potentials import AngleHarmonic, BondHarmonic, ImproperPeriodic, ProperTorsionPeriodic
from tests.test_molpot.test_forcefield.conftest import place_edges

DTYPE = torch.float64
_TOL = 1e-8

#: Cutoff (Å) of the ``lj/cut`` and open ``pair:coulomb`` styles declared
#: below, and the build radius of every placed ``edges`` namespace here. Past
#: the largest intramolecular separation of any nonbonded fixture in this file
#: (0.95 Å for the four-atom chain, 1.9 Å for the triangle), so the list holds
#: every pair the deleted molpot enumerator produced (``composable-a0-03b``).
#: Bonded-only fixtures declare no pair style and carry no ``edges``.
NONBONDED_CUTOFF = 10.0


def _autograd_forces(pot: torch.nn.Module, pos: torch.Tensor, **kwargs) -> torch.Tensor:
    leaf = pos.clone().requires_grad_(True)
    energy = pot(pos=leaf, **kwargs)
    (grad,) = torch.autograd.grad(energy, leaf)
    return -grad


class TestBondedEnergyForces:
    def test_bond_matches_autograd(self) -> None:
        pot = BondHarmonic(
            k=torch.tensor([120.0], dtype=DTYPE), r0=torch.tensor([1.09], dtype=DTYPE)
        )
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.3, 0.4]], dtype=DTYPE)
        index = torch.tensor([[0], [1]], dtype=torch.long)
        types = torch.tensor([0], dtype=torch.long)

        energy, forces = pot.energy_forces(pos=pos, bond_index=index, bond_types=types)
        expected_energy = pot(pos=pos, bond_index=index, bond_types=types)
        expected_forces = _autograd_forces(pot, pos, bond_index=index, bond_types=types)

        assert torch.allclose(energy, expected_energy, atol=_TOL, rtol=0.0)
        assert torch.allclose(forces, expected_forces, atol=_TOL, rtol=0.0)

    def test_angle_matches_autograd(self) -> None:
        pot = AngleHarmonic(
            k=torch.tensor([80.0], dtype=DTYPE), theta0=torch.tensor([1.91], dtype=DTYPE)
        )
        pos = torch.tensor([[1.0, 0.2, 0.0], [0.0, 0.0, 0.0], [0.1, 1.0, 0.3]], dtype=DTYPE)
        index = torch.tensor([[0], [1], [2]], dtype=torch.long)
        types = torch.tensor([0], dtype=torch.long)

        energy, forces = pot.energy_forces(pos=pos, angle_index=index, angle_types=types)
        expected_energy = pot(pos=pos, angle_index=index, angle_types=types)
        expected_forces = _autograd_forces(pot, pos, angle_index=index, angle_types=types)

        assert torch.allclose(energy, expected_energy, atol=_TOL, rtol=0.0)
        assert torch.allclose(forces, expected_forces, atol=_TOL, rtol=0.0)

    def test_angle_near_linear_float32_stays_finite(self) -> None:
        """float32 near-linear angle: analytic force must not NaN.

        The squared sine ``|u × v|²`` is evaluated as the cross-product norm,
        matching the energy path. The algebraic identity ``uu*vv - uv*uv``
        cancels catastrophically here (``-5.96e-8`` in float32) and its
        ``sqrt`` is NaN even after the ``1e-12`` softening.
        """
        pot = AngleHarmonic(
            k=torch.tensor([80.0], dtype=torch.float32),
            theta0=torch.tensor([1.91], dtype=torch.float32),
        )
        u = torch.tensor([[0.57310677, 0.54094744, -0.39190584]], dtype=torch.float32)
        v = torch.tensor([[-0.57321101, -0.54081559, 0.39198062]], dtype=torch.float32)
        pos = torch.cat([u, torch.zeros_like(u), v], dim=0)
        index = torch.tensor([[0], [1], [2]], dtype=torch.long)
        types = torch.tensor([0], dtype=torch.long)

        energy, forces = pot.energy_forces(pos=pos, angle_index=index, angle_types=types)
        assert torch.isfinite(energy).all()
        assert torch.isfinite(forces).all()

    def test_proper_matches_autograd(self) -> None:
        pot = ProperTorsionPeriodic(
            k=torch.tensor([[1.5, 0.8]], dtype=DTYPE),
            periodicity=torch.tensor([1, 3], dtype=torch.long),
            phase=torch.tensor([[0.3, 0.0]], dtype=DTYPE),
            idivf=torch.tensor([1.0], dtype=DTYPE),
        )
        pos = torch.tensor(
            [[0.7, 0.1, 0.0], [0.0, 0.0, 0.0], [0.2, 0.8, 0.1], [1.0, 0.9, 0.4]], dtype=DTYPE
        )
        index = torch.tensor([[0], [1], [2], [3]], dtype=torch.long)
        types = torch.tensor([0], dtype=torch.long)

        energy, forces = pot.energy_forces(pos=pos, proper_index=index, proper_types=types)
        expected_energy = pot(pos=pos, proper_index=index, proper_types=types)
        expected_forces = _autograd_forces(pot, pos, proper_index=index, proper_types=types)

        assert torch.allclose(energy, expected_energy, atol=_TOL, rtol=0.0)
        assert torch.allclose(forces, expected_forces, atol=_TOL, rtol=0.0)

    def test_improper_matches_autograd(self) -> None:
        pot = ImproperPeriodic(
            k=torch.tensor([[1.2, 0.6]], dtype=DTYPE),
            periodicity=torch.tensor([1, 2], dtype=torch.long),
            phase=torch.tensor([[0.0, 0.0]], dtype=DTYPE),
            idivf=torch.tensor([1.0], dtype=DTYPE),
        )
        pos = torch.tensor(
            [[0.9, 0.1, 0.0], [0.0, 0.0, 0.0], [0.2, 0.8, 0.1], [1.0, 0.9, 0.4]], dtype=DTYPE
        )
        index = torch.tensor([[1], [0], [2], [3]], dtype=torch.long)  # center first
        types = torch.tensor([0], dtype=torch.long)

        energy, forces = pot.energy_forces(pos=pos, improper_index=index, improper_types=types)
        expected_energy = pot(pos=pos, improper_index=index, improper_types=types)
        expected_forces = _autograd_forces(pot, pos, improper_index=index, improper_types=types)

        assert torch.allclose(energy, expected_energy, atol=_TOL, rtol=0.0)
        assert torch.allclose(forces, expected_forces, atol=_TOL, rtol=0.0)

    def test_force_loss_reaches_parameters(self) -> None:
        k = torch.tensor([120.0], dtype=DTYPE, requires_grad=True)
        r0 = torch.tensor([1.09], dtype=DTYPE, requires_grad=True)
        pot = BondHarmonic(k=k, r0=r0)
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.3, 0.4]], dtype=DTYPE)
        _, forces = pot.energy_forces(
            pos=pos,
            bond_index=torch.tensor([[0], [1]], dtype=torch.long),
            bond_types=torch.tensor([0], dtype=torch.long),
        )
        torch.mean(forces**2).backward()
        assert k.grad is not None
        assert r0.grad is not None


def _bonded_batch() -> TensorDict:
    """A four-atom chain carrying every bonded term (one molecule)."""
    pos = torch.tensor(
        [[0.7, 0.1, 0.0], [0.0, 0.0, 0.0], [0.2, 0.8, 0.1], [1.0, 0.9, 0.4]], dtype=DTYPE
    )
    return TensorDict(
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
            "angles": TensorDict(
                {
                    "atomi": torch.tensor([0, 1], dtype=torch.long),
                    "atomj": torch.tensor([1, 2], dtype=torch.long),
                    "atomk": torch.tensor([2, 3], dtype=torch.long),
                },
                batch_size=[2],
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
            "impropers": TensorDict(
                {
                    "atomi": torch.tensor([1], dtype=torch.long),  # center
                    "atomj": torch.tensor([0], dtype=torch.long),
                    "atomk": torch.tensor([2], dtype=torch.long),
                    "atoml": torch.tensor([3], dtype=torch.long),
                },
                batch_size=[1],
            ),
        },
        batch_size=[],
    )


def _bonded_potentials() -> ForceField:
    return (
        ForceField(name="test", units="real")
        .def_bondstyle("harmonic")
        .def_anglestyle("harmonic")
        .def_dihedralstyle("periodic")
        .def_improperstyle("periodic")
        .with_params(
            bonds={
                "k": torch.full((3,), 120.0, dtype=DTYPE),
                "r0": torch.full((3,), 1.09, dtype=DTYPE),
            },
            angles={
                "k": torch.full((2,), 80.0, dtype=DTYPE),
                "theta0": torch.full((2,), 1.91, dtype=DTYPE),
            },
            propers={
                "k": torch.tensor([[1.5, 0.8]], dtype=DTYPE),
                "periodicity": torch.tensor([1, 3], dtype=torch.long),
                "phase": torch.tensor([[0.3, 0.0]], dtype=DTYPE),
                "idivf": torch.tensor([1.0], dtype=DTYPE),
            },
            impropers={
                "k": torch.tensor([[1.2, 0.6]], dtype=DTYPE),
                "periodicity": torch.tensor([1, 2], dtype=torch.long),
                "phase": torch.tensor([[0.0, 0.0]], dtype=DTYPE),
                "idivf": torch.tensor([1.0], dtype=DTYPE),
            },
        )
    )


def _full_batch() -> TensorDict:
    """Four-atom chain plus LJ/Coulomb atom params (nonbonded frozen), with ``edges``."""
    batch = _bonded_batch()
    batch["atoms", "Z"] = torch.tensor([6, 1, 1, 1], dtype=torch.long)
    return place_edges(batch, cutoff=NONBONDED_CUTOFF)


def _full_potentials(*, charge: float = 0.0) -> ForceField:
    """Bonded (trainable) + nonbonded (frozen) ForceField."""
    return (
        ForceField(name="test", units="real")
        .def_bondstyle("harmonic")
        .def_anglestyle("harmonic")
        .def_dihedralstyle("periodic")
        .def_improperstyle("periodic")
        .def_pairstyle("lj/cut", cutoff=NONBONDED_CUTOFF)
        .def_pairstyle("coulomb", cutoff=NONBONDED_CUTOFF)
        .with_params(
            bonds={
                "k": torch.full((3,), 120.0, dtype=DTYPE, requires_grad=True),
                "r0": torch.full((3,), 1.09, dtype=DTYPE, requires_grad=True),
            },
            angles={
                "k": torch.full((2,), 80.0, dtype=DTYPE, requires_grad=True),
                "theta0": torch.full((2,), 1.91, dtype=DTYPE, requires_grad=True),
            },
            propers={
                "k": torch.tensor([[1.5, 0.8]], dtype=DTYPE, requires_grad=True),
                "periodicity": torch.tensor([1, 3], dtype=torch.long),
                "phase": torch.tensor([[0.3, 0.0]], dtype=DTYPE, requires_grad=True),
                "idivf": torch.tensor([1.0], dtype=DTYPE, requires_grad=True),
            },
            impropers={
                "k": torch.tensor([[1.2, 0.6]], dtype=DTYPE, requires_grad=True),
                "periodicity": torch.tensor([1, 2], dtype=torch.long),
                "phase": torch.tensor([[0.0, 0.0]], dtype=DTYPE, requires_grad=True),
                "idivf": torch.tensor([1.0], dtype=DTYPE, requires_grad=True),
            },
            atoms={
                "epsilon": torch.full((4,), 0.1, dtype=DTYPE),
                "sigma": torch.full((4,), 3.0, dtype=DTYPE),
                "q": torch.full((4,), charge, dtype=DTYPE),
            },
        )
    )


class TestPotentialsBondedForces:
    def test_calc_energy_forces_matches_calc_forces(self) -> None:
        ff = _bonded_potentials()
        pots = ff.to_potentials()
        batch = _bonded_batch()

        energy, forces = pots.calc_energy_forces(batch)
        assert torch.allclose(energy, pots.calc_energy(batch), atol=_TOL, rtol=0.0)
        assert torch.allclose(forces, pots.calc_forces(batch).detach(), atol=_TOL, rtol=0.0)

    def test_bonded_terms_only(self) -> None:
        """A bonded-only ForceField assembles no LJ / Coulomb terms."""
        ff = _bonded_potentials()
        pots = ff.to_potentials()
        terms = pots.term_energies(_bonded_batch())
        assert set(terms) == {"bonds", "angles", "propers", "impropers"}

    def test_batched_matches_calc_forces(self) -> None:
        """Two-molecule batch: per-graph energy reduction + global force scatter."""
        single = _bonded_batch()
        pos = torch.cat([single["atoms", "pos"], single["atoms", "pos"] + 5.0], dim=0)
        batch = TensorDict(
            {
                "atoms": TensorDict(
                    {"pos": pos, "batch": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])},
                    batch_size=[8],
                ),
                "bonds": TensorDict(
                    {
                        "atomi": torch.tensor([0, 1, 2, 4, 5, 6]),
                        "atomj": torch.tensor([1, 2, 3, 5, 6, 7]),
                    },
                    batch_size=[6],
                ),
                "angles": TensorDict(
                    {
                        "atomi": torch.tensor([0, 1, 4, 5]),
                        "atomj": torch.tensor([1, 2, 5, 6]),
                        "atomk": torch.tensor([2, 3, 6, 7]),
                    },
                    batch_size=[4],
                ),
                "propers": TensorDict(
                    {
                        "atomi": torch.tensor([0, 4]),
                        "atomj": torch.tensor([1, 5]),
                        "atomk": torch.tensor([2, 6]),
                        "atoml": torch.tensor([3, 7]),
                    },
                    batch_size=[2],
                ),
                "impropers": TensorDict(
                    {
                        "atomi": torch.tensor([1, 5]),
                        "atomj": torch.tensor([0, 4]),
                        "atomk": torch.tensor([2, 6]),
                        "atoml": torch.tensor([3, 7]),
                    },
                    batch_size=[2],
                ),
            },
            batch_size=[],
        )
        ff = (
            ForceField(name="test", units="real")
            .def_bondstyle("harmonic")
            .def_anglestyle("harmonic")
            .def_dihedralstyle("periodic")
            .def_improperstyle("periodic")
            .with_params(
                bonds={
                    "k": torch.full((6,), 120.0, dtype=DTYPE),
                    "r0": torch.full((6,), 1.09, dtype=DTYPE),
                },
                angles={
                    "k": torch.full((4,), 80.0, dtype=DTYPE),
                    "theta0": torch.full((4,), 1.91, dtype=DTYPE),
                },
                propers={
                    "k": torch.tensor([[1.5, 0.8], [1.5, 0.8]], dtype=DTYPE),
                    "periodicity": torch.tensor([1, 3], dtype=torch.long),
                    "phase": torch.tensor([[0.3, 0.0], [0.3, 0.0]], dtype=DTYPE),
                    "idivf": torch.tensor([1.0, 1.0], dtype=DTYPE),
                },
                impropers={
                    "k": torch.tensor([[1.2, 0.6], [1.2, 0.6]], dtype=DTYPE),
                    "periodicity": torch.tensor([1, 2], dtype=torch.long),
                    "phase": torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=DTYPE),
                    "idivf": torch.tensor([1.0, 1.0], dtype=DTYPE),
                },
            )
        )
        pots = ff.to_potentials()
        energy, forces = pots.calc_energy_forces(batch)
        assert energy.shape == (2,)
        assert torch.allclose(energy, pots.calc_energy(batch), atol=_TOL, rtol=0.0)
        assert torch.allclose(forces, pots.calc_forces(batch).detach(), atol=_TOL, rtol=0.0)

    def test_training_style_backward_full_forcefield(self) -> None:
        """``loss = E + mean(F²)`` back-propagates once with frozen nonbonded heads.

        Regresses the double-backward failure: computing nonbonded forces with
        ``create_graph=False`` freed the nonbonded energy graph, so a subsequent
        ``loss.backward()`` raised "backward through the graph a second time".
        """
        pots = _full_potentials().to_potentials()
        energy, forces = pots.calc_energy_forces(_full_batch())
        loss = energy.sum() + forces.square().mean()
        loss.backward()
        assert pots.bond.k.grad is not None
        assert pots.bond.r0.grad is not None
        assert pots.angle.k.grad is not None
        assert pots.angle.theta0.grad is not None
        assert pots._charges().grad is None  # frozen

    def test_nonbonded_scaling_matches_calc_forces(self) -> None:
        """Analytic LJ/Coulomb forces honour 1–4 scaling on a real topology.

        The four-atom chain only contributes nonbonded energy through its
        1–4 pair (scales LJ 0.5, Coulomb 5/6), so this pins the scaled
        analytic path against the autograd backend.
        """
        pots = _full_potentials(charge=0.5).to_potentials()
        batch = _full_batch()
        energy, forces = pots.calc_energy_forces(batch)
        assert torch.allclose(energy, pots.calc_energy(batch), atol=_TOL, rtol=0.0)
        assert torch.allclose(forces, pots.calc_forces(batch).detach(), atol=_TOL, rtol=0.0)


def _three_atom_positions() -> torch.Tensor:
    """One three-atom triangle with all three pair distances inside the cutoff."""
    return torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.2, 0.1], [0.3, 1.4, 0.2]], dtype=DTYPE)


def _nonbonded_batch() -> TensorDict:
    """Three-atom molecule (one graph) with no bonded topology, carrying ``edges``."""
    batch = TensorDict(
        {
            "atoms": TensorDict(
                {
                    "pos": _three_atom_positions(),
                    "batch": torch.zeros(3, dtype=torch.long),
                },
                batch_size=[3],
            ),
        },
        batch_size=[],
    )
    return place_edges(batch, cutoff=NONBONDED_CUTOFF)


def _nonbonded_pair_batch() -> TensorDict:
    """Two separated three-atom molecules packed into one batch (B = 2).

    The list is built per graph and rebased, so it holds the two molecules'
    intramolecular pairs and never bridges them — the same set the deleted
    ``_open`` enumerator produced, and the reason the 8.7 Å intermolecular
    separations never appear even though they are inside
    :data:`NONBONDED_CUTOFF`.
    """
    single = _three_atom_positions()
    pos = torch.cat([single, single + 5.0], dim=0)
    batch = TensorDict(
        {
            "atoms": TensorDict(
                {
                    "pos": pos,
                    "batch": torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long),
                },
                batch_size=[6],
            ),
        },
        batch_size=[],
    )
    return place_edges(batch, cutoff=NONBONDED_CUTOFF)


def _nonbonded_potentials(
    *, lj: bool = True, coulomb: bool = True, batched: bool = False
) -> ForceField:
    """LJ and/or open-Coulomb ForceField over three-atom molecules (nonbonded only)."""
    epsilon = torch.tensor([0.10, 0.05, 0.08], dtype=DTYPE)
    sigma = torch.tensor([3.0, 2.5, 2.8], dtype=DTYPE)
    charge = torch.tensor([0.5, -0.25, -0.25], dtype=DTYPE)
    if batched:
        epsilon = torch.cat([epsilon, epsilon])
        sigma = torch.cat([sigma, sigma])
        charge = torch.cat([charge, charge])
    ff = ForceField(name="test", units="real")
    if lj:
        ff = ff.def_pairstyle("lj/cut", cutoff=NONBONDED_CUTOFF)
    if coulomb:
        ff = ff.def_pairstyle("coulomb", cutoff=NONBONDED_CUTOFF)
    params: dict[str, torch.Tensor] = {}
    if lj:
        params["epsilon"] = epsilon
        params["sigma"] = sigma
    if coulomb:
        params["q"] = charge
    return ff.with_params(atoms=params)


class TestNonbondedEnergyForces:
    def test_lj_matches_calc_forces(self) -> None:
        pots = _nonbonded_potentials(coulomb=False).to_potentials()
        batch = _nonbonded_batch()
        energy, forces = pots.calc_energy_forces(batch)
        assert torch.allclose(energy, pots.calc_energy(batch), atol=_TOL, rtol=0.0)
        assert torch.allclose(forces, pots.calc_forces(batch).detach(), atol=_TOL, rtol=0.0)

    def test_coulomb_matches_calc_forces(self) -> None:
        pots = _nonbonded_potentials(lj=False).to_potentials()
        batch = _nonbonded_batch()
        energy, forces = pots.calc_energy_forces(batch)
        assert torch.allclose(energy, pots.calc_energy(batch), atol=_TOL, rtol=0.0)
        assert torch.allclose(forces, pots.calc_forces(batch).detach(), atol=_TOL, rtol=0.0)

    def test_lj_coulomb_batched_matches_calc_forces(self) -> None:
        pots = _nonbonded_potentials(batched=True).to_potentials()
        batch = _nonbonded_pair_batch()
        energy, forces = pots.calc_energy_forces(batch)
        assert energy.shape == (2,)
        assert torch.allclose(energy, pots.calc_energy(batch), atol=_TOL, rtol=0.0)
        assert torch.allclose(forces, pots.calc_forces(batch).detach(), atol=_TOL, rtol=0.0)
