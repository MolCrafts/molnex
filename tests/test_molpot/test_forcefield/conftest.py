"""Shared fixtures for the ForceField unit tests."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict


@pytest.fixture
def fake_pol_styles(monkeypatch: pytest.MonkeyPatch) -> tuple[str, str]:
    """Two induced-polarization pair styles, registered for the test only.

    ``composable-a0-01`` registers no ``Contribution.POL`` style in tree — the
    first one (``pair:induced/thole``) arrives with ``composable-a1``. The
    composition rules that reject a second polarization term, or a
    polarization term without a permanent electrostatic partner, therefore
    have to be exercised against a stand-in. Both stand-ins are Thole-shaped:
    one ``POL`` term, an ``INDUCED`` exclusion class, a hard real-space cutoff
    and a variational response.

    Returns:
        The two registered style keys, in registration order.
    """
    from molpot.contract import (
        Contribution,
        CutoffKind,
        Derivative,
        ExclusionClass,
        InteractionContract,
        TermSpec,
    )
    from molpot.forcefield.schema import ParamSpec, PotentialMeta, StyleSchema

    keys: list[str] = []
    for name in ("fake/pol", "fake/pol2"):
        schema = StyleSchema(
            category="pair",
            name=name,
            index_arity=2,
            energy_formula="-0.5 * sum_i mu_i . E_i",
            params=(ParamSpec("alpha", "[n_atoms]", "Å³", "isotropic polarizability"),),
            contract=InteractionContract(
                inputs=("atoms.pos", "atoms.q", "atoms.alpha"),
                cutoff=CutoffKind.HARD,
                exclusion=ExclusionClass.INDUCED,
                derivatives=frozenset({Derivative.FORCES, Derivative.VIRIAL}),
                terms=(TermSpec("induced", Contribution.POL),),
                response="variational",
            ),
        )
        monkeypatch.setitem(PotentialMeta._REGISTRY, schema.key, schema)
        keys.append(schema.key)
    return keys[0], keys[1]


# ----------------------------------------------------------------------------
# OPLS butane (n-C4H10) — the shared exclusion-table fixture
# ----------------------------------------------------------------------------

#: Atom order of the butane fixture: ``C0 C1 C2 C3`` then the hydrogens
#: ``H4 H5 H6`` on ``C0``, ``H7 H8`` on ``C1``, ``H9 H10`` on ``C2`` and
#: ``H11 H12 H13`` on ``C3``.
BUTANE_N_ATOMS = 14

#: The 13 covalent bonds of n-butane, ``(atomi, atomj)`` with ``i < j``.
#: Angles (24) and proper torsions (27) are *derived* from this list by
#: :func:`butane_angles` / :func:`butane_propers`, so the topology has a single
#: source of truth.
BUTANE_BONDS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (0, 5),
    (0, 6),
    (1, 7),
    (1, 8),
    (2, 9),
    (2, 10),
    (3, 11),
    (3, 12),
    (3, 13),
)

#: Per-pair nonbonded scale factors for the 91 undirected pairs of the butane
#: fixture, in ``torch.triu_indices(14, 14, offset=1)`` order, under
#: ``OPLSAAForceField().nonbonded_scaling`` (1-2 = 1-3 = 0, 1-4 = 0.5 for both
#: ``q`` and ``lj``, so one vector serves both kinds).
#:
#: **Hard-coded golden.** Captured from the pre-step
#: ``Potentials._pair_scales`` at commit ``29f7f004``
#: (``src/molpot/forcefield/potentials.py`` byte-identical at ``97c36852``).
#: That method was deleted in ``composable-a0-02``, so the capture is not
#: re-runnable and these values are frozen; full provenance in the module
#: docstring of ``tests/test_molpot/test_forcefield/test_exclusions.py``.
#:
#: 37 zeros (13 bonds + 24 angle ends), 27 halves (27 torsion ends), 27 ones.
# fmt: off
BUTANE_OPLS_SCALES: tuple[float, ...] = (
    0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0,  # (0, j), j = 1..13
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5,  # (1, j), j = 2..13
    0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # (2, j), j = 3..13
    1.0, 1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0,  # (3, j), j = 4..13
    0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0,  # (4, j), j = 5..13
    0.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0,  # (5, j), j = 6..13
    0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0,  # (6, j), j = 7..13
    0.0, 0.5, 0.5, 1.0, 1.0, 1.0,  # (7, j), j = 8..13
    0.5, 0.5, 1.0, 1.0, 1.0,  # (8, j), j = 9..13
    0.0, 0.5, 0.5, 0.5,  # (9, j), j = 10..13
    0.5, 0.5, 0.5,  # (10, j), j = 11..13
    0.0, 0.0,  # (11, j), j = 12..13
    0.0,  # (12, j), j = 13..13
)
# fmt: on

#: A fixed all-atom butane geometry (Å). Every one of the 91 pairs lies between
#: 1.09 Å and 6.04 Å, so a 10 Å neighbour list keeps all of them and the
#: exclusion vector above lines up with the full half list.
BUTANE_POSITIONS: tuple[tuple[float, float, float], ...] = (
    (0.000000, 0.000000, 0.000000),
    (1.268427, 0.855565, 0.000000),
    (2.536855, 0.000000, 0.000000),
    (3.805282, 0.855565, 0.000000),
    (-0.974926, -0.487463, 0.000000),
    (-0.291315, -0.582630, 0.873944),
    (-0.291315, -0.582630, -0.873944),
    (1.268427, 1.460188, 0.906935),
    (1.268427, 1.460188, -0.906935),
    (2.536855, -0.604623, 0.906935),
    (2.536855, -0.604623, -0.906935),
    (4.780208, 1.343028, 0.000000),
    (4.096597, 1.438195, 0.873944),
    (4.096597, 1.438195, -0.873944),
)


def butane_neighbors() -> dict[int, list[int]]:
    """Adjacency of :data:`BUTANE_BONDS`, each neighbour list sorted."""
    adjacency: dict[int, list[int]] = {i: [] for i in range(BUTANE_N_ATOMS)}
    for i, j in BUTANE_BONDS:
        adjacency[i].append(j)
        adjacency[j].append(i)
    return {atom: sorted(neighbors) for atom, neighbors in adjacency.items()}


def butane_angles() -> list[tuple[int, int, int]]:
    """The 24 ``(i, j, k)`` angles of butane — every neighbour pair of a centre."""
    adjacency = butane_neighbors()
    triples: list[tuple[int, int, int]] = []
    for centre in range(BUTANE_N_ATOMS):
        neighbors = adjacency[centre]
        for a in range(len(neighbors)):
            for b in range(a + 1, len(neighbors)):
                triples.append((neighbors[a], centre, neighbors[b]))
    return triples


def butane_propers() -> list[tuple[int, int, int, int]]:
    """The 27 ``(i, j, k, l)`` proper torsions of butane, ``j-k`` the central bond."""
    adjacency = butane_neighbors()
    quads: list[tuple[int, int, int, int]] = []
    for j, k in BUTANE_BONDS:
        for i in adjacency[j]:
            if i == k:
                continue
            for end in adjacency[k]:
                if end == j:
                    continue
                quads.append((i, j, k, end))
    return quads


@pytest.fixture
def butane_topology() -> dict[str, torch.Tensor]:
    """Column-form butane topology plus its complete undirected pair list.

    Returns:
        ``bond_index`` ``(2, 13)``, ``angle_index`` ``(3, 24)``,
        ``proper_index`` ``(4, 27)`` and ``edge_index`` ``(91, 2)`` — the latter
        every unordered pair of the 14 atoms in
        ``torch.triu_indices(14, 14, offset=1)`` order, which is the order
        :data:`BUTANE_OPLS_SCALES` was captured in.
    """
    triu = torch.triu_indices(BUTANE_N_ATOMS, BUTANE_N_ATOMS, offset=1)
    return {
        "bond_index": torch.tensor(BUTANE_BONDS, dtype=torch.long).t().contiguous(),
        "angle_index": torch.tensor(butane_angles(), dtype=torch.long).t().contiguous(),
        "proper_index": torch.tensor(butane_propers(), dtype=torch.long).t().contiguous(),
        "edge_index": triu.t().contiguous(),
    }


@pytest.fixture
def butane_scales() -> torch.Tensor:
    """:data:`BUTANE_OPLS_SCALES` as an fp64 ``(91,)`` tensor (hard-coded golden)."""
    return torch.tensor(BUTANE_OPLS_SCALES, dtype=torch.float64)


@pytest.fixture
def butane_batch() -> TensorDict:
    """Post-collate butane batch: ``atoms`` + ``bonds`` / ``angles`` / ``propers``."""
    angle_index = torch.tensor(butane_angles(), dtype=torch.long).t().contiguous()
    proper_index = torch.tensor(butane_propers(), dtype=torch.long).t().contiguous()
    bond_index = torch.tensor(BUTANE_BONDS, dtype=torch.long).t().contiguous()
    return TensorDict(
        {
            "atoms": TensorDict(
                {
                    "pos": torch.tensor(BUTANE_POSITIONS, dtype=torch.float64),
                    "Z": torch.tensor([6, 6, 6, 6] + [1] * 10, dtype=torch.long),
                    "batch": torch.zeros(BUTANE_N_ATOMS, dtype=torch.long),
                },
                batch_size=[BUTANE_N_ATOMS],
            ),
            "bonds": TensorDict(
                {"atomi": bond_index[0], "atomj": bond_index[1]},
                batch_size=[bond_index.shape[1]],
            ),
            "angles": TensorDict(
                {
                    "atomi": angle_index[0],
                    "atomj": angle_index[1],
                    "atomk": angle_index[2],
                },
                batch_size=[angle_index.shape[1]],
            ),
            "propers": TensorDict(
                {
                    "atomi": proper_index[0],
                    "atomj": proper_index[1],
                    "atomk": proper_index[2],
                    "atoml": proper_index[3],
                },
                batch_size=[proper_index.shape[1]],
            ),
        },
        batch_size=[],
    )
