"""Shared fixtures for the ForceField unit tests.

The butane geometry and topology themselves live in ``tests/conftest.py``:
the two-molecule periodic box built on them is the fixture both the molpot
symmetry gates and the molix NVE drift gate run, so it belongs at the
conftest common to both packages rather than in this leaf one. What stays
here is what only these tests use — the exclusion-scale golden and the
column-form topology / batch fixtures built around it.
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from tests.conftest import (
    BUTANE_BONDS,
    BUTANE_N_ATOMS,
    BUTANE_POSITIONS,
    butane_angles,
    butane_propers,
    place_edges,
)


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
# Butane exclusion table — the hard-coded 1-2 / 1-3 / 1-4 scale golden
# ----------------------------------------------------------------------------

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

#: Build radius (Å) of the butane fixture's placed neighbour list. Past the
#: largest of the 91 separations (6.04 Å), so the list holds **every** pair —
#: which is what ``BUTANE_OPLS_SCALES`` was captured against, and what the
#: cutoff-less open ``pair:coulomb`` of ``_butane_forcefield`` asks for. It is
#: also the ``lj/cut`` cutoff that fixture declares, so LJ and Coulomb read the
#: same rows.
BUTANE_LIST_CUTOFF = 10.0


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
    """Post-collate butane batch: ``atoms`` + ``bonds`` / ``angles`` / ``propers`` + ``edges``.

    The ``edges`` namespace is built by :func:`place_edges` at
    :data:`BUTANE_LIST_CUTOFF`, which keeps all 91 pairs — the same set the
    deleted molpot enumerator produced, so ``BUTANE_OPLS_SCALES`` still lines
    up row for row (as a set; the placed list is not in ``triu`` order and the
    energies the golden pins are sums).
    """
    angle_index = torch.tensor(butane_angles(), dtype=torch.long).t().contiguous()
    proper_index = torch.tensor(butane_propers(), dtype=torch.long).t().contiguous()
    bond_index = torch.tensor(BUTANE_BONDS, dtype=torch.long).t().contiguous()
    batch = TensorDict(
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
    return place_edges(batch, cutoff=BUTANE_LIST_CUTOFF)
