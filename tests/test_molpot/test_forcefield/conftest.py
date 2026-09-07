"""Shared fixtures for the ForceField unit tests."""

from __future__ import annotations

import pytest


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
