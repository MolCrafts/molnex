"""Parity test: ``EwaldMultipoleEnergy.self_contribution`` vs the oracle.

Five constants — ``energy_q, phi_q, energy_u, energy_Q, field_u`` — must
reproduce the analytic forms in :func:`tests._oracles.screened_coulomb.self_corrections`
to ≤1e-6 across a range of σ.
"""

from __future__ import annotations

import pytest

from molpot.potentials import EwaldMultipoleEnergy
from tests._oracles.screened_coulomb import self_corrections as oracle_self_corrections


@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("prefactor", [14.399645, 90.4756])
def test_self_contribution_matches_oracle(sigma: float, prefactor: float) -> None:
    pot = EwaldMultipoleEnergy(sigma=sigma, prefactor=prefactor)
    actual = pot.self_contribution()
    expected = oracle_self_corrections(sigma=sigma, prefactor=prefactor)

    for key in ("energy_q", "phi_q", "energy_u", "energy_Q", "field_u"):
        assert abs(actual[key] - expected[key]) < 1e-6, (
            f"{key}: got {actual[key]}, expected {expected[key]}"
        )
