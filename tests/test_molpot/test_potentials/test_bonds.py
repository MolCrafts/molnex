"""BondHarmonic canonical bond_index [2, N] contract + edge≠bond enforcement.

See spec graph-connectivity-alignment-03-bonds: covalent connectivity is a
canonical COO ``bond_index`` ``[2, num_bonds]`` (distinct from the geometric
``edge_index`` ``[E, 2]``). A geometric cutoff edge must never be silently
consumed as a bond list — the potential raises instead.
"""

import math

import pytest
import torch

from molpot.potentials.bonds import BondHarmonic


def _potential():
    # one bond type: k=2.0, r0=1.0
    return BondHarmonic(k=torch.tensor([2.0]), r0=torch.tensor([1.0]))


def test_canonical_bond_index_energy_matches_analytic():
    pot = _potential()
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    bond_index = torch.tensor([[0], [1]], dtype=torch.long)  # [2, 1] COO
    bond_types = torch.tensor([0], dtype=torch.long)
    e = pot(pos=pos, bond_index=bond_index, bond_types=bond_types)
    # 0.5 * 2.0 * (1.5 - 1.0)^2 = 0.25
    assert math.isclose(float(e), 0.25, rel_tol=1e-6)


def test_empty_bonds_returns_zero():
    pot = _potential()
    pos = torch.tensor([[0.0, 0.0, 0.0]])
    bond_index = torch.zeros(2, 0, dtype=torch.long)  # [2, 0]
    bond_types = torch.zeros(0, dtype=torch.long)
    e = pot(pos=pos, bond_index=bond_index, bond_types=bond_types)
    assert float(e) == 0.0


def test_rejects_edge_index_as_bond_index():
    # A geometric edge_index [E, 2] (here E=3) must be rejected, not silently
    # mis-indexed into [2, N] logic (the scientist-confirmed latent bug).
    pot = _potential()
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    edge_index = torch.tensor([[0, 1], [1, 2], [0, 2]], dtype=torch.long)  # [E, 2]
    bond_types = torch.tensor([0, 0, 0], dtype=torch.long)
    with pytest.raises(ValueError, match="bond_index"):
        pot(pos=pos, bond_index=edge_index, bond_types=bond_types)


def test_missing_bond_types_raises():
    pot = _potential()
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    bond_index = torch.tensor([[0], [1]], dtype=torch.long)
    with pytest.raises(ValueError):
        pot(pos=pos, bond_index=bond_index)


def test_no_edge_index_or_bonds_fallback():
    # The 3-way fallback is gone: passing edge_index via the data dict must NOT
    # be silently picked up as bond_index.
    pot = _potential()
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    data = {"pos": pos, "edge_index": torch.tensor([[0, 1]], dtype=torch.long)}
    with pytest.raises(ValueError):
        pot(data=data)
