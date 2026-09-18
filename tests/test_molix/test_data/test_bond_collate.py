"""Covalent bond_index collation: atom-offset rebase + BondHarmonic additivity.

See spec graph-connectivity-alignment-03-bonds. bond_index ([2, N] COO) is
offset via the spec-01 registry so multi-molecule batches index global atoms;
a topological energy must therefore equal the sum of per-molecule energies.
"""

import torch

from molix.data.bond_adapter import bond_index_from_columns
from molix.data.collate import collate_molecules
from molpot.potentials.bonds import BondHarmonic
from tests.conftest import frame


def _mol(pos, bonds, types):
    """One frame: all-hydrogen geometry plus a bond block."""
    return frame(
        Z=[1] * len(pos),
        pos=pos,
        bonds=(bonds[0], bonds[1]),
        bond_types=types,
    )


def test_bond_index_offset_and_types_preserved():
    m1 = _mol([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], [[0], [1]], [0])
    m2 = _mol(
        [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.9, 0.9, 0.0]],
        [[0, 1], [1, 2]],
        [0, 1],
    )
    batch = collate_molecules([m1, m2])
    bi = batch["bonds", "bond_index"]
    assert bi.shape == (2, 3)  # 1 + 2 bonds, COO
    # m2's bonds are offset by +2 atoms (m1 has 2 atoms).
    assert bi.tolist() == [[0, 2, 3], [1, 3, 4]]
    assert batch["bonds", "bond_types"].tolist() == [0, 0, 1]


def test_two_molecule_bondharmonic_equals_sum():
    pot = BondHarmonic(k=torch.tensor([2.0, 3.0]), r0=torch.tensor([1.0, 1.0]))
    m1 = _mol([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], [[0], [1]], [0])
    m2 = _mol(
        [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.9, 0.9, 0.0]],
        [[0, 1], [1, 2]],
        [0, 1],
    )

    def energy(m):
        one = collate_molecules([m])
        return pot(
            pos=one["atoms", "pos"],
            bond_index=one["bonds", "bond_index"],
            bond_types=one["bonds", "bond_types"],
        )

    batch = collate_molecules([m1, m2])
    batched = pot(
        pos=batch["atoms", "pos"],
        bond_index=batch["bonds", "bond_index"],
        bond_types=batch["bonds", "bond_types"],
    )
    assert torch.allclose(batched, energy(m1) + energy(m2), atol=1e-6)


def test_bond_index_from_columns_roundtrip():
    atomi = torch.tensor([0, 1])
    atomj = torch.tensor([1, 2])
    types = torch.tensor([0, 1])
    bi, bt = bond_index_from_columns(atomi, atomj, types)
    assert bi.tolist() == [[0, 1], [1, 2]]
    assert bt.tolist() == [0, 1]
    # feeds straight into a frame for collation
    batch = collate_molecules(
        [frame(Z=[1, 1, 1], pos=torch.zeros(3, 3), bonds=(atomi, atomj), bond_types=types)]
    )
    assert torch.equal(batch["bonds", "bond_index"], bi)
