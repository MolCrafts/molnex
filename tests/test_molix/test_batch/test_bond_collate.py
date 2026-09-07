"""Covalent bond_index collation: atom-offset rebase + BondHarmonic additivity.

See spec graph-connectivity-alignment-03-bonds. bond_index ([2, N] COO) is
offset via the spec-01 registry so multi-molecule batches index global atoms;
a topological energy must therefore equal the sum of per-molecule energies.
"""

import torch

from molix.batch.bond_adapter import bond_index_from_columns
from molix.batch.collate import collate_molecules, collate_packed
from molix.store import BlockDataset, PackedView
from molpot.potentials.bonds import BondHarmonic
from tests.test_molix.conftest import unit_a, unit_b, unit_samples, write_block


def _mol(pos, bonds, types):
    pos = torch.tensor(pos)
    return {
        "Z": torch.ones(pos.shape[0], dtype=torch.long),
        "pos": pos,
        "bond_index": torch.tensor(bonds, dtype=torch.long),  # [2, n]
        "bond_types": torch.tensor(types, dtype=torch.long),
    }


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
        return pot(pos=m["pos"], bond_index=m["bond_index"], bond_types=m["bond_types"])

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
    # feeds straight into a sample for collation
    sample = {
        "Z": torch.ones(3, dtype=torch.long),
        "pos": torch.zeros(3, 3),
        "bond_index": bi,
        "bond_types": bt,
    }
    batch = collate_molecules([sample])
    assert torch.equal(batch["bonds", "bond_index"], bi)


def test_block_bonds_roundtrip_equals_molecules(tmp_path):
    """Bonds survive a sealed block and ``collate_packed`` equals the oracle.

    ``collate_molecules`` over the per-frame sample dicts is the equivalence
    oracle for the packed path on the ``bonds`` namespace (ac-004).
    """
    units = [unit_a(), unit_b()]
    ds = BlockDataset.load_from_disk(write_block(tmp_path / "b", units).parent)
    samples = unit_samples(units)

    # per-sample unpack restores bond_index / bond_types
    assert torch.equal(ds[1]["bond_index"], samples[1]["bond_index"])
    assert torch.equal(ds[1]["bond_types"], samples[1]["bond_types"])

    frames = [0, 2]
    fast = collate_packed(PackedView(ds.frame_payload(frames)), range(len(frames)))
    oracle = collate_molecules([samples[f] for f in frames])
    assert torch.equal(fast["bonds", "bond_index"], oracle["bonds", "bond_index"])
    assert torch.equal(fast["bonds", "bond_types"], oracle["bonds", "bond_types"])
