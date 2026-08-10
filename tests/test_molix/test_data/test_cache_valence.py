"""PackedCache + collate_packed round-trip for valence column buckets.

Spec: learnable-classical-ff-02-valence-topology (ac-003).
"""

from __future__ import annotations

import pytest
import torch

from molix.data.cache import PackedCache
from molix.data.collate import collate_molecules, collate_packed
from molix.data.dataset import MmapDataset


def _mol(n_atoms: int, **families: dict) -> dict:
    sample: dict = {
        "Z": torch.ones(n_atoms, dtype=torch.long),
        "pos": torch.zeros(n_atoms, 3),
    }
    for name, cols in families.items():
        sample[name] = {k: torch.as_tensor(v, dtype=torch.long) for k, v in cols.items()}
    return sample


def test_packed_cache_angles_roundtrip_and_collate_packed(tmp_path):
    samples = [
        _mol(
            3,
            angles={"atomi": [0], "atomj": [1], "atomk": [2], "type": [0]},
        ),
        _mol(
            4,
            angles={"atomi": [0, 1], "atomj": [1, 2], "atomk": [2, 3], "type": [1, 0]},
        ),
    ]
    sink = tmp_path / "valence.pt"
    PackedCache(sink).save(samples)
    ds = MmapDataset(sink)

    # unpack restores nested columns (local indices)
    u1 = ds[1]
    assert isinstance(u1["angles"], dict)
    assert torch.equal(u1["angles"]["atomi"], samples[1]["angles"]["atomi"])
    assert torch.equal(u1["angles"]["atomj"], samples[1]["angles"]["atomj"])
    assert torch.equal(u1["angles"]["atomk"], samples[1]["angles"]["atomk"])
    assert torch.equal(u1["angles"]["type"], samples[1]["angles"]["type"])

    # single-file layout still (one .pt sink)
    assert sink.is_file()
    assert not any(p.is_dir() for p in tmp_path.iterdir() if p.name.startswith("valence"))

    indices = [0, 1]
    fast = collate_packed(ds.packed_view(), indices)
    oracle = collate_molecules([ds[i] for i in indices])
    for col in ("atomi", "atomj", "atomk", "type"):
        assert torch.equal(fast["angles"][col], oracle["angles"][col])
    # rebased: m2 +3
    assert fast["angles"]["atomi"].tolist() == [0, 3, 4]
    assert list(fast["angles"].batch_size) == list(oracle["angles"].batch_size)


def test_packed_cache_propers_impropers_roundtrip(tmp_path):
    samples = [
        _mol(
            4,
            propers={
                "atomi": [0],
                "atomj": [1],
                "atomk": [2],
                "atoml": [3],
                "type": [0],
            },
            impropers={
                "atomi": [1],
                "atomj": [0],
                "atomk": [2],
                "atoml": [3],
            },
        ),
        _mol(
            5,
            propers={
                "atomi": [0],
                "atomj": [1],
                "atomk": [2],
                "atoml": [3],
                "type": [1],
            },
            impropers={
                "atomi": [2],
                "atomj": [0],
                "atomk": [1],
                "atoml": [4],
            },
        ),
    ]
    sink = tmp_path / "torsions.pt"
    PackedCache(sink).save(samples)
    ds = MmapDataset(sink)

    assert torch.equal(ds[0]["impropers"]["atomi"], torch.tensor([1]))
    assert torch.equal(ds[1]["propers"]["atoml"], torch.tensor([3]))

    fast = collate_packed(ds.packed_view(), [0, 1])
    oracle = collate_molecules([ds[0], ds[1]])
    for fam in ("propers", "impropers"):
        for col in oracle[fam].keys():
            assert torch.equal(fast[fam][col], oracle[fam][col]), f"{fam}.{col}"
    # improper centers rebased: 1, 2+4
    assert fast["impropers"]["atomi"].tolist() == [1, 6]


def test_packed_cache_mixed_valence_all_or_none(tmp_path):
    samples = [
        _mol(3, angles={"atomi": [0], "atomj": [1], "atomk": [2]}),
        _mol(2),  # missing angles
    ]
    with pytest.raises(ValueError, match="angles"):
        PackedCache(tmp_path / "bad.pt").save(samples)


def test_bonds_and_angles_together_in_cache(tmp_path):
    """Bonds COO path coexists with angle columns in one packed file."""
    samples = [
        {
            "Z": torch.ones(3, dtype=torch.long),
            "pos": torch.zeros(3, 3),
            "bond_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            "bond_types": torch.tensor([0, 1], dtype=torch.long),
            "angles": {
                "atomi": torch.tensor([0], dtype=torch.long),
                "atomj": torch.tensor([1], dtype=torch.long),
                "atomk": torch.tensor([2], dtype=torch.long),
            },
        },
        {
            "Z": torch.ones(2, dtype=torch.long),
            "pos": torch.zeros(2, 3),
            "bond_index": torch.tensor([[0], [1]], dtype=torch.long),
            "bond_types": torch.tensor([0], dtype=torch.long),
            "angles": {
                "atomi": torch.tensor([0], dtype=torch.long),
                "atomj": torch.tensor([0], dtype=torch.long),
                "atomk": torch.tensor([1], dtype=torch.long),
            },
        },
    ]
    sink = tmp_path / "both.pt"
    PackedCache(sink).save(samples)
    ds = MmapDataset(sink)
    fast = collate_packed(ds.packed_view(), [0, 1])
    oracle = collate_molecules([ds[0], ds[1]])
    assert torch.equal(fast["bonds", "bond_index"], oracle["bonds", "bond_index"])
    assert torch.equal(fast["angles"]["atomi"], oracle["angles"]["atomi"])
