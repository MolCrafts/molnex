"""Public-API regression for valence topology collate namespaces.

Spec: `learnable-classical-ff-02-valence-topology`.

Hard-coded goldens only — no third-party oracles. Pins:

1. Nested pre-collate → nested TensorDict columns post-collate
2. Atom-offset rebase: second molecule indices = local + n_atoms_0
3. Improper center-first (atomi = center, molrs)
4. Optional stack helpers build COO [arity, N] without making COO the schema
5. Bonds collate still works alongside angles
6. PackedCache round-trip + collate_packed leaf equality

Provenance
----------
    capture command  : PYTHONPATH=src python regressions/learnable-classical-ff-02-valence-topology.py
    date             : 2026-08-10
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch


def main() -> int:
    from molix.data.cache import PackedCache
    from molix.data.collate import collate_molecules, collate_packed
    from molix.data.dataset import MmapDataset
    from molix.datasets._valence_columns import (
        stack_angle_index,
        stack_improper_index,
        stack_proper_index,
    )

    # --- Fixtures: m1 (3 atoms, 1 angle), m2 (4 atoms, 2 angles) ---
    m1 = {
        "Z": torch.ones(3, dtype=torch.long),
        "pos": torch.zeros(3, 3),
        "bond_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        "bond_types": torch.tensor([0, 1], dtype=torch.long),
        "angles": {
            "atomi": torch.tensor([0], dtype=torch.long),
            "atomj": torch.tensor([1], dtype=torch.long),
            "atomk": torch.tensor([2], dtype=torch.long),
            "type": torch.tensor([0], dtype=torch.long),
        },
        "propers": {
            "atomi": torch.tensor([0], dtype=torch.long),
            "atomj": torch.tensor([1], dtype=torch.long),
            "atomk": torch.tensor([2], dtype=torch.long),
            "atoml": torch.tensor([0], dtype=torch.long),  # reuse atom 0 for fixture
            "type": torch.tensor([0], dtype=torch.long),
        },
        "impropers": {
            "atomi": torch.tensor([1], dtype=torch.long),  # center
            "atomj": torch.tensor([0], dtype=torch.long),
            "atomk": torch.tensor([2], dtype=torch.long),
            "atoml": torch.tensor([0], dtype=torch.long),
        },
    }
    m2 = {
        "Z": torch.ones(4, dtype=torch.long),
        "pos": torch.zeros(4, 3),
        "bond_index": torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        "bond_types": torch.tensor([0, 0, 1], dtype=torch.long),
        "angles": {
            "atomi": torch.tensor([0, 1], dtype=torch.long),
            "atomj": torch.tensor([1, 2], dtype=torch.long),
            "atomk": torch.tensor([2, 3], dtype=torch.long),
            "type": torch.tensor([1, 0], dtype=torch.long),
        },
        "propers": {
            "atomi": torch.tensor([0], dtype=torch.long),
            "atomj": torch.tensor([1], dtype=torch.long),
            "atomk": torch.tensor([2], dtype=torch.long),
            "atoml": torch.tensor([3], dtype=torch.long),
            "type": torch.tensor([1], dtype=torch.long),
        },
        "impropers": {
            "atomi": torch.tensor([2], dtype=torch.long),  # center
            "atomj": torch.tensor([0], dtype=torch.long),
            "atomk": torch.tensor([1], dtype=torch.long),
            "atoml": torch.tensor([3], dtype=torch.long),
        },
    }

    batch = collate_molecules([m1, m2])

    # --- 1. Nested column access + rebase (m2 offset = 3) ---
    assert batch["angles"]["atomi"].tolist() == [0, 3, 4]
    assert batch["angles"]["atomj"].tolist() == [1, 4, 5]
    assert batch["angles"]["atomk"].tolist() == [2, 5, 6]
    assert batch["angles"]["type"].tolist() == [0, 1, 0]
    assert list(batch["angles"].batch_size) == [3]
    assert "angle_index" not in batch["angles"].keys()

    # --- 2. Propers rebased ---
    assert batch["propers"]["atomi"].tolist() == [0, 3]
    assert batch["propers"]["atoml"].tolist() == [0, 6]
    assert batch["propers"]["type"].tolist() == [0, 1]

    # --- 3. Impropers: atomi = center, rebased ---
    assert batch["impropers"]["atomi"].tolist() == [1, 5]  # 1, 2+3

    # --- 4. Bonds still green ---
    assert batch["bonds", "bond_index"].tolist() == [
        [0, 1, 3, 4, 5],
        [1, 2, 4, 5, 6],
    ]
    assert batch["bonds", "bond_types"].tolist() == [0, 1, 0, 0, 1]

    # --- 5. Stack helpers (kernel-local only) ---
    a_idx = stack_angle_index(batch["angles"])
    assert a_idx.shape == (3, 3)
    assert a_idx.tolist() == [[0, 3, 4], [1, 4, 5], [2, 5, 6]]
    p_idx = stack_proper_index(batch["propers"])
    assert p_idx.shape == (4, 2)
    i_idx = stack_improper_index(batch["impropers"])
    assert i_idx[0].tolist() == [1, 5]  # centers at row 0

    # --- 6. PackedCache round-trip + collate_packed ---
    with tempfile.TemporaryDirectory() as td:
        sink = Path(td) / "valence.pt"
        PackedCache(sink).save([m1, m2])
        assert sink.is_file()
        ds = MmapDataset(sink)
        u1 = ds[1]
        assert torch.equal(u1["angles"]["atomi"], m2["angles"]["atomi"])
        assert torch.equal(u1["impropers"]["atomi"], m2["impropers"]["atomi"])

        fast = collate_packed(ds.packed_view(), [0, 1])
        oracle = collate_molecules([ds[0], ds[1]])
        for fam in ("angles", "propers", "impropers"):
            for col in oracle[fam].keys():
                assert torch.equal(fast[fam][col], oracle[fam][col]), f"{fam}.{col}"
        assert torch.equal(fast["bonds", "bond_index"], oracle["bonds", "bond_index"])

    # --- 7. No molpy as batch store in collate/cache ---
    import molix.data.cache as cache_mod
    import molix.data.collate as collate_mod

    for mod in (collate_mod, cache_mod):
        text = Path(mod.__file__).read_text()
        assert "from molpy" not in text
        assert "import molrs" not in text

    print("learnable-classical-ff-02-valence-topology: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
