"""Valence topology collate: angles / propers / impropers as column TensorDicts.

Spec: learnable-classical-ff-02-valence-topology.

Pre-collate samples carry nested column dicts under ``angles`` / ``propers`` /
``impropers`` (atomi/atomj/atomk[/atoml], optional type). Post-collate the same
namespaces are nested TensorDicts with 1-D long columns rebased by atom offset.
Packed COO ``angle_index [3, N]`` is intentionally NOT the collate schema.
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from molix.data.collate import collate_molecules


def _mol(
    n_atoms: int,
    *,
    angles: dict | None = None,
    propers: dict | None = None,
    impropers: dict | None = None,
    bond_index: torch.Tensor | None = None,
    bond_types: torch.Tensor | None = None,
) -> dict:
    sample: dict = {
        "Z": torch.ones(n_atoms, dtype=torch.long),
        "pos": torch.zeros(n_atoms, 3),
    }
    if angles is not None:
        sample["angles"] = {
            k: torch.as_tensor(v, dtype=torch.long) for k, v in angles.items()
        }
    if propers is not None:
        sample["propers"] = {
            k: torch.as_tensor(v, dtype=torch.long) for k, v in propers.items()
        }
    if impropers is not None:
        sample["impropers"] = {
            k: torch.as_tensor(v, dtype=torch.long) for k, v in impropers.items()
        }
    if bond_index is not None:
        sample["bond_index"] = bond_index.long()
    if bond_types is not None:
        sample["bond_types"] = bond_types.long()
    return sample


def test_angles_column_rebase_two_molecules():
    """Second molecule's angle indices equal local + n_atoms_0 (ac-001)."""
    m1 = _mol(
        3,
        angles={
            "atomi": [0],
            "atomj": [1],  # central
            "atomk": [2],
            "type": [0],
        },
    )
    m2 = _mol(
        4,
        angles={
            "atomi": [0, 1],
            "atomj": [1, 2],
            "atomk": [2, 3],
            "type": [1, 0],
        },
    )
    batch = collate_molecules([m1, m2])

    angles = batch["angles"]
    assert isinstance(angles, TensorDict)
    # Nested access form preferred by the contract.
    atomi = batch["angles"]["atomi"]
    atomj = batch["angles"]["atomj"]
    atomk = batch["angles"]["atomk"]
    assert atomi.dtype == torch.long
    assert atomi.shape == (3,)  # 1 + 2 angles
    # m1 local, m2 rebased by +3 atoms.
    assert atomi.tolist() == [0, 3, 4]
    assert atomj.tolist() == [1, 4, 5]
    assert atomk.tolist() == [2, 5, 6]
    assert batch["angles"]["type"].tolist() == [0, 1, 0]
    # Prefer batch_size=[N] when all leaves share length N.
    assert list(angles.batch_size) == [3]
    # Primary schema is columns, not packed angle_index [3, N].
    assert "angle_index" not in angles.keys()


def test_propers_and_impropers_columns_with_center_first():
    """Propers/impropers expose atomi..atoml; impropers.atomi is center (ac-002)."""
    m1 = _mol(
        4,
        propers={
            "atomi": [0],
            "atomj": [1],
            "atomk": [2],
            "atoml": [3],
            "type": [0],
        },
        impropers={
            "atomi": [1],  # center (molrs)
            "atomj": [0],
            "atomk": [2],
            "atoml": [3],
            "type": [2],
        },
    )
    m2 = _mol(
        5,
        propers={
            "atomi": [0, 1],
            "atomj": [1, 2],
            "atomk": [2, 3],
            "atoml": [3, 4],
            "type": [1, 0],
        },
        impropers={
            "atomi": [2],
            "atomj": [0],
            "atomk": [1],
            "atoml": [3],
            "type": [0],
        },
    )
    batch = collate_molecules([m1, m2])

    # m2 offset = 4
    assert batch["propers"]["atomi"].tolist() == [0, 4, 5]
    assert batch["propers"]["atomj"].tolist() == [1, 5, 6]
    assert batch["propers"]["atomk"].tolist() == [2, 6, 7]
    assert batch["propers"]["atoml"].tolist() == [3, 7, 8]
    assert batch["propers"]["type"].tolist() == [0, 1, 0]
    assert list(batch["propers"].batch_size) == [3]

    assert batch["impropers"]["atomi"].tolist() == [1, 6]  # centers: 1, 2+4
    assert batch["impropers"]["atomj"].tolist() == [0, 4]
    assert batch["impropers"]["atomk"].tolist() == [2, 5]
    assert batch["impropers"]["atoml"].tolist() == [3, 7]
    assert batch["impropers"]["type"].tolist() == [2, 0]
    assert list(batch["impropers"].batch_size) == [2]


def test_valence_all_or_none_raises():
    """Mixed presence of a valence family across samples raises ValueError."""
    m1 = _mol(3, angles={"atomi": [0], "atomj": [1], "atomk": [2]})
    m2 = _mol(2)  # no angles
    with pytest.raises(ValueError, match="angles"):
        collate_molecules([m1, m2])


def test_empty_angles_namespace():
    """Zero-count angles on every sample still emit empty columns."""
    empty = {
        "atomi": torch.zeros(0, dtype=torch.long),
        "atomj": torch.zeros(0, dtype=torch.long),
        "atomk": torch.zeros(0, dtype=torch.long),
    }
    m1 = _mol(2, angles=empty)
    m2 = _mol(3, angles=empty)
    batch = collate_molecules([m1, m2])
    assert batch["angles"]["atomi"].shape == (0,)
    assert batch["angles"]["atomj"].shape == (0,)
    assert batch["angles"]["atomk"].shape == (0,)
    assert list(batch["angles"].batch_size) == [0]


def test_no_valence_omits_namespaces():
    """Samples without valence keys do not grow angles/propers/impropers."""
    batch = collate_molecules([_mol(2), _mol(3)])
    assert "angles" not in batch.keys()
    assert "propers" not in batch.keys()
    assert "impropers" not in batch.keys()


def test_bonds_still_work_alongside_angles():
    """Existing bond_index path stays green next to angles (ac-006)."""
    # bond_index is COO [2, N]: m1 has bonds 0-1 and 1-2; m2 has bond 0-1.
    m1 = _mol(
        3,
        angles={"atomi": [0], "atomj": [1], "atomk": [2], "type": [0]},
        bond_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        bond_types=torch.tensor([0, 1], dtype=torch.long),
    )
    m2 = _mol(
        2,
        angles={"atomi": [0], "atomj": [0], "atomk": [1], "type": [1]},
        bond_index=torch.tensor([[0], [1]], dtype=torch.long),
        bond_types=torch.tensor([0], dtype=torch.long),
    )
    batch = collate_molecules([m1, m2])
    assert batch["bonds", "bond_index"].tolist() == [[0, 1, 3], [1, 2, 4]]
    assert batch["bonds", "bond_types"].tolist() == [0, 1, 0]
    assert batch["angles"]["atomi"].tolist() == [0, 3]


def test_collate_does_not_import_molpy_as_batch_store():
    """Topology leaves are plain torch tensors inside TensorDict (ac-004)."""
    from pathlib import Path

    import molix.data.cache as cache_mod
    import molix.data.collate as collate_mod

    for mod in (collate_mod, cache_mod):
        src = Path(mod.__file__).read_text()
        assert "from molpy" not in src
        assert "import molpy" not in src
        assert "ForceField" not in src
        assert "import molrs" not in src
