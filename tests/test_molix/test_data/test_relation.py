"""PairRelation: the covalent relation of each cutoff pair."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from molix.data import PairRelation


def _batch(n: int, pairs: list[tuple[int, int]], **valence: dict[str, list[int]]) -> TensorDict:
    data = {
        "atoms": TensorDict({"Z": torch.ones(n, dtype=torch.long)}, batch_size=[n]),
        "pairs": TensorDict(
            {"pair_index": torch.tensor(pairs, dtype=torch.long)}, batch_size=[len(pairs)]
        ),
    }
    for namespace, columns in valence.items():
        size = len(next(iter(columns.values())))
        data[namespace] = TensorDict(
            {name: torch.tensor(values) for name, values in columns.items()}, batch_size=[size]
        )
    return TensorDict(data, batch_size=[])


def _as_bond_index(batch: TensorDict) -> TensorDict:
    """Bonds as the collate writes them: ``bond_index`` ``(2, Nb)``, no ``atomi``/``atomj``."""
    bonds = batch["bonds"]
    batch["bonds"] = TensorDict(
        {"bond_index": torch.stack([bonds["atomi"], bonds["atomj"]])}, batch_size=[]
    )
    return batch


class TestPairRelation:
    @pytest.mark.parametrize("spelling", ["atomi/atomj", "bond_index"])
    def test_chain_pairs_read_their_bond_distance_in_either_orientation(self, spelling):
        # 0-1-2-3-4: (0,1) is 1-2, (0,2) 1-3, (0,3) 1-4, (0,4) none.
        batch = _batch(
            5,
            [(0, 1), (2, 0), (0, 3), (4, 0)],
            bonds={"atomi": [0, 1, 2, 3], "atomj": [1, 2, 3, 4]},
            angles={"atomi": [0, 1, 2], "atomj": [1, 2, 3], "atomk": [2, 3, 4]},
            propers={"atomi": [0, 1], "atomj": [1, 2], "atomk": [2, 3], "atoml": [3, 4]},
        )
        if spelling == "bond_index":
            batch = _as_bond_index(batch)
        relation = PairRelation().execute(batch)["pairs", "relation"]
        assert relation.dtype == torch.int8
        assert relation.tolist() == [1, 2, 3, 0]

    def test_the_closest_relation_wins_on_a_three_ring(self):
        # On a triangle every bonded pair is also the two ends of an angle.
        batch = _batch(
            3,
            [(0, 2)],
            bonds={"atomi": [0, 1, 0], "atomj": [1, 2, 2]},
            angles={"atomi": [0], "atomj": [1], "atomk": [2]},
        )
        assert PairRelation().execute(batch)["pairs", "relation"].tolist() == [1]
