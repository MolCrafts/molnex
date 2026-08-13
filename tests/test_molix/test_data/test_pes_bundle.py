"""Unit tests for :class:`molix.data.pes_bundle.PesBundle`."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from molix.data.group_split import group_split_indices
from molix.data.pes_bundle import PesBundle

# Water geometries in Å (acceptance ac-008).
_WATER_Z = torch.tensor([8, 1, 1], dtype=torch.long)
_WATER_POS = [
    [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2399, 0.9266, 0.0]],
    [[0.0, 0.0, 0.0], [1.02, 0.0, 0.0], [-0.25, 0.97, 0.0]],
]
_POS_ATOL = 1e-8


def _water_bond_index() -> torch.Tensor:
    return torch.tensor([[0, 0], [1, 2]], dtype=torch.long)


def _water_topology() -> TensorDict:
    n_atoms = 3
    return TensorDict(
        {
            "atoms": TensorDict({"Z": _WATER_Z.clone()}, batch_size=[n_atoms]),
            "bonds": TensorDict({"bond_index": _water_bond_index()}, batch_size=[]),
        },
        batch_size=[],
    )


def _water_pos() -> torch.Tensor:
    return torch.tensor(_WATER_POS)


def _water_conformer(
    pos: torch.Tensor,
    *,
    bond_index: torch.Tensor | None = None,
) -> TensorDict:
    td = _water_topology()
    if bond_index is not None:
        td["bonds", "bond_index"] = bond_index
    td["atoms", "pos"] = pos
    return td


def _has_pos(td: TensorDict) -> bool:
    if "pos" in td.keys():
        return True
    return "atoms" in td.keys() and "pos" in td["atoms"].keys()


class TestPesBundle:
    def test_constructs_with_z_and_stacked_pos(self) -> None:
        pos = _water_pos()
        bundle = PesBundle(_water_topology(), pos, group_id="water")
        assert bundle.group_id == "water"
        assert bundle.n_atoms == 3
        assert bundle.n_conformers == 2
        assert bundle.pos.shape == (2, 3, 3)
        assert torch.equal(bundle.topology["atoms", "Z"], _WATER_Z)
        assert torch.allclose(bundle.pos, pos, atol=_POS_ATOL, rtol=0.0)

    def test_rejects_atoms_pos_in_topology(self) -> None:
        topology = _water_topology()
        topology["atoms", "pos"] = torch.zeros(3, 3)
        with pytest.raises(ValueError):
            PesBundle(topology, _water_pos(), group_id="water")

    def test_rejects_edges_namespace_in_topology(self) -> None:
        topology = _water_topology()
        topology["edges"] = TensorDict({}, batch_size=[])
        with pytest.raises(ValueError):
            PesBundle(topology, _water_pos(), group_id="water")

    def test_rejects_pos_shape_not_c_n_3(self) -> None:
        pos_missing_c = torch.tensor(_WATER_POS[0])
        with pytest.raises(ValueError):
            PesBundle(_water_topology(), pos_missing_c, group_id="water")

    def test_rejects_zero_conformers(self) -> None:
        pos = torch.zeros(0, 3, 3)
        with pytest.raises(ValueError):
            PesBundle(_water_topology(), pos, group_id="water")

    def test_rejects_atom_count_mismatch(self) -> None:
        pos = torch.zeros(2, 4, 3)
        with pytest.raises(ValueError):
            PesBundle(_water_topology(), pos, group_id="water")

    def test_rejects_unknown_targets_key(self) -> None:
        targets = TensorDict({"hof": torch.zeros(2)}, batch_size=[2])
        with pytest.raises(ValueError):
            PesBundle(_water_topology(), _water_pos(), group_id="water", targets=targets)

    def test_requires_group_id(self) -> None:
        with pytest.raises(TypeError):
            PesBundle(_water_topology(), _water_pos())

    def test_topology_batch_has_no_pos(self) -> None:
        bundle = PesBundle(_water_topology(), _water_pos(), group_id="water")
        topo = bundle.topology_batch()
        assert not _has_pos(topo)
        assert torch.equal(topo["atoms", "Z"], _WATER_Z)
        assert torch.equal(topo["bonds", "bond_index"], _water_bond_index())

    def test_conformation_batch_shares_bond_index(self) -> None:
        pos = _water_pos()
        bundle = PesBundle(_water_topology(), pos, group_id="water")
        conf0 = bundle.conformation_batch(0)
        conf1 = bundle.conformation_batch(1)
        assert torch.equal(conf0["bonds", "bond_index"], conf1["bonds", "bond_index"])
        assert not torch.equal(conf0["atoms", "pos"], conf1["atoms", "pos"])
        assert torch.allclose(conf0["atoms", "pos"], pos[0], atol=_POS_ATOL, rtol=0.0)
        assert torch.allclose(conf1["atoms", "pos"], pos[1], atol=_POS_ATOL, rtol=0.0)

    def test_conformation_batch_out_of_range_raises(self) -> None:
        bundle = PesBundle(_water_topology(), _water_pos(), group_id="water")
        with pytest.raises(IndexError):
            bundle.conformation_batch(2)

    def test_iter_length_equals_n_conformers(self) -> None:
        bundle = PesBundle(_water_topology(), _water_pos(), group_id="water")
        assert len(bundle) == 2
        assert len(list(bundle)) == 2

    def test_from_conformer_list_stacks_matching_graphs(self) -> None:
        pos = _water_pos()
        bundle = PesBundle.from_conformer_list(
            [_water_conformer(pos[0]), _water_conformer(pos[1])],
            group_id="water",
        )
        assert bundle.n_conformers == 2
        assert not _has_pos(bundle.topology_batch())

    def test_from_conformer_list_rejects_empty(self) -> None:
        with pytest.raises(ValueError):
            PesBundle.from_conformer_list([], group_id="water")

    def test_from_conformer_list_rejects_unequal_bond_index(self) -> None:
        pos = _water_pos()
        other = torch.tensor([[0], [1]], dtype=torch.long)
        with pytest.raises(ValueError):
            PesBundle.from_conformer_list(
                [_water_conformer(pos[0]), _water_conformer(pos[1], bond_index=other)],
                group_id="water",
            )

    def test_conformer_group_ids_repeats_group_id(self) -> None:
        bundle = PesBundle(_water_topology(), _water_pos(), group_id="m0")
        assert bundle.conformer_group_ids() == ("m0", "m0")

    def test_group_split_indices_does_not_leak_conformers(self) -> None:
        bundles = [PesBundle(_water_topology(), _water_pos(), group_id=f"m{k}") for k in range(10)]
        ids = [g for bundle in bundles for g in bundle.conformer_group_ids()]
        train, val, test = group_split_indices(ids, ratios=(0.8, 0.1, 0.1), seed=0)

        train_g = {ids[i] for i in train}
        val_g = {ids[i] for i in val}
        test_g = {ids[i] for i in test}
        assert len(train_g) == 8
        assert len(val_g) == 1
        assert len(test_g) == 1
        assert train_g.isdisjoint(val_g)
        assert train_g.isdisjoint(test_g)
        assert val_g.isdisjoint(test_g)

        split_of: dict[int, str] = {}
        for name, idxs in (("train", train), ("val", val), ("test", test)):
            for i in idxs:
                split_of[i] = name
        for k in range(10):
            assert split_of[2 * k] == split_of[2 * k + 1]
