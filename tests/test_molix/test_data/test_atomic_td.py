import torch
from molix.data.atom_td import AtomTD


class TestAtomTD:
    def test_create_required_fields(self):
        td = AtomTD.create(
            Z=torch.tensor([1, 8, 1]),
            xyz=torch.zeros(3, 3),
            batch=torch.tensor([0, 0, 0]),
        )
        assert td.Z.dtype == torch.int64
        assert td.xyz.shape == torch.Size([3, 3])
        assert td.batch.tolist() == [0, 0, 0]

    def test_create_optional_bond_fields(self):
        td = AtomTD.create(
            Z=torch.tensor([1, 8]),
            xyz=torch.zeros(2, 3),
            batch=torch.tensor([0, 0]),
            bond_i=torch.tensor([0]),
            bond_j=torch.tensor([1]),
            bond_dist=torch.tensor([1.0]),
        )
        assert td.bond_i.tolist() == [0]
        assert td.bond_j.tolist() == [1]
        assert td.bond_dist.shape == torch.Size([1])
