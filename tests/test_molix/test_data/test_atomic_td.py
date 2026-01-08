import torch

from molix.data.atomic_td import AtomicTD, set_default_dtype, get_default_dtype


class TestConfig:
    def test_set_get_default_dtype(self):
        original = get_default_dtype()
        try:
            set_default_dtype(torch.float64)
            assert get_default_dtype() == torch.float64
        finally:
            set_default_dtype(original)


class TestAtomicTD:
    def test_create_required_fields(self):
        td = AtomicTD.create(
            Z=torch.tensor([1, 8, 1]),
            xyz=torch.zeros(3, 3),
            batch=torch.tensor([0, 0, 0]),
        )
        assert td["atoms", "Z"].dtype == torch.int64
        assert td["atoms", "xyz"].shape == torch.Size([3, 3])
        assert td["graph", "batch"].tolist() == [0, 0, 0]

    def test_create_optional_bond_fields(self):
        td = AtomicTD.create(
            Z=torch.tensor([1, 8]),
            xyz=torch.zeros(2, 3),
            batch=torch.tensor([0, 0]),
            bond_i=torch.tensor([0]),
            bond_j=torch.tensor([1]),
            bond_dist=torch.tensor([1.0]),
        )
        assert td["bonds", "i"].tolist() == [0]
        assert td["bonds", "j"].tolist() == [1]
        assert td["bonds", "dist"].shape == torch.Size([1])
