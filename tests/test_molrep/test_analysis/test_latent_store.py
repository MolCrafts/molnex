import torch

from molrep.analysis import AtomLatentTable


class TestAtomLatentTable:
    def test_basic(self):
        t = AtomLatentTable(
            features=torch.zeros(3, 4),
            molecule_id=["a", "a", "b"],
            ref_atom_type=torch.tensor([0, 0, 1]),
        )
        assert t.n_atoms == 3
        assert t.dim == 4
