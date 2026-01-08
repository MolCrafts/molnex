import torch
from tensordict import TensorDict

from molpot.heads.heads import ForceHead


class TestForceHead:
    def test_requires_grad(self):
        head = ForceHead()
        atoms_x = torch.zeros(2, 3, requires_grad=False)
        td = TensorDict(
            {
                ("target", "energy"): torch.zeros(1),
                ("atoms", "x"): atoms_x,
            },
            batch_size=[],
        )
        try:
            head(td)
        except RuntimeError as exc:
            assert "requires_grad" in str(exc)
        else:
            raise AssertionError("Expected RuntimeError when atoms.x has no gradients")

    def test_forward_forces(self):
        head = ForceHead()
        atoms_x = torch.randn(2, 3, requires_grad=True)
        energy = atoms_x.pow(2).sum().unsqueeze(0)
        td = TensorDict(
            {
                ("target", "energy"): energy,
                ("atoms", "x"): atoms_x,
            },
            batch_size=[],
        )
        out = head(td)
        assert ("atoms", "f") in out.keys(include_nested=True)
        assert out["atoms", "f"].shape == atoms_x.shape
