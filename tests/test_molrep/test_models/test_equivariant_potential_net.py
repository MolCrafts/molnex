import torch
from tensordict import TensorDict

from molrep.models.equivariant_potential import EquivariantPotentialNet


def _make_tensordict():
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    edge_i = torch.tensor([0, 0, 1], dtype=torch.int64)
    edge_j = torch.tensor([1, 2, 2], dtype=torch.int64)
    diff = positions[edge_i] - positions[edge_j]
    dist = torch.norm(diff, dim=-1)
    return TensorDict(
        {
            ("atoms", "Z"): torch.tensor([1, 1, 8], dtype=torch.int64),
            ("atoms", "xyz"): positions.clone(),
            ("graph", "batch"): torch.tensor([0, 0, 0], dtype=torch.int64),
            ("pairs", "i"): edge_i,
            ("pairs", "j"): edge_j,
            ("pairs", "dist"): dist,
            ("pairs", "diff"): diff,
        },
        batch_size=[],
    )


def _make_model():
    hidden_dim = 16
    equivariant_dim = 8
    return EquivariantPotentialNet(
        num_atom_types=10,
        hidden_dim=hidden_dim,
        equivariant_dim=equivariant_dim,
        num_blocks=2,
        lmax=2,
        cutoff=5.0,
        num_rbf=6,
    )



class TestEquivariantPotentialNet:
    def test_forward_adds_energy(self):
        td = _make_tensordict()
        model = _make_model()
        out = model(td)
        assert ("target", "energy") in out.keys(include_nested=True)
        assert out["target", "energy"].shape == torch.Size([1])

    def test_autograd_forces(self):
        td = _make_tensordict()
        td["atoms", "xyz"].requires_grad_(True)
        model = _make_model()
        energy = model(td)["target", "energy"].sum()
        forces = -torch.autograd.grad(energy, td["atoms", "xyz"])[0]
        assert forces.shape == td["atoms", "xyz"].shape
        assert torch.isfinite(forces).all()
