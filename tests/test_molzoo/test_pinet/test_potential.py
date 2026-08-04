"""Unit tests for molzoo.pinet.potential.PiNetPotential (forces + energy)."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from molzoo.pinet import PiNet, PiNetPotential

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _batch(
    *, n_graphs: int = 2, atoms_per_graph: int = 6, dtype: torch.dtype = torch.float32
) -> TensorDict:
    n_atoms = n_graphs * atoms_per_graph
    z_cycle = torch.tensor([1, 6, 7, 8], device=DEVICE)
    z = z_cycle[torch.arange(n_atoms, device=DEVICE) % 4]
    pos = torch.randn(n_atoms, 3, dtype=dtype, device=DEVICE)
    atom_batch = torch.arange(n_graphs, device=DEVICE).repeat_interleave(atoms_per_graph)
    edges = [
        (g * atoms_per_graph + i, g * atoms_per_graph + j)
        for g in range(n_graphs)
        for i in range(atoms_per_graph)
        for j in range(atoms_per_graph)
        if i != j
    ]
    edge_index = torch.tensor(edges, dtype=torch.long, device=DEVICE)
    return TensorDict(
        atoms=TensorDict(
            Z=z,
            pos=pos,
            batch=atom_batch,
            forces=torch.zeros(n_atoms, 3, dtype=dtype, device=DEVICE),
            batch_size=[n_atoms],
        ),
        edges=TensorDict(edge_index=edge_index, batch_size=[edge_index.shape[0]]),
        graphs=TensorDict(
            num_atoms=torch.full((n_graphs,), atoms_per_graph, dtype=torch.long, device=DEVICE),
            energy=torch.zeros(n_graphs, dtype=dtype, device=DEVICE),
            batch_size=[n_graphs],
        ),
        batch_size=[],
    )


def _model(*, compute_forces: bool = True) -> PiNetPotential:
    torch.manual_seed(0)
    return PiNetPotential(
        atom_types=[1, 6, 7, 8],
        r_max=4.0,
        n_basis=4,
        pp_nodes=[16, 16],
        pi_nodes=[16],
        ii_nodes=[16, 16],
        depth=3,
        rank=3,
        hidden_dim=16,
        compute_forces=compute_forces,
    ).to(DEVICE)


def test_energy_and_force_shapes():
    model = PiNetPotential(
        atom_types=[1, 6, 7, 8],
        r_max=4.0,
        n_basis=3,
        pp_nodes=[8, 8],
        pi_nodes=[8, 8],
        ii_nodes=[8, 8],
        depth=2,
        rank=3,
        hidden_dim=8,
        compute_forces=True,
    )
    n = 4
    batch = TensorDict(
        atoms=TensorDict(
            Z=torch.tensor([1, 6, 7, 8]),
            pos=torch.randn(n, 3),
            batch=torch.zeros(n, dtype=torch.long),
            batch_size=[n],
        ),
        edges=TensorDict(
            edge_index=torch.tensor(
                [[0, 1], [1, 0], [0, 2], [2, 0], [1, 3], [3, 1], [2, 3], [3, 2]],
                dtype=torch.long,
            ),
            batch_size=[8],
        ),
        graphs=TensorDict(num_atoms=torch.tensor([n]), batch_size=[1]),
        batch_size=[],
    )
    batch = model(batch)
    assert batch["graphs", "energy"].shape == (1,)
    assert batch["atoms", "forces"].shape == (n, 3)


def test_encoder_kwarg_composition():
    enc = PiNet(
        atom_types=[1, 6, 7, 8],
        r_max=4.0,
        n_basis=3,
        pp_nodes=[8, 8],
        pi_nodes=[8, 8],
        ii_nodes=[8, 8],
        depth=2,
        rank=1,
    )
    model = PiNetPotential(encoder=enc, hidden_dim=8, compute_forces=False)
    assert model.encoder is enc


def test_func_forces_match_autograd_reference():
    model = _model(compute_forces=True)
    model.eval()
    batch = _batch()

    b1 = model(batch.clone())
    f_func = b1["atoms", "forces"].detach()

    # Energy-only model for the reference path (same architecture / weights).
    ref_model = _model(compute_forces=False)
    ref_model.load_state_dict(
        {k: v for k, v in model.state_dict().items()},
        strict=False,
    )
    # Copy only shared params (force pipeline has no extra params).
    ref_model.load_state_dict(model.state_dict())
    ref_model.eval()

    ref = batch.clone()
    pos = ref["atoms", "pos"].detach().clone().requires_grad_(True)
    ref["atoms", "pos"] = pos
    with torch.enable_grad():
        ref = ref_model(ref)
        energy = ref["graphs", "energy"].sum()
    f_autograd = -torch.autograd.grad(energy, pos)[0].detach()

    assert f_func.shape == f_autograd.shape
    assert torch.allclose(f_func, f_autograd, atol=1e-5, rtol=1e-5), (
        f"max abs diff {(f_func - f_autograd).abs().max().item():.3e}"
    )


def test_eval_and_train_forward_match():
    model = _model(compute_forces=True)
    torch.manual_seed(1)
    batch = _batch()

    model.train()
    out_train = model(batch.clone())
    model.eval()
    out_eval = model(batch.clone())

    for key in (("graphs", "energy"), ("atoms", "energy"), ("atoms", "forces")):
        assert torch.allclose(out_eval[key], out_train[key].detach(), atol=1e-6, rtol=1e-6), key


def test_force_loss_single_backward_populates_param_grads():
    model = _model(compute_forces=True)
    model.train()
    batch = model(_batch().clone())
    batch["atoms", "forces"].pow(2).mean().backward()

    n_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert n_grad > 0, "no parameter received a gradient from the force loss"


def test_no_lazy_linear_in_potential():
    model = _model(compute_forces=True)
    for m in model.modules():
        assert type(m).__name__ != "LazyLinear"
