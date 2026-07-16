"""Force path of ``PiNetPotential`` — functorch (``torch.func.grad``).

``PiNetPotential`` derives forces only via functorch, which traces the force into
the forward graph so ``energy → force → loss`` is a single backward (no
double-backward barrier). Function-level gates:

  * NUM  — functorch forces match a plain ``torch.autograd.grad`` reference
    (scientific correctness of the force derivation).
  * G3c  — one force-loss ``.backward()`` populates parameter grads, no error.

The ``torch.compile`` execution of this path is an integration concern, not a
unit test, so it is not gated here.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from molzoo.pinet import PiNetPotential

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


def _model() -> PiNetPotential:
    torch.manual_seed(0)
    model = PiNetPotential(
        atom_types=[1, 6, 7, 8],
        r_max=4.0,
        n_basis=4,
        pp_nodes=[16, 16],
        pi_nodes=[16],
        ii_nodes=[16, 16],
        depth=3,
        rank=3,
        hidden_dim=16,
    ).to(DEVICE)
    model(_batch(), compute_forces=False)  # materialise lazy params
    return model


def test_functorch_forces_match_autograd_reference():
    """functorch forces == a plain -autograd.grad(E, pos) reference."""
    model = _model()
    model.eval()
    batch = _batch()

    f_functorch = model(batch.clone(), compute_forces=True)["forces"].detach()

    ref = batch.clone()
    pos = ref["atoms", "pos"].detach().clone().requires_grad_(True)
    ref["atoms", "pos"] = pos
    with torch.enable_grad():
        energy = model(ref, compute_forces=False)["energy"].sum()
    f_autograd = -torch.autograd.grad(energy, pos)[0].detach()

    assert f_functorch.shape == f_autograd.shape
    assert torch.allclose(f_functorch, f_autograd, atol=1e-5, rtol=1e-5), (
        f"max abs diff {(f_functorch - f_autograd).abs().max().item():.3e}"
    )


def test_eval_single_pass_matches_train_two_pass():
    """eval() takes the fused grad(has_aux) single pass; outputs must be
    identical (same keys, same values) to the training two-pass path."""
    model = _model()
    torch.manual_seed(1)
    batch = _batch()

    model.train()
    out_train = model(batch.clone(), compute_forces=True)
    model.eval()
    out_eval = model(batch.clone(), compute_forces=True)

    assert set(out_eval) == set(out_train)
    for key in ("energy", "atomic_energy", "forces"):
        assert torch.allclose(out_eval[key], out_train[key].detach(), atol=1e-6, rtol=1e-6), (
            f"{key} diverges between eval single-pass and train two-pass"
        )


def test_force_loss_single_backward_populates_param_grads():
    """A force-loss single backward trains params with no double-backward error."""
    model = _model()
    model.train()
    batch = _batch()

    out = model(batch.clone(), compute_forces=True)
    loss = (out["forces"] - batch["atoms", "forces"]).pow(2).mean()
    loss.backward()  # must not raise the double-backward RuntimeError

    n_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert n_grad > 0, "no parameter received a gradient from the force loss"
