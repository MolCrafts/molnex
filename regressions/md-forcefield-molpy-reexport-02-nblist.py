"""Public-API MIC geometry written by NeighborList.build.

Hard-coded goldens (Nordlund MIC, 2026-08-16, no oracle):
L=10 Å, r_s=(0.5,5,5), r_t=(9.5,5,5) → raw=(9,0,0), edge_diff=(-1,0,0),
edge_dist=1, rsq=1. Live edge_diff stays on the pos graph (F ≠ 0).
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from molix.md import NeighborList


def main() -> None:
    cell = torch.eye(3, dtype=torch.float64) * 10.0
    pos = torch.tensor([[0.5, 5.0, 5.0], [9.5, 5.0, 5.0]], dtype=torch.float64)
    pos.requires_grad_(True)
    raw = pos[1] - pos[0]
    assert torch.allclose(raw, torch.tensor([9.0, 0.0, 0.0], dtype=torch.float64))
    nl = NeighborList(cell=cell, cutoff=2.0, positions=pos.detach(), skin=0.0)
    batch = TensorDict(
        {
            "atoms": TensorDict({"pos": pos}, batch_size=[2]),
            "graphs": TensorDict({"cell": cell}, batch_size=[]),
        },
        batch_size=[],
    )
    nl.build(batch)
    src = batch["edges", "edge_index"][:, 0]
    tgt = batch["edges", "edge_index"][:, 1]
    live = (src == 0) & (tgt == 1)
    d = batch["edges", "edge_diff"][live][0]
    expected = torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float64)
    if not torch.allclose(d, expected, atol=1e-12):
        raise AssertionError(f"edge_diff={d.tolist()} expected {expected.tolist()}")
    r = float(batch["edges", "edge_dist"][live][0])
    if abs(r - 1.0) > 1e-12:
        raise AssertionError(f"edge_dist={r} expected 1")
    if abs(float(d.dot(d)) - 1.0) > 1e-12:
        raise AssertionError(f"rsq={float(d.dot(d))} expected 1")
    energy = 0.5 * (d * d).sum()
    (grad,) = torch.autograd.grad(energy, pos)
    if float(grad.abs().sum()) == 0.0:
        raise AssertionError("detached edge_diff froze the force")
    print("md-forcefield-molpy-reexport-02-nblist: all hard-coded goldens OK")


if __name__ == "__main__":
    main()
