"""Public-API edge_vectors consumes edge_diff, not shifts.

Hard-coded L=10 MIC pair (2026-08-16). No ASE / e3nn / mace-torch.
"""

from __future__ import annotations

import torch

from molzoo.mace.geometry import edge_lengths, edge_vectors


def main() -> None:
    pos = torch.tensor([[0.5, 0.0, 0.0], [9.5, 0.0, 0.0]], dtype=torch.float64)
    pos.requires_grad_(True)
    edge_index = torch.tensor([[0, 1]])
    imaged = torch.tensor([[-1.0, 0.0, 0.0]], dtype=torch.float64)
    vec = edge_vectors(pos, edge_index, edge_diff=imaged)
    if not torch.allclose(vec[0], imaged[0], atol=1e-12):
        raise AssertionError(f"edge_diff not consumed: {vec[0].tolist()}")
    if abs(float(edge_lengths(vec)[0]) - 1.0) > 1e-12:
        raise AssertionError("length != 1")
    try:
        edge_vectors(pos.detach(), edge_index, shifts=imaged)
    except TypeError:
        pass
    else:
        raise AssertionError("shifts= should raise TypeError")
    anti = edge_vectors(pos.detach(), edge_index, edge_diff=torch.tensor([[5.0, 0.0, 0.0]]))
    if not torch.allclose(anti[0], torch.tensor([5.0, 0.0, 0.0], dtype=torch.float64)):
        raise AssertionError("non-MIC edge_diff was re-wrapped")
    print("md-forcefield-molpy-reexport-04-mace: goldens OK")


if __name__ == "__main__":
    main()
