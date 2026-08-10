"""Tests for AtomTypeReadout isolation + overfit path."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from molrep.chem import AtomTypeReadout, ChemEncoder


def _batch(Z: list[int], bonds: list[tuple[int, int]] | None = None) -> TensorDict:
    n = len(Z)
    atoms = TensorDict({"Z": torch.tensor(Z, dtype=torch.long)}, batch_size=[n])
    td = TensorDict({"atoms": atoms}, batch_size=[])
    if bonds:
        atomi = torch.tensor([a for a, _ in bonds], dtype=torch.long)
        atomj = torch.tensor([b for _, b in bonds], dtype=torch.long)
        td["bonds"] = TensorDict(
            {"atomi": atomi, "atomj": atomj},
            batch_size=[len(bonds)],
        )
    return td


class TestAtomTypeReadout:
    def test_forward_shapes(self):
        enc = ChemEncoder(atom_dim=8, bond_dim=4)
        probe = AtomTypeReadout(enc, num_types=3)
        batch = _batch([6, 1], bonds=[(0, 1)])
        out = probe(batch)
        assert out["logits"].shape == (2, 3)
        assert out["pred_type_id"].shape == (2,)

    def test_labels_do_not_affect_encoder(self):
        enc = ChemEncoder(atom_dim=8, bond_dim=4)
        probe = AtomTypeReadout(enc, num_types=2)
        b1 = _batch([6, 8], bonds=[(0, 1)])
        b2 = _batch([6, 8], bonds=[(0, 1)])
        b2["atoms", "atom_type_id"] = torch.tensor([0, 1], dtype=torch.long)
        f1 = probe.encode(b1)
        f2 = probe.encode(b2)
        assert torch.allclose(f1, f2)

    def test_overfit_fixture_accuracy(self):
        torch.manual_seed(0)
        enc = ChemEncoder(atom_dim=16, bond_dim=8)
        probe = AtomTypeReadout(enc, num_types=2)
        batch = _batch([6, 6, 1, 1], bonds=[(0, 1), (2, 3)])
        y = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        opt = torch.optim.Adam(probe.parameters(), lr=0.05)
        for _ in range(80):
            opt.zero_grad()
            logits = probe(batch)["logits"]
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()
            opt.step()
        pred = probe(batch)["pred_type_id"]
        acc = float((pred == y).float().mean().item())
        assert acc >= 0.95
