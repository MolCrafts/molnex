#!/usr/bin/env python
"""Regression: overfit AtomTypeReadout to overall_accuracy == 1.0 (offline)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from molrep.chem import AtomTypeReadout, ChemEncoder, TypingRecoveryMetrics  # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    enc = ChemEncoder(atom_dim=16, bond_dim=8)
    probe = AtomTypeReadout(enc, num_types=2)
    n = 6
    batch = TensorDict(
        {
            "atoms": TensorDict(
                {"Z": torch.tensor([6, 6, 6, 1, 1, 1], dtype=torch.long)},
                batch_size=[n],
            ),
            "bonds": TensorDict(
                {
                    "atomi": torch.tensor([0, 1, 3, 4], dtype=torch.long),
                    "atomj": torch.tensor([1, 2, 4, 5], dtype=torch.long),
                },
                batch_size=[4],
            ),
        },
        batch_size=[],
    )
    y = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    opt = torch.optim.Adam(probe.parameters(), lr=0.08)
    for _ in range(120):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(probe(batch)["logits"], y)
        loss.backward()
        opt.step()
    pred = probe(batch)["pred_type_id"]
    m = TypingRecoveryMetrics(num_types=2)
    m.update(pred, y)
    r = m.compute()
    assert r.overall_accuracy == 1.0, r.overall_accuracy
    print("mm-param-val-06-typing-recovery: OK")


if __name__ == "__main__":
    main()
