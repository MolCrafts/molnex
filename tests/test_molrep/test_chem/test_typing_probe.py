"""Tests for AtomTypeReadout isolation + overfit path."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from molrep.chem import AtomTypeReadout
from molrep.encoders.message_passing import MessagePassingEncoder, MessagePassingEncoderSpec

#: Floating dtype every module under test is built at.
DTYPE = torch.float32


def _batch(Z: list[int], bonds: list[tuple[int, int]] | None = None) -> TensorDict:
    n = len(Z)
    atoms = TensorDict({"Z": torch.tensor(Z, dtype=torch.long)}, batch_size=[n])
    td = TensorDict({"atoms": atoms}, batch_size=[])
    index = torch.tensor(bonds or [], dtype=torch.long).reshape(-1, 2).T.contiguous()
    td["bonds"] = TensorDict({"bond_index": index}, batch_size=[])
    return td


class TestAtomTypeReadout:
    def test_forward_shapes(self):
        enc = MessagePassingEncoder(MessagePassingEncoderSpec(atom_dim=8, bond_dim=4), dtype=DTYPE)
        probe = AtomTypeReadout(enc, num_types=3, dtype=DTYPE)
        batch = _batch([6, 1], bonds=[(0, 1)])
        out = probe(batch)
        assert out["atoms", "logits"].shape == (2, 3)
        assert out["atoms", "pred_type_id"].shape == (2,)

    def test_labels_do_not_affect_encoder(self):
        enc = MessagePassingEncoder(MessagePassingEncoderSpec(atom_dim=8, bond_dim=4), dtype=DTYPE)
        probe = AtomTypeReadout(enc, num_types=2, dtype=DTYPE)
        b1 = _batch([6, 8], bonds=[(0, 1)])
        b2 = _batch([6, 8], bonds=[(0, 1)])
        b2["atoms", "atom_type_id"] = torch.tensor([0, 1], dtype=torch.long)
        f1 = probe.encode(b1)
        f2 = probe.encode(b2)
        assert torch.allclose(f1, f2)
