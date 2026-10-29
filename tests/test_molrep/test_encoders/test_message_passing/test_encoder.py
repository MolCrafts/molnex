"""Unit tests for molrep.encoders.message_passing.encoder.MessagePassingEncoder."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from molrep.encoders.message_passing import (
    DiscreteChannel,
    MessagePassingEncoder,
    MessagePassingEncoderSpec,
)

DTYPE = torch.float64
BONDS = [(0, 1), (0, 2), (0, 3)]


def _frame(bonds=BONDS) -> TensorDict:
    return TensorDict(
        {
            "atoms": TensorDict({"Z": torch.tensor([8, 1, 1, 6])}, batch_size=[4]),
            "bonds": TensorDict(
                {
                    "bond_index": torch.tensor(bonds).T.contiguous(),
                    "bond_type": torch.tensor([1, 1, 2]),
                },
                batch_size=[],
            ),
        },
        batch_size=[],
    )


def _encoder(**fields) -> MessagePassingEncoder:
    torch.manual_seed(0)
    spec = MessagePassingEncoderSpec(atom_dim=8, bond_dim=6, n_blocks=2, **fields)
    return MessagePassingEncoder(spec, dtype=DTYPE)


class TestMessagePassingEncoder:
    def test_one_row_per_atom_and_per_bond(self):
        out = _encoder()(_frame())
        assert out.atom.shape == (4, 8) and out.bond.shape == (3, 6)

    def test_reversing_a_bond_changes_nothing(self):
        encoder = _encoder()
        one, two = encoder(_frame()), encoder(_frame([(j, i) for i, j in BONDS]))
        torch.testing.assert_close(one.atom, two.atom)
        torch.testing.assert_close(one.bond, two.bond)

    def test_a_declared_channel_changes_the_features(self):
        channel = DiscreteChannel(name="bond_type", num_classes=5)
        encoder = _encoder(bond_channels=(channel,))
        frame = _frame()
        one = encoder(frame).bond
        frame["bonds", "bond_type"] = torch.tensor([1, 1, 1])
        assert not torch.allclose(encoder(frame).bond[2], one[2])

    def test_a_missing_declared_column_is_refused_by_name(self):
        encoder = _encoder(atom_channels=(DiscreteChannel(name="formal_charge", num_classes=5),))
        with pytest.raises(KeyError, match="formal_charge"):
            encoder(_frame())

    def test_a_padded_bond_row_stays_zero(self):
        frame = _frame()
        frame["bonds", "mask"] = torch.tensor([True, True, False])
        assert _encoder()(frame).bond[2].abs().max() == 0.0
