"""Tests for :class:`molpot.heads.type.TypeHead`.

Moved from the stale ``tests/test_molpot/test_readout/`` mirror — the
production module is ``src/molpot/heads/type.py``, so the mirror is
``tests/test_molpot/test_heads/test_type.py``.
"""

from __future__ import annotations

import torch

from molpot.heads.type import TypeHead

#: Floating dtype every module under test is built at.
DTYPE = torch.float32


class TestTypeHead:
    def test_forward_logits(self):
        head = TypeHead(hidden_dim=4, num_types=5, dtype=DTYPE)
        h = torch.ones(3, 4)
        out = head(h)
        assert out.shape == torch.Size([3, 5])

    def test_all_parameters_are_built_at_the_given_dtype(self):
        """Every parameter is fp64 when the head is built at fp64.

        A layer that ignored the constructor dtype would leave the head
        mixed-precision, and its first forward would die on a dtype-mismatched
        matmul.
        """
        head = TypeHead(hidden_dim=4, num_types=5, dtype=torch.float64)

        assert {p.dtype for p in head.parameters()} == {torch.float64}

    def test_forward_runs_at_the_given_dtype(self):
        """A head built at fp64 consumes fp64 features and emits fp64 logits."""
        head = TypeHead(hidden_dim=4, num_types=5, dtype=torch.float64)

        out = head(torch.ones(3, 4, dtype=torch.float64))

        assert out.shape == torch.Size([3, 5])
        assert out.dtype == torch.float64

    def test_decode_with_confidence(self):
        head = TypeHead(hidden_dim=4, num_types=3, dtype=DTYPE)
        logits = torch.tensor([[5.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        indices, confidence = head.decode_with_confidence(logits)
        assert indices.tolist() == [0, 1]
        assert float(confidence[0]) > float(confidence[1])
