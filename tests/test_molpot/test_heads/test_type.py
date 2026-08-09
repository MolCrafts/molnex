"""Tests for :class:`molpot.heads.type.TypeHead`.

Moved from the stale ``tests/test_molpot/test_readout/`` mirror — the
production module is ``src/molpot/heads/type.py``, so the mirror is
``tests/test_molpot/test_heads/test_type.py``.
"""

from __future__ import annotations

import torch

from molpot.heads.type import TypeHead


class TestTypeHead:
    def test_forward_logits(self):
        head = TypeHead(hidden_dim=4, num_types=5)
        h = torch.ones(3, 4)
        out = head(h)
        assert out.shape == torch.Size([3, 5])

    def test_all_parameters_honour_the_fp64_precision(self, fp64):
        """Every parameter is fp64 when the head is built under fp64.

        ``config["ftype"]`` is the single source of truth for the working
        precision; a layer that ignores it leaves the head mixed-precision
        and its first forward dies on a dtype-mismatched matmul.
        """
        head = TypeHead(hidden_dim=4, num_types=5)

        assert {p.dtype for p in head.parameters()} == {torch.float64}

    def test_forward_runs_under_the_fp64_precision(self, fp64):
        """A head built at fp64 consumes fp64 features and emits fp64 logits."""
        head = TypeHead(hidden_dim=4, num_types=5)

        out = head(torch.ones(3, 4, dtype=torch.float64))

        assert out.shape == torch.Size([3, 5])
        assert out.dtype == torch.float64
