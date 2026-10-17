"""Unit tests for :mod:`molrep.cueq`: operations pinned to a kernel method."""

from __future__ import annotations

import cuequivariance as cue
import cuequivariance_torch as cuet

from molrep import cueq

IRREPS = cue.Irreps("O3", "4x0e+4x1o")


class TestLinear:
    def test_defaults_to_the_fused_kernel(self):
        """``cuet.Linear`` alone picks ``naive``; the molnex op picks ``fused_tp``."""
        assert cueq.Linear(IRREPS, IRREPS, layout=cue.ir_mul).method == "fused_tp"

    def test_keeps_the_parent_state_dict_keys(self):
        """No parameter or buffer of its own: checkpoints load by the parent's keys."""
        ours = cueq.Linear(IRREPS, IRREPS, layout=cue.ir_mul)
        parent = cuet.Linear(IRREPS, IRREPS, layout=cue.ir_mul, method="fused_tp")
        assert isinstance(ours, cuet.Linear)
        assert set(ours.state_dict()) == set(parent.state_dict())


class TestChannelWiseTensorProduct:
    def test_defaults_to_the_uniform_1d_kernel(self):
        tp = cueq.ChannelWiseTensorProduct(
            IRREPS, cue.Irreps("O3", "0e+1o"), layout=cue.ir_mul, shared_weights=False
        )
        assert tp.method == "uniform_1d"
