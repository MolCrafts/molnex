"""Tests for MultiLayerFeatureMixer."""

from __future__ import annotations

import cuequivariance as cue
import cuequivariance_torch as cuet
import pytest
import torch

from molix import config
from molrep.interaction.polar.mixer import MultiLayerFeatureMixer


@pytest.fixture(autouse=True)
def fp64():
    previous = config["ftype"]
    config.set_precision("fp64")
    yield
    config.set_precision("fp64" if previous == torch.float64 else "fp32")


class TestMultiLayerFeatureMixer:
    """Test the Polar per-layer feature mixer."""

    def test_forward_is_sum_of_per_layer_linears(self):
        """``(2, N, dim)`` mixes to ``(N, dim)`` as ``W0 h0 + W1 h1``."""
        torch.manual_seed(0)
        irreps = "4x0e+4x1o"
        dim = cue.Irreps("O3", irreps).dim
        mixer = MultiLayerFeatureMixer(irreps=irreps, num_layers=2)
        stacked = torch.randn(2, 3, dim, dtype=torch.float64)
        out = mixer(stacked)
        expect = mixer.linears[0](stacked[0]) + mixer.linears[1](stacked[1])
        assert out.shape == (3, dim)
        assert torch.allclose(out, expect)
        assert isinstance(mixer.linears[0], cuet.Linear)

    def test_stack_length_must_match_num_layers(self):
        """A silent truncate would mix the wrong Polar layers."""
        irreps = "4x0e"
        mixer = MultiLayerFeatureMixer(irreps=irreps, num_layers=2)
        stacked = torch.zeros(3, 1, cue.Irreps("O3", irreps).dim, dtype=torch.float64)
        with pytest.raises(ValueError, match="3 layers.*num_layers=2"):
            mixer(stacked)

    def test_rejects_zero_layers(self):
        """``num_layers >= 1`` — Polar-1-M dump is 2."""
        with pytest.raises(ValueError, match="num_layers"):
            MultiLayerFeatureMixer(irreps="4x0e", num_layers=0)

    def test_polar_one_m_hidden_dim_literal(self):
        """Dump hidden ``512x0e+512x1o`` is dim 2048."""
        assert cue.Irreps("O3", "512x0e+512x1o").dim == 2048
