"""Unit tests for :mod:`molrep.interaction.radial`.

Only the first-layer split contract lives here: the radial basis and envelope
behaviour is covered where their consumers are tested
(``test_mace/test_block.py``, ``test_polar/test_field.py``).
"""

import pytest
import torch

from molrep.interaction.radial import RadialMLP


class TestRadialMLPFirstLayerSplit:
    """Evaluating layer 0 from a partition of its inputs is the same function.

    The MACE-family interactions rely on this to avoid concatenating
    ``[edge_feats, source_emb[src], target_emb[tgt]]`` per edge — 1056 channels
    at Polar-1-M widths, which is both a multi-GiB tensor and the source of an
    Inductor ``int32`` index constant that overflows above ~2·10⁶ edges.
    """

    def test_split_parts_plus_bias_reproduce_forward(self):
        torch.manual_seed(0)
        mlp = RadialMLP([12, 16, 5]).double()
        x = torch.randn(7, 12, dtype=torch.float64)
        widths = [4, 5, 3]
        weights = mlp.first_layer_parts(widths)
        hidden = sum(part @ w for part, w in zip(x.split(widths, dim=-1), weights, strict=True))
        bias = mlp.first_layer_bias()
        assert bias is not None
        got = mlp.forward_from_first(hidden + bias)
        torch.testing.assert_close(got, mlp(x), atol=1e-12, rtol=0)

    def test_single_layer_network_needs_no_activation(self):
        """``forward_from_first`` on a one-layer net is the identity."""
        mlp = RadialMLP([6, 3]).double()
        hidden = torch.randn(4, 3, dtype=torch.float64)
        torch.testing.assert_close(mlp.forward_from_first(hidden), hidden, atol=0, rtol=0)

    def test_widths_must_partition_the_input(self):
        with pytest.raises(ValueError, match="do not partition"):
            RadialMLP([12, 16, 5]).first_layer_parts([4, 4])
