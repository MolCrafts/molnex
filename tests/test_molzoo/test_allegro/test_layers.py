"""Unit tests for molzoo.allegro.layers."""

from __future__ import annotations

import cuequivariance as cue
import pytest
import torch

from molix import config
from molix.F.scatter import scatter_sum_compile_safe
from molzoo.allegro.layers import Contracter, ProductTypeEmbedding, WeightedChannels

TYPE_EDGES = torch.tensor([[0, 1], [1, 0]])


class TestProductTypeEmbedding:
    def test_centre_and_neighbour_tables_gate_the_basis(self):
        layer = ProductTypeEmbedding(num_types=2, num_basis=3, dim=4)
        types = torch.tensor([0, 1])
        basis = torch.randn(2, 3, dtype=config.ftype)
        expected = torch.cat(
            [layer.center_embed.weight[[0, 1]], layer.neighbor_embed.weight[[1, 0]]], dim=-1
        ) * layer.basis_linear(basis)
        torch.testing.assert_close(layer(types, TYPE_EDGES, basis), expected)

    def test_rejects_an_odd_width(self):
        with pytest.raises(ValueError, match="even"):
            ProductTypeEmbedding(num_types=2, num_basis=3, dim=5)


class TestWeightedChannels:
    def test_weights_are_channel_major_and_output_ir_mul(self):
        """``w`` reads ``(u, R)``: irrep ``r`` of channel ``c`` is scaled by ``w[c, r]``."""
        layer = WeightedChannels(irrep_dims=[1, 3], channels=2)
        basis = torch.arange(1.0, 5.0, dtype=config.ftype).unsqueeze(0)  # (1, D=4)
        weights = torch.tensor([[10.0, 20.0, 30.0, 40.0]], dtype=config.ftype)  # (u=2, R=2)
        out = layer(basis, weights).view(1, 4, 2)  # (E, D, u)
        expected = torch.tensor(
            [[[10.0, 30.0], [40.0, 80.0], [60.0, 120.0], [80.0, 160.0]]], dtype=config.ftype
        )
        torch.testing.assert_close(out, expected)

    def test_env_fuse_matches_weighted_scatter(self):
        """``index`` sums the per-pair rows into ``num_rows`` buckets (no edge message)."""
        torch.manual_seed(0)
        layer = WeightedChannels(irrep_dims=[1, 3, 5], channels=4)
        basis = torch.randn(20, 9, dtype=config.ftype)
        weights = torch.randn(20, layer.weight_numel, dtype=config.ftype)
        source = torch.randint(0, 5, (20,))
        fused = layer(basis, weights, index=source, num_rows=5)
        expected = scatter_sum_compile_safe(layer(basis, weights), source, 5)
        torch.testing.assert_close(fused, expected, atol=1e-12, rtol=0)

    def test_index_without_rows_is_rejected(self):
        layer = WeightedChannels(irrep_dims=[1], channels=1)
        with pytest.raises(ValueError, match="num_rows"):
            layer(
                torch.ones(1, 1, dtype=config.ftype),
                torch.ones(1, 1, dtype=config.ftype),
                index=torch.tensor([0]),
            )


SH = cue.Irreps("O3", "1x0e+1x1o+1x2e")


U = 3


def _contracter(out: str = "1x0e+1x1o+1x2e") -> Contracter:
    torch.manual_seed(0)
    return Contracter(irreps_in1=SH, irreps_in2=SH, irreps_out=cue.Irreps("O3", out), channels=U)


class TestContracter:
    def test_weights_are_channel_by_path(self):
        """The upstream ``(u, paths)`` layout the package reader normalises onto."""
        tp = _contracter()
        assert tuple(tp.weights.shape) == (U, tp.num_paths)

    def test_scalar_output_of_scalar_inputs_is_a_weighted_product(self):
        tp = _contracter("1x0e")
        x1 = torch.zeros(1, 9 * U, dtype=config.ftype)
        x2 = torch.zeros(1, 9 * U, dtype=config.ftype)
        x1[0, :U] = torch.tensor([1.0, 2.0, 3.0])
        x2[0, :U] = torch.tensor([4.0, 5.0, 6.0])
        out = tp(x1, x2, torch.tensor([0]))
        # path (0e, 0e, 0e) is path 0 (enumeration order out, in1, in2); C = 1
        expected = tp.weights[:, 0] * torch.tensor([4.0, 10.0, 18.0], dtype=config.ftype)
        torch.testing.assert_close(out[0], expected)

    def test_gathers_the_atom_operand_by_index(self):
        tp = _contracter()
        x1 = torch.randn(2, 9 * U, dtype=config.ftype)
        x2 = torch.randn(3, 9 * U, dtype=config.ftype)
        torch.testing.assert_close(
            tp(x1, x2, torch.tensor([2, 0])), tp(x1, x2[[2, 0]], torch.tensor([0, 1]))
        )

    def test_no_path_raises(self):
        with pytest.raises(ValueError, match="no tensor-product path"):
            Contracter(
                irreps_in1=cue.Irreps("O3", "1x0e"),
                irreps_in2=cue.Irreps("O3", "1x0e"),
                irreps_out=cue.Irreps("O3", "1x1o"),
                channels=U,
            )
