"""Unit tests for molzoo.pinet.layers."""

from __future__ import annotations

import pytest
import torch

from molix import config
from molzoo.pinet.layers import (
    DotLayer,
    EquivarLayer,
    GCBlock,
    InvarLayer,
    IPLayer,
    OutLayer,
    PILayer,
    PIXLayer,
    ResUpdate,
    ScaleLayer,
    compute_d5,
)
from tests.conftest import random_rotation_matrix

CART_EDGES = torch.tensor([[0, 1], [2, 3]])


class TestComputeD5:
    def test_hand_values_for_the_z_axis(self):
        d5 = compute_d5(torch.tensor([[0.0, 0.0, 1.0]]))
        torch.testing.assert_close(d5, torch.tensor([[-1 / 3, -1 / 3, 0.0, 0.0, 0.0]]))

    def test_first_two_components_are_traceless(self):
        d3 = torch.nn.functional.normalize(torch.randn(6, 3, dtype=torch.float64), dim=-1)
        d5 = compute_d5(d3)
        x2, y2, z2 = d3.square().unbind(-1)
        torch.testing.assert_close(-(d5[:, 0] + d5[:, 1]), (2 * z2 - x2 - y2) / 3)


class TestPIXLayer:
    def test_unweighted_takes_the_target_property(self):
        px = torch.arange(24.0).reshape(4, 3, 2)
        torch.testing.assert_close(PIXLayer(channels=2)(CART_EDGES, px), px[CART_EDGES[:, 1]])

    def test_weighted_mixes_both_endpoints(self):
        layer = PIXLayer(channels=2, weighted=True)
        px = torch.randn(4, 3, 2, dtype=layer.wi.weight.dtype)
        expected = layer.wi(px[CART_EDGES[:, 0]]) + layer.wj(px[CART_EDGES[:, 1]])
        torch.testing.assert_close(layer(CART_EDGES, px), expected)


class TestScaleLayer:
    def test_scales_every_component_of_a_channel(self):
        px = torch.ones(2, 3, 2)
        out = ScaleLayer()(px, torch.tensor([[2.0, 3.0], [4.0, 5.0]]))
        torch.testing.assert_close(out[:, 1], torch.tensor([[2.0, 3.0], [4.0, 5.0]]))


class TestDotLayer:
    def test_unweighted_is_the_squared_norm(self):
        x = torch.randn(3, 3, 2)
        torch.testing.assert_close(DotLayer(channels=2)(x), (x * x).sum(1))

    def test_weighted_output_is_rotation_invariant(self):
        layer = DotLayer(channels=4, weighted=True)
        x = torch.randn(5, 3, 4, dtype=layer.wi.weight.dtype)
        rot = random_rotation_matrix(dtype=x.dtype)
        torch.testing.assert_close(layer(torch.einsum("ij,njc->nic", rot, x)), layer(x))


PAIR_EDGES = torch.tensor([[0, 1], [1, 0], [0, 2], [2, 1]])


class TestPILayer:
    def test_contracts_the_weights_with_the_basis(self):
        layer = PILayer([3], in_dim=4, n_basis=2)
        prop = torch.randn(3, 2, dtype=config.ftype)
        basis = torch.randn(4, 2, dtype=config.ftype)
        weights = layer.ff_layer(torch.cat([prop[PAIR_EDGES[:, 0]], prop[PAIR_EDGES[:, 1]]], -1))
        expected = torch.einsum("eob,eb->eo", weights.reshape(4, 3, 2), basis)
        torch.testing.assert_close(layer(PAIR_EDGES, prop, basis), expected)

    def test_a_vanishing_basis_gives_no_interaction(self):
        layer = PILayer([5], in_dim=4, n_basis=3)
        prop = torch.randn(3, 2, dtype=config.ftype)
        out = layer(PAIR_EDGES, prop, torch.zeros(4, 3, dtype=config.ftype))
        assert torch.count_nonzero(out) == 0


class TestIPLayer:
    def test_sums_onto_the_source_atom(self):
        inter = torch.tensor([[1.0], [10.0], [100.0], [1000.0]])
        out = IPLayer()(PAIR_EDGES, inter, 4)
        torch.testing.assert_close(out, torch.tensor([[101.0], [10.0], [1000.0], [0.0]]))


class TestResUpdate:
    def test_equal_widths_add(self):
        old, new = torch.randn(3, 4), torch.randn(3, 4)
        torch.testing.assert_close(ResUpdate(in_dim=4, out_dim=4)(old, new), old + new)

    def test_unequal_widths_project_the_old_state(self):
        layer = ResUpdate(in_dim=2, out_dim=5)
        old = torch.randn(3, 2, dtype=layer.transform.weight.dtype)
        new = torch.zeros(3, 5, dtype=old.dtype)
        torch.testing.assert_close(layer(old, new), layer.transform(old))


GC_EDGES = torch.tensor([[0, 1], [1, 0], [0, 2], [2, 0], [1, 3], [3, 1]])


N_ATOMS, N_EDGES, C = 4, 6, 8


def _inputs(rank: int, generator_seed: int = 0):
    torch.manual_seed(generator_seed)
    ftype = config.ftype
    d3 = torch.nn.functional.normalize(torch.randn(N_EDGES, 3, dtype=ftype), dim=-1)
    return {
        "edge_index": GC_EDGES,
        "basis": torch.randn(N_EDGES, 3, dtype=ftype),
        "p1": torch.randn(N_ATOMS, 4, dtype=ftype),
        "p3": torch.randn(N_ATOMS, 3, 1, dtype=ftype) if rank >= 3 else None,
        "d3": d3 if rank >= 3 else None,
        "p5": torch.randn(N_ATOMS, 5, 1, dtype=ftype) if rank >= 5 else None,
        "d5": compute_d5(d3) if rank >= 5 else None,
    }


def _block(rank: int, weighted: bool = False) -> GCBlock:
    return GCBlock(
        rank=rank,
        weighted=weighted,
        pp_nodes=[C, C],
        pi_nodes=[C, C],
        ii_nodes=[C, C],
        n_basis=3,
        p1_in_dim=4,
        activation="tanh",
    )


class TestInvarLayer:
    def test_shapes(self):
        layer = InvarLayer(p1_in_dim=4, pp_nodes=[C], pi_nodes=[C], ii_nodes=[6], n_basis=3)
        x = _inputs(1)
        p1, i1 = layer(GC_EDGES, x["p1"], x["basis"])
        assert p1.shape == (N_ATOMS, C) and i1.shape == (N_EDGES, 6)


class TestEquivarLayer:
    def test_vector_output_rotates_with_the_input(self):
        layer = EquivarLayer(in_channels=1, out_channels=C, weighted=True)
        x = _inputs(3)
        gate = torch.randn(N_EDGES, C, dtype=config.ftype)
        rot = random_rotation_matrix(dtype=config.ftype)
        px, _, dotted = layer(GC_EDGES, x["p3"], gate, x["d3"])
        px_r, _, dotted_r = layer(
            GC_EDGES, torch.einsum("ij,njc->nic", rot, x["p3"]), gate, x["d3"] @ rot.T
        )
        torch.testing.assert_close(px_r, torch.einsum("ij,njc->nic", rot, px))
        torch.testing.assert_close(dotted_r, dotted)


class TestGCBlock:
    @pytest.mark.parametrize("rank", [1, 3, 5])
    def test_tracks_present_exactly_up_to_the_rank(self, rank):
        x = _inputs(rank)
        out = _block(rank)(**x)
        assert out.p1.shape == (N_ATOMS, C)
        assert out.i1.shape == (N_EDGES, C * (rank // 2 + 1))
        assert (out.p3 is None) == (rank < 3) and (out.i3 is None) == (rank < 3)
        assert (out.p5 is None) == (rank < 5) and (out.i5 is None) == (rank < 5)
        if rank >= 3:
            assert out.p3.shape == (N_ATOMS, 3, C)
        if rank == 5:
            assert out.p5.shape == (N_ATOMS, 5, C)

    def test_rejects_an_unknown_rank(self):
        with pytest.raises(ValueError, match="rank"):
            _block(2)


class TestOutLayer:
    def test_adds_onto_the_running_output(self):
        head = OutLayer([8], in_dim=4)
        px = torch.randn(3, 4, dtype=config.ftype)
        prev = torch.randn(3, 1, dtype=config.ftype)
        torch.testing.assert_close(head(px, prev) - head(px, torch.zeros_like(prev)), prev)

    def test_final_projection_has_no_bias(self):
        assert OutLayer([8], in_dim=4).out_units.bias is None
