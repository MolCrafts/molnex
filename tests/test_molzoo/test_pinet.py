"""Tests for the PiNet encoder."""

from __future__ import annotations

import pytest
import torch

from molrep.embedding.cutoff import CosineCutoff, HalfCosineCutoff, TanhCutoff
from molrep.embedding.radial import GaussianBasis, PolynomialBasis
from molrep.interaction.pinet import DotLayer, PIXLayer
from molrep.utils.equivariance import random_rotation_matrix, rotate_vectors
from molzoo import PiNet
from tests.symmetry_helpers import (
    make_graph_batch,
    permute_graph,
    rotate_graph,
    translate_graph,
)


def _graph():
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.1, 0.1, 0.0],
            [0.3, 1.2, 0.2],
            [1.4, 1.1, -0.1],
        ],
        dtype=torch.float32,
    )
    z = torch.tensor([1, 6, 7, 8], dtype=torch.long)
    edge_index = torch.tensor(
        [[0, 1], [1, 0], [0, 2], [2, 0], [1, 3], [3, 1], [2, 3], [3, 2]],
        dtype=torch.long,
    )
    batch = torch.zeros(4, dtype=torch.long)
    return make_graph_batch(pos, z, edge_index, batch)


def _encoder(rank: int = 3, weighted: bool = False) -> PiNet:
    torch.manual_seed(0)
    enc = PiNet(
        atom_types=[1, 6, 7, 8],
        r_max=4.0,
        n_basis=3,
        pp_nodes=[8, 8],
        pi_nodes=[8, 8],
        ii_nodes=[8, 8],
        depth=2,
        rank=rank,
        weighted=weighted,
    )
    enc.eval()
    return enc


def _rotate_p3(p3: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    n, layers, comps, channels = p3.shape
    flat = p3.permute(0, 1, 3, 2).reshape(-1, comps)
    return rotate_vectors(flat, r).reshape(n, layers, channels, comps).permute(0, 1, 3, 2)


class TestBasis:
    def test_cutoff_types_shape_and_boundary(self):
        r = torch.tensor([0.0, 1.0, 4.0, 5.0])
        _cutoffs = {
            "f1": CosineCutoff,
            "f2": TanhCutoff,
            "hip": HalfCosineCutoff,
        }
        for cutoff_type, cls in _cutoffs.items():
            out = cls(r_cut=4.0)(r)
            assert out.shape == r.shape
            assert out[-1] == 0.0

    def test_polynomial_basis(self):
        fc = torch.tensor([0.5, 0.25])
        basis = PolynomialBasis(3)(torch.ones_like(fc), fc=fc)
        torch.testing.assert_close(basis, torch.stack([fc, fc**2, fc**3], dim=1))

    def test_gaussian_basis(self):
        r = torch.tensor([0.0, 1.0])
        fc = torch.ones_like(r)
        basis = GaussianBasis(r_cut=2.0, n_basis=4, gamma=3.0)(r, fc=fc)
        assert basis.shape == (2, 4)


class TestPiNetLayers:
    def test_pix_unweighted_gathers_target_property(self):
        px = torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2)
        edge_index = torch.tensor([[0, 1], [2, 3]])
        src, dst = edge_index[:, 0], edge_index[:, 1]
        out = PIXLayer(channels=2, weighted=False)(src, dst, px)
        torch.testing.assert_close(out, px[dst])

    def test_dot_weighted_shape(self):
        x = torch.randn(5, 3, 4)
        out = DotLayer(channels=4, weighted=True)(x)
        assert out.shape == (5, 4)


class TestPiNetEncoder:
    @pytest.mark.parametrize("rank", [1, 3, 5])
    def test_rank_output_shapes(self, rank):
        enc = _encoder(rank=rank)
        out = enc(_graph())
        assert out["atoms", "node_features"].shape == (4, 2, 8)
        assert out["edges", "i1_features"].shape == (8, 2, 8 * (rank // 2 + 1))
        assert ("atoms", "p3_features") in out.keys(include_nested=True) or rank == 1
        if rank >= 3:
            assert out["atoms", "p3_features"].shape == (4, 2, 3, 8)
            assert out["edges", "i3_features"].shape == (8, 2, 3, 8)
        if rank >= 5:
            assert out["atoms", "p5_features"].shape == (4, 2, 5, 8)
            assert out["edges", "i5_features"].shape == (8, 2, 5, 8)

    def test_translation_invariance(self):
        enc = _encoder(rank=5)
        g = _graph()
        shifted = translate_graph(g, torch.tensor([10.0, -3.0, 0.5]))
        with torch.no_grad():
            ref = enc(g.clone())["atoms", "node_features"]
            got = enc(shifted)["atoms", "node_features"]
        torch.testing.assert_close(ref, got, atol=1e-5, rtol=1e-5)

    def test_p3_rotation_equivariance(self):
        enc = _encoder(rank=3)
        g = _graph()
        torch.manual_seed(2)
        r = random_rotation_matrix()
        with torch.no_grad():
            ref = enc(g.clone())["atoms", "p3_features"]
            got = enc(rotate_graph(g, r))["atoms", "p3_features"]
        torch.testing.assert_close(_rotate_p3(ref, r), got, atol=1e-4, rtol=1e-4)

    def test_permutation_equivariance(self):
        enc = _encoder(rank=3)
        g = _graph()
        perm = torch.tensor([2, 0, 3, 1], dtype=torch.long)
        with torch.no_grad():
            ref = enc(g.clone())["atoms", "node_features"]
            got = enc(permute_graph(g, perm))["atoms", "node_features"]
        torch.testing.assert_close(ref[perm], got, atol=1e-5, rtol=1e-5)
