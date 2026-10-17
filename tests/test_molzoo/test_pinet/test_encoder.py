"""Unit tests for molzoo.pinet.encoder.PiNetEncoder."""

from __future__ import annotations

import pytest
import torch

from molzoo.pinet import PiNetEncoder, PiNetSpec
from tests.conftest import (
    permute_graph,
    random_rotation_matrix,
    rotate_graph,
    rotate_vectors,
    translate_graph,
)
from tests.test_molzoo.test_pinet.conftest import DTYPE, TINY

ATOL = 1e-10


def _encoder(rank: int = 3, **overrides) -> PiNetEncoder:
    torch.manual_seed(0)
    return PiNetEncoder(PiNetSpec(**{**TINY, "rank": rank, **overrides}), dtype=DTYPE).eval()


def _rotate_p3(p3: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    n, layers, comps, channels = p3.shape
    flat = p3.permute(0, 1, 3, 2).reshape(-1, comps)
    return rotate_vectors(flat, r).reshape(n, layers, channels, comps).permute(0, 1, 3, 2)


class TestPiNetEncoder:
    @pytest.mark.parametrize("rank", [1, 3, 5])
    def test_written_shapes_follow_the_rank(self, rank, cluster):
        out = _encoder(rank)(cluster)
        assert out["atoms", "node_features"].shape == (4, 2, 8)
        assert out["atoms", "p1_block_outputs"].shape == (4, 2, 8)
        assert out["pairs", "i1_features"].shape == (12, 2, 8 * (rank // 2 + 1))
        assert ("p3_features" in out["atoms"].keys()) == (rank >= 3)
        assert ("p5_features" in out["atoms"].keys()) == (rank >= 5)
        if rank == 5:
            assert out["atoms", "p5_features"].shape == (4, 2, 5, 8)
            assert out["pairs", "i5_features"].shape == (12, 2, 5, 8)

    def test_emit_off_drops_only_the_property_tracks(self, cluster):
        lean = _encoder(5, emit_property_features=False)(cluster.clone())
        full = _encoder(5)(cluster.clone())
        for key in ("node_features", "p1_block_outputs"):
            assert torch.equal(lean["atoms", key], full["atoms", key])
        assert not {"p3_features", "p5_features"} & set(lean["atoms"].keys())
        assert not {"i1_features", "i3_features", "i5_features"} & set(lean["pairs"].keys())

    def test_translation_invariance(self, cluster):
        enc = _encoder(5)
        ref = enc(cluster.clone())["atoms", "node_features"]
        got = enc(translate_graph(cluster, torch.tensor([10.0, -3.0, 0.5], dtype=torch.float64)))
        torch.testing.assert_close(got["atoms", "node_features"], ref, atol=ATOL, rtol=0)

    def test_p3_rotation_equivariance(self, cluster):
        enc = _encoder(3)
        torch.manual_seed(2)
        r = random_rotation_matrix(dtype=torch.float64)
        ref = enc(cluster.clone())["atoms", "p3_features"]
        got = enc(rotate_graph(cluster, r))["atoms", "p3_features"]
        torch.testing.assert_close(got, _rotate_p3(ref, r), atol=ATOL, rtol=0)

    def test_permutation_equivariance(self, cluster):
        enc = _encoder(3)
        perm = torch.tensor([2, 0, 3, 1])
        ref = enc(cluster.clone())["atoms", "node_features"]
        got = enc(permute_graph(cluster, perm))["atoms", "node_features"]
        torch.testing.assert_close(got, ref[perm], atol=ATOL, rtol=0)

    def test_node_attrs_index_the_sorted_table(self):
        enc = _encoder(1)
        attrs = enc.node_attrs(torch.tensor([8, 1]), torch.float64)
        torch.testing.assert_close(
            attrs, torch.tensor([[0, 0, 0, 1], [1, 0, 0, 0]], dtype=torch.float64)
        )
