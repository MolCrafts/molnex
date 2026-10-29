"""Unit tests for molrep.encoders.allegro.encoder.AllegroEncoder.

The symmetries of ``pairs.edge_features`` — translation and rotation
invariance of the scalar track, permutation equivariance of the rows — are
asserted here, once, and not again through the potential.
"""

from __future__ import annotations

import pytest
import torch

from molrep.encoders.allegro import AllegroEncoder, AllegroEncoderSpec
from tests.conftest import (
    permute_graph,
    rotate_graph,
    rotation_matrix_x,
    rotation_matrix_z,
    translate_graph,
)
from tests.test_molrep.test_encoders.test_allegro.conftest import DTYPE, TINY

#: Rigid translation (Å), ``Rz(γ) @ Rx(β)`` angles (rad) and a relabelling of
#: the five cluster atoms — all fixed, no RNG.
TRANSLATION = [0.31, -0.72, 1.13]
ROTATION_Z = 0.7
ROTATION_X = 0.4
PERMUTATION = [3, 0, 4, 1, 2]

#: Scalar features are dimensionless activations; fp64 invariance holds to ~1e-13.
INVARIANCE_ATOL = 1e-10


def _encoder(**overrides) -> AllegroEncoder:
    torch.manual_seed(0)
    return AllegroEncoder(AllegroEncoderSpec(**{**TINY, **overrides}), dtype=DTYPE)


def _edge_features(encoder: AllegroEncoder, batch) -> torch.Tensor:
    with torch.no_grad():
        return encoder(batch.clone())["pairs", "edge_features"]


class TestAllegroEncoder:
    @pytest.mark.parametrize(
        ("l_max", "num_layers", "parity", "paths"),
        # Contracter.num_paths of upstream AllegroModel builds (nequip-allegro 0.8.3).
        [
            (2, 2, True, [11, 3]),
            (2, 3, True, [15, 15, 3]),
            (2, 3, False, [11, 11, 3]),
            (1, 2, True, [4, 2]),
            (1, 3, True, [5, 5, 2]),
            (3, 4, False, [23, 23, 23, 4]),
            (3, 3, True, [34, 34, 4]),
        ],
    )
    def test_tensor_product_paths_match_upstream(self, l_max, num_layers, parity, paths):
        enc = _encoder(l_max=l_max, num_layers=num_layers, parity=parity)
        assert [tp.num_paths for tp in enc.tps] == paths

    def test_writes_the_densenet_stack(self, cluster):
        enc = _encoder()
        out = enc(cluster)["pairs", "edge_features"]
        assert out.shape == (20, enc.output_dim) and enc.output_dim == 8 * 3

    def test_exposed_tensor_track_matches_its_irreps(self, cluster):
        enc = _encoder(expose_tensor_track=True)
        out = enc(cluster)["pairs", "edge_tensor_features"]
        assert out.shape == (20, enc.tensor_track_irreps.dim)

    def test_types_follow_the_table_order(self):
        enc = _encoder(atomic_numbers=[8, 1, 6])
        assert enc.node_types(torch.tensor([1, 6, 8])).tolist() == [1, 2, 0]

    def test_edge_features_are_translation_invariant(self, cluster):
        enc = _encoder()
        moved = translate_graph(cluster, torch.tensor(TRANSLATION, dtype=torch.float64))
        torch.testing.assert_close(
            _edge_features(enc, moved), _edge_features(enc, cluster), atol=INVARIANCE_ATOL, rtol=0.0
        )

    def test_edge_features_are_rotation_invariant(self, cluster):
        """The written track is the scalar one: a rotation of the system cannot move it."""
        enc = _encoder()
        rotation = rotation_matrix_z(ROTATION_Z, torch.float64) @ rotation_matrix_x(
            ROTATION_X, torch.float64
        )
        torch.testing.assert_close(
            _edge_features(enc, rotate_graph(cluster, rotation)),
            _edge_features(enc, cluster),
            atol=INVARIANCE_ATOL,
            rtol=0.0,
        )

    def test_edge_features_are_permutation_equivariant(self, cluster):
        """Relabelling the atoms keeps every edge row's features: the pair counts, not the label."""
        enc = _encoder()
        perm = torch.tensor(PERMUTATION, dtype=torch.long)
        torch.testing.assert_close(
            _edge_features(enc, permute_graph(cluster, perm)),
            _edge_features(enc, cluster),
            atol=INVARIANCE_ATOL,
            rtol=0.0,
        )
