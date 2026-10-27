"""Tests for molzoo.mace.encoder — the spec-driven MACE backbone.

Each case pins one primitive at a time: the encoder exposes
``element_index`` / ``node_attrs`` / ``initial_node_features`` /
``angular_features`` / ``radial_features`` / ``conditioning`` /
``layer_features`` and the caller composes them; ``forward`` is that
composition on the batch schema, writing ``atoms.node_features``. The
symmetries of the features — translation and rotation invariance of the
scalar blocks, permutation equivariance — are asserted here, once, on the
primitives, and not again through ``forward`` or a product.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from tensordict import TensorDict

import molrep.eq as meq
from molix.compile import Compiler
from molzoo.mace.encoder import MACEEncoder
from tests.conftest import (
    permute_graph,
    rotate_graph,
    rotation_matrix_x,
    rotation_matrix_z,
    translate_graph,
)
from tests.test_molzoo.test_mace.conftest import (
    CLUSTER_Z,
    DTYPE,
    TINY_MATPES_KWARGS,
    TINY_OMOL_KWARGS,
)

#: Rigid translation applied by the invariance test (Å).
TRANSLATION = [0.31, -0.72, 1.13]

#: Rotation angles (rad) of the ``Rz(γ) @ Rx(β)`` test rotation.
ROTATION_Z = 0.7
ROTATION_X = 0.4

#: A fixed relabelling of the five cluster atoms (no RNG).
PERMUTATION = [3, 0, 4, 1, 2]

#: Scalar features are dimensionless activations; fp64 invariance holds to ~1e-13.
INVARIANCE_ATOL = 1e-10

#: ``Z = [8, 1, 6, 1]`` against the ascending table ``[1, 6, 8]``.
ONE_HOT_Z = [8, 1, 6, 1]
EXPECTED_ONE_HOT = [
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
]


def _rotation_matrix() -> torch.Tensor:
    """``Rz(0.7) @ Rx(0.4)`` in fp64 — a fixed, general SO(3) element."""
    return rotation_matrix_z(ROTATION_Z, torch.float64) @ rotation_matrix_x(
        ROTATION_X, torch.float64
    )


def _layer_inputs(encoder: MACEEncoder, batch: TensorDict) -> dict[str, Any]:
    """Compose the encoder primitives into ``layer_features`` keyword arguments."""
    Z = batch["atoms", "Z"]
    edge_index = batch["pairs", "pair_index"]
    vectors = batch["pairs", "pair_diff"]
    lengths = batch["pairs", "pair_dist"]

    elements = encoder.element_index(Z)
    edge_feats, cutoff = encoder.radial_features(lengths, Z, edge_index)
    return {
        "node_feats": encoder.initial_node_features(elements),
        "elements": elements,
        "edge_attrs": encoder.angular_features(vectors),
        "edge_feats": edge_feats,
        "edge_index": edge_index,
        "cutoff": cutoff,
    }


def _scalar_blocks(encoder: MACEEncoder, batch: TensorDict) -> torch.Tensor:
    """The ``0e`` block of every layer, ``(N, L, F)`` — the first ``F`` columns in ``ir_mul``."""
    n_features = TINY_MATPES_KWARGS["num_features"]
    per_layer = encoder.layer_features(**_layer_inputs(encoder, batch))
    return torch.stack([h[:, :n_features] for h in per_layer], dim=1)


@pytest.fixture
def matpes_encoder(tiny_matpes_spec) -> MACEEncoder:
    """``MACEEncoder`` on the tiny MatPES spec."""
    return MACEEncoder(tiny_matpes_spec, dtype=DTYPE).eval()


@pytest.fixture
def omol_encoder(tiny_omol_spec) -> MACEEncoder:
    """``MACEEncoder`` on the tiny OMOL spec."""
    return MACEEncoder(tiny_omol_spec, dtype=DTYPE).eval()


class TestMACEEncoder:
    """Test the configuration-driven MACE backbone."""

    # -- construction ----------------------------------------------------------

    def test_the_block_stack_follows_the_spec(self, matpes_encoder, tiny_off_spec):
        """MatPES: density blocks, a residual from the second layer on; OFF: the agnostic stack."""
        assert [type(b).__name__ for b in matpes_encoder.interactions] == [
            "DensityInteraction",
            "DensityResidualInteraction",
        ]
        assert [p.use_sc for p in matpes_encoder.products] == [False, True]
        off = MACEEncoder(tiny_off_spec, dtype=DTYPE)
        assert [type(b).__name__ for b in off.interactions] == [
            "AgnosticInteraction",
            "AgnosticResidualInteraction",
        ]
        assert all(b.density_fn is None for b in off.interactions)
        assert off.distance_transform is None

    # -- element table primitives --------------------------------------------

    def test_element_index_is_the_table_row(self, matpes_encoder):
        """``Z`` is looked up in the ascending table; ``index32`` is the same rows as int32."""
        elements = matpes_encoder.element_index(torch.tensor(ONE_HOT_Z, dtype=torch.long))
        expected = torch.tensor(EXPECTED_ONE_HOT, dtype=torch.float64).argmax(dim=-1)
        assert torch.equal(elements.index, expected)
        assert elements.index32.dtype == torch.int32
        assert torch.equal(elements.index32.long(), expected)

    def test_node_attrs_is_the_hard_coded_one_hot(self, matpes_encoder):
        """``Z`` is looked up in the ascending table and one-hot encoded."""
        attrs = matpes_encoder.node_attrs(torch.tensor(ONE_HOT_Z, dtype=torch.long), torch.float64)
        expected = torch.tensor(EXPECTED_ONE_HOT, dtype=torch.float64)
        assert torch.equal(attrs, expected)

    def test_node_attrs_honours_the_requested_dtype(self, matpes_encoder):
        """The dtype argument is the caller's, not the module's."""
        attrs = matpes_encoder.node_attrs(torch.tensor(ONE_HOT_Z, dtype=torch.long), torch.float32)
        assert attrs.dtype == torch.float32

    # -- conditioning ---------------------------------------------------------

    def test_conditioning_rejects_a_column_on_an_unconditioned_spec(self, matpes_encoder, cluster):
        """MatPES has no charge/spin embedding — a column would be silently ignored."""
        node_feats = torch.zeros(len(CLUSTER_Z), TINY_MATPES_KWARGS["num_features"])
        with pytest.raises(ValueError):
            matpes_encoder.conditioning(
                node_feats, torch.tensor([0]), None, cluster["atoms", "batch"], 1
            )
        assert matpes_encoder.conditioning(
            node_feats, None, None, cluster["atoms", "batch"], 1
        ) is (node_feats)

    def test_conditioning_defaults_to_the_neutral_singlet(self, omol_encoder, cluster):
        """On a conditioned spec a missing column is charge 0 / spin 1 (spin 0 is untrained)."""
        node_feats = torch.zeros(len(CLUSTER_Z), TINY_OMOL_KWARGS["num_features"])
        batch = cluster["atoms", "batch"]
        explicit = omol_encoder.conditioning(
            node_feats, torch.tensor([0]), torch.tensor([1]), batch, 1
        )
        defaulted = omol_encoder.conditioning(node_feats, None, None, batch, 1)
        assert explicit.shape == (len(CLUSTER_Z), TINY_OMOL_KWARGS["num_features"])
        assert torch.equal(explicit, defaulted)
        other = omol_encoder.conditioning(
            node_feats, torch.tensor([1]), torch.tensor([1]), batch, 1
        )
        assert not torch.equal(other, explicit)

    def test_forward_applies_the_conditioning_semantics(self, omol_encoder, cluster, omol_cluster):
        """``forward`` reads the columns through ``conditioning``: defaulted when absent."""
        bare = omol_encoder(cluster.clone())["atoms", "node_features"]
        explicit = omol_encoder(omol_cluster.clone())["atoms", "node_features"]
        assert torch.equal(bare, explicit)

    # -- layer_features -------------------------------------------------------

    def test_layer_features_is_one_block_per_interaction(self, matpes_encoder, cluster):
        """A sequence, not a stacked tensor — MatPES layer widths differ; fp64 rows per atom."""
        feats = matpes_encoder.layer_features(**_layer_inputs(matpes_encoder, cluster))
        assert [tuple(f.shape) for f in feats] == [
            (len(CLUSTER_Z), meq.Irreps("O3", irreps).dim) for irreps in matpes_encoder.layer_irreps
        ]
        assert all(f.dtype == torch.float64 for f in feats)

    def test_scalar_blocks_are_translation_invariant(self, matpes_encoder, cluster):
        """Every layer's ``0e`` block is unmoved by a rigid shift."""
        moved = translate_graph(cluster, torch.tensor(TRANSLATION, dtype=torch.float64))
        torch.testing.assert_close(
            _scalar_blocks(matpes_encoder, moved),
            _scalar_blocks(matpes_encoder, cluster),
            atol=INVARIANCE_ATOL,
            rtol=0.0,
        )

    def test_scalar_blocks_are_rotation_invariant_with_shifts(
        self, matpes_encoder, periodic_cluster
    ):
        """``S = n · h`` rotates with the box, so every layer's ``0e`` block stays put."""
        turned = rotate_graph(periodic_cluster, _rotation_matrix())
        torch.testing.assert_close(
            _scalar_blocks(matpes_encoder, turned),
            _scalar_blocks(matpes_encoder, periodic_cluster),
            atol=INVARIANCE_ATOL,
            rtol=0.0,
        )

    def test_scalar_blocks_are_permutation_equivariant(self, matpes_encoder, cluster):
        """Relabelling the atoms relabels the rows: ``f(Px) = P f(x)``.

        The element rows (``ElementIndex.index``) and the per-element maps
        built on them are what a wrong gather here would show up in.
        """
        perm = torch.tensor(PERMUTATION, dtype=torch.long)
        torch.testing.assert_close(
            _scalar_blocks(matpes_encoder, permute_graph(cluster, perm)),
            _scalar_blocks(matpes_encoder, cluster)[perm],
            atol=INVARIANCE_ATOL,
            rtol=0.0,
        )

    # -- forward: the encoder contract ----------------------------------------

    def test_forward_writes_scalar_features_per_layer(self, matpes_encoder, cluster):
        """``(N, L, F)``: each layer's ``0e`` block, i.e. its first ``F`` columns."""
        features = matpes_encoder(cluster.clone())["atoms", "node_features"]
        assert features.shape == (
            len(CLUSTER_Z),
            TINY_MATPES_KWARGS["num_interactions"],
            TINY_MATPES_KWARGS["num_features"],
        )
        torch.testing.assert_close(
            features, _scalar_blocks(matpes_encoder, cluster), atol=INVARIANCE_ATOL, rtol=0.0
        )

    def test_forward_runs_without_the_spec_attribute(self, matpes_encoder, cluster):
        """``_spec`` is provenance only — a pydantic attribute read on the hot path
        is a dynamo graph break — so every primitive must run after it is gone."""
        del matpes_encoder._spec
        features = matpes_encoder(cluster.clone())["atoms", "node_features"]
        assert features.shape[1] == TINY_MATPES_KWARGS["num_interactions"]

    @pytest.mark.slow
    def test_forward_compiles_to_one_graph(self, matpes_encoder, cluster):
        """The encoder is a hot path: 1 dynamo graph, 0 breaks after the eager gate."""
        matpes_encoder(cluster.clone())
        with torch.no_grad():
            explanation = Compiler.explain(matpes_encoder, cluster.clone())
        assert (explanation.graph_count, explanation.graph_break_count) == (1, 0), (
            explanation.break_sites
        )
