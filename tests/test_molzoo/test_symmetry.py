"""Symmetry test suite: translation, rotation, and permutation tests for encoders and pipelines.

Tests three physical symmetries that molecular models must satisfy:
1. Translation invariance — features/energy unchanged under rigid shifts
2. Rotation invariance/equivariance — scalar features/energy invariant, forces equivariant
3. Permutation equivariance — features permute with atom reordering

Generic graph-transform helpers live in ``tests.symmetry_helpers`` and are
reused by every symmetry test in this repo (encoder, head, pipeline).
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from molpot.derivation import EnergyAggregation, ForceDerivation
from molpot.heads import AtomicEnergyMLP
from molpot.pooling import EdgeToNodePooling, LayerPooling
from molrep.embedding.node import DiscreteEmbeddingSpec
from molrep.utils.equivariance import random_rotation_matrix, rotate_vectors
from molzoo import MACE, Allegro
from tests.symmetry_helpers import (
    make_graph_batch,
    permute_graph,
    recompute_edge_geometry,
    rotate_graph,
    translate_graph,
)

# ---------------------------------------------------------------------------
# Pipeline builder (encoder → energy → forces)
# ---------------------------------------------------------------------------


def make_pipeline(encoder, is_edge_encoder: bool = False):
    """Build an energy+force pipeline from an encoder.

    Computes edge_diff from pos inside the forward pass so the functorch force
    closure can trace gradients pos → geometry → energy for force derivation.
    """
    layer_pool = LayerPooling("mean")
    edge_to_node = EdgeToNodePooling("mean") if is_edge_encoder else None
    energy_mlp = AtomicEnergyMLP(hidden_dim=16)
    energy_agg = EnergyAggregation(pooling="sum")
    force_deriv = ForceDerivation()

    def _energy_and_feats(batch: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        edge_index = batch["edges", "edge_index"]
        pos = batch["atoms", "pos"]

        recompute_edge_geometry(batch)

        result = encoder(batch)

        if is_edge_encoder:
            ef = result["edges", "edge_features"]
            # Allegro emits a DenseNet stack of per-layer scalars in a single
            # flat ``(E, F·(L+1))`` tensor. Reshape into ``(E, L+1, F)`` so the
            # pipeline's ``LayerPooling("mean")`` reduces over the layer axis
            # and yields the same per-edge feature dim ``F`` the downstream
            # ``AtomicEnergyMLP(hidden_dim=F)`` expects.
            F_dim = encoder.num_scalar_features
            ef = ef.view(ef.shape[0], -1, F_dim)
            feats = layer_pool(ef)
            node_feats = edge_to_node(feats, edge_index, num_nodes=pos.shape[0])
        else:
            node_feats = layer_pool(result["atoms", "node_features"])

        atom_energy = energy_mlp(node_feats)
        atom_batch = batch["atoms", "batch"]
        mol_energy = energy_agg(atom_energy, atom_batch, num_graphs=int(atom_batch.max()) + 1)
        return mol_energy, node_feats

    def forward(batch: TensorDict):
        pos = batch["atoms", "pos"].detach()

        # Eager pass for energy + features (using the original geometry).
        b0 = batch.clone()
        b0["atoms", "pos"] = pos
        mol_energy, node_feats = _energy_and_feats(b0)

        # Forces via functorch: F = -∂E/∂pos, energy traced as a pure
        # function of positions (edge geometry recomputed inside).
        def energy_of_pos(p: torch.Tensor) -> torch.Tensor:
            b = batch.clone()
            b["atoms", "pos"] = p
            return _energy_and_feats(b)[0].sum()

        forces = force_deriv(energy_of_pos, pos)

        return mol_energy, forces, node_feats

    return forward


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_molecule():
    """5-atom chain with 8 edges, 2 molecules (3+2)."""
    torch.manual_seed(42)
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.3, 0.0],
            [2.5, 0.0, 0.1],
            [4.0, 0.5, 0.0],
            [5.3, 0.2, 0.1],
        ],
        dtype=torch.float32,
    )
    Z = torch.tensor([6, 1, 8, 6, 1], dtype=torch.long)
    edge_index = torch.tensor(
        [
            [0, 1],
            [1, 0],
            [1, 2],
            [2, 1],
            [2, 3],
            [3, 2],
            [3, 4],
            [4, 3],
        ],
        dtype=torch.long,
    )
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    return make_graph_batch(pos, Z, edge_index, batch)


@pytest.fixture
def mace_encoder():
    torch.manual_seed(0)
    encoder = MACE(
        node_attr_specs=[DiscreteEmbeddingSpec(input_key="Z", num_classes=10, emb_dim=16)],
        num_elements=10,
        num_features=16,
        r_max=8.0,
        num_interactions=2,
    )
    encoder.eval()
    return encoder


@pytest.fixture
def allegro_encoder():
    torch.manual_seed(0)
    encoder = Allegro(
        num_elements=10,
        num_scalar_features=16,
        num_tensor_features=8,
        r_max=8.0,
        num_layers=2,
        type_embed_dim=16,
        latent_mlp_depth=1,
        latent_mlp_width=16,
        avg_num_neighbors=4.0,
    )
    encoder.eval()
    return encoder


SEEDS = [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Translation Invariance
# ---------------------------------------------------------------------------


class TestTranslationInvariance:
    """Shifting all atoms by a constant vector must not change features or energy."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_mace_encoder(self, mace_encoder, small_molecule, seed):
        torch.manual_seed(seed)
        t = torch.randn(3) * 10.0

        with torch.no_grad():
            ref = mace_encoder(small_molecule.clone())["atoms", "node_features"]
            shifted = mace_encoder(translate_graph(small_molecule, t))["atoms", "node_features"]

        assert torch.allclose(ref, shifted, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_allegro_encoder(self, allegro_encoder, small_molecule, seed):
        torch.manual_seed(seed)
        t = torch.randn(3) * 10.0

        with torch.no_grad():
            ref = allegro_encoder(small_molecule.clone())["edges", "edge_features"]
            shifted = allegro_encoder(translate_graph(small_molecule, t))["edges", "edge_features"]

        # Tolerance is float32-ULP — `(pos+t)[j]-(pos+t)[i]` is not
        # bit-exactly `pos[j]-pos[i]` for `t ≈ 10*randn`, so the encoder
        # input `edge_diff` differs by ~ULP and propagates linearly.
        assert torch.allclose(ref, shifted, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_mace_pipeline_energy_and_forces(self, mace_encoder, small_molecule, seed):
        torch.manual_seed(seed)
        t = torch.randn(3) * 10.0
        pipeline = make_pipeline(mace_encoder, is_edge_encoder=False)

        batch_ref = small_molecule.clone()
        batch_ref["atoms", "pos"] = batch_ref["atoms", "pos"].clone().requires_grad_(True)
        e_ref, f_ref, _ = pipeline(batch_ref)

        batch_t = translate_graph(small_molecule, t)
        batch_t["atoms", "pos"] = batch_t["atoms", "pos"].clone().requires_grad_(True)
        e_t, f_t, _ = pipeline(batch_t)

        assert torch.allclose(e_ref, e_t, atol=1e-5, rtol=1e-5)
        assert torch.allclose(f_ref, f_t, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_allegro_pipeline_energy_and_forces(self, allegro_encoder, small_molecule, seed):
        torch.manual_seed(seed)
        t = torch.randn(3) * 10.0
        pipeline = make_pipeline(allegro_encoder, is_edge_encoder=True)

        batch_ref = small_molecule.clone()
        batch_ref["atoms", "pos"] = batch_ref["atoms", "pos"].clone().requires_grad_(True)
        e_ref, f_ref, _ = pipeline(batch_ref)

        batch_t = translate_graph(small_molecule, t)
        batch_t["atoms", "pos"] = batch_t["atoms", "pos"].clone().requires_grad_(True)
        e_t, f_t, _ = pipeline(batch_t)

        assert torch.allclose(e_ref, e_t, atol=1e-5, rtol=1e-5)
        assert torch.allclose(f_ref, f_t, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Rotation Invariance / Equivariance
# ---------------------------------------------------------------------------


class TestRotationEquivariance:
    """Scalar features and energy are rotation-invariant. Forces are rotation-equivariant.

    The MACE encoder propagates a *pure-scalar* node state between layers and
    keeps mixed-l components only transiently inside each tensor-product message
    (contracted back to invariants by the ProductHead). Rotation invariance is
    therefore exact to float precision on both CPU and GPU — these tests were
    previously xfailed under a misdiagnosis ("missing cuequivariance_ops_torch
    GPU kernel"); the real cause was non-equivariant scalar->mixed-l plain
    linear layers in the encoder. See molrep/readout/product.py and
    molzoo/mace.py for the fix.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_mace_encoder_scalar_invariance(self, mace_encoder, small_molecule, seed):
        torch.manual_seed(seed)
        R = random_rotation_matrix()

        with torch.no_grad():
            ref = mace_encoder(small_molecule.clone())["atoms", "node_features"]
            rotated = mace_encoder(rotate_graph(small_molecule, R))["atoms", "node_features"]

        assert torch.allclose(ref, rotated, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_allegro_encoder_scalar_invariance(self, allegro_encoder, small_molecule, seed):
        torch.manual_seed(seed)
        R = random_rotation_matrix()

        with torch.no_grad():
            ref = allegro_encoder(small_molecule.clone())["edges", "edge_features"]
            rotated = allegro_encoder(rotate_graph(small_molecule, R))["edges", "edge_features"]

        assert torch.allclose(ref, rotated, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_mace_pipeline_energy_invariance(self, mace_encoder, small_molecule, seed):
        torch.manual_seed(seed)
        R = random_rotation_matrix()
        pipeline = make_pipeline(mace_encoder, is_edge_encoder=False)

        batch_ref = small_molecule.clone()
        batch_ref["atoms", "pos"] = batch_ref["atoms", "pos"].clone().requires_grad_(True)
        e_ref, _, _ = pipeline(batch_ref)

        batch_r = rotate_graph(small_molecule, R)
        batch_r["atoms", "pos"] = batch_r["atoms", "pos"].clone().requires_grad_(True)
        e_r, _, _ = pipeline(batch_r)

        assert torch.allclose(e_ref, e_r, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_allegro_pipeline_energy_invariance(self, allegro_encoder, small_molecule, seed):
        torch.manual_seed(seed)
        R = random_rotation_matrix()
        pipeline = make_pipeline(allegro_encoder, is_edge_encoder=True)

        batch_ref = small_molecule.clone()
        batch_ref["atoms", "pos"] = batch_ref["atoms", "pos"].clone().requires_grad_(True)
        e_ref, _, _ = pipeline(batch_ref)

        batch_r = rotate_graph(small_molecule, R)
        batch_r["atoms", "pos"] = batch_r["atoms", "pos"].clone().requires_grad_(True)
        e_r, _, _ = pipeline(batch_r)

        assert torch.allclose(e_ref, e_r, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_mace_pipeline_force_equivariance(self, mace_encoder, small_molecule, seed):
        """F(Rx) = R @ F(x)"""
        torch.manual_seed(seed)
        R = random_rotation_matrix()
        pipeline = make_pipeline(mace_encoder, is_edge_encoder=False)

        batch_ref = small_molecule.clone()
        batch_ref["atoms", "pos"] = batch_ref["atoms", "pos"].clone().requires_grad_(True)
        _, f_ref, _ = pipeline(batch_ref)

        batch_r = rotate_graph(small_molecule, R)
        batch_r["atoms", "pos"] = batch_r["atoms", "pos"].clone().requires_grad_(True)
        _, f_r, _ = pipeline(batch_r)

        f_ref_rotated = rotate_vectors(f_ref, R)
        assert torch.allclose(f_ref_rotated, f_r, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_allegro_pipeline_force_equivariance(self, allegro_encoder, small_molecule, seed):
        """F(Rx) = R @ F(x)"""
        torch.manual_seed(seed)
        R = random_rotation_matrix()
        pipeline = make_pipeline(allegro_encoder, is_edge_encoder=True)

        batch_ref = small_molecule.clone()
        batch_ref["atoms", "pos"] = batch_ref["atoms", "pos"].clone().requires_grad_(True)
        _, f_ref, _ = pipeline(batch_ref)

        batch_r = rotate_graph(small_molecule, R)
        batch_r["atoms", "pos"] = batch_r["atoms", "pos"].clone().requires_grad_(True)
        _, f_r, _ = pipeline(batch_r)

        f_ref_rotated = rotate_vectors(f_ref, R)
        assert torch.allclose(f_ref_rotated, f_r, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Permutation Equivariance
# ---------------------------------------------------------------------------


class TestPermutationEquivariance:
    """Reordering atoms must reorder outputs correspondingly."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_mace_encoder_node_feature_permutation(self, mace_encoder, small_molecule, seed):
        torch.manual_seed(seed)
        n = small_molecule["atoms", "Z"].shape[0]
        perm = torch.randperm(n)

        with torch.no_grad():
            ref = mace_encoder(small_molecule.clone())["atoms", "node_features"]
            permuted = mace_encoder(permute_graph(small_molecule, perm))["atoms", "node_features"]

        # f(perm(x)) should equal perm(f(x))
        assert torch.allclose(ref[perm], permuted, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_allegro_encoder_edge_feature_permutation(self, allegro_encoder, small_molecule, seed):
        """After permuting atoms, edge features for the same atom pairs must match."""
        torch.manual_seed(seed)
        n = small_molecule["atoms", "Z"].shape[0]
        perm = torch.randperm(n)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(n)

        with torch.no_grad():
            ref_batch = small_molecule.clone()
            ref_result = allegro_encoder(ref_batch)
            ref_features = ref_result["edges", "edge_features"]
            ref_edges = ref_batch["edges", "edge_index"]

            perm_batch = permute_graph(small_molecule, perm)
            perm_result = allegro_encoder(perm_batch)
            perm_features = perm_result["edges", "edge_features"]
            perm_edges = perm_batch["edges", "edge_index"]

        # Build lookup: (src, dst) -> feature for both
        ref_dict = {}
        for i in range(ref_edges.shape[0]):
            key = (ref_edges[i, 0].item(), ref_edges[i, 1].item())
            ref_dict[key] = ref_features[i]

        for i in range(perm_edges.shape[0]):
            # Map permuted indices back to original
            src_orig = perm[perm_edges[i, 0]].item()
            dst_orig = perm[perm_edges[i, 1]].item()
            ref_feat = ref_dict[(src_orig, dst_orig)]
            assert torch.allclose(ref_feat, perm_features[i], atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_mace_pipeline_energy_permutation_invariance(self, mace_encoder, small_molecule, seed):
        torch.manual_seed(seed)
        n = small_molecule["atoms", "Z"].shape[0]
        perm = torch.randperm(n)
        pipeline = make_pipeline(mace_encoder, is_edge_encoder=False)

        batch_ref = small_molecule.clone()
        batch_ref["atoms", "pos"] = batch_ref["atoms", "pos"].clone().requires_grad_(True)
        e_ref, _, _ = pipeline(batch_ref)

        batch_p = permute_graph(small_molecule, perm)
        batch_p["atoms", "pos"] = batch_p["atoms", "pos"].clone().requires_grad_(True)
        e_p, _, _ = pipeline(batch_p)

        assert torch.allclose(e_ref, e_p, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_allegro_pipeline_energy_permutation_invariance(
        self, allegro_encoder, small_molecule, seed
    ):
        torch.manual_seed(seed)
        n = small_molecule["atoms", "Z"].shape[0]
        perm = torch.randperm(n)
        pipeline = make_pipeline(allegro_encoder, is_edge_encoder=True)

        batch_ref = small_molecule.clone()
        batch_ref["atoms", "pos"] = batch_ref["atoms", "pos"].clone().requires_grad_(True)
        e_ref, _, _ = pipeline(batch_ref)

        batch_p = permute_graph(small_molecule, perm)
        batch_p["atoms", "pos"] = batch_p["atoms", "pos"].clone().requires_grad_(True)
        e_p, _, _ = pipeline(batch_p)

        assert torch.allclose(e_ref, e_p, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_mace_pipeline_force_permutation_equivariance(self, mace_encoder, small_molecule, seed):
        """F(perm(x))[i] = F(x)[perm[i]]"""
        torch.manual_seed(seed)
        n = small_molecule["atoms", "Z"].shape[0]
        perm = torch.randperm(n)
        pipeline = make_pipeline(mace_encoder, is_edge_encoder=False)

        batch_ref = small_molecule.clone()
        batch_ref["atoms", "pos"] = batch_ref["atoms", "pos"].clone().requires_grad_(True)
        _, f_ref, _ = pipeline(batch_ref)

        batch_p = permute_graph(small_molecule, perm)
        batch_p["atoms", "pos"] = batch_p["atoms", "pos"].clone().requires_grad_(True)
        _, f_p, _ = pipeline(batch_p)

        assert torch.allclose(f_ref[perm], f_p, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_allegro_pipeline_force_permutation_equivariance(
        self, allegro_encoder, small_molecule, seed
    ):
        """F(perm(x))[i] = F(x)[perm[i]]"""
        torch.manual_seed(seed)
        n = small_molecule["atoms", "Z"].shape[0]
        perm = torch.randperm(n)
        pipeline = make_pipeline(allegro_encoder, is_edge_encoder=True)

        batch_ref = small_molecule.clone()
        batch_ref["atoms", "pos"] = batch_ref["atoms", "pos"].clone().requires_grad_(True)
        _, f_ref, _ = pipeline(batch_ref)

        batch_p = permute_graph(small_molecule, perm)
        batch_p["atoms", "pos"] = batch_p["atoms", "pos"].clone().requires_grad_(True)
        _, f_p, _ = pipeline(batch_p)

        assert torch.allclose(f_ref[perm], f_p, atol=1e-5, rtol=1e-5)
