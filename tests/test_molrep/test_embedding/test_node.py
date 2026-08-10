"""Tests for molrep.embedding.node module."""

import pytest
import torch

from molrep.embedding.node import (
    ContinuousEmbeddingSpec,
    DiscreteEmbeddingSpec,
    JointEmbedding,
    JointEmbeddingSpec,
    JointFeatureEmbedding,
    JointFeatureSpec,
)


class TestDiscreteEmbeddingSpec:
    """Test DiscreteEmbeddingSpec configuration."""

    def test_valid_spec(self):
        """Test creation with valid parameters."""
        spec = DiscreteEmbeddingSpec(input_key="Z", num_classes=100, emb_dim=64)
        assert spec.input_key == "Z"
        assert spec.num_classes == 100
        assert spec.emb_dim == 64

    def test_invalid_num_classes(self):
        """Test validation for num_classes."""
        with pytest.raises(ValueError):
            DiscreteEmbeddingSpec(input_key="Z", num_classes=0, emb_dim=64)

    def test_invalid_emb_dim(self):
        """Test validation for emb_dim."""
        with pytest.raises(ValueError):
            DiscreteEmbeddingSpec(input_key="Z", num_classes=100, emb_dim=0)


class TestContinuousEmbeddingSpec:
    """Test ContinuousEmbeddingSpec configuration."""

    def test_valid_spec(self):
        """Test creation with valid parameters."""
        spec = ContinuousEmbeddingSpec(input_key="pos", in_dim=3, emb_dim=32, use_bias=False)
        assert spec.input_key == "pos"
        assert spec.in_dim == 3
        assert spec.emb_dim == 32
        assert spec.use_bias is False

    def test_invalid_in_dim(self):
        """Test validation for in_dim."""
        with pytest.raises(ValueError):
            ContinuousEmbeddingSpec(input_key="pos", in_dim=0, emb_dim=32)


class TestJointEmbeddingSpec:
    """Test JointEmbeddingSpec configuration."""

    def test_valid_spec(self):
        """Test creation with valid parameters."""
        d_spec = DiscreteEmbeddingSpec(input_key="Z", num_classes=10, emb_dim=16)
        c_spec = ContinuousEmbeddingSpec(input_key="attr", in_dim=5, emb_dim=16)
        j_spec = JointEmbeddingSpec(specs=[d_spec, c_spec], out_dim=32, output_key="node_feats")

        assert j_spec.out_dim == 32
        assert j_spec.output_key == "node_feats"

    def test_duplicate_keys(self):
        """Test validation for duplicate keys."""
        d_spec1 = DiscreteEmbeddingSpec(input_key="Z", num_classes=10, emb_dim=16)
        d_spec2 = DiscreteEmbeddingSpec(input_key="Z", num_classes=5, emb_dim=16)
        with pytest.raises(ValueError, match="specs contain duplicate keys"):
            JointEmbeddingSpec(specs=[d_spec1, d_spec2], out_dim=32, output_key="node_feats")


class TestJointEmbedding:
    """Test JointEmbedding module."""

    def test_initialization(self):
        """Test JointEmbedding initialization and parameter setup."""
        specs = [
            DiscreteEmbeddingSpec(input_key="Z", num_classes=10, emb_dim=16),
            ContinuousEmbeddingSpec(input_key="pos", in_dim=3, emb_dim=32),
        ]
        joint = JointEmbedding(embedding_specs=specs, out_dim=64)

        assert len(joint.embedders) == 2
        assert isinstance(joint.embedders[0], torch.nn.Embedding)
        assert isinstance(joint.embedders[1], torch.nn.Sequential)

    def test_forward_pass(self):
        """Test forward pass correctness and output shape."""
        specs = [
            DiscreteEmbeddingSpec(input_key="Z", num_classes=10, emb_dim=16),
            ContinuousEmbeddingSpec(input_key="pos", in_dim=3, emb_dim=32),
        ]
        out_dim = 64
        joint = JointEmbedding(embedding_specs=specs, out_dim=out_dim)

        n_atoms = 5
        z = torch.randint(0, 10, (n_atoms,))
        pos = torch.randn(n_atoms, 3)

        output = joint(Z=z, pos=pos)
        assert isinstance(output, torch.Tensor)
        assert output.shape == (n_atoms, out_dim)

    def test_empty_specs_error(self):
        """Test error when no specs are provided."""
        with pytest.raises(ValueError, match="No feature embeddings configured."):
            JointEmbedding(embedding_specs=[], out_dim=64)

    def test_gradient_flow(self):
        """Test that gradients flow through the joint embedding."""
        specs = [ContinuousEmbeddingSpec(input_key="attr", in_dim=5, emb_dim=16)]
        joint = JointEmbedding(embedding_specs=specs, out_dim=32)

        attr = torch.randn(5, 5, requires_grad=True)

        output = joint(attr=attr)
        loss = output.sum()
        loss.backward()

        assert attr.grad is not None
        assert not torch.isnan(attr.grad).any()

    @pytest.mark.parametrize(
        "spec_kinds",
        [("discrete",), ("continuous",), ("discrete", "continuous")],
        ids=["discrete", "continuous", "both"],
    )
    def test_all_parameters_honour_the_fp64_precision(self, fp64, spec_kinds):
        """Every parameter is fp64 when the module is built under fp64.

        ``config["ftype"]`` is the single source of truth for the working
        precision; a layer that ignores it leaves the module mixed-precision
        and its first forward dies on a dtype-mismatched matmul.
        """
        available = {
            "discrete": DiscreteEmbeddingSpec(input_key="Z", num_classes=10, emb_dim=16),
            "continuous": ContinuousEmbeddingSpec(input_key="pos", in_dim=3, emb_dim=16),
        }
        joint = JointEmbedding(
            embedding_specs=[available[kind] for kind in spec_kinds],
            out_dim=32,
        )

        assert {p.dtype for p in joint.parameters()} == {torch.float64}

    def test_forward_runs_under_the_fp64_precision(self, fp64):
        """A module built at fp64 consumes fp64 inputs and emits fp64."""
        specs = [
            DiscreteEmbeddingSpec(input_key="Z", num_classes=10, emb_dim=16),
            ContinuousEmbeddingSpec(input_key="pos", in_dim=3, emb_dim=32),
        ]
        joint = JointEmbedding(embedding_specs=specs, out_dim=64)

        n_atoms = 5
        z = torch.zeros(n_atoms, dtype=torch.long)
        pos = torch.zeros(n_atoms, 3, dtype=torch.float64)

        output = joint(Z=z, pos=pos)
        assert output.shape == (n_atoms, 64)
        assert output.dtype == torch.float64


class TestJointFeatureSpec:
    """Test JointFeatureSpec configuration."""

    def test_valid_categorical_spec(self):
        """A categorical feature carries its vocabulary size."""
        spec = JointFeatureSpec(
            name="total_charge", kind="categorical", emb_dim=4, num_classes=5, offset=2
        )
        assert spec.name == "total_charge"
        assert spec.kind == "categorical"
        assert spec.num_classes == 5
        assert spec.offset == 2
        assert spec.per == "graph"

    def test_valid_continuous_spec(self):
        """A continuous feature carries its input dimension."""
        spec = JointFeatureSpec(name="total_spin", kind="continuous", emb_dim=4, in_dim=1)
        assert spec.kind == "continuous"
        assert spec.in_dim == 1
        assert spec.use_bias is True

    def test_invalid_emb_dim(self):
        """Test validation for emb_dim."""
        with pytest.raises(ValueError):
            JointFeatureSpec(name="total_spin", kind="continuous", emb_dim=0, in_dim=1)


class TestJointFeatureEmbedding:
    """Test JointFeatureEmbedding module."""

    #: One spec per ``kind``; ``per="graph"`` matches the OMOL charge/spin use.
    SPECS = {
        "categorical": JointFeatureSpec(
            name="total_charge", kind="categorical", emb_dim=4, num_classes=5, offset=2
        ),
        "continuous": JointFeatureSpec(name="total_spin", kind="continuous", emb_dim=4, in_dim=1),
    }

    @pytest.mark.parametrize(
        "spec_kinds",
        [("categorical",), ("continuous",), ("categorical", "continuous")],
        ids=["categorical", "continuous", "both"],
    )
    def test_all_parameters_honour_the_fp64_precision(self, fp64, spec_kinds):
        """Every parameter is fp64 when the module is built under fp64."""
        embedding = JointFeatureEmbedding(
            feature_specs=[self.SPECS[kind] for kind in spec_kinds],
            out_dim=8,
        )

        assert {p.dtype for p in embedding.parameters()} == {torch.float64}

    def test_forward_runs_under_the_fp64_precision(self, fp64):
        """A module built at fp64 fuses fp64 per-graph features into per-atom ones."""
        embedding = JointFeatureEmbedding(
            feature_specs=[self.SPECS["categorical"], self.SPECS["continuous"]],
            out_dim=8,
        )

        n_atoms = 3
        output = embedding(
            batch=torch.zeros(n_atoms, dtype=torch.long),
            total_charge=torch.zeros(1, dtype=torch.long),
            total_spin=torch.ones(1, dtype=torch.float64),
        )

        assert output.shape == (n_atoms, 8)
        assert output.dtype == torch.float64

    def test_empty_specs_error(self):
        """Test error when no feature specs are provided."""
        with pytest.raises(ValueError, match="feature_specs must be non-empty."):
            JointFeatureEmbedding(feature_specs=[], out_dim=8)
