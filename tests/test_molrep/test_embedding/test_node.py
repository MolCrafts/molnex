"""Tests for molrep.embedding.node module."""

import pytest
import torch
from molrep.embedding.node import (
    DiscreteEmbeddingSpec,
    ContinuousEmbeddingSpec,
    JointEmbeddingSpec,
    JointEmbedding,
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
            ContinuousEmbeddingSpec(input_key="pos", in_dim=3, emb_dim=32)
        ]
        joint = JointEmbedding(embedding_specs=specs, out_dim=64)
        
        assert len(joint.embedders) == 2
        assert isinstance(joint.embedders[0], torch.nn.Embedding)
        assert isinstance(joint.embedders[1], torch.nn.Sequential)

    def test_forward_pass(self):
        """Test forward pass correctness and output shape."""
        specs = [
            DiscreteEmbeddingSpec(input_key="Z", num_classes=10, emb_dim=16),
            ContinuousEmbeddingSpec(input_key="pos", in_dim=3, emb_dim=32)
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
        specs = [
            ContinuousEmbeddingSpec(input_key="attr", in_dim=5, emb_dim=16)
        ]
        joint = JointEmbedding(embedding_specs=specs, out_dim=32)
        
        attr = torch.randn(5, 5, requires_grad=True)
        
        output = joint(attr=attr)
        loss = output.sum()
        loss.backward()
        
        assert attr.grad is not None
        assert not torch.isnan(attr.grad).any()
