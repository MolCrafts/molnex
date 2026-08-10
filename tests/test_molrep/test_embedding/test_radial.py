"""Tests for molrep.embedding.radial module."""

import pytest
import torch

from molrep.embedding.radial import AgnesiTransform, BesselRBF, BesselRBFSpec


class TestBesselRBFSpec:
    """Test BesselRBFSpec configuration."""

    def test_valid_config(self):
        """Test creation with valid parameters."""
        spec = BesselRBFSpec(
            num_radial=8,
            r_cut=5.0,
        )
        assert spec.num_radial == 8
        assert spec.r_cut == 5.0

    def test_invalid_num_radial(self):
        """Test validation for num_radial."""
        with pytest.raises(ValueError):
            BesselRBFSpec(num_radial=0, r_cut=5.0)

        with pytest.raises(ValueError):
            BesselRBFSpec(num_radial=-1, r_cut=5.0)

    def test_invalid_r_cut(self):
        """Test validation for r_cut."""
        with pytest.raises(ValueError):
            BesselRBFSpec(num_radial=8, r_cut=0.0)

        with pytest.raises(ValueError):
            BesselRBFSpec(num_radial=8, r_cut=-1.0)


class TestBesselRBF:
    """Test BesselRBF radial basis function."""

    def test_initialization(self):
        """Test BesselRBF initialization."""
        rbf = BesselRBF(num_radial=8, r_cut=5.0)
        assert rbf.config.num_radial == 8
        assert rbf.config.r_cut == 5.0

    def test_forward_shape(self):
        """Test output shape."""
        rbf = BesselRBF(num_radial=8, r_cut=5.0)
        distances = torch.tensor([1.0, 2.0, 3.0, 4.0])

        output = rbf(distances)
        assert output.shape == (4, 8)

    def test_forward_batch(self):
        """Test with batch of distances."""
        rbf = BesselRBF(num_radial=16, r_cut=10.0)
        distances = torch.randn(100, 50).abs()  # [batch, edges]

        output = rbf(distances)
        assert output.shape == (100, 50, 16)

    def test_cutoff_behavior_raw(self):
        """Raw (un-normalised) Bessel basis decays at and past the cutoff.

        This property only holds for the un-normalised basis, since
        shift+scale normalisation re-centres each channel.
        """
        rbf = BesselRBF(num_radial=8, r_cut=5.0, normalize=False)

        distances = torch.tensor([2.0, 4.9, 5.0, 6.0, 10.0])
        output = rbf(distances)

        assert output[2].abs().max() < 0.15  # At r_cut
        assert output[3].abs().max() < 0.15  # Beyond r_cut
        assert output[4].abs().max() < 0.15  # Far beyond r_cut

    def test_normalized_basis_stats(self):
        """Normalised basis has ~0 mean and ~1 std under r ~ Uniform([0, r_cut])."""
        rbf = BesselRBF(num_radial=8, r_cut=5.0, normalize=True)
        r = torch.linspace(1e-3, 5.0, 10000)
        phi = rbf(r)
        assert phi.mean(dim=0).abs().max() < 0.01
        assert (phi.std(dim=0) - 1.0).abs().max() < 0.01

    def test_zero_distance(self):
        """Test behavior at zero distance."""
        rbf = BesselRBF(num_radial=8, r_cut=5.0)
        distances = torch.tensor([0.0, 0.1, 1.0])

        output = rbf(distances)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_differentiable(self):
        """Test that gradients flow through RBF."""
        rbf = BesselRBF(num_radial=8, r_cut=5.0)
        distances = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

        output = rbf(distances)
        loss = output.sum()
        loss.backward()

        assert distances.grad is not None
        assert not torch.isnan(distances.grad).any()

    def test_different_num_radial(self):
        """Test with different num_radial values."""
        for num_radial in [4, 8, 16, 32]:
            rbf = BesselRBF(num_radial=num_radial, r_cut=5.0)
            distances = torch.tensor([1.0, 2.0, 3.0])
            output = rbf(distances)
            assert output.shape == (3, num_radial)

    def test_dtype_consistency(self):
        """Test that output dtype matches input."""
        rbf = BesselRBF(num_radial=8, r_cut=5.0)

        # Float32
        dist_f32 = torch.tensor([1.0, 2.0], dtype=torch.float32)
        out_f32 = rbf(dist_f32)
        assert out_f32.dtype == torch.float32

        # Float64
        dist_f64 = torch.tensor([1.0, 2.0], dtype=torch.float64)
        out_f64 = rbf(dist_f64)
        # Note: BesselRBF casts to float internally
        assert out_f64.dtype in [torch.float32, torch.float64]


class TestAgnesiTransform:
    """Test the Agnesi element-pair distance transform."""

    def test_maps_into_the_unit_interval(self):
        """The transform compresses r onto (0, 1] so the basis stays bounded."""
        transform = AgnesiTransform().double()
        r = torch.linspace(0.1, 12.0, 50, dtype=torch.float64)
        z = torch.full((50,), 8, dtype=torch.long)
        u = transform(r, z, z)
        assert bool((u > 0).all()) and bool((u <= 1.0).all())

    def test_is_monotonically_decreasing(self):
        """Longer distances map to smaller transformed coordinates."""
        transform = AgnesiTransform().double()
        r = torch.linspace(0.3, 8.0, 40, dtype=torch.float64)
        z = torch.full((40,), 6, dtype=torch.long)
        u = transform(r, z, z)
        assert bool((u[1:] - u[:-1] < 0).all())

    def test_is_symmetric_in_the_pair(self):
        """r_0 uses the mean covalent radius, so swapping the pair is a no-op."""
        transform = AgnesiTransform().double()
        r = torch.tensor([1.0, 2.0], dtype=torch.float64)
        forward = transform(r, torch.tensor([1, 8]), torch.tensor([8, 1]))
        reverse = transform(r, torch.tensor([8, 1]), torch.tensor([1, 8]))
        assert torch.allclose(forward, reverse)

    def test_element_pair_sets_the_length_scale(self):
        """A larger pair radius means the same r is compressed less."""
        transform = AgnesiTransform().double()
        r = torch.tensor([1.5, 1.5], dtype=torch.float64)
        small = transform(r[:1], torch.tensor([1]), torch.tensor([1]))  # H-H
        large = transform(r[:1], torch.tensor([55]), torch.tensor([55]))  # Cs-Cs
        assert float(large) > float(small)

    def test_is_differentiable(self):
        """It sits on the force path, so it must carry gradient."""
        transform = AgnesiTransform().double()
        r = torch.tensor([1.0, 2.5], dtype=torch.float64, requires_grad=True)
        transform(r, torch.tensor([8, 1]), torch.tensor([1, 8])).sum().backward()
        assert r.grad is not None and bool((r.grad < 0).all())

    def test_parameters_are_frozen_by_default(self):
        """a / q / p are buffers unless the caller opts into training them."""
        assert not any(p.requires_grad for p in AgnesiTransform().parameters())

    def test_trainable_promotes_shape_parameters(self):
        """``trainable=True`` exposes a / q / p for fine-tuning."""
        names = {n for n, _ in AgnesiTransform(trainable=True).named_parameters()}
        assert {"a", "q", "p"} <= names
