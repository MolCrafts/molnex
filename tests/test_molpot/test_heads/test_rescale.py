"""Tests for ``molpot.heads.GlobalRescale`` and ``PerSpeciesScaleShift``."""

from __future__ import annotations

import pytest
import torch

from molpot.heads import GlobalRescale, PerSpeciesScaleShift


class TestGlobalRescale:
    def test_identity_default(self):
        m = GlobalRescale()
        x = torch.randn(5)
        torch.testing.assert_close(m(x), x)

    def test_affine(self):
        m = GlobalRescale(scale=2.0, shift=1.0)
        x = torch.tensor([-1.0, 0.0, 1.0])
        torch.testing.assert_close(m(x), torch.tensor([-1.0, 1.0, 3.0]))

    def test_buffers_not_trainable(self):
        m = GlobalRescale(scale=2.0, shift=0.5)
        params = list(m.parameters())
        assert params == []
        buffers = {name for name, _ in m.named_buffers()}
        assert buffers == {"scale", "shift"}


class TestPerSpeciesScaleShift:
    def test_shapes_and_defaults(self):
        m = PerSpeciesScaleShift(num_species=5)
        x = torch.tensor([1.0, 2.0, 3.0])
        Z = torch.tensor([0, 1, 2])
        # defaults: scale=1, shift=0 → identity
        torch.testing.assert_close(m(x, Z), x)

    def test_explicit_shift_scale(self):
        scales = [1.0, 2.0, 0.5]
        shifts = [0.0, 1.0, -1.0]
        m = PerSpeciesScaleShift(num_species=3, scales=scales, shifts=shifts)
        x = torch.tensor([10.0, 10.0, 10.0])
        Z = torch.tensor([0, 1, 2])
        expected = torch.tensor([10.0, 21.0, 4.0])
        torch.testing.assert_close(m(x, Z), expected)

    def test_trainable_flag(self):
        m = PerSpeciesScaleShift(
            num_species=3,
            scales=[1.0, 1.0, 1.0],
            shifts=[0.0, 0.0, 0.0],
            trainable=True,
        )
        assert isinstance(m.scales, torch.nn.Parameter)
        assert isinstance(m.shifts, torch.nn.Parameter)

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="length 5"):
            PerSpeciesScaleShift(num_species=5, scales=[1.0, 2.0])
