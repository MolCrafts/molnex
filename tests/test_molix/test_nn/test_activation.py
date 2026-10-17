"""Unit tests for molix.nn.activation."""

from __future__ import annotations

import pytest
import torch.nn as nn

from molix.nn.activation import activation_from_name


class TestActivationFromName:
    def test_names_resolve_case_insensitively(self):
        assert activation_from_name("Tanh") is nn.Tanh
        assert activation_from_name("swish") is nn.SiLU

    def test_a_module_class_passes_through(self):
        assert activation_from_name(nn.ELU) is nn.ELU

    def test_none_means_no_activation(self):
        assert activation_from_name(None) is None

    def test_an_unknown_name_raises(self):
        with pytest.raises(ValueError, match="unsupported activation"):
            activation_from_name("sigmoidish")
