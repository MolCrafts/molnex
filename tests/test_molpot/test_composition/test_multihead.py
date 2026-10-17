""":class:`molpot.composition.multihead.MultiHead` — merging head outputs.

Fixtures are local rather than in ``conftest.py``: a ``batch`` fixture there
would be injected into every test in this directory, and ``test_sonata.py``
already has tests taking that argument name.
"""

from __future__ import annotations

import pytest
import torch

from molpot.composition.heads import (
    ChargeHead,
    ChargeTransferParameterHead,
    RepulsionParameterHead,
    TSScalingHead,
)
from molpot.composition.multihead import MultiHead


@pytest.fixture
def node_features():
    return torch.randn(5, 16)


@pytest.fixture
def batch():
    return torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)


class TestMultiHead:
    def test_merges_outputs(self, node_features, batch):
        multi = MultiHead(
            {
                "rep": RepulsionParameterHead(feature_dim=16, dtype=torch.float32),
                "ct": ChargeTransferParameterHead(feature_dim=16, dtype=torch.float32),
                "charge": ChargeHead(feature_dim=16, dtype=torch.float32),
            }
        )
        out = multi(node_features, batch=batch)
        assert "eps_rep" in out
        assert "lam_rep" in out
        assert "eps_ct" in out
        assert "lam_ct" in out
        assert "charge" in out

    def test_duplicate_key_raises(self, node_features):
        multi = MultiHead(
            {
                "a": RepulsionParameterHead(feature_dim=16, dtype=torch.float32),
                "b": RepulsionParameterHead(feature_dim=16, dtype=torch.float32),
            }
        )
        with pytest.raises(ValueError, match="Duplicate key"):
            multi(node_features)

    def test_empty_heads_raises(self):
        with pytest.raises(ValueError, match="at least one head"):
            MultiHead({})

    def test_with_ts_head(self, node_features):
        Z = torch.tensor([1, 6, 8, 1, 6], dtype=torch.long)
        multi = MultiHead(
            {
                "ts": TSScalingHead(
                    feature_dim=16,
                    c6_free=torch.rand(10) * 10,
                    alpha_free=torch.rand(10) * 5,
                    r_star_free=torch.rand(10) * 2 + 1.0,
                    dtype=torch.float32,
                ),
            }
        )
        out = multi(node_features, Z=Z)
        assert "c6" in out
        assert "alpha" in out
        assert "r_star" in out
