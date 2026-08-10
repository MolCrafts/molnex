"""Unit tests for molzoo.pinet.spec.PiNetSpec."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from molzoo.pinet import PiNet, PiNetSpec


def test_defaults_build_encoder():
    spec = PiNetSpec()
    enc = PiNet.from_spec(spec)
    assert enc.depth == spec.depth
    assert enc.feature_dim == spec.ii_nodes[-1]


def test_rejects_bad_rank():
    with pytest.raises(ValidationError):
        PiNetSpec(rank=2)
