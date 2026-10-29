"""RondoSpec: encoder and heads, nothing about the functional form or a training stage."""

from __future__ import annotations

import pytest

from molzoo.rondo.spec import validate


class TestValidate:
    def test_each_setting_lands_on_its_own_spec(self):
        spec = validate(width=64, n_layers=2, bonded_hidden=32, hidden=16)
        assert (spec.encoder.width, spec.encoder.n_layers) == (64, 2)
        assert (spec.heads.bonded_hidden, spec.heads.hidden) == (32, 16)

    def test_an_unknown_setting_is_refused_by_name(self):
        with pytest.raises(TypeError, match="n_blocks"):
            validate(n_blocks=3)
