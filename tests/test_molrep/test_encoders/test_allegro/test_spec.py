"""Unit tests for molrep.encoders.allegro.spec."""

from __future__ import annotations

import ast
import inspect

import pytest
from pydantic import ValidationError

import molrep.encoders.allegro.spec as spec_module
from molrep.encoders.allegro import AllegroEncoderSpec


class TestAllegroEncoderSpec:
    def test_rejects_duplicate_atomic_numbers(self):
        with pytest.raises(ValidationError, match="unique"):
            AllegroEncoderSpec(atomic_numbers=[1, 1], r_max=4.0)

    def test_the_embedding_width_defaults_to_the_scalar_width(self):
        spec = AllegroEncoderSpec(atomic_numbers=[1], r_max=4.0, num_scalar_features=12)
        assert spec.embed_dim == 12

    def test_spec_module_imports_no_torch(self):
        tree = ast.parse(inspect.getsource(spec_module))
        roots = {
            (n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
        }
        roots |= {
            a.name.split(".")[0]
            for n in ast.walk(tree)
            if isinstance(n, ast.Import)
            for a in n.names
        }
        assert roots <= {"__future__", "typing", "pydantic"}
