"""Unit tests for molrep.encoders.pinet.spec."""

from __future__ import annotations

import ast
import inspect

import pytest
from pydantic import ValidationError

import molrep.encoders.pinet.spec as spec_module
from molrep.encoders.pinet import PiNetEncoderSpec


class TestPiNetEncoderSpec:
    def test_rejects_an_unknown_rank(self):
        with pytest.raises(ValidationError):
            PiNetEncoderSpec(rank=2)

    def test_rejects_an_unsorted_element_table(self):
        with pytest.raises(ValidationError, match="ascending"):
            PiNetEncoderSpec(atomic_numbers=[6, 1])

    def test_rejects_mismatched_gate_widths(self):
        with pytest.raises(ValidationError, match="pp_nodes"):
            PiNetEncoderSpec(pp_nodes=[8, 8], ii_nodes=[8, 4])

    def test_spec_module_imports_no_torch(self):
        """The spec is a validator only; building one must not pull in torch."""
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
