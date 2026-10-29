"""Unit tests for molzoo.pinet.spec."""

from __future__ import annotations

import ast
import inspect

import pytest
from pydantic import ValidationError

import molzoo.pinet.spec as spec_module
from molzoo.pinet import PiNetSpec


class TestPiNetSpec:
    def test_rejects_an_unknown_field(self):
        with pytest.raises(ValidationError):
            PiNetSpec(atom_types=[1, 6])
        with pytest.raises(ValidationError):
            PiNetSpec(atomic_energies=[0.0])

    def test_spec_module_imports_no_torch(self):
        """The spec is a validator only; it imports pydantic and the torch-free encoder spec."""
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
        modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        assert roots <= {"__future__", "typing", "pydantic", "molrep"}
        assert {m for m in modules if m and m.startswith("molrep")} == {
            "molrep.encoders.pinet.spec"
        }
