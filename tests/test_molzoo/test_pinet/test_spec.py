"""Unit tests for molzoo.pinet.spec."""

from __future__ import annotations

import ast
import inspect

import pytest
from pydantic import ValidationError

import molzoo.pinet.spec as spec_module
from molzoo.pinet import PiNetSpec


class TestPiNetSpec:
    def test_rejects_an_unknown_rank(self):
        with pytest.raises(ValidationError):
            PiNetSpec(rank=2)

    def test_rejects_an_unsorted_element_table(self):
        with pytest.raises(ValidationError, match="ascending"):
            PiNetSpec(atomic_numbers=[6, 1])

    def test_rejects_mismatched_gate_widths(self):
        with pytest.raises(ValidationError, match="pp_nodes"):
            PiNetSpec(pp_nodes=[8, 8], ii_nodes=[8, 4])

    def test_rejects_atomic_energies_of_the_wrong_length(self):
        with pytest.raises(ValidationError, match="one entry per atomic number"):
            PiNetSpec(atomic_numbers=[1, 6], atomic_energies=[0.0])

    def test_rejects_an_unknown_field(self):
        with pytest.raises(ValidationError):
            PiNetSpec(atom_types=[1, 6])

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
