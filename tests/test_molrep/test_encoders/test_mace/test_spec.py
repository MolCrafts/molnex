"""Unit tests for molrep.encoders.mace.spec."""

from __future__ import annotations

import ast
import inspect

import pytest
from pydantic import ValidationError

import molrep.encoders.mace.spec as spec_module
from molrep.encoders.mace.spec import MACEEncoderSpec

_SWITCHES = dict(interaction="density", distance_transform="none", conditioning="none")


class TestMACEEncoderSpec:
    def test_rejects_an_unsorted_element_table(self):
        with pytest.raises(ValidationError, match="ascending"):
            MACEEncoderSpec(atomic_numbers=[6, 1], **_SWITCHES)

    def test_an_agnostic_stack_needs_the_neighbour_count(self):
        with pytest.raises(ValidationError, match="avg_num_neighbors"):
            MACEEncoderSpec(atomic_numbers=[1, 6], **{**_SWITCHES, "interaction": "agnostic"})

    def test_conditioning_sizes_exist_iff_the_stack_is_conditioned(self):
        with pytest.raises(ValidationError, match="charge_spin"):
            MACEEncoderSpec(atomic_numbers=[1, 6], **{**_SWITCHES, "conditioning": "charge_spin"})
        with pytest.raises(ValidationError, match="takes no"):
            MACEEncoderSpec(atomic_numbers=[1, 6], charge_classes=3, **_SWITCHES)

    def test_a_bare_encoder_spec_states_no_body_schedule(self):
        with pytest.raises(NotImplementedError):
            MACEEncoderSpec(atomic_numbers=[1, 6], **_SWITCHES).body_schedule()

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
