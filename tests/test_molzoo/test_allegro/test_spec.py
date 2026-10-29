"""Unit tests for molzoo.allegro.spec."""

from __future__ import annotations

import ast
import inspect

import pytest
from pydantic import ValidationError

import molzoo.allegro.spec as spec_module
from molzoo.allegro import AllegroSpec, NequipConfig
from molzoo.allegro.spec import NequipMLP


def _config(**overrides) -> NequipConfig:
    fields = {
        "atomic_numbers": [1, 6, 8],
        "r_max": 5.0,
        "num_bessels": 8,
        "bessel_trainable": False,
        "polynomial_cutoff_p": 6,
        "radial_chemical_embed_dim": 64,
        "scalar_embed_mlp": NequipMLP(depth=1, width=48, nonlinearity="silu"),
        "l_max": 2,
        "parity": True,
        "num_layers": 2,
        "num_scalar_features": 64,
        "num_tensor_features": 16,
        "allegro_mlp": NequipMLP(depth=1, width=48, nonlinearity="silu"),
        "readout_mlp": NequipMLP(depth=0),
        "avg_num_neighbors": 20.0,
        "per_type_energy_shifts": [-1.0],
    }
    return NequipConfig(**{**fields, **overrides})


class TestAllegroSpec:
    def test_rejects_a_type_table_of_the_wrong_length(self):
        with pytest.raises(ValidationError, match="one per type"):
            AllegroSpec(atomic_numbers=[1, 6, 8], r_max=4.0, per_type_energy_shifts=[0.0, 1.0])

    def test_a_scalar_or_one_entry_table_is_broadcast_to_every_type(self):
        spec = AllegroSpec(
            atomic_numbers=[1, 6, 8],
            r_max=4.0,
            per_type_energy_scales=2.0,
            per_type_energy_shifts=[-1.5],
        )
        assert (spec.per_type_energy_scales, spec.per_type_energy_shifts) == (
            [2.0, 2.0, 2.0],
            [-1.5, -1.5, -1.5],
        )

    def test_rejects_an_unknown_field(self):
        with pytest.raises(ValidationError):
            AllegroSpec(atomic_numbers=[1], r_max=4.0, num_elements=3)

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
        modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        assert roots <= {"__future__", "typing", "pydantic", "molrep"}
        assert {m for m in modules if m and m.startswith("molrep")} == {
            "molrep.encoders.allegro.spec"
        }

    def test_from_package_keeps_the_default_width_of_a_depth_zero_mlp(self):
        spec = AllegroSpec.from_package(_config())
        assert (spec.allegro_mlp_hidden_layers_width, spec.readout_mlp_hidden_layers_width) == (
            48,
            32,
        )
        assert (spec.readout_mlp_hidden_layers_depth, spec.zbl) == (0, False)

    def test_from_package_reads_zbl_with_the_package_cutoff_exponent(self):
        spec = AllegroSpec.from_package(_config(pair_potential="ZBL", zbl_polynomial_cutoff_p=4))
        assert (spec.zbl, spec.zbl_polynomial_cutoff_p) == (True, 4)


class TestNequipConfig:
    def test_ignores_what_this_port_does_not_read(self):
        assert _config(model_dtype="float32").num_layers == 2
