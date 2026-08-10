"""ForceSpec schema + JSON round-trip tests."""

from __future__ import annotations

import json

from molix.ff_export.force_spec import ForceSpec


class TestForceSpec:
    def test_construct_and_to_dict(self):
        spec = ForceSpec(
            backend="openmm",
            forces=[
                {
                    "type": "HarmonicBondForce",
                    "case": "direct_unit_scale",
                    "parameters": [{"type_index": 0, "k": 41840.0, "r0": 0.15}],
                }
            ],
            scaling={"scale_q_14": 5.0 / 6.0, "scale_lj_14": 0.5},
            metadata={"unit_system_source": "class_i_canonical"},
        )
        d = spec.to_dict()
        assert d["backend"] == "openmm"
        assert d["forces"][0]["type"] == "HarmonicBondForce"
        assert d["scaling"]["scale_lj_14"] == 0.5

    def test_to_dict_is_json_serializable(self):
        spec = ForceSpec(
            backend="openmm",
            forces=[{"type": "HarmonicAngleForce", "parameters": []}],
            scaling=None,
            metadata={},
        )
        payload = json.dumps(spec.to_dict())
        assert "HarmonicAngleForce" in payload

    def test_from_dict_round_trip(self):
        original = ForceSpec(
            backend="openmm",
            forces=[
                {
                    "type": "PeriodicTorsionForce",
                    "case": "form_reparameterize",
                    "parameters": [
                        {
                            "type_index": 0,
                            "term_index": 0,
                            "periodicity": 2,
                            "phase": 0.0,
                            "k": 4.184,
                        }
                    ],
                }
            ],
            scaling={"scale_q_14": 5.0 / 6.0, "scale_lj_14": 0.5},
            metadata={"note": "roundtrip"},
        )
        restored = ForceSpec.from_dict(original.to_dict())
        assert restored == original
