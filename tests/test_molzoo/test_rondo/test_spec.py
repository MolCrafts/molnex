"""RondoSpec: encoder and heads, nothing about the functional form or a training stage."""

from __future__ import annotations

from molzoo.rondo import RondoSpec


class TestRondoSpec:
    def test_defaults_declare_byteffs_input_channels(self):
        spec = RondoSpec()
        assert [c.name for c in spec.encoder.atom_channels] == [
            "formal_charge",
            "total_charge",
            "ring_connectivity",
            "min_ring_size",
        ]
        assert [c.name for c in spec.encoder.bond_channels] == ["bond_type", "in_ring"]
