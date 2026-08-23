"""PotentialMeta style registry — improper schemas must be registered."""

from __future__ import annotations

from molpot.forcefield.schema import PotentialMeta, StyleSchema


class TestPotentialMeta:
    def test_schema_for_improper_periodic_is_registered(self):
        view = PotentialMeta.schema_for(["improper:periodic"])
        schema = view["improper:periodic"]
        assert isinstance(schema, StyleSchema)
        assert schema.energy_formula != "(unregistered style)"

    def test_schema_for_improper_harmonic_is_registered(self):
        view = PotentialMeta.schema_for(["improper:harmonic"])
        schema = view["improper:harmonic"]
        assert isinstance(schema, StyleSchema)
        assert schema.energy_formula != "(unregistered style)"
