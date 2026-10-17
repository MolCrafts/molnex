"""Tests for the official Allegro key table (molzoo.allegro.checkpoint)."""

from __future__ import annotations

from molzoo.allegro import AllegroPotential, AllegroSpec
from molzoo.allegro.checkpoint import allegro_keys
from tests.test_molzoo.test_allegro.conftest import TINY


class TestAllegroKeys:
    def test_every_target_prefix_names_a_module_of_the_model(self):
        modules = dict(AllegroPotential(AllegroSpec(**TINY)).named_modules())
        targets = {v.rstrip(".") for v in allegro_keys.values() if v is not None}
        assert targets <= set(modules)
