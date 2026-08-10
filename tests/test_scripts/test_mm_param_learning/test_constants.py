"""Unit tests for mm_param_learning constants."""

from __future__ import annotations

from scripts.mm_param_learning.constants import (
    EXPERIMENT_SLUGS,
    MOLHUB_COORDINATES,
    VALIDATION_STAGES,
)


class TestConstants:
    def test_four_experiment_slugs(self):
        assert set(EXPERIMENT_SLUGS) == {
            "potential-parity",
            "zinc-typing-recovery",
            "phalkethoh-mm-energy",
            "latent-analysis",
        }
        assert set(MOLHUB_COORDINATES) == set(EXPERIMENT_SLUGS)

    def test_coordinates_are_dataset_prefixed(self):
        for slug, coord in MOLHUB_COORDINATES.items():
            assert coord.startswith("dataset:"), (slug, coord)

    def test_six_validation_stages(self):
        assert len(VALIDATION_STAGES) == 6
        nums = [n for n, _ in VALIDATION_STAGES]
        assert nums == ["1", "2", "3", "4", "5", "6"]
        titles = " ".join(t for _, t in VALIDATION_STAGES).lower()
        assert "inventory" in titles
        assert "b0" in titles or "parity" in titles
        assert "typing" in titles
        assert "energy" in titles or "b1" in titles
        assert "latent" in titles
        assert "gate" in titles or "synthesis" in titles
