"""Tests for molzoo.mace.utils — official archive helpers."""

from __future__ import annotations

import pytest
import torch

from molzoo.mace.utils import read_archive


class TestReadArchive:
    def test_state_dict_mapping_passes_through(self) -> None:
        """A plain tensor dict is returned as ``state`` with an empty config."""
        weight = torch.arange(4, dtype=torch.float64)
        archive = read_archive({"linear.weight": weight})
        assert set(archive) == {"state", "config"}
        assert archive["config"] == {}
        assert torch.equal(archive["state"]["linear.weight"], weight)

    def test_rejects_unconverted_contractions(self) -> None:
        with pytest.raises(ValueError, match="CG basis"):
            read_archive(
                {"products.0.symmetric_contractions.contractions.0.weights_max": torch.ones(2)}
            )

    def test_preserves_envelope_metadata(self, tmp_path) -> None:
        path = tmp_path / "converted.pt"
        config = {"kspace_cutoff_factor": 1.5}
        torch.save({"state_dict": {"linear.weight": torch.ones(2)}, "config": config}, path)
        assert read_archive(path)["config"] == config

    def test_serialized_cutoff_wins_over_envelope_heuristic(self) -> None:
        archive = read_archive(
            {
                "state_dict": {"kspace_cutoff": torch.tensor(1.25)},
                "config": {"kspace_cutoff_factor": 1.5, "kspace_cutoff": 9.0},
            }
        )
        assert archive["config"]["kspace_cutoff"] == 1.25
        assert archive["config"]["kspace_cutoff_factor"] == 1.5

    def test_converted_weights_replace_not_overlay(self, tmp_path) -> None:
        path = tmp_path / "converted.pt"
        torch.save({"linear.weight": torch.ones(2)}, path)
        original = {
            "state_dict": {
                "products.0.symmetric_contractions.contractions.0.weights_max": torch.zeros(5)
            },
            "config": {"kspace_cutoff_factor": 1.5},
        }
        archive = read_archive(original, converted_weights=path)
        assert set(archive["state"]) == {"linear.weight"}
        assert archive["config"]["kspace_cutoff_factor"] == 1.5

    def test_pickle_requires_explicit_trust(self, tmp_path) -> None:
        path = tmp_path / "model.pt"
        torch.save(torch.nn.Linear(2, 3), path)
        with pytest.raises(ValueError, match="trusted=True"):
            read_archive(path)
        assert set(read_archive(path, trusted=True)["state"]) == {"weight", "bias"}

    def test_missing_file_does_not_trigger_pickle_fallback(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            read_archive(tmp_path / "missing.pt", trusted=True)

    @pytest.mark.parametrize(
        "state", [{}, {"weight": 1}, {"state_dict": {"w": torch.ones(1)}, "config": 5}]
    )
    def test_invalid_schema_is_not_silently_filtered(self, state) -> None:
        with pytest.raises(TypeError):
            read_archive(state)
