"""Tests for molzoo.mace.utils — the MACE official-archive reader."""

from __future__ import annotations

import json

import pytest
import torch
from pydantic import ValidationError

from molzoo.mace.utils import MaceArchive

#: The metadata every typed config needs, as a cueq ``state_dict`` carries it.
META_STATE: dict[str, torch.Tensor] = {
    "atomic_numbers": torch.tensor([1, 8]),
    "atomic_energies_fn.atomic_energies": torch.tensor([[-13.6, -2041.0]], dtype=torch.float64),
    "r_max": torch.tensor(5.0, dtype=torch.float64),
}

#: The same metadata as a sidecar json carries it.
META_JSON: dict[str, object] = {
    "atomic_numbers": [1, 8],
    "atomic_energies": [-13.6, -2041.0],
    "r_max": 5.0,
}


def _state(**extra: torch.Tensor) -> dict[str, torch.Tensor]:
    return {**META_STATE, "linear.weight": torch.arange(4, dtype=torch.float64), **extra}


class TestMaceArchive:
    def test_a_state_dict_yields_a_typed_config_from_its_buffers(self) -> None:
        """``atomic_numbers`` / ``E0`` / ``r_max`` come off the state's own buffers."""
        state, cfg = MaceArchive().read(_state())
        assert (cfg.atomic_numbers, cfg.atomic_energies, cfg.r_max) == (
            [1, 8],
            [-13.6, -2041.0],
            5.0,
        )
        assert torch.equal(state["linear.weight"], torch.arange(4, dtype=torch.float64))

    def test_envelope_config_is_read_with_its_aliases(self) -> None:
        """An envelope's ``config`` is typed; upstream spellings resolve to the molnex fields."""
        _, cfg = MaceArchive().read(
            {"state_dict": _state(), "config": {"max_ell": 3, "atomic_inter_scale": 0.8}}
        )
        assert (cfg.l_max, cfg.scale) == (3, 0.8)

    def test_serialized_cutoff_wins_over_envelope_heuristic(self) -> None:
        _, cfg = MaceArchive().read(
            {
                "state_dict": _state(kspace_cutoff=torch.tensor(1.25)),
                "config": {"kspace_cutoff_factor": 1.5, "kspace_cutoff": 9.0},
            }
        )
        assert (cfg.kspace_cutoff, cfg.kspace_cutoff_factor) == (1.25, 1.5)

    def test_explicit_config_json_wins_over_the_archive(self, tmp_path) -> None:
        path = tmp_path / "weights.pt"
        torch.save({"state_dict": _state(), "config": {"num_bessel": 8}}, path)
        config = tmp_path / "config.json"
        config.write_text(json.dumps({"num_bessel": 10, "max_ell": 2}))
        _, cfg = MaceArchive().read(path, config)
        assert (cfg.num_bessel, cfg.l_max) == (10, 2)

    def test_a_config_json_finds_the_weights_beside_it(self, tmp_path) -> None:
        """``<stem>_config.json`` → ``<stem>_cueq_state.pt``."""
        torch.save(_state(), tmp_path / "matpes_cueq_state.pt")
        config = tmp_path / "matpes_config.json"
        config.write_text(json.dumps({**META_JSON, "max_ell": 2}))
        state, cfg = MaceArchive().read(config)
        assert "linear.weight" in state and cfg.l_max == 2

    def test_a_config_json_without_weights_raises(self, tmp_path) -> None:
        config = tmp_path / "matpes_config.json"
        config.write_text(json.dumps(META_JSON))
        with pytest.raises(FileNotFoundError):
            MaceArchive().read(config)

    def test_a_directory_resolves_its_one_config_and_weights(self, tmp_path) -> None:
        torch.save(_state(), tmp_path / "x_cueq_state.pt")
        (tmp_path / "x_config.json").write_text(json.dumps({"max_ell": 2}))
        _, cfg = MaceArchive().read(tmp_path)
        assert cfg.l_max == 2

    def test_an_ambiguous_directory_raises(self, tmp_path) -> None:
        torch.save(_state(), tmp_path / "a_cueq_state.pt")
        torch.save(_state(), tmp_path / "b_cueq_state.pt")
        with pytest.raises(ValueError, match="ambiguous"):
            MaceArchive().read(tmp_path)

    def test_converted_weights_replace_not_overlay(self, tmp_path) -> None:
        path = tmp_path / "converted.pt"
        torch.save(_state(), path)
        original = {
            "state_dict": {
                "products.0.symmetric_contractions.contractions.0.weights_max": torch.zeros(5)
            },
            "config": {"kspace_cutoff_factor": 1.5},
        }
        state, cfg = MaceArchive(converted_weights=path).read(original)
        assert set(state) == set(_state())
        assert cfg.kspace_cutoff_factor == 1.5

    def test_rejects_unconverted_contractions(self) -> None:
        with pytest.raises(ValueError, match="CG basis"):
            MaceArchive().read(
                _state(
                    **{
                        "products.0.symmetric_contractions.contractions.0.weights_max": torch.ones(
                            2
                        )
                    }
                )
            )

    def test_a_pickled_module_reads_like_a_state_dict(self, tmp_path) -> None:
        path = tmp_path / "model.pt"
        torch.save(torch.nn.Linear(2, 3), path)
        config = tmp_path / "meta.json"
        config.write_text(json.dumps(META_JSON))
        state, _ = MaceArchive().read(path, config)
        assert set(state) == {"weight", "bias"}

    def test_a_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            MaceArchive().read(tmp_path / "missing.pt")

    def test_missing_metadata_is_a_validation_error(self) -> None:
        """A bare state without the element table cannot become a config."""
        with pytest.raises(ValidationError):
            MaceArchive().read({"linear.weight": torch.ones(2)})

    @pytest.mark.parametrize(
        "state", [{}, {"weight": 1}, {"state_dict": {"w": torch.ones(1)}, "config": 5}]
    )
    def test_invalid_schema_is_not_silently_filtered(self, state) -> None:
        with pytest.raises(TypeError):
            MaceArchive().read(state)
