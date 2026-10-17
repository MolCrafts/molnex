"""Tests for molzoo.allegro.utils (the ``.nequip.zip`` reader)."""

from __future__ import annotations

import sys
import zipfile

import pytest
import torch

from molzoo.allegro.utils import NequipPackage
from tests.test_molzoo.test_allegro.conftest import WEIGHTS_DIR

needs_assets = pytest.mark.skipif(
    not (WEIGHTS_DIR / "allegro_small_default.nequip.zip").exists(),
    reason="MOLNEX_ALLEGRO_WEIGHTS_DIR not set",
)

SCALES = "model.func.per_type_energy_scale_shift.scales"
SHIFTS = "model.func.per_type_energy_scale_shift.shifts"


class TestNequipPackage:
    def test_rejects_a_zip_that_is_not_a_package(self, tmp_path):
        path = tmp_path / "x.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("x/byteorder", "little")
        with pytest.raises(ValueError, match="not a nequip package"):
            NequipPackage().read(path)

    def test_normalises_the_upstream_layout(self):
        state = {
            SCALES: torch.tensor([[2.0]]),
            SHIFTS: torch.tensor([[1.0], [2.0], [3.0]]),
            "model.func.allegro.tps.0.weights": torch.ones(4),
            "model.func.allegro.tps.0.w3j": torch.zeros(3),
            "model.func.allegro.tps.1.weights": torch.ones(4, 2),
            "model.func.allegro.tps.1.w3j": torch.zeros(3),
            "model.func.edge_readout.mlp_module.0.weight": torch.ones(2, 2),
        }
        out = NequipPackage._normalise_layout(state, n_types=3, num_layers=2)
        assert not any(key.endswith(".w3j") for key in out)
        torch.testing.assert_close(out[SCALES], torch.tensor([2.0, 2.0, 2.0]))
        torch.testing.assert_close(out[SHIFTS], torch.tensor([1.0, 2.0, 3.0]))
        assert out["model.func.allegro.tps.0.weights"].shape == (4, 1)
        assert out["model.func.allegro.tps.1.weights"].shape == (4, 2)

    def test_drops_an_absent_table(self):
        out = NequipPackage._normalise_layout({SHIFTS: torch.zeros(0)}, n_types=3, num_layers=1)
        assert out == {}

    @pytest.mark.slow
    @needs_assets
    def test_recovers_the_architecture_without_importing_upstream(self):
        state, spec = NequipPackage().read(WEIGHTS_DIR / "allegro_small_default.nequip.zip")
        assert not {m.split(".")[0] for m in sys.modules} & {"nequip", "allegro", "e3nn"}
        assert spec.atomic_numbers == [1, 6, 8]
        assert (spec.num_layers, spec.l_max, spec.parity, spec.zbl) == (2, 2, True, False)
        assert state["model.func.allegro.tps.0.weights"].dim() == 2
