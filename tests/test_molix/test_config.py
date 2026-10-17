"""Precision presets of molix.config."""

import pytest
import torch

from molix.config import MolnexConfig


class TestMolnexConfig:
    @pytest.mark.parametrize("mode,dtype", [("fp32", torch.float32), ("fp64", torch.float64)])
    def test_preset_writes_the_dtype_and_leaves_matmul_policy_alone(self, mode, dtype) -> None:
        previous = torch.get_float32_matmul_precision()
        cfg = MolnexConfig()
        cfg.set_precision(mode)
        assert cfg["ftype"] == dtype
        assert cfg["use_amp"] is False
        assert torch.get_float32_matmul_precision() == previous

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported precision mode"):
            MolnexConfig().set_precision("fp8")
