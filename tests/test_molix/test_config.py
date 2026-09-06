"""Precision preset and explicit process-wide GEMM override contract."""

import pytest
import torch

from molix.config import MolnexConfig


class TestMolnexConfig:
    @pytest.mark.parametrize("mode,dtype", [("fp32", torch.float32), ("fp64", torch.float64)])
    def test_strict_matmul_override(self, mode, dtype) -> None:
        previous = torch.get_float32_matmul_precision()
        try:
            cfg = MolnexConfig()
            cfg.set_precision(mode, matmul_precision="highest")
            assert cfg["ftype"] == dtype
            assert cfg["use_amp"] is False
            assert torch.get_float32_matmul_precision() == "highest"
        finally:
            torch.set_float32_matmul_precision(previous)

    def test_invalid_override_has_no_side_effects(self) -> None:
        cfg = MolnexConfig()
        cfg["ftype"] = torch.float64
        previous = torch.get_float32_matmul_precision()
        with pytest.raises(ValueError, match="matmul_precision"):
            cfg.set_precision("fp32", matmul_precision="invalid")
        assert cfg["ftype"] == torch.float64
        assert torch.get_float32_matmul_precision() == previous
