"""Tests for molix.export.export_model() -- AOT Inductor export workflow.

Acceptance criteria:
    ac-001  export_model creates .so, .pt, .meta.json in export_dir
    ac-002  exported .so loadable via aoti_load
    ac-003  CPU export produces correct output (atol=1e-5)
    ac-004  CUDA export produces correct output (skip if no GPU)
    ac-005  device='auto' selects CUDA when available
    ac-006  full test suite passes (verified by CI)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from molix.export import export_model


def _load_and_run(so_path: str, inputs: torch.Tensor) -> torch.Tensor:
    """Load an exported .so and run inference, returning the first output tensor."""
    device_type = "cuda" if inputs.is_cuda else "cpu"
    runner_cls = (
        torch._C._aoti.AOTIModelContainerRunnerCuda
        if device_type == "cuda"
        else torch._C._aoti.AOTIModelContainerRunnerCpu
    )
    runner = runner_cls(so_path, 1)
    return runner.run([inputs])[0]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_mlp() -> nn.Sequential:
    """A small 3-layer MLP (10 -> 32 -> 16 -> 5) with seeded weights."""
    torch.manual_seed(42)
    return nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 5),
    )


@pytest.fixture
def example_input() -> torch.Tensor:
    """Deterministic example input (batch=4, features=10)."""
    torch.manual_seed(99)
    return torch.randn(4, 10)


# ---------------------------------------------------------------------------
# ac-001: File creation
# ---------------------------------------------------------------------------


class TestExportCreatesFiles:
    """export_model creates .so, .pt, .meta.json in export_dir."""

    def test_all_expected_files_exist(self, tmp_path: Path, small_mlp: nn.Sequential) -> None:
        """ac-001: .so, .pt, .meta.json all present after export."""
        x = torch.randn(4, 10)
        export_dir = export_model(
            small_mlp, (x,), export_dir=tmp_path / "export_001", name="my_model"
        )

        assert isinstance(export_dir, Path)
        assert (export_dir / "my_model.so").is_file()
        assert (export_dir / "my_model.pt").is_file()
        assert (export_dir / "my_model.meta.json").is_file()

    def test_meta_json_content(self, tmp_path: Path, small_mlp: nn.Sequential) -> None:
        """meta.json contains device, input shapes/dtypes, model class name."""
        x = torch.randn(4, 10)
        export_dir = export_model(
            small_mlp, (x,), export_dir=tmp_path / "export_meta", name="meta_check"
        )

        with open(export_dir / "meta_check.meta.json") as f:
            meta: dict = json.load(f)

        assert "device" in meta
        assert "input_shapes" in meta
        assert "input_dtypes" in meta
        assert "model_class" in meta
        assert isinstance(meta["device"], str)
        assert isinstance(meta["input_shapes"], list)
        assert isinstance(meta["input_dtypes"], list)
        assert meta["input_shapes"] == [[4, 10]]
        assert meta["input_dtypes"] == ["torch.float32"]
        assert meta["model_class"] == "Sequential"

    def test_default_name_is_model(self, tmp_path: Path, small_mlp: nn.Sequential) -> None:
        """Default name='model' produces model.so / model.pt / model.meta.json."""
        x = torch.randn(4, 10)
        export_dir = export_model(small_mlp, (x,), export_dir=tmp_path / "export_default_name")

        assert (export_dir / "model.so").is_file()
        assert (export_dir / "model.pt").is_file()
        assert (export_dir / "model.meta.json").is_file()

    def test_export_dir_as_string(self, tmp_path: Path, small_mlp: nn.Sequential) -> None:
        """export_dir accepts str (not just Path)."""
        x = torch.randn(4, 10)
        dir_str: str = str(tmp_path / "export_str")
        export_dir = export_model(small_mlp, (x,), export_dir=dir_str, name="from_str")

        assert isinstance(export_dir, Path)
        assert (export_dir / "from_str.so").is_file()


# ---------------------------------------------------------------------------
# ac-002: Loadability
# ---------------------------------------------------------------------------


class TestExportLoadable:
    """Exported .so loadable via AOTIModelContainerRunner."""

    def test_runner_loads_and_returns_tensor(
        self, tmp_path: Path, small_mlp: nn.Sequential
    ) -> None:
        """ac-002: exported .so loads via AOTIModelContainerRunner and returns correct shape."""
        x = torch.randn(4, 10)
        export_dir = export_model(
            small_mlp, (x,), export_dir=tmp_path / "export_load", device="cpu", name="loadable"
        )

        so_path = str(export_dir / "loadable.so")
        result = _load_and_run(so_path, x)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (4, 5)


# ---------------------------------------------------------------------------
# ac-003: CPU correctness
# ---------------------------------------------------------------------------


class TestExportCpuCorrectness:
    """CPU export produces output matching original model."""

    def test_output_matches_original_model(self, tmp_path: Path, small_mlp: nn.Sequential) -> None:
        """ac-003: exported model output matches original within atol=1e-5."""
        small_mlp.eval()
        x = torch.randn(4, 10)

        with torch.no_grad():
            expected = small_mlp(x)

        export_dir = export_model(
            small_mlp, (x,), export_dir=tmp_path / "export_cpu", device="cpu", name="cpu_model"
        )
        actual = _load_and_run(str(export_dir / "cpu_model.so"), x)

        assert torch.allclose(expected, actual, atol=1e-5), (
            f"Output mismatch. max diff = {(expected - actual).abs().max().item()}"
        )

    def test_multiple_batches_match(self, tmp_path: Path, small_mlp: nn.Sequential) -> None:
        """Exported model handles different batch sizes correctly."""
        small_mlp.eval()
        x_small = torch.randn(2, 10)
        x_large = torch.randn(8, 10)

        with torch.no_grad():
            expected_small = small_mlp(x_small)
            expected_large = small_mlp(x_large)

        export_dir = export_model(
            small_mlp,
            (x_large,),
            export_dir=tmp_path / "export_multi_batch",
            device="cpu",
            name="multi",
        )
        so_path = str(export_dir / "multi.so")
        actual_small = _load_and_run(so_path, x_small)
        actual_large = _load_and_run(so_path, x_large)

        assert torch.allclose(expected_small, actual_small, atol=1e-5)
        assert torch.allclose(expected_large, actual_large, atol=1e-5)


# ---------------------------------------------------------------------------
# ac-004: CUDA correctness
# ---------------------------------------------------------------------------


class TestExportCudaCorrectness:
    """CUDA export produces correct output (skipped if no GPU)."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_output_matches_original(self, tmp_path: Path, small_mlp: nn.Sequential) -> None:
        """ac-004: CUDA exported model output matches original on GPU."""
        model = small_mlp.cuda()
        model.eval()
        x = torch.randn(4, 10, device="cuda")

        with torch.no_grad():
            expected = model(x)

        export_dir = export_model(
            model, (x,), export_dir=tmp_path / "export_cuda", device="cuda", name="cuda_model"
        )
        actual = _load_and_run(str(export_dir / "cuda_model.so"), x)

        assert torch.allclose(expected, actual, atol=1e-5), (
            f"CUDA output mismatch. max diff = {(expected - actual).abs().max().item()}"
        )


# ---------------------------------------------------------------------------
# ac-005: device="auto"
# ---------------------------------------------------------------------------


class TestExportDeviceAuto:
    """device='auto' selects CUDA when available, CPU otherwise."""

    def test_auto_device_in_meta(self, tmp_path: Path, small_mlp: nn.Sequential) -> None:
        """ac-005: meta.json device field reflects auto-detection."""
        x = torch.randn(4, 10)
        export_dir = export_model(
            small_mlp, (x,), export_dir=tmp_path / "export_auto", device="auto", name="auto_model"
        )

        with open(export_dir / "auto_model.meta.json") as f:
            meta: dict = json.load(f)

        expected_device = "cuda" if torch.cuda.is_available() else "cpu"
        assert meta["device"] == expected_device, (
            f"Expected device={expected_device!r}, got {meta['device']!r}"
        )


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestExportErrors:
    """Error handling for invalid inputs."""

    def test_non_module_raises_typeerror(self, tmp_path: Path) -> None:
        """Passing a non-nn.Module as model raises TypeError."""
        with pytest.raises(TypeError):
            export_model("not_a_module", (torch.randn(4, 10),), export_dir=tmp_path / "fail_type")

    def test_non_tuple_example_inputs_raises_typeerror(
        self, tmp_path: Path, small_mlp: nn.Sequential
    ) -> None:
        """Passing non-tuple example_inputs raises TypeError."""
        with pytest.raises(TypeError):
            export_model(small_mlp, torch.randn(4, 10), export_dir=tmp_path / "fail_inputs")  # type: ignore[arg-type]

    def test_nonexistent_parent_dir_raises(self, tmp_path: Path, small_mlp: nn.Sequential) -> None:
        """Export_dir whose parent does not exist raises an error."""
        x = torch.randn(4, 10)
        bad_path = tmp_path / "does_not_exist" / "subdir"

        with pytest.raises((FileNotFoundError, OSError, RuntimeError)):
            export_model(small_mlp, (x,), export_dir=bad_path)

    def test_unsupported_device_raises(self, tmp_path: Path, small_mlp: nn.Sequential) -> None:
        """Unsupported device string raises RuntimeError."""
        x = torch.randn(4, 10)
        with pytest.raises(RuntimeError):
            export_model(
                small_mlp, (x,), export_dir=tmp_path / "fail_device", device="invalid_device"
            )


# ---------------------------------------------------------------------------
# Lifecycle / side-effects
# ---------------------------------------------------------------------------


class TestExportLifecycle:
    """Side-effects of export_model on the original model."""

    def test_model_set_to_eval(self, tmp_path: Path, small_mlp: nn.Sequential) -> None:
        """Model should be in eval mode after export returns."""
        small_mlp.train()
        assert small_mlp.training  # sanity

        x = torch.randn(4, 10)
        export_model(small_mlp, (x,), export_dir=tmp_path / "export_eval")

        assert not small_mlp.training, "Model should be in eval mode after export"

    def test_exported_pt_loadable_as_state_dict(
        self, tmp_path: Path, small_mlp: nn.Sequential
    ) -> None:
        """The .pt file contains a valid state_dict matching the original."""
        original_state = {k: v.clone() for k, v in small_mlp.state_dict().items()}

        x = torch.randn(4, 10)
        export_dir = export_model(
            small_mlp, (x,), export_dir=tmp_path / "export_pt", device="cpu", name="weights"
        )

        loaded_state = torch.load(export_dir / "weights.pt", weights_only=True)
        assert isinstance(loaded_state, dict)
        for key in original_state:
            assert key in loaded_state, f"Missing key {key} in saved state_dict"
            assert torch.equal(original_state[key], loaded_state[key]), (
                f"Mismatch for parameter {key}"
            )


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


class TestExportImmutability:
    """Original model unchanged after export."""

    def test_state_dict_unchanged(self, tmp_path: Path, small_mlp: nn.Sequential) -> None:
        """Model's state_dict parameters are unchanged after export (CPU path — under
        ``device="auto"`` with CUDA available the model is intentionally moved to GPU,
        per spec §设备自动检测)."""
        original_state = {k: v.clone() for k, v in small_mlp.state_dict().items()}

        x = torch.randn(4, 10)
        export_model(small_mlp, (x,), export_dir=tmp_path / "export_immut", device="cpu")

        for key in original_state:
            assert torch.equal(original_state[key], small_mlp.state_dict()[key]), (
                f"Parameter {key} changed after export"
            )
