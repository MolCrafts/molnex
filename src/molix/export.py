"""AOT Inductor model export for MolNex.

Wraps ``torch._export.aot_compile()`` to produce deployable .so files
with weights and metadata.  Users call ``export_model(model, inputs, dir)``
and get a self-contained export directory — no knowledge of AOT Inductor
internals required.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from molix import logger as _logger_mod

logger = _logger_mod.getLogger(__name__)

_VALID_DEVICES = frozenset({"auto", "cuda", "cpu"})


def export_model(
    model: nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    export_dir: str | Path,
    *,
    device: str = "auto",
    name: str = "model",
) -> Path:
    """Export an nn.Module to an AOT-compiled shared library with weights and metadata.

    Args:
        model: The PyTorch module to export.  Set to eval mode during export.
        example_inputs: Example input tensors for tracing.  Must be a tuple.
        export_dir: Directory to write ``{name}.so``, ``{name}.pt``, and
            ``{name}.meta.json``.  The leaf directory is created if missing;
            its parent must already exist.
        device: Target device (``"auto"``, ``"cuda"``, or ``"cpu"``).
            ``"auto"`` picks CUDA when available, else CPU.
        name: Base name for the exported artifact files (default ``"model"``).

    Returns:
        The export directory as a :class:`Path`.

    Raises:
        TypeError: If *model* is not an :class:`nn.Module` or *example_inputs*
            is not a :class:`tuple`.
        RuntimeError: If *device* is unrecognised.
        FileNotFoundError: If the parent of *export_dir* does not exist.
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    if not isinstance(example_inputs, tuple):
        raise TypeError(f"example_inputs must be a tuple, got {type(example_inputs).__name__}")
    if device not in _VALID_DEVICES:
        raise RuntimeError(f"device must be one of {sorted(_VALID_DEVICES)}, got {device!r}")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    export_dir = Path(export_dir)
    export_dir.mkdir(exist_ok=True)

    model.eval()
    target_model = model.to(device)
    device_inputs = tuple(
        inp.to(device) if isinstance(inp, torch.Tensor) else inp for inp in example_inputs
    )

    so_path = str(export_dir / f"{name}.so")

    with torch.no_grad():
        torch._export.aot_compile(
            target_model,
            args=device_inputs,
            options={"aot_inductor.output_path": so_path},
        )

    torch.save(target_model.state_dict(), export_dir / f"{name}.pt")

    meta = {
        "device": device,
        "input_shapes": [
            list(inp.shape) if isinstance(inp, torch.Tensor) else None for inp in device_inputs
        ],
        "input_dtypes": [
            str(inp.dtype) if isinstance(inp, torch.Tensor) else type(inp).__name__
            for inp in device_inputs
        ],
        "model_class": model.__class__.__name__,
    }
    with open(export_dir / f"{name}.meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Exported model to {export_dir} (device={device})")
    return export_dir
