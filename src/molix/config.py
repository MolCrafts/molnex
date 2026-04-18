"""Global configuration for molix using external molcfg library."""

from __future__ import annotations

import torch
from molcfg import Config

# Global singleton instance. ``ftype`` / ``itype`` control tensor creation
# throughout the stack; ``use_amp`` / ``amp_dtype`` control the autocast
# behaviour of :class:`molix.core.steps.DefaultTrainStep` /
# :class:`DefaultEvalStep` and the :class:`torch.amp.GradScaler` owned by
# :class:`molix.core.trainer.Trainer`.
config = Config(
    {
        "ftype": torch.float32,
        "itype": torch.int64,
        "use_amp": False,
        "amp_dtype": torch.float16,
    }
)


_PRECISION_PRESETS: dict[str, dict[str, object]] = {
    "fp32":       {"ftype": torch.float32, "use_amp": False, "amp_dtype": torch.float16,   "matmul_precision": "high"},
    "fp64":       {"ftype": torch.float64, "use_amp": False, "amp_dtype": torch.float16,   "matmul_precision": "highest"},
    "fp16-mixed": {"ftype": torch.float32, "use_amp": True,  "amp_dtype": torch.float16,   "matmul_precision": "high"},
    "bf16-mixed": {"ftype": torch.float32, "use_amp": True,  "amp_dtype": torch.bfloat16,  "matmul_precision": "high"},
}


def set_precision(mode: str) -> None:
    """Set the global precision for model parameters and AMP autocast.

    Writes ``ftype``, ``use_amp``, and ``amp_dtype`` into the global
    :data:`config`. Must be called **before** model construction and Trainer
    instantiation, because model layers bake ``config.ftype`` into their
    parameters at ``__init__`` time and Trainer creates its ``GradScaler``
    based on ``config.use_amp`` at construction time.

    Args:
        mode: One of

            - ``"fp32"``: parameters in float32, no autocast (default).
            - ``"fp64"``: parameters in float64, no autocast.
            - ``"fp16-mixed"``: parameters in float32, autocast to float16
              with ``GradScaler``.
            - ``"bf16-mixed"``: parameters in float32, autocast to bfloat16.

    Raises:
        ValueError: If ``mode`` is not one of the supported presets. Pure
            (non-mixed) ``fp16`` / ``bf16`` are intentionally not supported.
    """
    if mode not in _PRECISION_PRESETS:
        supported = ", ".join(sorted(_PRECISION_PRESETS))
        raise ValueError(
            f"Unsupported precision mode {mode!r}. Supported modes: {supported}."
        )
    preset = _PRECISION_PRESETS[mode]
    for key, value in preset.items():
        config[key] = value
    torch.set_float32_matmul_precision(preset["matmul_precision"])
