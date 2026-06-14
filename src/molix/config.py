"""Global configuration for molix using external molcfg library.

``config["ftype"]`` is the **single source of truth** for the project's
floating-point dtype, consumed in two places and only two places:

1. **Model construction** — encoder / readout / head modules read
   ``config["ftype"]`` in ``__init__`` and bake it into their parameters.
2. **DataModule collate boundary** —
   :class:`molix.data.datamodule._CollateFn` captures ``config["ftype"]``
   at construction time (in the parent process, before workers spawn)
   and casts every emitted batch's floating-point tensors via
   :func:`molix.core.steps.batch_to`. Long index tensors
   (``edge_index``, ``Z``, ``batch``) are left untouched.

Dataset loaders (``QM9Source``, ``RevMD17Source``, ``_extxyz``, …) keep a
canonical fp32 representation on disk — the per-batch up-cast happens
at the collate boundary, so the :class:`~molix.data.cache.PackedCache`
stays compact and is reusable across precisions without re-baking.

Precision is set via :meth:`MolnexConfig.set_precision`, e.g.
``config.set_precision("fp64")``.
"""

from __future__ import annotations

import torch
from molcfg import Config

_PRECISION_PRESETS: dict[str, dict[str, object]] = {
    "fp32": {
        "ftype": torch.float32,
        "use_amp": False,
        "amp_dtype": torch.float16,
        "matmul_precision": "high",
    },
    "fp64": {
        "ftype": torch.float64,
        "use_amp": False,
        "amp_dtype": torch.float16,
        "matmul_precision": "highest",
    },
    "fp16-mixed": {
        "ftype": torch.float32,
        "use_amp": True,
        "amp_dtype": torch.float16,
        "matmul_precision": "high",
    },
    "bf16-mixed": {
        "ftype": torch.float32,
        "use_amp": True,
        "amp_dtype": torch.bfloat16,
        "matmul_precision": "high",
    },
}


class MolnexConfig(Config):
    """MolNex-wide :class:`molcfg.Config` with the precision contract.

    Subclasses :class:`molcfg.Config` to attach :meth:`set_precision` as a
    method on the root config object (``molcfg.Config`` uses ``__slots__``
    so dynamic attribute attachment isn't an option). Nested children
    remain plain :class:`Config`; that's fine because the precision
    contract only lives at the root.

    The class is named for the whole **MolNex** project (molix + molrep +
    molpot + molzoo), not just ``molix``, because every package consumes
    ``config["ftype"]`` — the singleton lives in ``molix.config`` only
    because that's the package at the bottom of the dependency graph.
    """

    __slots__ = ()

    def set_precision(self, mode: str) -> None:
        """Set the global precision for the entire project.

        Writes ``ftype``, ``use_amp``, and ``amp_dtype`` into this config.
        Must be called **before** both model construction and
        :class:`~molix.data.datamodule.DataModule` instantiation, because:

        - Model layers bake ``config["ftype"]`` into their parameters at
          ``__init__`` time.
        - :class:`~molix.data.datamodule._CollateFn` captures
          ``config["ftype"]`` at construction time so every batch is cast
          to the right dtype on its way out of the dataloader (long
          index tensors are preserved). This means ``set_precision``
          controls model *and* data dtype from a single call — the user
          never casts batches.
        - :class:`~molix.core.trainer.Trainer` creates its ``GradScaler``
          from ``config["use_amp"]`` at construction time.

        Args:
            mode: One of

                - ``"fp32"``: parameters and batches in float32, no autocast (default).
                - ``"fp64"``: parameters and batches in float64, no autocast.
                - ``"fp16-mixed"``: parameters in float32, autocast to float16
                  with ``GradScaler``. Batches stay float32; autocast handles
                  the per-op down-cast inside the forward pass.
                - ``"bf16-mixed"``: parameters in float32, autocast to bfloat16.

        Raises:
            ValueError: If ``mode`` is not one of the supported presets.
                Pure (non-mixed) ``fp16`` / ``bf16`` are intentionally not
                supported.
        """
        if mode not in _PRECISION_PRESETS:
            supported = ", ".join(sorted(_PRECISION_PRESETS))
            raise ValueError(f"Unsupported precision mode {mode!r}. Supported modes: {supported}.")
        preset = _PRECISION_PRESETS[mode]
        for key, value in preset.items():
            if key == "matmul_precision":
                continue
            self[key] = value
        torch.set_float32_matmul_precision(preset["matmul_precision"])


# Global singleton instance. ``ftype`` / ``itype`` control tensor creation
# throughout the stack; ``use_amp`` / ``amp_dtype`` control the autocast
# behaviour of :class:`molix.core.steps.DefaultTrainStep` /
# :class:`DefaultEvalStep` and the :class:`torch.amp.GradScaler` owned by
# :class:`molix.core.trainer.Trainer`.
config = MolnexConfig(
    {
        "ftype": torch.float32,
        "itype": torch.int64,
        "use_amp": False,
        "amp_dtype": torch.float16,
    }
)
