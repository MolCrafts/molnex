"""Generate fixtures for the C++ ModelRunner test suite.

Invoked as a ctest fixture before any subtest runs. Produces:

    <out_dir>/model_<device>/                  exported model dir
    <out_dir>/reference_<device>.pt            {input, output} reference
    <out_dir>/alt_<device>.pt                  alternative state_dict
    <out_dir>/reference_alt_<device>.pt        {input, output} with alt weights

`<device>` is `cpu`, plus `cuda` if a GPU is visible. The C++ tests pick
the device based on what's present.

This script *requires* a working editable install of MolNex so that
``molix.export.export_model`` is importable; the test harness is
expected to be run inside the project's dev environment.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn


class TinyLinear(nn.Module):
    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2)
        gen = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.linear.weight.copy_(torch.randn((2, 3), generator=gen))
            self.linear.bias.copy_(torch.randn((2,), generator=gen))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _export_with_runtime_constants(
    model: nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    export_dir: Path,
    device: str,
    name: str = "model",
) -> None:
    """Mirror of molix.export.export_model but with runtime constant folding.

    Default AOTI compilation folds small constants (like Linear bias) into
    the .so as literals, making them unreloadable via update_weights().
    `aot_inductor.use_runtime_constant_folding=True` keeps every parameter
    as an updateable constant buffer — required to exercise the reload
    path in this test suite.
    """
    import json

    export_dir.mkdir(parents=True, exist_ok=True)
    so_path = str(export_dir / f"{name}.so")
    model.eval()
    target = model.to(device)
    device_inputs = tuple(
        t.to(device) if isinstance(t, torch.Tensor) else t for t in example_inputs
    )
    with torch.no_grad():
        torch._export.aot_compile(
            target,
            args=device_inputs,
            options={
                "aot_inductor.output_path": so_path,
                "aot_inductor.use_runtime_constant_folding": True,
            },
        )
    torch.save(dict(target.state_dict()), export_dir / f"{name}.pt")
    meta = {
        "device": device,
        "input_shapes": [
            list(t.shape) if isinstance(t, torch.Tensor) else None for t in device_inputs
        ],
        "input_dtypes": [
            str(t.dtype) if isinstance(t, torch.Tensor) else type(t).__name__ for t in device_inputs
        ],
        "model_class": model.__class__.__name__,
    }
    (export_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=2))


def _emit_for_device(out_dir: Path, device: str, example: torch.Tensor) -> None:
    model_dir = out_dir / f"model_{device}"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True)

    base = TinyLinear(seed=0).to(device).eval()
    with torch.no_grad():
        ref_out = base(example.to(device))
    _export_with_runtime_constants(base, (example.to(device),), model_dir, device)
    torch.save(
        {"input": example.to(device), "output": ref_out},
        out_dir / f"reference_{device}.pt",
    )

    alt = TinyLinear(seed=42).to(device).eval()
    with torch.no_grad():
        alt_out = alt(example.to(device))
    # Save as plain dict (not OrderedDict) so torch::pickle_load in C++ can
    # round-trip it without a custom type resolver.
    torch.save(dict(alt.state_dict()), out_dir / f"alt_{device}.pt")
    torch.save(
        {"input": example.to(device), "output": alt_out},
        out_dir / f"reference_alt_{device}.pt",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    gen = torch.Generator().manual_seed(1234)
    example = torch.randn((4, 3), generator=gen)

    _emit_for_device(args.out_dir, "cpu", example)
    if torch.cuda.is_available():
        _emit_for_device(args.out_dir, "cuda", example)

    return 0


if __name__ == "__main__":
    sys.exit(main())
