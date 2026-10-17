"""pytest plugin: run the CPU-written molnex tests with CUDA as torch's default device.

The test fixtures build every tensor on the CPU and nothing in the conftests
selects a device, so on a GPU node the slow compile tests
trace the CPU path (cuEquivariance's naive fallback). Loaded with
``-p benchmarks._gh200_verify.cuda_default_plugin`` this makes every factory
call (``torch.tensor`` / ``torch.zeros`` / parameter init) land on CUDA, so
the same tests trace the kernel path. Best effort: a test helper that mixes a
CPU generator or an explicit CPU tensor into the batch fails on device
placement, which is a helper limitation, not a hot-path finding.
"""

import torch


def pytest_configure(config) -> None:  # noqa: ARG001
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
