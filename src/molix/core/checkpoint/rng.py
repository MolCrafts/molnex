"""Random number generator state capture and restoration."""

from __future__ import annotations

import random
from typing import Any

import torch

from molix import logger as _logger_mod

logger = _logger_mod.getLogger(__name__)


def capture_rng_states() -> dict[str, Any]:
    """Capture current RNG states for reproducible checkpoint resume.

    Captures states for: torch CPU, torch CUDA (all devices),
    numpy (if installed), and python stdlib random.

    Returns:
        Dictionary with RNG states keyed by source name.
    """
    states: dict[str, Any] = {
        "torch": torch.random.get_rng_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        states["cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np

        states["numpy"] = np.random.get_state()
    except ImportError:
        pass
    return states


def restore_rng_states(states: dict[str, Any]) -> None:
    """Restore RNG states from a previously captured dict.

    Args:
        states: Dictionary produced by :func:`capture_rng_states`.
    """
    if "torch" in states:
        torch.random.set_rng_state(states["torch"])
    if "cuda" in states:
        if not torch.cuda.is_available():
            logger.warning(
                "partial RNG restore: checkpoint contains CUDA RNG states "
                "but CUDA is unavailable — resumed run is not "
                "bit-reproducible on this machine"
            )
        elif torch.cuda.device_count() != len(states["cuda"]):
            logger.warning(
                "partial RNG restore: checkpoint has CUDA RNG states for "
                f"{len(states['cuda'])} device(s) but "
                f"{torch.cuda.device_count()} are visible — skipping CUDA "
                "restore"
            )
        else:
            torch.cuda.set_rng_state_all(states["cuda"])
    if "numpy" in states:
        try:
            import numpy as np

            np.random.set_state(states["numpy"])
        except ImportError:
            pass
    if "python" in states:
        random.setstate(states["python"])
