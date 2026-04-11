"""Global configuration for molix using external molcfg library."""

from __future__ import annotations

import torch
from molcfg import Config

# Global singleton instance
config = Config(
    {
        "ftype": torch.float32,
        "itype": torch.int64,
    }
)
