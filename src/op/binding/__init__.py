"""
High-performance PyTorch operations for neural network potentials
"""

import sys
from pathlib import Path
import torch

# Determine library name based on platform
if sys.platform == "win32":
    lib_name = "molnex_opLib.pyd"
elif sys.platform == "darwin":
    lib_name = "libmolnex_opLib.dylib"
else:
    lib_name = "libmolnex_opLib.so"

# Load the library
lib_path = Path(__file__).parent / lib_name
if lib_path.exists():
    torch.ops.load_library(lib_path)
else:
    raise ImportError(f"Could not find operator library: {lib_path}")

from .locality.neighbors import get_neighbor_pairs
from .scatter import scatter_sum, batch_add
from .pot import PMEkernel

__all__ = [
    "get_neighbor_pairs",
    "scatter_sum",
    "batch_add",
    "PMEkernel",
]
