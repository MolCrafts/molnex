"""MolNeX: TensorDict-first molecular ML framework"""

import sys
from pathlib import Path
import torch

# Load C++ library once at package level
_lib_loaded = False

def _load_library():
    """Load the C++ extension library"""
    global _lib_loaded
    if _lib_loaded:
        return
    
    if sys.platform == "win32":
        lib_name = "molnex_opLib.pyd"
    elif sys.platform == "darwin":
        lib_name = "libmolnex_opLib.dylib"
    else:
        lib_name = "libmolnex_opLib.so"
    
    lib_path = Path(__file__).parent / "op" / lib_name
    if not lib_path.exists():
        raise ImportError(
            f"Could not find operator library: {lib_path}\n"
            f"Build C++ extensions with: pip install -e ."
        )
    
    torch.ops.load_library(str(lib_path))
    _lib_loaded = True

_load_library()



# Import F and nn submodules
from . import F
from . import nn

__all__ = ["F", "nn"]
