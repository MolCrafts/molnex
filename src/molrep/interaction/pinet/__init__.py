"""PiNet graph-convolution layers (molrep interaction primitives).

Single-responsibility package: pure message-passing blocks with no energy /
force / TensorDict orchestration. Composition into encoders lives in
``molzoo.pinet``; energy heads live with the potential (``molzoo.pinet.potential``
today; planned home under ``molpot``).

Reference:
    Li et al. "PiNN: Equivariant Neural Network Suite for Modeling
    Electrochemical Systems", JCTC 2025.
    https://doi.org/10.1021/acs.jctc.4c01570
"""

from .blocks import EquivarLayer, GCBlock, InvarLayer, OutLayer
from .ff import FFLayer, activation_from_name
from .message import DotLayer, IPLayer, PILayer, PIXLayer, ScaleLayer
from .residual import ResUpdate

__all__ = [
    "FFLayer",
    "activation_from_name",
    "PILayer",
    "IPLayer",
    "PIXLayer",
    "ScaleLayer",
    "DotLayer",
    "ResUpdate",
    "InvarLayer",
    "EquivarLayer",
    "OutLayer",
    "GCBlock",
]
