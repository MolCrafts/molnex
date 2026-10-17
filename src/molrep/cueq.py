"""cuEquivariance operations pinned to a kernel method.

Each class is the ``cuequivariance_torch`` operation of the same name with its
execution method fixed to a CUDA kernel, so on GPU nothing runs cuEquivariance's
pure-PyTorch ``naive`` path — which is what ``cuet.Linear`` picks when no method
is given. On CPU cuEquivariance substitutes its own naive implementation. As in
cuEquivariance, the precision is fixed at construction: build at the wanted
``dtype`` rather than calling ``.double()`` afterwards.
"""

from __future__ import annotations

from typing import Any

import cuequivariance_torch as cuet

__all__ = [
    "ChannelWiseTensorProduct",
    "FullyConnectedTensorProduct",
    "Linear",
    "SegmentedPolynomial",
    "SphericalHarmonics",
    "SymmetricContraction",
]


class Linear(cuet.Linear):
    """``cuet.Linear`` on the ``fused_tp`` kernel."""

    def __init__(self, *args: Any, method: str = "fused_tp", **kwargs: Any) -> None:
        super().__init__(*args, method=method, **kwargs)


class ChannelWiseTensorProduct(cuet.ChannelWiseTensorProduct):
    """``cuet.ChannelWiseTensorProduct`` on the ``uniform_1d`` kernel."""

    def __init__(self, *args: Any, method: str = "uniform_1d", **kwargs: Any) -> None:
        super().__init__(*args, method=method, **kwargs)


class FullyConnectedTensorProduct(cuet.FullyConnectedTensorProduct):
    """``cuet.FullyConnectedTensorProduct`` on the ``fused_tp`` kernel."""

    def __init__(self, *args: Any, method: str = "fused_tp", **kwargs: Any) -> None:
        super().__init__(*args, method=method, **kwargs)


class SymmetricContraction(cuet.SymmetricContraction):
    """``cuet.SymmetricContraction`` on the ``uniform_1d`` kernel."""

    def __init__(self, *args: Any, method: str = "uniform_1d", **kwargs: Any) -> None:
        super().__init__(*args, method=method, **kwargs)


class SphericalHarmonics(cuet.SphericalHarmonics):
    """``cuet.SphericalHarmonics`` on the ``uniform_1d`` kernel."""

    def __init__(self, *args: Any, method: str = "uniform_1d", **kwargs: Any) -> None:
        super().__init__(*args, method=method, **kwargs)


class SegmentedPolynomial(cuet.SegmentedPolynomial):
    """``cuet.SegmentedPolynomial`` on a kernel method (default ``uniform_1d``)."""

    def __init__(self, *args: Any, method: str = "uniform_1d", **kwargs: Any) -> None:
        super().__init__(*args, method=method, **kwargs)
