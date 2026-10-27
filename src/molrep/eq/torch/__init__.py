# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Torch execution frontend ported from NVIDIA cuEquivariance 0.13.0.

Upstream: NVIDIA/cuEquivariance @ 49a741e00432b0b2f03526aa99599fcdf3903148.
Public names match ``cuequivariance_torch``: ``import molrep.eq.torch as cuet``.
Kernels stay in ``cuequivariance_ops_torch`` (``torch.ops.cuequivariance.*``).
"""

import importlib.resources

__version__ = importlib.resources.files(__package__).joinpath("VERSION").read_text().strip()

from .primitives.transpose import TransposeSegments, TransposeIrrepsLayout
from .primitives.segmented_polynomial import SegmentedPolynomial
from .primitives.deprecated import (
    TensorProduct,
    EquivariantTensorProduct,
    SymmetricTensorProduct,
    IWeightedSymmetricTensorProduct,
)
from .operations.tp_channel_wise import ChannelWiseTensorProduct
from .operations.tp_fully_connected import FullyConnectedTensorProduct
from .operations.linear import Linear
from .operations.symmetric_contraction import SymmetricContraction
from .operations.rotation import (
    Rotation,
    encode_rotation_angle,
    vector_to_euler_angles,
    Inversion,
)
from .operations.spherical_harmonics import SphericalHarmonics
from molrep.eq.torch import layers


def onnx_custom_translation_table():
    r"""
    Returns ONNX translation table for custom operations from cuequivariance_ops_torch
    to be passed to torch.onnx.export as 'custom_translation_table' argument.

    Example:
    >>> import molrep.eq.torch
    >>> # cueq_custon_onnx_table = molrep.eq.torch.onnx_custom_translation_table()
    >>> # onnx_program = torch.onnx.export(module, inputs, custom_translation_table=cueq_custom_onnx_table)
    """
    from cuequivariance_ops_torch.onnx import op_table

    return op_table


def register_tensorrt_plugins():
    r"""
    Registers TensorRT plugins for custom operations from cuequivariance_ops_torch

    Example:
    >>> import molrep.eq.torch
    >>> # molrep.eq.torch.register_plugins()
    """
    from cuequivariance_ops_torch.tensorrt import register_plugins

    register_plugins()


__all__ = [
    "TransposeSegments",
    "TransposeIrrepsLayout",
    "SegmentedPolynomial",
    "TensorProduct",
    "EquivariantTensorProduct",
    "SymmetricTensorProduct",
    "IWeightedSymmetricTensorProduct",
    "ChannelWiseTensorProduct",
    "FullyConnectedTensorProduct",
    "Linear",
    "SymmetricContraction",
    "Rotation",
    "encode_rotation_angle",
    "vector_to_euler_angles",
    "Inversion",
    "SphericalHarmonics",
    "layers",
    "onnx_custom_translation_table",
    "register_tensorrt_plugins",
]
