# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Definitions for components of modules."""

# 网络组件包入口：暴露MLP/CNN/记忆/归一化/注意力编码器等基础网络组件
from .cnn import CNN
from .memory import HiddenState, Memory
from .mlp import MLP
from .normalization import EmpiricalDiscountedVariationNormalization, EmpiricalNormalization
from .attn_encoder import AttentionEncoder

__all__ = [
    "CNN",
    "MLP",
    "EmpiricalDiscountedVariationNormalization",
    "EmpiricalNormalization",
    "HiddenState",
    "Memory",
    "AttentionEncoder",
]
