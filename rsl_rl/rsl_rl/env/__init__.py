# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Submodule defining the environment definitions."""

# 环境包入口：暴露向量化环境抽象基类VecEnv
from .vec_env import VecEnv

__all__ = ["VecEnv"]
