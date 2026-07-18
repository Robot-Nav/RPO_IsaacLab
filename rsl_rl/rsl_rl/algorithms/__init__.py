# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation of different learning algorithms."""

# 算法包入口：暴露PPO、蒸馏、PPO-AMP三种强化学习算法
from .distillation import Distillation
from .ppo import PPO
from .ppo_amp import PPOAMP

__all__ = ["PPO", "Distillation", "PPOAMP"]
