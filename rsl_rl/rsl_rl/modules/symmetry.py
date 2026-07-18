# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# 对称性配置解析工具
# 将env对象注入symmetry_cfg，供后续对称数据增强函数处理不同观测项
# 利用四足机器人左右对称性扩展样本，提升样本效率与策略对称性

from __future__ import annotations

from rsl_rl.env import VecEnv


def resolve_symmetry_config(alg_cfg: dict, env: VecEnv) -> dict:
    """Resolve the symmetry configuration.

    将env对象注入symmetry_cfg["_env"]字段，供对称数据增强函数查询观测项的对称映射关系。

    Args:
        alg_cfg: Algorithm configuration dictionary.
        env: Environment object.

    Returns:
        The resolved algorithm configuration dictionary.
    """
    # 将env注入配置便于对称函数处理不同观测项
    # Note: This is used by the symmetry function for handling different observation terms
    if "symmetry_cfg" in alg_cfg and alg_cfg["symmetry_cfg"] is not None:
        alg_cfg["symmetry_cfg"]["_env"] = env
    else:
        alg_cfg["symmetry_cfg"] = None
    return alg_cfg
