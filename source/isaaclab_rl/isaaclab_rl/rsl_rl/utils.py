# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""处理 RSL-RL 跨版本的已弃用配置(适配 rsl-rl 3.x)。"""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

from packaging import version

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg

_V4_0_0 = version.parse("4.0.0")
_V5_0_0 = version.parse("5.0.0")
_MODEL_CFG_NAMES = ("actor", "critic", "student", "teacher")


def handle_deprecated_rsl_rl_cfg(agent_cfg: "RslRlBaseRunnerCfg", installed_version) -> "RslRlBaseRunnerCfg":
    """处理 RSL-RL 跨版本的已弃用配置。

    对于 rsl-rl < 4.0.0:
      - 要求 ``policy`` 配置存在
      - 迁移已弃用的 ``empirical_normalization`` 到 ``actor_obs_normalization`` / ``critic_obs_normalization``
      - 移除仅 rsl-rl >= 4.0.0 支持的 ``optimizer`` 参数
      - 清除 rsl-rl >= 4.0.0 的 model 配置(actor/critic/student/teacher)
    对于 rsl-rl >= 4.0.0:
      - 使用 ``policy`` 推断缺失的 model 配置,然后清除 ``policy``
    """
    installed_version = version.parse(installed_version)

    # rsl-rl < 4.0.0 分支(当前安装版本 3.3.0 走此分支)
    if installed_version < _V4_0_0:
        # policy 配置必须存在
        if not hasattr(agent_cfg, "policy") or _is_missing(agent_cfg.policy):
            raise ValueError(
                "The `policy` configuration is required for rsl-rl < 4.0.0. Please specify the `policy`"
                " configuration or update rsl-rl."
            )

        # 迁移已弃用的 empirical_normalization
        if _has_non_missing_attr(agent_cfg, "empirical_normalization"):
            _handle_empirical_normalization(agent_cfg.policy, agent_cfg)

        # 移除 optimizer 参数(仅 rsl-rl >= 4.0.0 支持)
        from isaaclab_rl.rsl_rl.rl_cfg import RslRlPpoAlgorithmCfg

        if hasattr(agent_cfg.algorithm, "optimizer") and isinstance(agent_cfg.algorithm, RslRlPpoAlgorithmCfg):
            if agent_cfg.algorithm.optimizer != "adam":
                print(
                    "[WARNING]: The `optimizer` parameter for PPO is only available for rsl-rl >= 4.0.0."
                    " Defaulting to `adam` optimizer."
                )
            del agent_cfg.algorithm.optimizer

        # 清除 rsl-rl >= 4.0.0 的 model 配置
        for model_name in _MODEL_CFG_NAMES:
            if _has_non_missing_attr(agent_cfg, model_name):
                print(
                    f"[WARNING]: The `{model_name}` model configuration is only used for rsl-rl >= 4.0.0."
                    " Consider updating rsl-rl or use the `policy` configuration."
                )
                setattr(agent_cfg, model_name, MISSING)

    # rsl-rl >= 4.0.0 分支
    else:
        # 使用 policy 推断新的 model 配置
        if _has_non_missing_attr(agent_cfg, "policy"):
            print(
                "[WARNING]: The `policy` configuration is deprecated for rsl-rl >= 4.0.0. Please use"
                " `actor` and `critic` model configurations instead."
            )

            if _has_non_missing_attr(agent_cfg, "empirical_normalization"):
                _handle_empirical_normalization(agent_cfg.policy, agent_cfg)

            # 移除已弃用的 policy 配置
            agent_cfg.policy = MISSING

        # rsl-rl >= 5.0.0 处理 distribution_cfg
        if installed_version >= _V5_0_0:
            for model_name in _MODEL_CFG_NAMES:
                if _has_non_missing_attr(agent_cfg, model_name):
                    _update_distribution_cfg(getattr(agent_cfg, model_name))

    return agent_cfg


def _is_missing(value) -> bool:
    return isinstance(value, type(MISSING))


def _has_non_missing_attr(obj, attr_name: str) -> bool:
    return hasattr(obj, attr_name) and not _is_missing(getattr(obj, attr_name))


def _handle_empirical_normalization(policy_cfg, agent_cfg):
    """将已弃用的 empirical_normalization 迁移到 policy 的 obs_normalization 字段。"""
    print(
        "[WARNING]: The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization`"
        " and `critic_obs_normalization` as part of the `policy` configuration instead."
    )
    if _is_missing(policy_cfg.actor_obs_normalization):
        policy_cfg.actor_obs_normalization = agent_cfg.empirical_normalization
    if _is_missing(policy_cfg.critic_obs_normalization):
        policy_cfg.critic_obs_normalization = agent_cfg.empirical_normalization
    agent_cfg.empirical_normalization = MISSING


def _update_distribution_cfg(model_cfg):
    """rsl-rl >= 5.0.0 将旧的随机参数迁移到 distribution_cfg。"""
    if hasattr(model_cfg, "distribution_cfg") and model_cfg.distribution_cfg is not None:
        return
    if hasattr(model_cfg, "stochastic") and model_cfg.stochastic is True:
        print(
            "[WARNING]: The `distribution_cfg` configuration is now used to specify the output distribution."
            " Consider updating the configuration to use `distribution_cfg`."
        )
    # 移除已弃用的随机参数
    for key in ("stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"):
        if hasattr(model_cfg, key):
            delattr(model_cfg, key)
