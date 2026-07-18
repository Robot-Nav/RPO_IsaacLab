# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Parkour AMP environment wiring :class:`~robolab.tasks.manager_based.parkour.managers.MultiRewardCfg` to MultiReward."""
# Parkour障碍穿越环境，机器人在复杂地形（台阶/斜坡/窄道）上自主导航
# 在AmpEnv基础上替换为MultiRewardManager支持多奖励组课程调权，并将字典奖励转回RSL-RL所需张量
from __future__ import annotations

import torch

from isaaclab.envs import VecEnvStepReturn

from robolab.tasks.manager_based.amp.amp_env import AmpEnv
from robolab.tasks.manager_based.parkour.managers import DummyRewardCfg, MultiRewardCfg, MultiRewardManager


def _reward_dict_to_vector(rew: dict[str, torch.Tensor]) -> torch.Tensor:
    """Stack multi-group rewards for RSL-RL, which expects a tensor (usually one column)."""
    # MultiRewardManager返回按奖励组分组的字典，RSL-RL只接收张量形式，需要做一次堆叠转换
    tensors = tuple(rew.values())
    if not tensors:
        raise ValueError("MultiRewardManager produced an empty reward dict.")
    # 单奖励组保持(num_envs,)形状兼容旧调用；多奖励组沿最后一维堆叠成(num_envs, num_groups)
    out = torch.stack(tensors, dim=-1)
    return out.squeeze(-1) if out.shape[-1] == 1 else out


class ParkourEnv(AmpEnv):
    """Same as :class:`~robolab.tasks.manager_based.amp.amp_env.AmpEnv` but swaps in
    :class:`~robolab.tasks.manager_based.parkour.managers.MultiRewardManager` when ``cfg.rewards``
    is a :class:`~robolab.tasks.manager_based.parkour.managers.MultiRewardCfg`.

    Enables curriculum helpers that adjust per-environment reward weights via
    ``get_per_env_term_weights`` / ``set_term_weight_for_envs``.

    RSL-RL expects tensor rewards from ``step``; ``MultiRewardManager.compute`` returns a dict, so we
    convert it here without touching ``rsl_rl``.
    """
    # 关键设计：先以空DummyRewardCfg走父类load_managers完成观测/动作/终止等管理器装配，
    # 再用真正的MultiRewardCfg覆盖奖励管理器，避免父类对奖励配置的额外校验阻塞流程

    def load_managers(self):
        reward_group_cfg = None
        # 检测到多奖励组配置时先暂存并替换为占位配置，让父类装配其他管理器
        if isinstance(self.cfg.rewards, MultiRewardCfg):
            reward_group_cfg = self.cfg.rewards
            self.cfg.rewards = DummyRewardCfg()

        super().load_managers()

        # 装配完成后恢复真实奖励配置并用MultiRewardManager替换奖励管理器
        if reward_group_cfg is not None:
            self.cfg.rewards = reward_group_cfg
            self.reward_manager = MultiRewardManager(self.cfg.rewards, self)
            print("[INFO] Multi Reward Manager: ", self.reward_manager)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        obs, rew, terminated, truncated, extras = super().step(action)
        # 父类step返回字典奖励时转回张量，同步刷新reward_buf供RSL-RL读取
        if isinstance(rew, dict):
            rew = _reward_dict_to_vector(rew)
            self.reward_buf = rew
        return obs, rew, terminated, truncated, extras
