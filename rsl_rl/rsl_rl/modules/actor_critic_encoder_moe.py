# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# 编码器+混合专家Actor-Critic实现
# 在EncoderActorCritic基础上，将Actor/Critic的trunk替换为MoeLayer
# 适用于图像感知任务中提升模型容量而不显著增加推理开销

from __future__ import annotations

from typing import Any

import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules.actor_critic_encoder import EncoderActorCritic
from rsl_rl.modules.moe import MoeLayer


class EncoderMoEActorCritic(EncoderActorCritic):
    """Encoder actor-critic where trunk actor and critic are MoE layers.

    在EncoderActorCritic基础上，将Actor/Critic的trunk MLP替换为MoeLayer，
    通过门控加权多专家输出提升模型容量，适合图像感知任务。
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        num_moe_experts: int = 8,
        moe_gate_hidden_dims: list[int] | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        self.num_moe_experts = num_moe_experts
        self.moe_gate_hidden_dims = list(moe_gate_hidden_dims or [])
        gh = self.moe_gate_hidden_dims if self.moe_gate_hidden_dims else "[] (linear gate)"
        print(
            "MoE encoder actor-critic:"
            f" num_experts={self.num_moe_experts}, gate_hidden_dims={gh},"
            f" actor_hidden_dims={kwargs.get('actor_hidden_dims')},"
            f" critic_hidden_dims={kwargs.get('critic_hidden_dims')},"
            f" activation={kwargs.get('activation', 'elu')}"
        )
        super().__init__(obs, obs_groups, num_actions, **kwargs)

    def _build_actor(
        self,
        num_actor_obs: int,
        num_actions: int,
        actor_hidden_dims: tuple[int] | list[int],
        activation: str,
    ) -> nn.Module:
        if self.state_dependent_std:
            raise NotImplementedError("EncoderMoEActorCritic does not support state_dependent_std=True")
        return MoeLayer(
            num_actor_obs,
            self.num_moe_experts,
            num_actions,
            activation=activation,
            expert_hidden_dims=list(actor_hidden_dims),
            gate_hidden_dims=self.moe_gate_hidden_dims,
        )

    def _build_critic(
        self, num_critic_obs: int, critic_hidden_dims: tuple[int] | list[int], activation: str
    ) -> nn.Module:
        return MoeLayer(
            num_critic_obs,
            self.num_moe_experts,
            1,
            activation=activation,
            expert_hidden_dims=list(critic_hidden_dims),
            gate_hidden_dims=self.moe_gate_hidden_dims,
        )
