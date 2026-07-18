# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# MoE混合专家Actor-Critic实现
# Actor和Critic均替换为MoeLayer：多专家并行MLP + 门控网络加权聚合
# 不支持state_dependent_std，仅使用scalar/log标准差参数化

from __future__ import annotations

from typing import Any

import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.modules.moe import MoeLayer


class MoEActorCritic(ActorCritic):
    """Actor-critic with MoE actor and MoE critic (scalar/std noise only; no state_dependent_std).

    用MoeLayer替换标准MLP实现Actor与Critic，通过门控网络对多专家输出加权，
    提升模型容量而不显著增加推理计算量。
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
            "MoE actor-critic:"
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
            raise NotImplementedError("MoEActorCritic does not support state_dependent_std=True")
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
