# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Definitions for neural-network components for RL-agents."""

# 模块包入口：暴露Actor-Critic变体、MoE、RND、师生网络、AMP判别器、对称性等策略网络
from .actor_critic import ActorCritic
from .actor_critic_cnn import ActorCriticCNN
from .actor_critic_encoder import Conv2dHeadModel, EncoderActorCritic
from .actor_critic_encoder_moe import EncoderMoEActorCritic
from .actor_critic_moe import MoEActorCritic
from .actor_critic_recurrent import ActorCriticRecurrent
from .moe import MoeLayer
from .actor_critic_attn_enc import ActorCriticAttnEnc
from .rnd import RandomNetworkDistillation, resolve_rnd_config
from .student_teacher import StudentTeacher
from .student_teacher_recurrent import StudentTeacherRecurrent
from .symmetry import resolve_symmetry_config
from .amp import AMPDiscriminator, resolve_amp_config

__all__ = [
    "ActorCritic",
    "ActorCriticCNN",
    "Conv2dHeadModel",
    "EncoderActorCritic",
    "EncoderMoEActorCritic",
    "MoEActorCritic",
    "MoeLayer",
    "ActorCriticRecurrent",
    "ActorCriticAttnEnc",
    "RandomNetworkDistillation",
    "StudentTeacher",
    "StudentTeacherRecurrent",
    "resolve_rnd_config",
    "resolve_symmetry_config",
    "AMPDiscriminator",
    "resolve_amp_config",
]
