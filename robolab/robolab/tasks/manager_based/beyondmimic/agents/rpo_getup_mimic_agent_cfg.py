
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# RPO GetupMimic 起身训练agent配置
# 与BeyondMimic同结构PPO，迭代次数翻倍至30000（起身任务样本效率较低，需更长训练）
# 其余超参与BeyondMimic保持一致，便于对比与迁移

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class RPOGetupMimicPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24  # 每环境每次rollout步数，与BeyondMimic一致
    max_iterations = 30000  # 起身任务难度高，迭代数加倍
    save_interval = 200
    experiment_name = "rpo_getup_mimic"
    wandb_project = "rpo_getup_mimic"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,  # 初始动作噪声标准差，探索期较大
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        actor_obs_normalization=False,  # 关闭归一化，命令向量含物理量纲不宜归一化
        critic_obs_normalization=False,
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,  # PPO裁剪比率，限制策略更新幅度
        entropy_coef=0.005,  # 熵正则系数，鼓励探索
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",  # 自适应学习率，按目标KL动态调整
        gamma=0.99,
        lam=0.95,  # GAElambda，偏差-方差权衡
        desired_kl=0.01,  # 目标KL散度，adaptive学习率以此为基准
        max_grad_norm=1.0,  # 梯度裁剪阈值
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None
    )
