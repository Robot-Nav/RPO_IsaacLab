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

# RPO AMP训练agent配置：定义RSL-RL的PPO+AMP训练超参与网络结构
# 包含AMP判别器配置（hidden_dims、学习率、梯度惩罚、风格奖励缩放）
# 对称性配置（数据增强+镜像损失）与PPO算法超参（clip、entropy、KL自适应等）
# obs_groups映射：policy→actor，critic→critic，disc/disc_demo→判别器策略侧与参考侧

import os
from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoActorCriticRecurrentCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg
from robolab import ROBOLAB_ROOT_DIR
import torch
from robolab.tasks.manager_based.amp.mdp.symmetry import rpo


@configclass
class RslRlAmpCfg:
    """Configuration class for the AMP (Adversarial Motion Priors) in the training
    """
    # AMP训练配置：判别器超参、损失类型、梯度惩罚等
    # grad_penalty_scale：梯度惩罚系数，防止判别器梯度爆炸
    # task_style_lerp：任务奖励与风格奖励的插值因子，0=纯风格，1=纯任务

    disc_obs_buffer_size: int = 1000
    """Size of the replay buffer for storing discriminator observations"""

    grad_penalty_scale: float = 10.0
    """Scale for the gradient penalty in AMP training"""

    disc_trunk_weight_decay: float = 1.0e-4
    """Weight decay for the discriminator trunk network"""

    disc_linear_weight_decay: float = 1.0e-2
    """Weight decay for the discriminator linear network"""

    disc_learning_rate: float = 1.0e-5
    """Learning rate for the discriminator networks"""

    disc_max_grad_norm: float = 1.0
    """Maximum gradient norm for the discriminator networks"""

    @configclass
    class AMPDiscriminatorCfg:
        """Configuration for the AMP discriminator network."""
        # 判别器网络结构：hidden_dims定义MLP层维，activation为激活函数
        # style_reward_scale：风格奖励缩放因子，影响策略学习参考动作的强度
        # task_style_lerp：任务奖励与风格奖励线性插值，平衡跟踪与风格

        hidden_dims: list[int] = MISSING
        """The hidden dimensions of the AMP discriminator network."""

        activation: str = "elu"
        """The activation function for the AMP discriminator network."""

        style_reward_scale: float = 1.0
        """Scale for the style reward in the training"""

        task_style_lerp: float = 0.0
        """Linear interpolation factor for the task style reward in the AMP training."""

    amp_discriminator: AMPDiscriminatorCfg = AMPDiscriminatorCfg()
    """Configuration for the AMP discriminator network."""

    loss_type: Literal["GAN", "LSGAN", "WGAN"] = "LSGAN"
    """Type of loss function used for the AMP discriminator (e.g., 'GAN', 'LSGAN', 'WGAN')"""
    # 损失类型：LSGAN最小二乘损失，训练更稳定；GAN原始交叉熵；WGAN Wasserstein距离


@configclass
class RslRlPpoActorCriticConv2dCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks with convolutional layers."""

    class_name: str = "ActorCriticConv2d"
    """The policy class name. Default is ActorCriticConv2d."""

    conv_layers_params: list[dict] = [
        {"out_channels": 4, "kernel_size": 3, "stride": 2},
        {"out_channels": 8, "kernel_size": 3, "stride": 2},
        {"out_channels": 16, "kernel_size": 3, "stride": 2},
    ]
    """List of convolutional layer parameters for the convolutional network."""

    conv_linear_output_size: int = 16
    """Output size of the linear layer after the convolutional features are flattened."""


@configclass
class RslRlPpoAmpAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the AMP algorithm."""
    
    class_name: str = "PPOAmp"
    """The algorithm class name. Default is PPOAmp."""

    amp_cfg: RslRlAmpCfg = RslRlAmpCfg()
    """Configuration for the AMP (Adversarial Motion Priors) in the training."""


@configclass
class RslRlOnPolicyRunnerAmpCfg(RslRlOnPolicyRunnerCfg):
    # AMP训练runner主配置：组合policy网络、PPO算法、AMP判别器、对称性配置
    # obs_groups将环境观测组映射到训练侧：policy/critic/disc/disc_demo四组
    # 对称性：use_data_augmentation=True启用镜像数据增强，use_mirror_loss=True启用镜像损失
    # AMP：style_reward_scale=1.5放大风格奖励，task_style_lerp=0.6任务奖励占比60%
    class_name = "AMPRunner"
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "rpo_amp"
    wandb_project = "rpo_amp"
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
        "discriminator": ["disc"],
        "discriminator_demonstration": ["disc_demo"]
    }
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        activation="elu",
    )
    algorithm = RslRlPpoAmpAlgorithmCfg(
        class_name="PPOAMP",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            mirror_loss_coeff=0.2,
            data_augmentation_func=rpo.compute_symmetric_states
        ),
        amp_cfg=RslRlAmpCfg(
            disc_obs_buffer_size=100,
            grad_penalty_scale=10.0,
            disc_trunk_weight_decay=1.0e-3,
            disc_linear_weight_decay=1.0e-1,
            disc_learning_rate=1.0e-4,
            disc_max_grad_norm=1.0,
            amp_discriminator=RslRlAmpCfg.AMPDiscriminatorCfg(
                hidden_dims=[1024, 512],
                activation="elu",
                style_reward_scale=1.5,
                task_style_lerp=0.6
            ),
            loss_type="LSGAN"
        ),
    )


@configclass
class RslRlOnPolicyRunnerAmpRoughCfg(RslRlOnPolicyRunnerAmpCfg):
    # AMP粗糙地形训练runner配置：继承AMP配置，增大迭代次数与保存间隔
    # PPO超参、AMP判别器、对称性配置均继承自RslRlOnPolicyRunnerAmpCfg
    max_iterations = 40000
    save_interval = 1000
    experiment_name = "rpo_amp_rough"
    wandb_project = "rpo_amp_rough"


@configclass
class RslRlOnPolicyRunnerDanceAmpCfg(RslRlOnPolicyRunnerAmpCfg):
    # 舞蹈AMP训练配置：数据已修复（clip软限位+MuJoCo重算key_body_pos），核心是防disc_loss崩溃
    # disc_lr=1e-6大幅放慢判别器；buffer=1000增加参考数据多样性；style_scale=1.5恢复风格信号
    # task_style_lerp=0.5任务/风格各半，确保策略兼顾速度跟踪与动作模仿
    max_iterations = 5000
    experiment_name = "rpo_amp_dance"
    wandb_project = "rpo_amp_dance"

    algorithm = RslRlPpoAmpAlgorithmCfg(
        class_name="PPOAMP",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            mirror_loss_coeff=0.2,
            data_augmentation_func=rpo.compute_symmetric_states
        ),
        amp_cfg=RslRlAmpCfg(
            disc_obs_buffer_size=1000,           # 增大buffer，让判别器看到更多参考数据
            grad_penalty_scale=10.0,
            disc_trunk_weight_decay=1.0e-3,
            disc_linear_weight_decay=1.0e-1,
            disc_learning_rate=1.0e-6,           # 大幅降低判别器学习率，防止disc_loss崩溃
            disc_max_grad_norm=1.0,
            amp_discriminator=RslRlAmpCfg.AMPDiscriminatorCfg(
                hidden_dims=[1024, 512],
                activation="elu",
                style_reward_scale=1.5,          # 数据修复后恢复正常风格奖励权重
                task_style_lerp=0.5              # 任务/风格各50%，确保兼顾跟踪与模仿
            ),
            loss_type="LSGAN"
        ),
    )


@configclass
class RslRlOnPolicyRunnerDanceSingleAmpCfg(RslRlOnPolicyRunnerDanceAmpCfg):
    # 单舞蹈AMP训练：8段舞蹈混合(权重3:1:1...)，dance1_subject2主导
    # 与父类(RPODanceAmpCfg)差异：
    #   task_style_lerp=0.3：70%风格，策略主动模仿舞蹈而非站着不动
    #   style_reward_scale=3.0：放大风格信号，与任务奖励(1.0+0.5=1.5)竞争
    #   disc_learning_rate=5e-7：8段~30k帧极大降低过拟合风险，lr可适度放宽
    #   grad_penalty_scale=15.0：保护判别器泛化
    #   max_iterations=30000：AMP对抗收敛需要更多轮
    experiment_name = "rpo_amp_dance_single"
    wandb_project = "rpo_amp_dance_single"
    max_iterations = 30000

    algorithm = RslRlPpoAmpAlgorithmCfg(
        class_name="PPOAMP",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.02,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            mirror_loss_coeff=0.2,
            data_augmentation_func=rpo.compute_symmetric_states
        ),
        amp_cfg=RslRlAmpCfg(
            disc_obs_buffer_size=3000,            # 增大buffer，8段舞蹈更丰富
            grad_penalty_scale=15.0,
            disc_trunk_weight_decay=1.0e-3,
            disc_linear_weight_decay=1.0e-1,
            disc_learning_rate=5.0e-7,            # 8段~30k帧，判别器不太容易过拟合
            disc_max_grad_norm=1.0,
            amp_discriminator=RslRlAmpCfg.AMPDiscriminatorCfg(
                hidden_dims=[1024, 512],
                activation="elu",
                style_reward_scale=3.0,           # 风格信号与任务信号(1.5)形成2:1竞争
                task_style_lerp=0.3               # 70%风格+30%任务，策略主动模仿舞蹈
            ),
            loss_type="LSGAN"
        ),
    )