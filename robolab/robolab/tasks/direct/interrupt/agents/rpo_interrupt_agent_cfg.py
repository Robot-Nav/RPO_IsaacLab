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

# RPO中断恢复agent配置：标准PPO + 左右对称镜像数据增强(10帧历史)
# 镜像映射复用base逻辑，actor obs 79维/帧，critic obs 140维/帧，各10帧堆叠
# mirror_loss_coeff=0.2高于attn_enc的0.1，更强约束对称性以稳定中断恢复

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (  # noqa:F401
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRndCfg,
    RslRlSymmetryCfg,
)
import torch
from tensordict import TensorDict
from functools import lru_cache

from robolab.tasks.direct.base import (  # noqa:F401
    BaseAgentCfg,
)


def generate_joint_mirror(start_idx):
    """生成关节镜像索引与符号(23关节左右对称)。

    布局：[hip_yaw_L, hip_yaw_R, torso, (成对关节_L,R)...]。
    返回(mirror_indices, mirror_signs)：indices左右置换，signs按关节类型反转(yaw/roll/pitch=-1,屈伸=+1)。
    """
    mirror_indices = []
    mirror_indices.extend([start_idx + 1, start_idx + 0])
    mirror_indices.append(start_idx + 2)  # torso居中，保持原位
    for i in range(start_idx + 3, start_idx + 23, 2):
        mirror_indices.extend([i + 1, i])  # 成对关节左右互换
    mirror_signs = [-1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1]
    return mirror_indices, mirror_signs

# 关节段镜像起点：joint_pos=9, joint_vel=32, action=55 (与base一致)
joint_pos_mirror_indices, joint_pos_mirror_signs = generate_joint_mirror(9)
joint_vel_mirror_indices, joint_vel_mirror_signs = generate_joint_mirror(32)
action_mirror_indices, action_mirror_signs = generate_joint_mirror(55)
# actor观测镜像：[0:3]角速度 + [3:6]投影重力 + [6:9]指令 + 关节位置 + 关节速度 + 动作 + [78]interrupt_mask
policy_obs_mirror_indices = [0, 1, 2,\
                             3, 4, 5,\
                             6, 7, 8]\
                            + joint_pos_mirror_indices + joint_vel_mirror_indices + action_mirror_indices\
                            + [78]
# 符号：角速度yaw反转/roll保持；重力y反转/z保持；指令vx反转/vy保持/yaw反转；interrupt_mask保持
policy_obs_mirror_signs = [-1, 1, -1,\
                           1, -1, 1,\
                           1, -1, -1] + joint_pos_mirror_signs + joint_vel_mirror_signs + action_mirror_signs\
                           + [1]
# critic扩展段：[79:82]线速度 + [83:84]接触(左右足) + [85:90]接触力 + [91:92]空中时间 + [93:94]足部高度
joint_acc_mirror_indices, joint_acc_mirror_signs = generate_joint_mirror(94)
joint_torques_mirror_indices, joint_torques_mirror_signs = generate_joint_mirror(117)
critic_obs_mirror_indices = policy_obs_mirror_indices +\
                            [79, 80, 81,\
                             83, 82,\
                             87, 88, 89, 84, 85, 86,\
                             91, 90,\
                             93, 92]\
                            + joint_acc_mirror_indices + joint_torques_mirror_indices
critic_obs_mirror_signs = policy_obs_mirror_signs +\
                           [1, -1, 1,\
                            1, 1,\
                            1, -1, 1, 1, -1, 1,\
                            1, 1,\
                            1, 1]\
                            + joint_acc_mirror_signs + joint_torques_mirror_signs
# 动作镜像：23维左右成对互换，yaw/roll/pitch反转
act_mirror_indices = [1, 0, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21]
act_mirror_signs = [-1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1]
# 历史帧堆叠镜像：每帧偏移79(actor)或140(critic)，10帧历史展开
policy_obs_mirror_indices_expanded = []
for i in range(10):
    offset = i * 79
    for idx in policy_obs_mirror_indices:
        policy_obs_mirror_indices_expanded.append(idx + offset)
policy_obs_mirror_signs_expanded = policy_obs_mirror_signs * 10

critic_obs_mirror_indices_expanded = []
for i in range(10):
    offset = i * 140
    for idx in critic_obs_mirror_indices:
        critic_obs_mirror_indices_expanded.append(idx + offset)
critic_obs_mirror_signs_expanded = critic_obs_mirror_signs * 10

@lru_cache(maxsize=None)
def get_policy_obs_mirror_signs_tensor(device, dtype):
    """缓存符号张量，避免每步重建。"""
    return torch.tensor(policy_obs_mirror_signs_expanded, device=device, dtype=dtype)

def mirror_policy_observation(policy_obs):
    """对actor观测做左右镜像：按索引重排并乘符号。"""
    mirrored_policy_obs = policy_obs[..., policy_obs_mirror_indices_expanded]
    signs = get_policy_obs_mirror_signs_tensor(device=policy_obs.device, dtype=policy_obs.dtype)
    mirrored_policy_obs = mirrored_policy_obs * signs
    return mirrored_policy_obs

@lru_cache(maxsize=None)
def get_critic_obs_mirror_signs_tensor(device, dtype):
    """缓存critic符号张量。"""
    return torch.tensor(critic_obs_mirror_signs_expanded, device=device, dtype=dtype)

def mirror_critic_observation(critic_obs):
    """对critic观测做左右镜像。"""
    mirrored_critic_obs = critic_obs[..., critic_obs_mirror_indices_expanded]
    signs = get_critic_obs_mirror_signs_tensor(device=critic_obs.device, dtype=critic_obs.dtype)
    mirrored_critic_obs = mirrored_critic_obs * signs
    return mirrored_critic_obs

@lru_cache(maxsize=None)
def get_act_mirror_signs_tensor(device, dtype):
    """缓存动作符号张量。"""
    return torch.tensor(act_mirror_signs, device=device, dtype=dtype)

def mirror_actions(actions):
    """对动作做左右镜像，生成对称动作标签。"""
    mirrored_actions = actions[..., act_mirror_indices]
    signs = get_act_mirror_signs_tensor(device=actions.device, dtype=actions.dtype)
    mirrored_actions = mirrored_actions * signs
    return mirrored_actions

def data_augmentation_func(env, obs, actions):
    """数据增强：原始batch与镜像batch拼接，等价2倍数据量。"""
    if obs is None:
        obs_aug = None
    else:
        obs_mirror = obs.clone()
        obs_mirror["policy"] = mirror_policy_observation(obs["policy"])
        if "critic" in obs.keys():
            obs_mirror["critic"] = mirror_critic_observation(obs["critic"])
        obs_aug = torch.cat([obs, obs_mirror], dim=0)
    if actions is None:
        actions_aug = None
    else:
        actions_aug = torch.cat((actions, mirror_actions(actions)), dim=0)
    return obs_aug, actions_aug


@configclass
class RPOInterruptAgentCfg(BaseAgentCfg):
    """RPO中断恢复agent配置：标准PPO + 10帧历史 + 对称性约束(mirror_loss=0.2)。"""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name: str = "rpo_interrupt"
        self.wandb_project: str = "rpo_interrupt"
        self.seed = 42
        self.num_steps_per_env = 24
        self.max_iterations = 9001
        self.save_interval = 1000
        self.actor_obs_normalization: True
        self.critic_obs_normalization: True
        self.algorithm = RslRlPpoAlgorithmCfg(
            class_name="PPO",
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",  # 自适应学习率，根据KL偏离desired_kl调整
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            normalize_advantage_per_mini_batch=False,
            symmetry_cfg=RslRlSymmetryCfg(
                use_data_augmentation=True,
                use_mirror_loss=True,
                mirror_loss_coeff=0.2,  # 镜像损失系数，高于attn_enc的0.1，更强对称约束
                data_augmentation_func=data_augmentation_func
            ),
            rnd_cfg=None,  # RslRlRndCfg()
        )
        self.clip_actions = 100.0