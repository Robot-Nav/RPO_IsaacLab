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

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Functions to specify the symmetry in the observation and action space for ANYmal."""
# RPO对称性约束：左右镜像增强数据，把单条轨迹扩展为原+镜像两条，加速策略收敛
# 镜像规则：左右关节互换、roll/yaw方向取反、y方向速度/重力取反、深度图/高度图按y翻转
from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# specify the functions that are available for import
__all__ = ["compute_symmetric_states"]


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Augments the given observations and actions by applying symmetry transformations.

    ``env`` is kept for compatibility with RSL-RL's symmetry callback signature.

    This function creates augmented versions of the provided observations and actions by applying
    four symmetrical transformations: original, left-right, front-back, and diagonal. The symmetry
    transformations are beneficial for reinforcement learning tasks by providing additional
    diverse data without requiring additional data collection.

    Args:
        env: The environment instance.
        obs: The original observation tensor dictionary. Defaults to None.
        actions: The original actions tensor. Defaults to None.

    Returns:
        Augmented observations and actions tensors, or None if the respective input was None.
    """
    # 对称增强：原batch + 左右镜像batch 拼接为2倍batch，无额外采样成本
    # observations
    if obs is not None:
        batch_size = obs.batch_size[0]
        # since we have 2 different symmetries, we need to augment the batch size by 2
        obs_aug = obs.repeat(2)

        # policy observation group
        # -- original
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        # -- left-right
        obs_aug["policy"][batch_size : 2 * batch_size] = _transform_policy_obs_left_right(obs["policy"])

        # critic observation group
        # -- original
        obs_aug["critic"][:batch_size] = obs["critic"][:]
        # -- left-right
        obs_aug["critic"][batch_size : 2 * batch_size] = _transform_critic_obs_left_right(env, obs["critic"])

    else:
        obs_aug = None

    # actions
    if actions is not None:
        batch_size = actions.shape[0]
        # since we have 2 different symmetries, we need to augment the batch size by 2
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        # -- original
        actions_aug[:batch_size] = actions[:]
        # -- left-right
        actions_aug[batch_size : 2 * batch_size] = _transform_actions_left_right(actions)

    else:
        actions_aug = None

    return obs_aug, actions_aug


"""
Symmetry functions for observations.
"""


def _height_scan_left_right_dims(env: ManagerBasedRLEnv) -> tuple[int, int, int]:
    """(history_length, ny, nx) from the running env cfg (dataclass fields are not on config *classes*)."""
    # 从运行时cfg解析高度扫描网格维度，用于左右镜像时正确翻转y轴
    cfg = getattr(env, "unwrapped", env).cfg
    pat = cfg.scene.height_scanner.pattern_cfg
    if pat.ordering != "xy":
        raise NotImplementedError(
            "height_scan L-R symmetry only supports GridPatternCfg ordering 'xy';"
            f" extend layouts if pattern uses ordering {pat.ordering!r}."
        )
    hist = cfg.observations.critic.height_scan.history_length
    res = float(pat.resolution)
    s0, s1 = float(pat.size[0]), float(pat.size[1])
    # Match ``isaaclab.sensors.ray_caster.patterns.grid_pattern`` (same arange endpoints / step).
    # 复用grid_pattern的arange端点和步长，保证维度与传感器实际输出一致
    nx = int(torch.arange(-s0 / 2, s0 / 2 + 1.0e-9, res).numel())
    ny = int(torch.arange(-s1 / 2, s1 / 2 + 1.0e-9, res).numel())
    return hist, ny, nx


def _transform_height_scan_left_right(env: ManagerBasedRLEnv, hs: torch.Tensor) -> torch.Tensor:
    """Mirror lateral (y) grid rows; scalars only reorder (unlike ``depth_image``, scan is flat w.r.t. spatial axes)."""
    # 高度扫描左右镜像：按y轴翻转网格行，标量值不变（地形高度本身无方向性）
    hist, ny, nx = _height_scan_left_right_dims(env)
    out = hs.view(hs.shape[0], hist, ny, nx).flip(dims=[2])
    return out.reshape(hs.shape)


def _transform_policy_obs_left_right(obs: TensorDict) -> TensorDict:
    """Left-right mirror for policy observations (``ObservationsCfg.PolicyCfg`` with ``concatenate_terms=False``)."""
    # 策略观测镜像：角速度x/z取反、重力y取反、速度命令y/z取反、关节左右互换
    obs = obs.clone()
    obs["base_ang_vel"] = _apply_xyz_sign(obs["base_ang_vel"], [-1, 1, -1])
    obs["projected_gravity"] = _apply_xyz_sign(obs["projected_gravity"], [1, -1, 1])
    obs["velocity_commands"] = _apply_xyz_sign(obs["velocity_commands"], [1, -1, -1])
    obs["joint_pos"] = _switch_joints_left_right_flat(obs["joint_pos"])
    obs["joint_vel"] = _switch_joints_left_right_flat(obs["joint_vel"])
    obs["actions"] = _switch_joints_left_right_flat(obs["actions"])
    if "depth_image" in obs:
        obs["depth_image"] = _transform_depth_obs_left_right(obs["depth_image"])
    return obs


def _transform_critic_obs_left_right(env: ManagerBasedRLEnv, obs: TensorDict) -> TensorDict:
    """Left-right mirror for critic observations."""
    # 评论员观测镜像：含特权信息base_lin_vel，y分量取反；额外处理height_scan
    obs = obs.clone()
    obs["base_lin_vel"] = _apply_xyz_sign(obs["base_lin_vel"], [1, -1, 1])
    obs["base_ang_vel"] = _apply_xyz_sign(obs["base_ang_vel"], [-1, 1, -1])
    obs["projected_gravity"] = _apply_xyz_sign(obs["projected_gravity"], [1, -1, 1])
    obs["velocity_commands"] = _apply_xyz_sign(obs["velocity_commands"], [1, -1, -1])
    obs["joint_pos"] = _switch_joints_left_right_flat(obs["joint_pos"])
    obs["joint_vel"] = _switch_joints_left_right_flat(obs["joint_vel"])
    obs["actions"] = _switch_joints_left_right_flat(obs["actions"])
    if "depth_image" in obs:
        obs["depth_image"] = _transform_depth_obs_left_right(obs["depth_image"])
    if "height_scan" in obs:
        obs["height_scan"] = _transform_height_scan_left_right(env, obs["height_scan"])
    return obs


def _transform_depth_obs_left_right(obs: torch.Tensor) -> torch.Tensor:
    """Apply a left-right symmetry transformation to depth image observations."""
    # 深度图镜像：沿最后一维（宽度方向）翻转，对应y轴左右镜像
    return torch.flip(obs, dims=(-1,))


def _apply_xyz_sign(obs: torch.Tensor, signs: list[int]) -> torch.Tensor:
    # 按signs对xyz三分量施加正负号，用于镜像时翻转y方向或反转roll/yaw
    obs_shape = obs.shape
    obs = obs.reshape(*obs_shape[:-1], -1, 3)
    obs = obs * torch.tensor(signs, device=obs.device, dtype=obs.dtype)
    return obs.reshape(obs_shape)


def _switch_joints_left_right_flat(joint_data: torch.Tensor) -> torch.Tensor:
    # 把扁平关节数据reshape为(num_joints=23)后做左右互换，再恢复原形状
    joint_data_shape = joint_data.shape
    joint_data = joint_data.reshape(*joint_data_shape[:-1], -1, 23)
    joint_data = _switch_joints_left_right(joint_data)
    return joint_data.reshape(joint_data_shape)


"""
Symmetry functions for actions.
"""


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the actions tensor.

    This function modifies the given actions tensor by applying transformations
    that represent a symmetry with respect to the left-right axis. This includes
    flipping the joint positions, joint velocities, and last actions for the
    ANYmal robot.

    Args:
        actions: The actions tensor to be transformed.

    Returns:
        The transformed actions tensor with left-right symmetry applied.
    """
    # 动作镜像：复用关节左右互换逻辑，保证镜像状态下的动作与镜像策略一致
    actions = actions.clone()
    actions[:] = _switch_joints_left_right(actions[:])
    return actions


"""
Helper functions for symmetry.

In Isaac Sim, the joint ordering is as follows:
[
'left_thigh_yaw_joint',   #0
'right_thigh_yaw_joint',  #1
'torso_joint',            #2
'left_thigh_roll_joint',  #3
'right_thigh_roll_joint', #4
'left_arm_pitch_joint',   #5
'right_arm_pitch_joint',  #6
'left_thigh_pitch_joint', #7
'right_thigh_pitch_joint',#8
'left_arm_roll_joint',    #9
'right_arm_roll_joint',   #10
'left_knee_joint',        #11
'right_knee_joint',       #12
'left_arm_yaw_joint',     #13
'right_arm_yaw_joint',    #14
'left_ankle_pitch_joint', #15
'right_ankle_pitch_joint',#16
'left_elbow_pitch_joint', #17
'right_elbow_pitch_joint',#18
'left_ankle_roll_joint',  #19
'right_ankle_roll_joint', #20
'left_elbow_yaw_joint',   #21
'right_elbow_yaw_joint'   #22
]

"""


def _switch_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the joint data tensor."""
    # 关节左右互换：先交换左右关节索引，再对roll/yaw类关节取反（镜像后旋转方向反转）
    joint_data_switched = joint_data.clone()
    # left <-- right
    joint_data_switched[..., [0, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]] = joint_data[..., [1, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]]
    # right <-- left
    joint_data_switched[..., [1, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]] = joint_data[..., [0, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]]
    # roll/yaw类关节镜像后方向反转：thigh_yaw/torso/thigh_roll/arm_roll/arm_yaw/ankle_roll/elbow_yaw
    joint_data_switched[..., [0,1,2,3,4,9,10,13,14,19,20,21,22]] = -1 * joint_data_switched[..., [0,1,2,3,4,9,10,13,14,19,20,21,22]]

    return joint_data_switched