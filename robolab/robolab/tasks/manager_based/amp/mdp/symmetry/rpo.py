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

# RPO机器人左右对称性变换：用于RSL-RL数据增强与镜像损失
# 通过左右镜像扩充样本，加速策略学习并保证步态对称
# 关键约定：23个关节顺序固定，镜像时左右关节互换并翻转yaw/roll方向符号

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
    # 对称性增强入口：原始+左右镜像，batch翻倍
    # 镜像逻辑：xyz分量按物理镜像规则翻转符号（如y→-y），关节按左右互换

    # observations
    if obs is not None:
        batch_size = obs.batch_size[0]
        # since we have 2 different symmetries, we need to augment the batch size by 2
        obs_aug = obs.repeat(2)

        # policy observation group
        # -- original
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        # -- left-right
        obs_aug["policy"][batch_size : 2 * batch_size] = _transform_policy_obs_left_right(env, obs["policy"])

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
def _history_length(env: ManagerBasedRLEnv, group_name: str) -> int:
    cfg = getattr(env, "unwrapped", env).cfg
    history_length = getattr(getattr(cfg.observations, group_name), "history_length", 0)
    return history_length if history_length is not None and history_length > 0 else 1


def _transform_policy_obs_left_right(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    """Left-right mirror for flat policy observations (``ObservationsCfg.PolicyCfg`` with ``concatenate_terms=True``)."""
    # policy观测镜像：base_ang_vel/projected_gravity/velocity_cmd + joint_pos/joint_vel/actions
    # 每个xyz向量按镜像规则翻转符号，关节部分按左右互换
    obs_shape = obs.shape
    history_length = _history_length(env, "policy")
    expected_dim = history_length * (3 + 3 + 3 + 23 + 23 + 23)
    assert obs_shape[-1] == expected_dim, f"Expected policy obs dim to be {expected_dim}, got {obs_shape[-1]}."
    obs = obs.clone()
    offset = 0
    term_dim = 3 * history_length
    # base_ang_vel: x→-x, z→-z（左右镜像绕y轴，y不变）
    obs[..., offset : offset + term_dim] = _apply_xyz_sign(obs[..., offset : offset + term_dim], [-1, 1, -1])
    offset += term_dim
    # projected_gravity: x→-x, z不变（y翻转）
    obs[..., offset : offset + term_dim] = _apply_xyz_sign(obs[..., offset : offset + term_dim], [1, -1, 1])
    offset += term_dim
    # velocity_commands: lin_x不变, lin_y→-lin_y, ang_z→-ang_z
    obs[..., offset : offset + term_dim] = _apply_xyz_sign(obs[..., offset : offset + term_dim], [1, -1, -1])
    offset += term_dim
    term_dim = 23 * history_length
    # joint_pos/joint_vel/actions：左右关节互换 + yaw/roll方向翻转
    obs[..., offset : offset + term_dim] = _switch_joints_left_right_flat(obs[..., offset : offset + term_dim])
    offset += term_dim
    obs[..., offset : offset + term_dim] = _switch_joints_left_right_flat(obs[..., offset : offset + term_dim])
    offset += term_dim
    obs[..., offset : offset + term_dim] = _switch_joints_left_right_flat(obs[..., offset : offset + term_dim])
    offset += term_dim
    return obs


def _transform_critic_obs_left_right(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    """Left-right mirror for flat critic observations.

    After the base terms (3x3 + 23x3) * history_length,
    optionally followed by height_scan (cols*rows) * history_length.
    """
    obs_shape = obs.shape
    history_length = _history_length(env, "critic")
    base_dim = history_length * (3 + 3 + 3 + 3 + 23 + 23 + 23)  # 81 * history_length
    total_dim = obs_shape[-1]
    has_height_scan = total_dim > base_dim
    if has_height_scan:
        # Detect grid: height_scan dims = (total - base) / history_length
        hs_per_frame = (total_dim - base_dim) // history_length
        _height_scan_cols = 16  # must match grid in env cfg
        _height_scan_rows = hs_per_frame // _height_scan_cols
        assert hs_per_frame == _height_scan_cols * _height_scan_rows, (
            f"height_scan points ({hs_per_frame}) doesn't match "
            f"grid {_height_scan_cols}x{_height_scan_rows}"
        )
    obs = obs.clone()
    offset = 0
    term_dim = 3 * history_length
    obs[..., offset:offset + term_dim] = _apply_xyz_sign(obs[..., offset:offset + term_dim], [1, -1, 1])
    offset += term_dim
    obs[..., offset:offset + term_dim] = _apply_xyz_sign(obs[..., offset:offset + term_dim], [-1, 1, -1])
    offset += term_dim
    obs[..., offset:offset + term_dim] = _apply_xyz_sign(obs[..., offset:offset + term_dim], [1, -1, 1])
    offset += term_dim
    obs[..., offset:offset + term_dim] = _apply_xyz_sign(obs[..., offset:offset + term_dim], [1, -1, -1])
    offset += term_dim
    term_dim = 23 * history_length
    obs[..., offset:offset + term_dim] = _switch_joints_left_right_flat(obs[..., offset:offset + term_dim])
    offset += term_dim
    obs[..., offset:offset + term_dim] = _switch_joints_left_right_flat(obs[..., offset:offset + term_dim])
    offset += term_dim
    obs[..., offset:offset + term_dim] = _switch_joints_left_right_flat(obs[..., offset:offset + term_dim])
    offset += term_dim
    # height_scan left-right mirror: flip columns within each row
    if has_height_scan:
        obs[..., offset:total_dim] = _mirror_height_scan(
            obs[..., offset:total_dim],
            history_length,
            _height_scan_cols,
            _height_scan_rows,
        )
    return obs


def _mirror_height_scan(
    hs: torch.Tensor, history_length: int, cols: int, rows: int
) -> torch.Tensor:
    """Left-right mirror of height_scan: reverse column order per row, per history frame.

    Grid ordering: xy (row-major). Shape: [..., history_length * cols * rows].
    """
    hs_shape = hs.shape
    hs = hs.reshape(*hs_shape[:-1], history_length, rows, cols)
    hs = torch.flip(hs, dims=[-1])  # flip columns
    return hs.reshape(hs_shape)


def _apply_xyz_sign(obs: torch.Tensor, signs: list[int]) -> torch.Tensor:
    obs_shape = obs.shape
    obs = obs.reshape(*obs_shape[:-1], -1, 3)
    obs = obs * torch.tensor(signs, device=obs.device, dtype=obs.dtype)
    return obs.reshape(obs_shape)


def _switch_joints_left_right_flat(joint_data: torch.Tensor) -> torch.Tensor:
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
    # 关节左右镜像：先互换左右关节索引，再翻转yaw/roll方向的符号
    # 步骤1：左右关节互换（如left_thigh_yaw ↔ right_thigh_yaw）
    # 步骤2：yaw/roll类关节符号翻转（镜像后方向相反）
    joint_data_switched = joint_data.clone()
    # left <-- right
    joint_data_switched[..., [0, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]] = joint_data[..., [1, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]]
    # right <-- left
    joint_data_switched[..., [1, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]] = joint_data[..., [0, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]]

    # yaw/roll类关节符号翻转：thigh_yaw/roll、arm_roll/yaw、ankle_roll、elbow_yaw等
    joint_data_switched[..., [0,1,2,3,4,9,10,13,14,19,20,21,22]] = -1 * joint_data_switched[..., [0,1,2,3,4,9,10,13,14,19,20,21,22]]

    return joint_data_switched