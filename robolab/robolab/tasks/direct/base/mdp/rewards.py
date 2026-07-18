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

# locomotion奖励函数集合：速度跟踪、姿态稳定、能耗、动作平滑、步态、关节偏差、中断相关等
# 多数奖励按躯干投影重力z分量做姿态门控（翻倒时奖励清零），避免策略在失败状态下钻空子

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from robolab.envs.base.base_env import BaseEnv


def track_lin_vel_xy_yaw_frame_exp(
    env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """线速度跟踪奖励：在yaw坐标系下计算xy线速度误差，exp核平滑后用姿态门控。"""
    asset: Articulation = env.scene[asset_cfg.name]
    # 把世界系线速度旋转到yaw-only坐标系，仅保留水平分量
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_error = torch.sum(torch.square(env.command_generator.command[:, :2] - vel_yaw[:, :2]), dim=1)
    reward = torch.exp(-lin_vel_error / std**2)
    # 姿态门控：躯干越接近直立奖励越满，翻倒时奖励归零
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_world_exp(
    env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """z向角速度跟踪奖励：世界系角速度与指令z分量误差的exp核。"""
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_generator.command[:, 2] - asset.data.root_ang_vel_w[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def lin_vel_z_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """z向线速度平方惩罚，抑制颠簸。"""
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_lin_vel_b[:, 2])
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def ang_vel_xy_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """xy角速度平方和惩罚，抑制翻滚/俯仰。"""
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def energy(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """关节功率（力矩*速度）绝对值之和，惩罚高能耗。"""
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.sum(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=-1)
    return reward


def action_rate_l2(env: BaseEnv) -> torch.Tensor:
    """相邻动作二阶差平方和，抑制抖动。"""
    return torch.sum(
        torch.square(
            env.action_buffer.buffer[:, -1, :] - env.action_buffer.buffer[:, -2, :]
        ),
        dim=1,
    )

def action_smoothness_l2(env: BaseEnv) -> torch.Tensor:
    """动作二阶差分平方和（jerk），鼓励平滑加速度。"""
    return torch.sum(
        torch.square(
            env.action_buffer.buffer[:, -3, :] - 2*env.action_buffer.buffer[:, -2, :] + env.action_buffer.buffer[:, -1, :]
        ),
        dim=1,
    )


def undesired_contacts(env: BaseEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """非预期接触数量：取历史3帧内最大法向力，>1N即计为接触。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > 1.0
    return torch.sum(is_contact, dim=1)


def flat_orientation_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """躯干水平姿态惩罚：投影重力xy分量的平方和。"""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def is_terminated(env: BaseEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.reset_terminated


def feet_air_time_positive_biped(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """双足步态腾空奖励：单足支撑时另一足腾空时长，clamp到threshold上限。

    threshold过大会鼓励过高抬腿，过小则步频过快。零指令时清零避免原地抬腿。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    is_contact = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    # 接触时取contact_time，腾空时取air_time
    in_mode_time = torch.where(is_contact, contact_time, air_time)
    single_stance = torch.sum(is_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, min=0.0, max=threshold)
    # no reward for zero command
    reward *= (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) > 0.01
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_slide(
    env: BaseEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """足部滑动惩罚：接触状态下足部水平速度乘以接触掩码。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def body_force(
    env: BaseEnv, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400
) -> torch.Tensor:
    """足部接触力超额惩罚：总力减阈值后clamp到max_reward，避免硬冲击损伤。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = torch.sum(torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=2), dim=1)
    reward = (reward - threshold).clamp(min=0.0, max=max_reward)
    return reward


def body_orientation_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """指定刚体姿态惩罚：将世界系重力旋转到刚体系后取xy分量平方和。"""
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = torch.stack(
        [
            math_utils.quat_apply_inverse(
                asset.data.body_quat_w[:, body_id, :], asset.data.GRAVITY_VEC_W
            )
            for body_id in asset_cfg.body_ids
            if body_id is not None
        ],
        dim=-1,
    )
    return torch.sum(torch.sum(torch.square(body_orientation[:, :2, :]), dim=1), dim=-1)


def feet_stumble(env: BaseEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """足部绊倒判定：水平接触力大于3倍垂直接触力时记为绊倒。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 3 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    )


def body_distance_y(
    env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), min: float = 0.2, max: float = 0.5
) -> torch.Tensor:
    """双足y向间距奖励：距离落入[min,max]区间满分，越界指数衰减。

    用于引导合理步宽，避免双腿过窄或过宽。
    """
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    # 将双足世界坐标转换到基座坐标系，仅取y向间距
    root_quat_w = asset.data.root_quat_w.unsqueeze(1).expand(-1, 2, -1)
    root_pos_w = asset.data.root_pos_w.unsqueeze(1).expand(-1, 2, -1)
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    feet_pos_b = math_utils.quat_apply_inverse(root_quat_w, feet_pos_w - root_pos_w)
    distance = torch.abs(feet_pos_b[:, 0, 1] - feet_pos_b[:, 1, 1])
    # 偏离[min,max]两侧时各自指数衰减
    d_min = torch.clamp(distance - min, -0.5, 0.)
    d_max = torch.clamp(distance - max, 0, 0.5)
    return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2


def feet_contact_without_cmd(env: BaseEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    # 零指令下双足同时接触奖励，鼓励静止站立
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    reward = (torch.sum(contacts, dim=-1) == 2).float()
    reward *= (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) < 0.01
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def undesired_foothold(env: BaseEnv, sensor_cfg: SceneEntityCfg, sensor_cfg1: SceneEntityCfg | None = None,
    sensor_cfg2: SceneEntityCfg | None = None, ankle_height: float = 0.035) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    # 接触状态下统计足底扫描命中点高于踝高+1cm的比例，惩罚落脚位置不当
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    undesired_contacts = torch.stack(
        [
            torch.sum(
                (env.scene[sensor.name].data.pos_w[:, 2].unsqueeze(1)
                - env.scene[sensor.name].data.ray_hits_w[..., 2]
                - ankle_height) > 0.01,
                dim=-1
            ) / float(env.scene[sensor.name].data.ray_hits_w.shape[1])
            for sensor in [sensor_cfg1, sensor_cfg2]
            if sensor is not None
        ],
        dim=-1,
    )
    reward = torch.where(contacts, undesired_contacts, 0.0)
    return reward.sum(dim=1)

def upward(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # 投影重力z分量为正（脚朝上）时奖励为负，鼓励躯干直立
    reward = -asset.data.projected_gravity_b[:, 2]
    return reward


def stand_still(
    env: BaseEnv,
    pos_cfg: SceneEntityCfg,
    vel_cfg: SceneEntityCfg,
    pos_weight: float = 1.0,
    vel_weight: float = 1.0,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene["robot"]
    cmd = (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    )
    body_lin_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    body_ang_vel = torch.abs(asset.data.root_ang_vel_b[:, 2])
    body_vel = body_ang_vel + body_lin_vel
    # 零指令且本体静止时才生效，惩罚关节偏离默认角与关节速度
    pos_reward = pos_weight * torch.sum(torch.abs
        (asset.data.joint_pos[:, pos_cfg.joint_ids] - asset.data.default_joint_pos[:, pos_cfg.joint_ids]), dim=1
    )
    vel_reward = vel_weight * torch.sum(torch.abs(asset.data.joint_vel[:, vel_cfg.joint_ids]), dim=1)
    reward = torch.where(
        torch.logical_or(cmd > 0.01, body_vel > 0.5),
        0.0,
        pos_reward + vel_reward,
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height(env: BaseEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), sensor_cfg1: SceneEntityCfg | None = None,
    sensor_cfg2: SceneEntityCfg | None = None, ankle_height: float = 0.035, threshold: float = 0.05):
    """
    Calculates reward based on the clearance of the swing leg from the ground during movement.
    Encourages appropriate lift of the feet during the swing phase of the gait.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    # 摆动足足底距地面高度，减去踝高后截断到[0,1]
    feet_height = torch.stack(
        [
            env.scene[sensor.name].data.pos_w[:, 2]
            - env.scene[sensor.name].data.ray_hits_w[..., 2].mean(dim=-1)
            for sensor in [sensor_cfg1, sensor_cfg2]
            if sensor is not None
        ],
        dim=-1,
    )
    feet_height = torch.clamp(feet_height - ankle_height, min=0.0, max=1.0)
    feet_height = torch.nan_to_num(feet_height, nan=1.0, posinf=1.0, neginf=0)
    # Compute single_stance mask
    single_stance = contacts.sum(dim=1) == 1
    # feet height should be closed to target feet height at the peak
    rew_pos = feet_height > threshold
    # 单足支撑且该足未接触（即摆动足）时，若高度超过阈值则给奖励
    reward = torch.where(torch.logical_and(~contacts, single_stance.unsqueeze(-1)), rew_pos.float(), 0.0).sum(dim=1)
    reward *= (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) > 0.01
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def joint_deviation_interrupt(env: BaseEnv, asset_cfg1: SceneEntityCfg, asset_cfg2: SceneEntityCfg, weight1: float, weight2: float) -> torch.Tensor:
    """Penalize joint deviation during interruption."""
    # extract the used quantities (to enable type-hinting)
    # 非中断环境下生效，惩罚两组关节相对默认角的偏差
    asset1: Articulation = env.scene[asset_cfg1.name]
    asset2: Articulation = env.scene[asset_cfg2.name]
    angle1 = asset1.data.joint_pos[:, asset_cfg1.joint_ids] - asset1.data.default_joint_pos[:, asset_cfg1.joint_ids]
    angle2 = asset2.data.joint_pos[:, asset_cfg2.joint_ids] - asset2.data.default_joint_pos[:, asset_cfg2.joint_ids]
    reward = weight1 * torch.sum(torch.abs(angle1), dim=1) + weight2 * torch.sum(torch.abs(angle2), dim=1)
    reward *= ~env.interrupt_mask
    return reward

def stand_still_interrupt(
    env: BaseEnv,
    pos_cfg: SceneEntityCfg,
    vel_cfg: SceneEntityCfg,
    interrupt_cfg: SceneEntityCfg,
    pos_weight: float = 1.0,
    vel_weight: float = 1.0,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene["robot"]
    cmd = (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    )
    body_lin_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    body_ang_vel = torch.abs(asset.data.root_ang_vel_b[:, 2])
    body_vel = body_ang_vel + body_lin_vel
    # 中断时排除被中断关节（手臂），避免对无法控制的关节施加惩罚
    pos_joint_ids = list(set(pos_cfg.joint_ids) - set(interrupt_cfg.joint_ids))
    vel_joint_ids = list(set(vel_cfg.joint_ids) - set(interrupt_cfg.joint_ids))
    pos_reward = torch.where(env.interrupt_mask,
                             pos_weight * torch.sum(torch.abs(asset.data.joint_pos[:, pos_joint_ids] - asset.data.default_joint_pos[:, pos_joint_ids]), dim=1),
                             pos_weight * torch.sum(torch.abs(asset.data.joint_pos[:, pos_cfg.joint_ids] - asset.data.default_joint_pos[:, pos_cfg.joint_ids]), dim=1))
    vel_reward = torch.where(env.interrupt_mask,
                             vel_weight * torch.sum(torch.abs(asset.data.joint_vel[:, vel_joint_ids]), dim=1),
                             vel_weight * torch.sum(torch.abs(asset.data.joint_vel[:, vel_cfg.joint_ids]), dim=1))
    reward = torch.where(
        torch.logical_or(cmd > 0.01, body_vel > 0.5),
        0.0,
        pos_reward + vel_reward,
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def action_penalty_interrupt(env: BaseEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize action magnitude during interruption."""
    # 中断时惩罚被中断关节的动作幅度，鼓励策略在中断期间输出零动作
    reward = torch.sum(
        torch.square(
            env.action_buffer.buffer[:, -1, asset_cfg.joint_ids]
        ),
        dim=1,
    )
    reward *= env.interrupt_mask
    return reward