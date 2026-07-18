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

# 注意力编码器locomotion环境：在BaseEnv基础上将高度扫描从观测向量剥离为独立perception输入
# 配合ActorCriticAttnEnc策略网络，用多头注意力编码地形高度图，actor/critic分别接入perception_a/perception_c

import torch
import numpy as np

from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.utils.buffers import CircularBuffer
from isaaclab.utils.math import quat_apply_inverse,quat_apply_yaw, quat_inv
from robolab.tasks.direct.base import (  # noqa:F401
    BaseEnv,
    BaseEnvCfg
)

class AttnEncEnv(BaseEnv):
    """注意力编码器环境：覆写观测计算与噪声缓冲，把高度扫描单独输出供感知分支使用。"""
    def __init__(self, cfg, render_mode, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.cfg: BaseEnvCfg

    def compute_current_observations(self):
        """计算actor与critic观测，可选将线速度纳入actor观测（vel_in_obs）。"""
        robot = self.robot
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        ang_vel = robot.data.root_ang_vel_b
        lin_vel = robot.data.root_lin_vel_b
        projected_gravity = robot.data.projected_gravity_b
        command = self.command_generator.command
        joint_pos = robot.data.joint_pos - robot.data.default_joint_pos
        joint_vel = robot.data.joint_vel - robot.data.default_joint_vel
        action = self.action_buffer.buffer[:, -1, :]
        if self.cfg.attn_enc.vel_in_obs:
            # 启用时把线速度塞入actor观测，用于特权信息监督学习
            current_actor_obs = torch.cat(
                [
                    ang_vel * self.obs_scales.ang_vel,
                    projected_gravity * self.obs_scales.projected_gravity,
                    command * self.obs_scales.commands,
                    joint_pos * self.obs_scales.joint_pos,
                    joint_vel * self.obs_scales.joint_vel,
                    action * self.obs_scales.actions,
                    lin_vel * self.obs_scales.lin_vel,
                ],
                dim=-1,
            )
        else:
            current_actor_obs = torch.cat(
                [
                    ang_vel * self.obs_scales.ang_vel,
                    projected_gravity * self.obs_scales.projected_gravity,
                    command * self.obs_scales.commands,
                    joint_pos * self.obs_scales.joint_pos,
                    joint_vel * self.obs_scales.joint_vel,
                    action * self.obs_scales.actions,
                ],
                dim=-1,
            )

        feet_contact = torch.max(torch.norm(net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1), dim=1)[0] > 1.0
        feet_contact_force = self.contact_sensor.data.net_forces_w[:, self.feet_cfg.body_ids, :]
        feet_air_time = self.contact_sensor.data.current_air_time[:, self.feet_cfg.body_ids]
        feet_height = torch.stack(
        [
            self.scene[sensor_cfg.name].data.pos_w[:, 2]
            - self.scene[sensor_cfg.name].data.ray_hits_w[..., 2].mean(dim=-1)
            for sensor_cfg in [self.left_feet_scanner_cfg, self.right_feet_scanner_cfg]
            if sensor_cfg is not None
        ],
        dim=-1,
        )
        feet_height = torch.clamp(feet_height - 0.04, min=0.0, max=1.0)
        feet_height = torch.nan_to_num(feet_height, nan=1.0, posinf=1.0, neginf=0)
        joint_torque = robot.data.applied_torque
        joint_acc = robot.data.joint_acc
        # 双足位置（基座坐标系），供critic估计足部相对位置
        root_quat_w = robot.data.root_quat_w.unsqueeze(1).expand(-1, 2, -1)
        root_pos_w = robot.data.root_pos_w.unsqueeze(1).expand(-1, 2, -1)
        feet_pos_w = robot.data.body_pos_w[:, self.feet_cfg.body_ids]
        feet_pos = quat_apply_inverse(root_quat_w, feet_pos_w - root_pos_w)
        if self.cfg.attn_enc.vel_in_obs:
            # actor已含线速度，critic不再重复
            current_critic_obs = torch.cat(
                [current_actor_obs, feet_contact.float(), feet_contact_force.flatten(1), feet_air_time.flatten(1), feet_height.flatten(1), joint_acc, joint_torque, feet_pos.flatten(1)], dim=-1
            )
        else:
            current_critic_obs = torch.cat(
                [current_actor_obs, lin_vel * self.obs_scales.lin_vel, feet_contact.float(), feet_contact_force.flatten(1), feet_air_time.flatten(1), feet_height.flatten(1), joint_acc, joint_torque, feet_pos.flatten(1)], dim=-1
            )

        return current_actor_obs, current_critic_obs

    def _get_observations(self):
        """组装观测：actor/critic本体观测 + 独立perception_a/perception_c高度扫描。"""
        current_actor_obs, current_critic_obs = self.compute_current_observations()
        if self.add_noise:
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec

        # The ray pattern is generated by iterating through x first, then y.
        # This means the flattened (L*W) dimension is ordered as [p(x0,y0), p(x1,y0), ..., p(xL,y0), p(x0,y1), ...].
        if self.cfg.scene_context.height_scanner.enable_height_scan:
            height_scan = (
                self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                - self.height_scanner.data.ray_hits_w[..., 2]
            )
            height_scan = torch.clamp(height_scan - self.cfg.normalization.height_scan_offset, min=-1.0, max=1.0)
            height_scan = torch.nan_to_num(height_scan, nan=1.0, posinf=1.0, neginf=-1.0)
            height_scan *= self.obs_scales.height_scan
            if self.cfg.scene_context.height_scanner.enable_height_scan_actor:
                # actor侧高度扫描需要噪声，critic侧使用干净版本作为特权信息
                height_scan_actor = height_scan.clone()
                if self.add_noise:
                    height_scan_actor += (2 * torch.rand_like(height_scan_actor) - 1) * self.height_scan_noise_vec
                if not self.cfg.attn_enc.use_attn_enc:
                    # 未启用注意力编码器时回退为向量拼接
                    current_actor_obs = torch.cat([current_actor_obs, height_scan_actor], dim=-1)

        self.actor_obs_buffer.append(current_actor_obs)
        self.critic_obs_buffer.append(current_critic_obs)

        actor_obs = self.actor_obs_buffer.buffer.reshape(self.num_envs, -1)
        critic_obs = self.critic_obs_buffer.buffer.reshape(self.num_envs, -1)

        actor_obs = torch.clip(actor_obs, -self.clip_obs, self.clip_obs)
        critic_obs = torch.clip(critic_obs, -self.clip_obs, self.clip_obs)

        if self.cfg.attn_enc.use_attn_enc:
            # 启用注意力编码器：高度扫描单独作为perception输入，由策略网络注意力分支处理
            observations = {"policy": actor_obs, "critic":critic_obs, "perception_a": height_scan_actor, "perception_c": height_scan}
        else:
            observations = {"policy": actor_obs, "critic":critic_obs}
        return observations

    def init_obs_buffer(self):
        """构建噪声向量，按vel_in_obs分支采用不同观测布局。"""
        if self.add_noise:
            if self.cfg.attn_enc.vel_in_obs:
                # actor观测布局：[ang_vel(3), lin_vel(3), proj_grav(3), cmd(3), joint_pos(N), joint_vel(N), action(N)]
                actor_obs, _ = self.compute_current_observations()
                noise_vec = torch.zeros_like(actor_obs[0])
                noise_scales = self.cfg.noise.noise_scales
                noise_vec[:3] = noise_scales.ang_vel * self.obs_scales.ang_vel
                noise_vec[3:6] = noise_scales.lin_vel * self.obs_scales.lin_vel
                noise_vec[6:9] = noise_scales.projected_gravity * self.obs_scales.projected_gravity
                noise_vec[9:12] = 0
                noise_vec[12 : 12 + self.num_actions] = noise_scales.joint_pos * self.obs_scales.joint_pos
                noise_vec[12 + self.num_actions : 12 + self.num_actions * 2] = (
                    noise_scales.joint_vel * self.obs_scales.joint_vel
                )
                noise_vec[12 + self.num_actions * 2 : 12 + self.num_actions * 3] = 0.0
            else:
                # 与BaseEnv一致的标准布局
                actor_obs, _ = self.compute_current_observations()
                noise_vec = torch.zeros_like(actor_obs[0])
                noise_scales = self.cfg.noise.noise_scales
                noise_vec[:3] = noise_scales.ang_vel * self.obs_scales.ang_vel
                noise_vec[3:6] = noise_scales.projected_gravity * self.obs_scales.projected_gravity
                noise_vec[6:9] = 0
                noise_vec[9 : 9 + self.num_actions] = noise_scales.joint_pos * self.obs_scales.joint_pos
                noise_vec[9 + self.num_actions : 9 + self.num_actions * 2] = (
                    noise_scales.joint_vel * self.obs_scales.joint_vel
                )
                noise_vec[9 + self.num_actions * 2 : 9 + self.num_actions * 3] = 0.0
            self.noise_scale_vec = noise_vec

            if self.cfg.scene_context.height_scanner.enable_height_scan:
                height_scan = (
                    self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                    - self.height_scanner.data.ray_hits_w[..., 2]
                )
                height_scan = torch.clamp(height_scan - self.cfg.normalization.height_scan_offset, min=-1.0, max=1.0)
                height_scan = torch.nan_to_num(height_scan, nan=1.0, posinf=1.0, neginf=-1.0)
                height_scan *= self.obs_scales.height_scan
                height_scan_noise_vec = torch.zeros_like(height_scan[0])
                height_scan_noise_vec[:] = noise_scales.height_scan * self.obs_scales.height_scan
                self.height_scan_noise_vec = height_scan_noise_vec

        self.actor_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.actor_obs_history_length, batch_size=self.num_envs, device=self.device
        )
        self.critic_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.critic_obs_history_length, batch_size=self.num_envs, device=self.device
        )
        