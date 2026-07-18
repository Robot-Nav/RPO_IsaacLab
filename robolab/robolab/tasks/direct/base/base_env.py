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

# Direct基础locomotion环境，基于DirectRLEnv实现速度跟踪、地形适应与姿态稳定
# 核心机制：PPO策略输出关节位置目标，PD控制器驱动关节；观测使用循环缓冲区堆叠历史帧
# 奖励由RewardManager统一调度，事件EventManager处理域随机化与重置扰动

from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence
from isaaclab.envs import DirectRLEnv
from isaaclab.assets.articulation import Articulation
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import EventManager, RewardManager
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.buffers import CircularBuffer
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
import isaaclab.sim as sim_utils

from .base_config import BaseEnvCfg


class BaseEnv(DirectRLEnv):
    """RPO四足/双足机器人基础locomotion环境。

    在IsaacLab DirectRLEnv基础上构建：策略输出23维关节位置增量，叠加默认关节角后送PD控制器；
    观测包含本体姿态、关节状态、指令与历史动作，critic额外获取线速度、接触力等特权信息。
    """
    cfg: BaseEnvCfg
    def __init__(self, cfg: BaseEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.reward_manager = RewardManager(self.cfg.reward, self)
        print("[INFO] Reward Manager: ", self.reward_manager)
        self.contact_sensor: ContactSensor = self.scene.sensors["contact_sensor"]
        if self.cfg.scene_context.height_scanner.enable_height_scan:
            self.height_scanner: RayCaster = self.scene.sensors["height_scanner"]

        # 左右脚RayCaster配置，用于足底高度采样
        self.left_feet_scanner_cfg = SceneEntityCfg("left_feet_scanner")
        self.right_feet_scanner_cfg = SceneEntityCfg("right_feet_scanner")

        # 速度指令生成器：按resampling_time_range周期重采样，heading_command模式下追踪航向角
        command_cfg = UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=self.cfg.commands.resampling_time_range,
            rel_standing_envs=self.cfg.commands.rel_standing_envs,
            rel_heading_envs=self.cfg.commands.rel_heading_envs,
            heading_command=self.cfg.commands.heading_command,
            heading_control_stiffness=self.cfg.commands.heading_control_stiffness,
            debug_vis=self.cfg.commands.debug_vis,
            ranges=self.cfg.commands.ranges,
        )
        self.command_generator = UniformVelocityCommand(cfg=command_cfg, env=self)

        self.init_buffers()

        # 启动时一次性应用startup模式事件（域随机化）
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.event_manager = EventManager(self.cfg.events, self)
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
        self._reset_idx(env_ids)

    def init_buffers(self):
        """初始化动作历史缓冲区、观测缩放因子与场景实体引用。"""
        self.extras = {}

        self.episode_length = np.ceil(self.max_episode_length_s / self.step_dt)
        self.num_actions = self.robot.data.default_joint_pos.shape[1]
        self.clip_actions = self.cfg.normalization.clip_actions
        self.clip_obs = self.cfg.normalization.clip_observations

        self.action_scale = self.cfg.robot.action_scale
        # 动作历史缓冲，用于动作平滑性奖励与观测输入
        self.action_buffer = CircularBuffer(
            max_len=self.cfg.robot.action_history_length, batch_size=self.num_envs, device=self.device
        )
        self.action_buffer.append(torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False))

        self.robot_cfg = SceneEntityCfg(name="robot")
        self.robot_cfg.resolve(self.scene)
        # 触发终止的碰撞体（躯干、大腿等不应触地的部位）
        self.termination_contact_cfg = SceneEntityCfg(
            name="contact_sensor", body_names=self.cfg.robot.terminate_contacts_body_names
        )
        self.termination_contact_cfg.resolve(self.scene)
        # 足部碰撞体，用于接触判定与步态奖励
        self.feet_cfg = SceneEntityCfg(name="contact_sensor", body_names=self.cfg.robot.feet_body_names)
        self.feet_cfg.resolve(self.scene)

        self.obs_scales = self.cfg.normalization.obs_scales
        self.add_noise = self.cfg.noise.add_noise

        self.init_obs_buffer()

    def init_obs_buffer(self):
        """构建观测噪声向量与actor/critic历史观测循环缓冲区。

        噪声向量按actor观测布局逐段填充，启用高度扫描时单独构建其噪声向量。
        """
        if self.add_noise:
            actor_obs, _ = self.compute_current_observations()
            noise_vec = torch.zeros_like(actor_obs[0])
            noise_scales = self.cfg.noise.noise_scales
            # actor观测布局：[角速度3, 投影重力3, 指令3, 关节角N, 关节速度N, 动作N]
            noise_vec[:3] = noise_scales.ang_vel * self.obs_scales.ang_vel
            noise_vec[3:6] = noise_scales.projected_gravity * self.obs_scales.projected_gravity
            noise_vec[6:9] = 0  # 指令不注入噪声
            noise_vec[9 : 9 + self.num_actions] = noise_scales.joint_pos * self.obs_scales.joint_pos
            noise_vec[9 + self.num_actions : 9 + self.num_actions * 2] = (
                noise_scales.joint_vel * self.obs_scales.joint_vel
            )
            noise_vec[9 + self.num_actions * 2 : 9 + self.num_actions * 3] = 0.0  # 动作项不噪声
            self.noise_scale_vec = noise_vec

            if self.cfg.scene_context.height_scanner.enable_height_scan:
                # 高度扫描 = 传感器z - 命中点z，截断到[-1,1]并做归一化
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

    def compute_current_observations(self):
        """计算当前时刻的actor与critic观测向量。

        actor观测：角速度、投影重力、指令、关节偏差、关节速度偏差、上一动作（用于策略输入）。
        critic观测：actor观测 + 线速度、足部接触、接触力、腾空时间、足部高度、关节加速度、关节力矩（特权信息）。
        """
        robot = self.robot
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        ang_vel = robot.data.root_ang_vel_b
        projected_gravity = robot.data.projected_gravity_b
        command = self.command_generator.command
        # 关节状态以默认角为基准，避免策略学到的偏置过大
        joint_pos = robot.data.joint_pos - robot.data.default_joint_pos
        joint_vel = robot.data.joint_vel - robot.data.default_joint_vel
        action = self.action_buffer.buffer[:, -1, :]
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

        root_lin_vel = robot.data.root_lin_vel_b
        # 接触阈值1N，足部接触布尔掩码用于步态奖励
        feet_contact = torch.max(torch.norm(net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1), dim=1)[0] > 1.0
        feet_contact_force = self.contact_sensor.data.net_forces_w[:, self.feet_cfg.body_ids, :]
        feet_air_time = self.contact_sensor.data.current_air_time[:, self.feet_cfg.body_ids]
        # 足部高度：脚踝z - 地面命中点z，减去脚踝0.04m偏置并截断到[0,1]
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
        current_critic_obs = torch.cat(
            [current_actor_obs, root_lin_vel * self.obs_scales.lin_vel, feet_contact.float(), feet_contact_force.flatten(1), feet_air_time.flatten(1), feet_height.flatten(1), joint_acc, joint_torque], dim=-1
        )

        return current_actor_obs, current_critic_obs


    def step(self, actions: torch.Tensor):
        """推进一个策略步：执行decimation次物理子步后计算观测、奖励与终止条件。"""
        actions = actions.to(self.device)

        self._pre_physics_step(actions)

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # 控制频率 = 仿真频率 / decimation；按render_interval周期渲染避免阻塞
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self._apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.command_generator.compute(self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

        self.obs_buf = self._get_observations()

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def update_terrain_levels(self, env_ids):
        """地形课程：行进距离超过半地形尺寸则升级，低于指令预期距离则降级。"""
        distance = torch.norm(self.robot.data.root_pos_w[env_ids, :2] - self.scene.env_origins[env_ids, :2], dim=1)
        move_up = distance > self.cfg.scene_context.terrain_generator.size[0] / 2
        move_down = (
            distance < torch.norm(self.command_generator.command[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
        )
        move_down *= ~move_up
        self.scene.terrain.update_env_origins(env_ids, move_up, move_down)
        extras = {"Curriculum/terrain_levels": torch.mean(self.scene.terrain.terrain_levels.float())}
        return extras

    def _setup_scene(self):
        self.robot: Articulation = self.scene["robot"]
        # GPU仿真下clone不拷贝源数据，节省显存
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

    def _pre_physics_step(self, actions: torch.Tensor):
        """策略动作预处理：截断、缩放并叠加默认关节角得到PD目标位置。"""
        self.action_buffer.append(actions)
        self.actions = actions.clone()
        self.actions = torch.clip(self.actions, -self.clip_actions, self.clip_actions).to(self.device)
        self.actions = self.actions * self.action_scale + self.robot.data.default_joint_pos

    def _apply_action(self) -> None:
        # 关节位置目标由PD控制器在PhysX侧求解
        self.robot.set_joint_position_target(self.actions)

    def _get_observations(self):
        """组装观测：注入噪声、拼接高度扫描、展平历史缓冲并截断到clip范围。"""
        current_actor_obs, current_critic_obs = self.compute_current_observations()
        if self.add_noise:
            # 均匀噪声U[-1,1] * scale，仿真传感器不确定性
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec

        if self.cfg.scene_context.height_scanner.enable_height_scan:
            height_scan = (
                    self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                    - self.height_scanner.data.ray_hits_w[..., 2]
                )
            height_scan = torch.clamp(height_scan - self.cfg.normalization.height_scan_offset, min=-1.0, max=1.0)
            height_scan = torch.nan_to_num(height_scan, nan=1.0, posinf=1.0, neginf=-1.0)
            height_scan *= self.obs_scales.height_scan
            # critic始终接入高度扫描作为特权信息
            current_critic_obs = torch.cat([current_critic_obs, height_scan], dim=-1)
            if self.add_noise:
                height_scan += (2 * torch.rand_like(height_scan) - 1) * self.height_scan_noise_vec
            # actor接入高度扫描时为非特权观测，需要叠加噪声
            if self.cfg.scene_context.height_scanner.enable_height_scan_actor:
                current_actor_obs = torch.cat([current_actor_obs, height_scan], dim=-1)

        self.actor_obs_buffer.append(current_actor_obs)
        self.critic_obs_buffer.append(current_critic_obs)

        # 历史缓冲展平：(history, dim) -> (history*dim)供网络处理
        actor_obs = self.actor_obs_buffer.buffer.reshape(self.num_envs, -1)
        critic_obs = self.critic_obs_buffer.buffer.reshape(self.num_envs, -1)

        actor_obs = torch.clip(actor_obs, -self.clip_obs, self.clip_obs)
        critic_obs = torch.clip(critic_obs, -self.clip_obs, self.clip_obs)

        observations = {"policy": actor_obs, "critic":critic_obs}
        return observations

    def _get_rewards(self):
        return self.reward_manager.compute(dt=self.step_dt)

    def _get_dones(self):
        """终止条件：终止碰撞体接触地面、躯干倾斜超阈、基座高度过低或超时。"""
        net_contact_forces = self.contact_sensor.data.net_forces_w_history
        if self.cfg.robot.terminate_contacts_body_names is not None:
            terminated_buf = torch.any(
                torch.max(
                    torch.norm(
                        net_contact_forces[:, :, self.termination_contact_cfg.body_ids],
                        dim=-1,
                    ),
                    dim=1,
                )[0]
                > 1.0,
                dim=1,
            )
        if self.cfg.robot.terminate_base_orientation is not None:
            # 反余弦计算躯干与重力夹角，超阈即判定翻倒
            terminated_buf |= torch.acos(-self.robot.data.projected_gravity_b[:, 2]).abs() > self.cfg.robot.terminate_base_orientation
        if self.cfg.robot.terminate_base_height is not None:
            terminated_buf |= self.robot.data.root_pos_w[:, 2] < self.cfg.robot.terminate_base_height
        time_out_buf = self.episode_length_buf >= self.episode_length
        return terminated_buf, time_out_buf

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """重置指定环境：地形课程更新、事件reset、奖励重置、缓冲区清零。"""
        if len(env_ids) == 0:
            return

        if self.cfg.scene_context.height_scanner.enable_height_scan:
            self.height_scanner.reset(env_ids)

        self.extras["log"] = dict()
        if self.cfg.scene_context.terrain_generator is not None:
            if self.cfg.scene_context.terrain_generator.curriculum:
                terrain_levels = self.update_terrain_levels(env_ids)
                self.extras["log"].update(terrain_levels)

        self.scene.reset(env_ids)
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(
                mode="reset",
                env_ids=env_ids,
                dt=self.step_dt,
                global_env_step_count=self._sim_step_counter // self.cfg.decimation,
            )

        reward_extras = self.reward_manager.reset(env_ids)
        self.extras["log"].update(reward_extras)
        # time_outs用于critic的bootstrap mask，超时回合不截断价值估计
        self.extras["time_outs"] = self.reset_time_outs

        self.command_generator.reset(env_ids)
        self.actor_obs_buffer.reset(env_ids)
        self.critic_obs_buffer.reset(env_ids)
        self.action_buffer.reset(env_ids)
        self.episode_length_buf[env_ids] = 0

        self.scene.write_data_to_sim()
        self.sim.forward()
