"""Sub-module containing Flat-Patch based velocity command generators."""
# 基于地形flat_patch的位姿/速度命令生成器：从地形patch采样目标点，按位置/航向误差计算机体系速度命令
# 支持按地形类型设置速度范围、随机速度覆盖、站立环境、直线行走等模式
from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.math import quat_apply_inverse, wrap_to_pi, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import PoseVelocityCommandCfg


class PoseVelocityCommand(CommandTerm):
    """Velocity command based on the 2D flat patch command generator."""
    # 命令生成器核心类：维护世界系/机体系命令缓冲，按地形类型分配速度范围，重采样时从地形patch取目标

    cfg: PoseVelocityCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: PoseVelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the robot and terrain assets
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # crete buffers to store the command
        # 命令缓冲：世界系目标位姿、机体系速度命令、最大速度限制、站立标志
        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_command_w = torch.zeros(self.num_envs, device=self.device)
        self.pos_command_b = torch.zeros_like(self.pos_command_w)
        self.heading_command_b = torch.zeros_like(self.heading_command_w)
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.max_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # -- metrics
        # 评估指标：误差累积与exp核跟踪精度，用于评估策略收敛情况
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["tracking_exp_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["tracking_exp_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

        # obtain the terrain asset
        self.terrain: TerrainImporter = env.scene["terrain"]

        # 按地形类型分配的速度范围（用于位置命令转速度时的限制）
        self.lin_vel_x_range = torch.zeros(self.num_envs, 2, device=self.device)
        self.lin_vel_y_range = torch.zeros(self.num_envs, 2, device=self.device)
        self.ang_vel_z_range = torch.zeros(self.num_envs, 2, device=self.device)

        # 随机速度地形的独立范围（覆盖位置命令，直接给速度命令）
        self.random_lin_vel_x_range = torch.zeros(self.num_envs, 2, device=self.device)
        self.random_lin_vel_y_range = torch.zeros(self.num_envs, 2, device=self.device)
        self.random_ang_vel_z_range = torch.zeros(self.num_envs, 2, device=self.device)
        self.random_velocity_indices = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.random_lin_vel_x = torch.zeros(self.num_envs, device=self.device)
        self.random_lin_vel_y = torch.zeros(self.num_envs, device=self.device)
        self.random_ang_vel_z = torch.zeros(self.num_envs, device=self.device)

        if self.cfg.velocity_ranges is not None:
            terrain_generator_cfg = self.terrain.cfg.terrain_generator
            proportions = np.array([sub_cfg.proportion for sub_cfg in terrain_generator_cfg.sub_terrains.values()])
            proportions /= np.sum(proportions)

            # find the sub-terrain index for each column
            # we generate the terrains based on their proportion (not randomly sampled)
            # 按子地形比例反推每列对应的子地形索引，便于按terrain_type找到env_indices
            sub_indices = []
            for index in range(terrain_generator_cfg.num_cols):
                sub_index = np.min(np.where(index / terrain_generator_cfg.num_cols + 0.001 < np.cumsum(proportions))[0])
                sub_indices.append(sub_index)
            sub_indices = np.array(sub_indices, dtype=np.int32)
            sub_terrains_names = list(terrain_generator_cfg.sub_terrains.keys())
            for key, value in self.cfg.velocity_ranges.items():
                if key in sub_terrains_names:
                    terrain_type_index = sub_terrains_names.index(key)
                    type_indices = np.where(sub_indices == terrain_type_index)[0]
                    for type_indice in type_indices:
                        env_indices = torch.where(self.terrain.terrain_types == type_indice)[0]
                        self.lin_vel_x_range[env_indices, 0] = value["lin_vel_x"][0]
                        self.lin_vel_x_range[env_indices, 1] = value["lin_vel_x"][1]
                        self.lin_vel_y_range[env_indices, 0] = value["lin_vel_y"][0]
                        self.lin_vel_y_range[env_indices, 1] = value["lin_vel_y"][1]
                        self.ang_vel_z_range[env_indices, 0] = value["ang_vel_z"][0]
                        self.ang_vel_z_range[env_indices, 1] = value["ang_vel_z"][1]
                else:
                    raise RuntimeError(f"Terrain type {key} not found in the terrain generator sub-terrain names.")

            if self.cfg.random_velocity_terrain is not None:
                for key in self.cfg.random_velocity_terrain:
                    terrain_type_index = sub_terrains_names.index(key)
                    type_indices = np.where(sub_indices == terrain_type_index)[0]
                    for type_indice in type_indices:
                        env_indices = torch.where(self.terrain.terrain_types == type_indice)[0]
                        self.random_velocity_indices[env_indices] = True

        self.random_lin_vel_x_range[:, 0] = self.cfg.ranges.lin_vel_x[0]
        self.random_lin_vel_x_range[:, 1] = self.cfg.ranges.lin_vel_x[1]
        self.random_lin_vel_y_range[:, 0] = self.cfg.ranges.lin_vel_y[0]
        self.random_lin_vel_y_range[:, 1] = self.cfg.ranges.lin_vel_y[1]
        self.random_ang_vel_z_range[:, 0] = self.cfg.ranges.ang_vel_z[0]
        self.random_ang_vel_z_range[:, 1] = self.cfg.ranges.ang_vel_z[1]

        # obtain the valid targets from the terrain
        if "target" not in self.terrain.flat_patches:
            raise RuntimeError(
                "The terrain-based command generator requires a valid flat patch under 'target' in the terrain."
                f" Found: {list(self.terrain.flat_patches.keys())}"
            )
        # valid targets: (terrain_level, terrain_type, num_patches, 3)
        # 地形patch预生成的可达目标点，按level/type索引取用，避免运行时射线检测
        self.valid_targets: torch.Tensor = self.terrain.flat_patches["target"]

    def __str__(self) -> str:
        msg = "PositionCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.vel_command_b

    @property
    def pose_command(self) -> torch.Tensor:
        """The desired base pose command in the base frame. Shape is (num_envs, 3)."""
        return torch.cat([self.pos_command_b[:, :2], self.vel_command_b[:, 0:1]], dim=1)

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # 评估指标更新：误差按步长归一化，exp核跟踪精度按回合长度归一化
        # logs data
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # logs data
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) / max_command_step
        )
        lin_vel_error = torch.sum(
            torch.square(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2]),
            dim=1,
        )
        self.metrics["tracking_exp_vel_xy"] += (
            torch.exp(-lin_vel_error / self.cfg.lin_vel_metrics_std**2) / self._env.max_episode_length
        )
        angular_vel_error = torch.square(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
        self.metrics["tracking_exp_vel_yaw"] += (
            torch.exp(-angular_vel_error / self.cfg.ang_vel_metrics_std**2) / self._env.max_episode_length
        )

    def _resample_command(self, env_ids: Sequence[int]):
        # 重采样：从地形patch取目标点，按速度范围采样max_command，随机速度地形单独采样
        # sample new position targets from the terrain
        ids = torch.randint(0, self.valid_targets.shape[2], size=(len(env_ids),), device=self.device)
        self.pos_command_w[env_ids] = self.valid_targets[
            self.terrain.terrain_levels[env_ids], self.terrain.terrain_types[env_ids], ids
        ]
        if self.cfg.straight_target_prob > 0.0:
            # 按概率强制y对齐当前机器人y，训练直线行走
            straight_mask = torch.rand(len(env_ids), device=self.device) < self.cfg.straight_target_prob
            self.pos_command_w[env_ids, 1] = torch.where(
                straight_mask,
                self.robot.data.root_pos_w[env_ids, 1],
                self.pos_command_w[env_ids, 1],
            )

        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self.max_command_b[env_ids, 0] = self.lin_vel_x_range[env_ids, 0] + r.uniform_(0.0, 1.0) * (
            self.lin_vel_x_range[env_ids, 1] - self.lin_vel_x_range[env_ids, 0]
        )
        # -- linear velocity - y direction
        self.max_command_b[env_ids, 1] = self.lin_vel_y_range[env_ids, 0] + r.uniform_(0.0, 1.0) * (
            self.lin_vel_y_range[env_ids, 1] - self.lin_vel_y_range[env_ids, 0]
        )
        # -- ang vel yaw - rotation around z
        self.max_command_b[env_ids, 2] = self.ang_vel_z_range[env_ids, 0] + r.uniform_(0.0, 1.0) * (
            self.ang_vel_z_range[env_ids, 1] - self.ang_vel_z_range[env_ids, 0]
        )
        # update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

        # Only update random velocities for envs that are currently being resampled AND are marked for random velocity
        # Create a mask for the current batch of env_ids
        # 仅对本批次重采样且标记为随机速度的env更新随机速度，避免影响其他env
        current_batch_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        current_batch_mask[env_ids] = True

        # Find intersection: envs in this batch that are also random velocity envs
        update_mask = current_batch_mask & self.random_velocity_indices
        random_velocity_env_ids = update_mask.nonzero(as_tuple=False).flatten()

        if len(random_velocity_env_ids) > 0:
            self.random_lin_vel_x[random_velocity_env_ids] = self.random_lin_vel_x_range[
                random_velocity_env_ids, 0
            ] + torch.rand(len(random_velocity_env_ids), device=self.device) * (
                self.random_lin_vel_x_range[random_velocity_env_ids, 1]
                - self.random_lin_vel_x_range[random_velocity_env_ids, 0]
            )
            self.random_lin_vel_y[random_velocity_env_ids] = self.random_lin_vel_y_range[
                random_velocity_env_ids, 0
            ] + torch.rand(len(random_velocity_env_ids), device=self.device) * (
                self.random_lin_vel_y_range[random_velocity_env_ids, 1]
                - self.random_lin_vel_y_range[random_velocity_env_ids, 0]
            )
            self.random_ang_vel_z[random_velocity_env_ids] = self.random_ang_vel_z_range[
                random_velocity_env_ids, 0
            ] + torch.rand(len(random_velocity_env_ids), device=self.device) * (
                self.random_ang_vel_z_range[random_velocity_env_ids, 1]
                - self.random_ang_vel_z_range[random_velocity_env_ids, 0]
            )
            # 角速度过小则归零，避免随机速度地形下出现无意义的小幅转向
            self.random_ang_vel_z *= torch.abs(self.random_ang_vel_z) > 0.5

    def _update_command(self):
        """Re-target the position command to the current root state."""
        # 每步重新计算：把世界系目标转为机体系，按刚度生成速度命令，再按max_command限幅
        target_vec = self.pos_command_w - self.robot.data.root_pos_w[:, :3]
        target_dist = torch.norm(target_vec[:, :2], dim=1)
        self.pos_command_b[:] = quat_apply_inverse(yaw_quat(self.robot.data.root_quat_w), target_vec)
        self.vel_command_b[:, :2] = self.pos_command_b[:, :2] * self.cfg.velocity_control_stiffness

        # set heading command to point towards target
        target_vec = self.pos_command_w - self.robot.data.root_pos_w
        target_direction = torch.atan2(target_vec[:, 1], target_vec[:, 0])

        # compute errors to find the closest direction to the current heading
        # this is done to avoid the discontinuity at the -pi/pi boundary
        # wrap_to_pi避免±pi边界跳变，保证航向误差连续
        self.heading_command_w = wrap_to_pi(target_direction - self.robot.data.heading_w)

        self.vel_command_b[:, 2] = self.heading_command_w * self.cfg.heading_control_stiffness

        # scale linear velocity so the dominant axis hits its limit and
        # the other axis preserves its ratio
        # 主轴限幅策略：哪个轴绝对值大就作为主轴限幅，另一轴按比例缩放保持方向
        vx = self.vel_command_b[:, 0]
        vy = self.vel_command_b[:, 1]
        min_x = (
            -self.max_command_b[:, 0]
            if not self.cfg.only_positive_lin_vel_x
            else torch.zeros_like(self.max_command_b[:, 0])
        )
        min_y = -self.max_command_b[:, 1]
        max_x = self.max_command_b[:, 0]
        max_y = self.max_command_b[:, 1]
        eps = 1e-6

        abs_vx = vx.abs()
        abs_vy = vy.abs()

        if not self.cfg.only_positive_lin_vel_x:
            # clamp each axis independently
            clamped_vx = torch.clamp(abs_vx, min=min_x, max=max_x)
            clamped_vy = torch.clamp(abs_vy, min=min_y, max=max_y)

            # compute scale for whichever axis is dominant
            scale_x = clamped_vx / (abs_vx + eps)
            scale_y = clamped_vy / (abs_vy + eps)
            scale = torch.where(abs_vx >= abs_vy, scale_x, scale_y)

            # apply scale and restore sign
            self.vel_command_b[:, 0] = vx * scale
            self.vel_command_b[:, 1] = vy * scale

        else:
            self.vel_command_b[:, 0] = torch.clamp(vx, min=min_x, max=max_x)
            self.vel_command_b[:, 1] = torch.clamp(vy, min=min_y, max=max_y)

        self.vel_command_b[:, 2] = torch.clamp(
            self.vel_command_b[:, 2],
            self.cfg.ranges.ang_vel_z[0],
            self.cfg.ranges.ang_vel_z[1],
        )
        # 到达目标距离阈值内命令归零，避免目标点附近震荡
        self.vel_command_b[:] *= (target_dist > self.cfg.target_dis_threshold).unsqueeze(-1)
        # 死区过滤：低于阈值的命令归零
        self.vel_command_b[:, :2] *= (
            (torch.norm(self.vel_command_b[:, :2], dim=1) > self.cfg.lin_vel_threshold).float().unsqueeze(-1)
        )
        self.vel_command_b[:, 2] *= (torch.abs(self.vel_command_b[:, 2]) > self.cfg.ang_vel_threshold).float()
        # 站立env命令置零
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0

        # 随机速度地形覆盖位置命令，直接用预采样的随机速度
        random_velocity_env_ids = self.random_velocity_indices.nonzero(as_tuple=False).flatten()
        self.vel_command_b[random_velocity_env_ids, 0] = self.random_lin_vel_x[random_velocity_env_ids]
        self.vel_command_b[random_velocity_env_ids, 1] = self.random_lin_vel_y[random_velocity_env_ids]
        self.vel_command_b[random_velocity_env_ids, 2] = self.random_ang_vel_z[random_velocity_env_ids]

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "flat_patch_visualizer"):
                # -- pose
                self.cfg.flat_patch_visualizer_cfg.markers["Goal"].radius = self.cfg.target_dis_threshold
                self.cfg.flat_patch_visualizer_cfg.markers["Patches"].radius = self.cfg.target_dis_threshold
                self.flat_patch_visualizer = VisualizationMarkers(self.cfg.flat_patch_visualizer_cfg)
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            # set their visibility to true
            self.flat_patch_visualizer.set_visibility(True)
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "flat_patch_visualizer"):
                self.flat_patch_visualizer.set_visibility(False)
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        if getattr(self.cfg, "patch_vis", True):
            flat_patches = self.valid_targets.reshape(-1, 3)
            poses = torch.cat([self.pos_command_w, flat_patches], dim=0)
            marker_indices = torch.cat(
                [
                    torch.zeros(self.num_envs, dtype=torch.int, device=self.device),
                    torch.ones(flat_patches.shape[0], dtype=torch.int, device=self.device),
                ],
                dim=0,
            )
            self.flat_patch_visualizer.visualize(poses, marker_indices=marker_indices)
        else:
            marker_indices = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
            self.flat_patch_visualizer.visualize(self.pos_command_w, marker_indices=marker_indices)
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 8.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat
