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

# 注意力编码器RPO环境配置：基于BaseEnvCfg组合ROUGH_HARD地形、注意力感知输入与扩展奖励项
# 启用actor/critic双路高度扫描，使用5帧历史(而非base的10帧)适配AttnEnc时序
# 额外奖励undesired_foothold约束足部落点于可行区域，强化复杂地形下的落足规划

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg
import matplotlib as mpl
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass
import math

from robolab.tasks.direct.base import mdp
from robolab.assets.robots import RPO_CFG
from robolab.tasks.direct.base import (  # noqa:F401
    BaseAgentCfg, 
    BaseEnvCfg, 
    RewardCfg, 
    HeightScannerCfg, 
    SceneContextCfg, 
    RobotCfg, 
    ObsScalesCfg, 
    NormalizationCfg, 
    CommandRangesCfg, 
    CommandsCfg, 
    NoiseScalesCfg, 
    NoiseCfg, 
    EventCfg,
    GRAVEL_TERRAINS_CFG,
    ROUGH_TERRAINS_CFG,
    ROUGH_HARD_TERRAINS_CFG,
    SceneCfg
)


@configclass
class RPORewardCfg(RewardCfg):
    """RPO AttnEnc奖励组合：在base奖励集基础上新增undesired_foothold落点约束，强化复杂地形足底规划。"""
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.25, params={"std": 0.5})  # 线速度跟踪，权重略高于base
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"std": 0.5})  # 角速度跟踪
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.05)  # 垂直速度惩罚，权重低于base以容忍地形起伏
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)  # roll/pitch角速度惩罚
    energy = RewTerm(func=mdp.energy, weight=-1e-4)  # 能耗惩罚
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)  # 关节力矩惩罚
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-2e-4)  # 关节速度惩罚
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)  # 关节加速度惩罚
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-2e-2)  # 相邻动作差分惩罚
    action_smoothness_l2 = RewTerm(func=mdp.action_smoothness_l2, weight=-2e-2)  # 二阶动作平滑性惩罚
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="(?!.*ankle_roll.*).*")},  # 除足踝外任何部位接触即惩罚
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)  # 躯干姿态归正
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)  # 终止惩罚，引导避免翻倒
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*"),
        },
    )
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "threshold": 500,  # 接触力超过500N开始计惩罚，避免硬着陆
            "max_reward": 400,
        },
    )
    feet_distance = RewTerm(
        func=mdp.body_distance_y,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]), "min": 0.10, "max": 0.50},  # 双足横向间距约束，防止交叉步
    )
    knee_distance = RewTerm(
        func=mdp.body_distance_y,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_knee.*"]), "min": 0.13, "max": 0.41},  # 膝盖间距约束，避免膝部碰撞
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll.*"])},  # 足部擦地惩罚
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)  # 关节位置软限位惩罚
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.03,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_thigh_yaw.*", ".*_thigh_roll.*"]
            )
        },  # 髋部yaw/roll偏离默认姿态惩罚
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*torso.*", ".*_arm_roll.*", ".*_arm_yaw.*", ".*_elbow_pitch.*", ".*_elbow_yaw.*"]
            )
        },  # 躯干与手臂偏离默认姿态强惩罚，促使手臂自然摆动
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.09,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_arm_pitch.*"],
            )
        },  # 手臂pitch偏离惩罚，维持前向摆臂自然
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_thigh_pitch.*", ".*_knee.*", ".*_ankle_pitch.*", ".*_ankle_roll.*"])},  # 腿部各关节偏离默认姿态
    )
    feet_contact_without_cmd = RewTerm(
        func=mdp.feet_contact_without_cmd,
        weight=0.1,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll.*"])},  # 零速指令下保持双足接触，鼓励静态稳定
    )
    upward = RewTerm(func=mdp.upward, weight=0.4)  # 投影重力z正向奖励，鼓励直立姿态
    stand_still = RewTerm(func=mdp.stand_still, weight=-0.2, params={"pos_cfg": SceneEntityCfg("robot", joint_names=[".*_arm.*", ".*_elbow.*", ".*torso.*", ".*_thigh.*", ".*_knee.*", ".*_ankle.*"]),
                                                                     "vel_cfg": SceneEntityCfg("robot", joint_names=[".*_arm.*", ".*_elbow.*", ".*torso.*", ".*_thigh.*", ".*_knee.*", ".*_ankle.*"]),
                                                                     "pos_weight": 0.0, "vel_weight": 0.04})  # 静止稳定性，pos_weight=0仅惩罚速度，零速指令下关节速度趋零
    feet_height = RewTerm(
        func=mdp.feet_height,
        weight=0.2,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*"),
                "sensor_cfg1": SceneEntityCfg("left_feet_scanner"),
                "sensor_cfg2": SceneEntityCfg("right_feet_scanner"),
                "ankle_height":0.04,"threshold":0.02})  # 足部抬升高度奖励，配合feet_scanner精确测量离地高度
    undesired_foothold = RewTerm(
        func=mdp.undesired_foothold,
        weight=-0.2,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
                "sensor_cfg1": SceneEntityCfg("left_feet_scanner"),
                "sensor_cfg2": SceneEntityCfg("right_feet_scanner"),
                "ankle_height":0.04})  # 落点不可行惩罚：足部接触时位置处于危险区域(如深坑边缘/障碍顶)即惩罚


color = [tuple(float(c) for c in mpl.colormaps["viridis"](i / 9.0)[:-1]) for i in range(10)]  # viridis colormap采10色，用于注意力可视化标记
markers = {}
for i in range(10):
    markers[f"hit_{i}"] = sim_utils.SphereCfg(
        radius=0.02,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color[i])
    )
@configclass
class AttnEncCfg:
    """注意力编码器开关与可视化标记配置。"""
    use_attn_enc: bool = False  # 启用注意力编码器分支
    vel_in_obs: bool = False  # 线速度是否保留在obs向量(否则单独切出)
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Attention",
        markers=markers,
    )


@configclass
class RPOAttnEncEnvCfg(BaseEnvCfg):
    """RPO AttnEnc环境配置：ROUGH_HARD地形 + actor/critic双路高度扫描 + 5帧历史，配合ActorCriticAttnEnc策略。"""

    reward = RPORewardCfg()
    attn_enc = AttnEncCfg(
            use_attn_enc=True,
            vel_in_obs=False,
        )

    def __post_init__(self):
        super().__post_init__()
        self.action_space = 23
        self.observation_space = 78  # actor obs维度(不含height_scan，剥离至perception_a)
        self.state_space = 145  # critic obs维度(含height_scan，但同样剥离至perception_c)
        self.scene_context.robot = RPO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene_context.height_scanner.prim_body_name = "base_link"
        self.scene_context.terrain_type = "generator"
        self.scene_context.terrain_generator = ROUGH_HARD_TERRAINS_CFG  # 使用高难度粗糙地形
        self.scene_context.height_scanner.enable_height_scan = True
        self.scene_context.height_scanner.enable_height_scan_actor = True  # actor侧也启用高度扫描(剥离至perception_a)
        self.scene_context.height_scanner.resolution = 0.1
        self.scene_context.height_scanner.size = (1.6, 1.0)  # 扫描区域1.6m×1.0m
        self.scene = SceneCfg(
            config=self.scene_context,
            physics_dt = self.sim.dt,
            step_dt = self.decimation * self.sim.dt
        )
        self.robot.terminate_contacts_body_names = ["torso_link", ".*_thigh_yaw_link", ".*_thigh_roll_link", ".*_elbow_.*_link", ".*_arm_.*_link"]  # 增加手臂/肘部接触终止
        self.robot.feet_body_names = [".*ankle_roll.*"]
        self.noise.add_noise = True
        self.events.add_base_mass.params["asset_cfg"].body_names = ["torso_link"]
        self.events.randomize_rigid_body_com.params["asset_cfg"].body_names = ["torso_link"]
        self.events.scale_link_mass.params["asset_cfg"].body_names = ["left_.*_link", "right_.*_link"]
        self.events.scale_actuator_gains.params["asset_cfg"].joint_names = [".*_joint"]
        self.events.scale_joint_parameters.params["asset_cfg"].joint_names = [".*_joint"]
        self.robot.action_scale = 0.25
        self.robot.actor_obs_history_length = 5  # AttnEnc使用5帧历史(而非base的10帧)
        self.robot.critic_obs_history_length = 5
        self.normalization.height_scan_offset = 0.75  # 高度扫描基准偏移，机器人base_link高度
        self.sim.physx.gpu_collision_stack_size = 2**29  # 增大GPU碰撞栈以承载复杂地形接触
        self.noise.noise_scales.joint_vel = 1.75
        self.noise.noise_scales.joint_pos = 0.03
        self.noise.noise_scales.lin_vel = 0.2
        self.noise.noise_scales.height_scan = 0.025
        self.commands.ranges = CommandRangesCfg(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.6, 0.6), ang_vel_z=(-1.57, 1.57), heading=(-math.pi, math.pi)
        )