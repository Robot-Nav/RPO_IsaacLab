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

# RPO中断恢复环境配置：在BaseEnvCfg基础上组合GRAVEL地形与手臂关节中断扰动
# InterruptCfg定义8个手臂关节的中断范围/课程参数，RPOInterruptEnvCfg注册中断相关奖励项
# 新增joint_deviation_interrupt(手臂偏离惩罚)、stand_still_interrupt(静止稳定性)、action_penalty_interrupt

from isaaclab.markers import VisualizationMarkersCfg
import isaaclab.sim as sim_utils
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

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
    """RPO Interrupt奖励组合：在base奖励集上新增中断相关奖励项。

    新增：joint_deviation_interrupt(手臂偏离加权惩罚)、stand_still_interrupt(静止时含手臂)、
    action_penalty_interrupt(中断态动作惩罚)；stand_still使用interrupt_cfg区分手臂/肘部。
    """
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0, params={"std": 0.5})  # 线速度跟踪
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"std": 0.5})  # 角速度跟踪
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2)  # 垂直速度惩罚
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1)  # roll/pitch角速度惩罚
    energy = RewTerm(func=mdp.energy, weight=-1e-4)  # 能耗
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)  # 关节力矩
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-2e-4)  # 关节速度
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)  # 关节加速度
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-2e-2)  # 动作差分
    action_smoothness_l2 = RewTerm(func=mdp.action_smoothness_l2, weight=-2e-2)  # 动作平滑
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="(?!.*ankle_roll.*).*")},  # 除足踝外接触惩罚
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)  # 躯干姿态
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)  # 终止惩罚
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"), "threshold": 0.4},  # 双足空中时间，threshold过滤过短腾空
    )
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
            "threshold": 500,  # 接触力超500N计惩罚
            "max_reward": 400,
        },
    )
    feet_distance = RewTerm(
        func=mdp.body_distance_y,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]), "min": 0.16, "max": 0.50},  # 双足横向间距
    )
    knee_distance = RewTerm(
        func=mdp.body_distance_y,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_knee.*"]), "min": 0.18, "max": 0.35},  # 膝盖间距
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll.*"])},  # 足部擦地
    )
    feet_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"])},  # 足部姿态惩罚，避免足部扭转
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)  # 关节限位
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.03,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_thigh_yaw.*", ".*_thigh_roll.*"]
            )
        },  # 髋部yaw/roll偏离
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*torso.*", ".*_elbow_yaw.*"]
            )
        },  # 躯干与肘yaw偏离强惩罚
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_thigh_pitch.*", ".*_knee.*", ".*_ankle_pitch.*", ".*_ankle_roll.*"])},  # 腿部偏离
    )
    joint_deviation_interrupt = RewTerm(
        func=mdp.joint_deviation_interrupt,
        weight=-1.0,
        params={
            "asset_cfg1": SceneEntityCfg(
                "robot", joint_names=[".*_arm_roll.*", ".*_arm_yaw.*", ".*_elbow_pitch.*"]
            ),  # 第一组：手臂roll/yaw + 肘pitch，权重1.0
            "asset_cfg2": SceneEntityCfg(
                "robot",
                joint_names=[".*_arm_pitch.*"],
            ),  # 第二组：手臂pitch，权重0.06(允许摆臂)
            "weight1": 1.0, "weight2": 0.06
        }
    )
    feet_contact_without_cmd = RewTerm(
        func=mdp.feet_contact_without_cmd,
        weight=0.1,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll.*"])},  # 零速指令下保持双足接触
    )
    upward = RewTerm(func=mdp.upward, weight=0.4)  # 直立姿态奖励
    stand_still = RewTerm(func=mdp.stand_still_interrupt, weight=-0.2, params={"pos_cfg": SceneEntityCfg("robot", joint_names=[".*_arm.*", ".*_elbow.*", ".*torso.*", ".*_thigh.*", ".*_knee.*", ".*_ankle.*"]),
                                                                               "vel_cfg": SceneEntityCfg("robot", joint_names=[".*_arm.*", ".*_elbow.*", ".*torso.*", ".*_thigh.*", ".*_knee.*", ".*_ankle.*"]),
                                                                               "interrupt_cfg": SceneEntityCfg("robot", joint_names=[".*_arm.*", ".*_elbow_pitch.*"]),  # 中断关节单独配置，静止时手臂目标归零
                                                                               "pos_weight": 1.0, "vel_weight": 0.04})
    feet_height = RewTerm(
        func=mdp.feet_height,
        weight=0.2,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*"),
                "sensor_cfg1": SceneEntityCfg("left_feet_scanner"),
                "sensor_cfg2": SceneEntityCfg("right_feet_scanner"),
                "ankle_height":0.04,"threshold":0.02})  # 足部抬升奖励
    action_penalty = RewTerm(func=mdp.action_penalty_interrupt, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_arm.*", ".*_elbow_pitch.*"])})  # 中断态手臂动作惩罚，抑制过度补偿


@configclass
class InterruptCfg:
    """中断扰动配置：定义可中断关节、采样范围、课程参数与可视化标记。"""
    use_interrupt: bool = False  # 是否启用中断
    max_curriculum: float = 1.0  # 课程最大幅度
    interrupt_ratio: float = 0.5  # 可中断env比例
    interrupt_joint_names: list = []  # 可中断关节名列表
    interrupt_scale : list = []  # 各关节均匀采样范围(scale)
    interrupt_lower_bound: list = []  # 各关节均匀采样下界
    interrupt_init_range: float = 0.2  # 课程clipping初始范围
    interrupt_update_step: int = 30  # 中断目标重采样周期(步)
    switch_prob: float = 0.005  # 随机切换中断态概率
    interrupt_vis: VisualizationMarkersCfg = VisualizationMarkersCfg(
        markers={
            "interrupt": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),  # 绿色=正常
            ),
            "no_interrupt": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),  # 红色=中断
            ),
        },
        prim_path="/Visuals/Command/interrupt",
    )

@configclass
class RPOInterruptEnvCfg(BaseEnvCfg):
    """RPO中断恢复环境配置：GRAVEL地形 + 8个手臂关节中断 + interrupt相关奖励。"""

    reward = RPORewardCfg()
    interrupt = InterruptCfg(
        use_interrupt = True,
        max_curriculum = 1.0,
        interrupt_ratio = 0.5,
        interrupt_joint_names = [
            "left_arm_pitch_joint",
            "left_arm_roll_joint",
            "left_arm_yaw_joint",
            "left_elbow_pitch_joint",
            "right_arm_pitch_joint",
            "right_arm_roll_joint",
            "right_arm_yaw_joint",
            "right_elbow_pitch_joint",
        ],  # 8个手臂关节(左右各4：arm pitch/roll/yaw + elbow pitch)
    interrupt_scale = [
            3.14, # Arm Pitch -1.57~1.57
            1.82, # Arm Roll, -0.25~1.57
            3.14, # Arm Yaw,  -1.57~1.57
            2.07, # Elbow Pitch, -0.5~1.57
            3.14, # Arm Pitch -1.57~1.57
            1.82, # Arm Roll, -1.57~0.25
            3.14, # Arm Yaw,  -1.57~1.57
            2.07, # Elbow Pitch, -0.5~1.57
        ], # Uniform Distribution Noise for each joint.  # 各关节均匀采样scale=上界-下界
    interrupt_lower_bound = [
            -1.57,
            -0.25,
            -1.57,
            -0.5,
            -1.57,
            -1.57,
            -1.57,
            -0.5,
        ],  # 各关节均匀采样下界
        interrupt_init_range = 0.2,
        interrupt_update_step = 30,
        switch_prob = 0.005,
    )
    interrupt_vis = VisualizationMarkersCfg(
        markers={
            "interrupt": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
            "no_interrupt": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        },
        prim_path="/Visuals/Command/interrupt",
    )

    def __post_init__(self):
        super().__post_init__()
        self.action_space = 23
        self.observation_space = 79  # actor obs维度(含1维interrupt_mask)
        self.state_space = 140  # critic obs维度
        self.scene_context.robot = RPO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene_context.height_scanner.prim_body_name = "base_link"
        self.scene_context.terrain_type = "generator"
        self.scene_context.terrain_generator = GRAVEL_TERRAINS_CFG  # 使用GRAVEL平缓地形，聚焦中断任务而非地形适应
        self.scene = SceneCfg(
            config=self.scene_context,
            physics_dt = self.sim.dt,
            step_dt = self.decimation * self.sim.dt
        )
        self.robot.terminate_contacts_body_names = ["torso_link", ".*_thigh_yaw_link", ".*_thigh_roll_link"]
        self.robot.feet_body_names = [".*ankle_roll.*"]
        self.events.add_base_mass.params["asset_cfg"].body_names = ["torso_link", "base_link"]
        self.events.randomize_rigid_body_com.params["asset_cfg"].body_names = ["torso_link", "base_link"]
        self.events.scale_link_mass.params["asset_cfg"].body_names = ["left_.*_link", "right_.*_link"]
        self.events.scale_actuator_gains.params["asset_cfg"].joint_names = [".*_joint"]
        self.events.scale_joint_parameters.params["asset_cfg"].joint_names = [".*_joint"]
        self.robot.action_scale = 0.25
        self.noise.noise_scales.joint_vel = 1.75
        self.noise.noise_scales.joint_pos = 0.03