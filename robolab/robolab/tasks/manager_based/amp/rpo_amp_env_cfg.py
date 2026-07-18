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

# RPO四足机器人专用AMP环境配置
# 在AmpEnvCfg基础上绑定RPO机器人资产、加载rpo_lab参考动作数据集、配置奖励权重与指令范围
# 关键参数：KEY_BODY_NAMES定义判别器关注的关键体，AMP_NUM_STEPS=3定义判别器观测步数
# 训练配置：速度跟踪+姿态/能量/接触惩罚，配合判别器输出的风格奖励训练自然步态

import os
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

import robolab.tasks.manager_based.amp.mdp as mdp
from robolab.tasks.manager_based.amp.managers import MotionDataTermCfg
from robolab.tasks.manager_based.amp.amp_env_cfg import AmpEnvCfg, MotionDataCfg

import isaaclab.terrains as terrain_gen
from robolab.tasks.direct.base.terrain_generator_cfg import ROUGH_TERRAINS_CFG

# Height scanner grid: 1.6m x 1.0m @ 0.1m resolution = 16 x 10 = 160 points per frame
_HEIGHT_SCAN_COLS = 16
_HEIGHT_SCAN_ROWS = 10

##
# Pre-defined configs
##

from robolab.assets.robots.roboparty import RPO_CFG
from robolab import ROBOLAB_ROOT_DIR

# NOTE: KEY_BODY_NAMES must match lab_key_body_names in robolab/scripts/tools/retarget/config/rpo.yaml
KEY_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_knee_link",
    "right_knee_link",
    "left_elbow_yaw_link",
    "right_elbow_yaw_link"
]
# 关键体名称：判别器观测中使用的关键身体部位，必须与retarget配置一致
ANIMATION_TERM_NAME = "animation"
AMP_NUM_STEPS = 3
# AMP判别器观测步数：策略侧与参考侧各取3步时序状态进行判别

@configclass
class RPOAmpRewards():
    """Reward terms for the MDP."""
    # RPO机器人奖励项：默认权重为0，由RPOAmpEnvCfg.__post_init__按需启用
    # 分组：Task速度跟踪、Alive存活、Base Link姿态、Joint能量/平滑/限位、Feet接触/间距/噪声
    # AMP风格奖励由判别器单独提供，不在此处

    # -- Task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0, params={"command_name": "base_velocity", "std": 0.5}
    )
    
    # -- Alive
    alive = RewTerm(func=mdp.is_alive, weight=0)
    
    # -- Base Link
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=0)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0)

    # -- Joint
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=0)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=0)
    smoothness_1 = RewTerm(func=mdp.smoothness_1, weight=0)
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0)
    joint_energy = RewTerm(func=mdp.joint_energy, weight=0)
    joint_regularization = RewTerm(func=mdp.joint_deviation_l1, weight=0)
    arm_pitch_mean_offset = RewTerm(
        func=mdp.paired_joints_mean_deviation_l1,
        weight=0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_arm_pitch_joint"],
            )
        },
    )
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=0.0,
    )
        
    # -- Feet
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    feet_distance_y = RewTerm(
        func=mdp.feet_distance_y,
        weight=0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                preserve_order=True,
            ),
            "min": 0.14,
            "max": 0.50,
        },
    )
    
    sound_suppression = RewTerm(
        func=mdp.sound_suppression_acc_per_foot,
        weight=0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=".*_ankle_roll_link",
            ),
        },
    )

    # -- other
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )


@configclass
class RPOAmpEnvCfg(AmpEnvCfg):
    # RPO机器人AMP环境配置：覆盖AmpEnvCfg各模块以适配RPO硬件特性与训练目标
    rewards: RPOAmpRewards = RPOAmpRewards()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------------------------------
        # Scene
        # ------------------------------------------------------
        # 绑定RPO机器人配置，prim_path使用ENV_REGEX_NS支持多环境克隆
        self.scene.robot = RPO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # plane terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # ------------------------------------------------------
        # motion data
        # ------------------------------------------------------
        # 参考动作数据：rpo_lab目录下的.pkl文件，按权重采样
        # 权重决定各动作被采样的概率，影响训练数据分布
        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            ROBOLAB_ROOT_DIR, "data", "motions", "rpo_lab"
        )
        self.motion_data.motion_dataset.motion_data_weights={

            "127_06": 16,

            "A1-_Stand_stageii": 6.5,

            "run_start_180_R_001__A345_M":4,
            "run_start_180_R_001__A345":4,

            "move_l":4.5,
            "move_r":5,

            "run_stop_180_R_001__A345_M":3,
            "run_stop_180_R_001__A345":3
        }

        # ------------------------------------------------------
        # animation
        # ------------------------------------------------------
        # 判别器参考侧观测步数：与AMP_NUM_STEPS一致，控制时序长度
        self.animation.animation.num_steps_to_use = AMP_NUM_STEPS

        # ------------------------------------------------------
        # Observations
        # ------------------------------------------------------
        # 判别器策略侧关键体配置：指定判别器需关注的关键体部位
        # preserve_order=True保证关键体顺序与参考数据一致

        # discriminator observations

        self.observations.disc.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(
                name="robot",
                body_names=KEY_BODY_NAMES,
                preserve_order=True
            )
        }
        self.observations.disc.history_length = AMP_NUM_STEPS
        
        # ------------------------------------------------------
        # Events
        # ------------------------------------------------------

        # ------------------------------------------------------
        # Rewards
        # ------------------------------------------------------
        # 奖励权重：任务跟踪正向，惩罚项负向；权重经调参平衡各目标
        # task
        self.rewards.track_lin_vel_xy_exp.weight = 1.25
        self.rewards.track_ang_vel_z_exp.weight = 1.25
        self.rewards.alive.weight = 0.15

        # base
        # self.rewards.lin_vel_z_l2.weight = -0.1
        self.rewards.ang_vel_xy_l2.weight = -0.1
        self.rewards.flat_orientation_l2.weight = -1.2
        
        # joint
        self.rewards.joint_vel_l2.weight = -2e-4
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.joint_pos_limits.weight = -1.0
        self.rewards.joint_energy.weight = -1e-4
        self.rewards.joint_torques_l2.weight = -1e-5
        self.rewards.arm_pitch_mean_offset.weight = -0.1
        
        # feet
        self.rewards.feet_slide.weight = -0.1
        self.rewards.sound_suppression.weight = -5e-5
        self.rewards.feet_distance_y.weight = 0.05


        self.rewards.undesired_contacts.weight = -10.0
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=["(?!.*ankle.*).*"],  # exclude ankle links
        )
        
        # ------------------------------------------------------
        # Commands
        # ------------------------------------------------------
        # 速度指令范围：前向速度[-0.5,2.5]覆盖前进/后退，侧向[-0.5,0.5]，转向[-1.5,1.5]
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 2.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
                
        # ------------------------------------------------------
        # Curriculum
        # ------------------------------------------------------
        
        
        
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            ".*_thigh_.*_link", "base_link", ".*_arm_.*_link", ".*_elbow_.*_link",
        ]
        if self.__class__.__name__ == "RPOAmpEnvCfg":
            # 关闭零权重奖励，避免子类PLAY误继承父类优化
            self.disable_zero_weight_rewards()


@configclass
class RPOAmpEnvCfg_PLAY(RPOAmpEnvCfg):
    # 推理/可视化配置：单环境、固定速度指令、关闭随机化与噪声
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.push_robot = None


@configclass
class RPOAmpRoughEnvCfg(RPOAmpEnvCfg):
    # RPO机器人AMP粗糙地形训练配置
    # 在RPOAmpEnvCfg基础上切换ROUGH_TERRAINS_CFG + height_scan特权critic
    # 含楼梯/坡/网格/随机起伏等多类地形，启用课程学习(curriculum=True)
    # 策略靠本体感知，critic额外获得高度扫描特权信息

    def __post_init__(self):
        super().__post_init__()

        # -- rough terrain
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG

        # -- height scanner sensor (privileged critic)
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 5.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(
                resolution=0.1, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )

        # -- height_scan observation term for critic
        self.observations.critic.height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-5.0, 5.0),
        )

        # 关闭零权重奖励
        self.disable_zero_weight_rewards()


@configclass
class RPOAmpRoughEnvCfg_PLAY(RPOAmpRoughEnvCfg):
    # 粗糙地形推理/可视化配置：单环境、固定速度指令、关闭随机化
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        self.observations.policy.enable_corruption = False
        self.events.push_robot = None


@configclass
class RPODanceAmpEnvCfg(RPOAmpEnvCfg):
    # RPO舞蹈AMP训练配置
    # 关键体保持与retarget配置(rpo.yaml)一致：6个，不修改disc.key_body_pos_b
    # 调整速度范围匹配舞蹈位移，降低姿态惩罚（舞蹈允许躯干倾斜），减弱推动随机化
    def __post_init__(self):
        super().__post_init__()

        # 舞蹈参考动作数据目录
        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            ROBOLAB_ROOT_DIR, "data", "motions", "rpo_dance_lab"
        )
        # 舞蹈动作权重：8段舞蹈均匀采样
        self.motion_data.motion_dataset.motion_data_weights = {
            "dance1_subject1": 1.0,
            "dance1_subject2": 1.0,
            "dance1_subject3": 1.0,
            "dance2_subject1": 1.0,
            "dance2_subject2": 1.0,
            "dance2_subject3": 1.0,
            "dance2_subject4": 1.0,
            "dance2_subject5": 1.0,
        }

        # 关键体不修改：继承父类的KEY_BODY_NAMES（6个，与.pkl数据一致）
        # 注意：elbow_yaw_link位置会因上臂运动而变化，仍有信息量

        # 任务奖励权重（匹配舞蹈位移速度）
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        self.rewards.alive.weight = 0.2

        # 降低姿态惩罚：舞蹈允许躯干倾斜（原-1.2过强会压制舞蹈动作）
        self.rewards.flat_orientation_l2.weight = -0.3

        # 速度范围：舞蹈平均速度低（root span 4m / 79s ≈ 0.05m/s），缩小范围让任务可完成
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # 减弱推动随机化：舞蹈单腿支撑阶段易被推倒（原±0.5过强）
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-0.5, 0.5)
        }

        # 关闭零权重奖励优化
        self.disable_zero_weight_rewards()


@configclass
class RPODanceSingleAmpEnvCfg(RPODanceAmpEnvCfg):
    # 单舞蹈训练配置：目标是dance1_subject2，但必须混合多段舞蹈防止判别器过拟合
    # 单段78秒舞蹈只有3945帧，判别器轻松记忆全部状态 → 必崩溃
    # 策略：8段舞蹈混合使用（~30k帧），dance1_subject2权重=3.0突出主目标
    def __post_init__(self):
        super().__post_init__()

        # 8段舞蹈混合：dance1_subject2权重3倍，策略偏向模仿目标舞蹈
        self.motion_data.motion_dataset.motion_data_weights = {
            "dance1_subject1": 1.0,
            "dance1_subject2": 3.0,
            "dance1_subject3": 1.0,
            "dance2_subject1": 1.0,
            "dance2_subject2": 1.0,
            "dance2_subject3": 1.0,
            "dance2_subject4": 1.0,
            "dance2_subject5": 1.0,
        }

        # 恢复平衡奖励（父类值：track_lin=1.0, track_ang=0.5, alive=0.2）
        # 策略需要站稳才能跳舞，过低的奖励会导致base_contact=100%
        # flat_orientation=-0.3保持不变：舞蹈允许躯干倾斜


@configclass
class RPODanceSingleAmpEnvCfg_PLAY(RPODanceSingleAmpEnvCfg):
    # 单舞蹈推理/可视化配置：单环境、零速度指令、关闭随机化
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None


@configclass
class RPODanceAmpEnvCfg_PLAY(RPODanceAmpEnvCfg):
    # 舞蹈推理/可视化配置：单环境、零速度指令、关闭随机化
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None