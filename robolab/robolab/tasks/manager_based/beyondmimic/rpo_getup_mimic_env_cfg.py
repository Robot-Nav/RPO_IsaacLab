
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

# RPO四足机器人起身模仿环境配置
# 参考动作为仰卧/俯卧翻身到站立，动作结束后需保持站立稳定（motion_ended后施加stand_still奖励）
# 锚点设为base_link，缩短推扰间隔以训练跌倒后的恢复能力

import os

from robolab.assets.robots import RPO_CFG
from robolab.tasks.manager_based.beyondmimic.beyondmimic_env_cfg import BeyondMimicEnvCfg

from isaaclab.utils import configclass
from robolab import ROBOLAB_ROOT_DIR

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm
import robolab.tasks.manager_based.beyondmimic.mdp as mdp
from isaaclab.managers import SceneEntityCfg


@configclass
class RPOGetupMimicEnvCfg(BeyondMimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # 注入RPO机器人资产到场景
        self.scene.robot = RPO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # 加载起身参考动作 getup_supin2prone.npz（仰卧到俯卧再到站立）
        self.commands.motion.motion_file = os.path.join(
            ROBOLAB_ROOT_DIR, "data", "motions", "rpo_bm", "getup_supin2prone.npz"
        )
        # 起身任务以base_link作为锚点，该link在跌倒姿态下位姿更稳定可参考
        self.commands.motion.anchor_body_name = "base_link"
        # 起身任务跟踪的身体部位，注释掉的部位在俯卧姿态下参考数据噪声较大
        self.commands.motion.body_names = [
            'left_thigh_yaw_link',
            'right_thigh_yaw_link',
            "base_link",
            'torso_link',
            'left_thigh_roll_link',
            'right_thigh_roll_link',
            # 'left_arm_pitch_link',
            # 'right_arm_pitch_link',
            'left_thigh_pitch_link',
            'right_thigh_pitch_link',
            # 'left_arm_roll_link',
            # 'right_arm_roll_link',
            'left_knee_link',
            'right_knee_link',
            'left_arm_yaw_link',
            'right_arm_yaw_link',
            # 'left_ankle_pitch_link',
            # 'right_ankle_pitch_link',
            # 'left_elbow_pitch_link',
            # 'right_elbow_pitch_link',
            'left_ankle_roll_link',
            'right_ankle_roll_link',
            'left_elbow_yaw_link',
            'right_elbow_yaw_link',
        ]

        # 动作播完后不重置环境，机器人保持末帧姿态，由stand_still奖励约束维持站立
        self.commands.motion.reset_on_motion_end = False

        # 强化身体位姿跟踪权重，起身末端姿态对齐关键
        self.rewards.motion_body_pos.weight = 2.0
        # 动作结束后施加关节位姿/速度惩罚，引导策略保持稳定站立姿态
        # pos_weight/vel_weight 控制位姿偏差与速度抑制的相对强度
        self.rewards.stand_still_after_motion = RewTerm(
            func=mdp.stand_still_after_motion,
            weight=-0.2,
            params={
                "command_name": "motion",
                "pos_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "vel_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "pos_weight": 1.0,
                "vel_weight": 0.04,
            },
        )

        # 起身任务推扰间隔放宽至5s，给策略充足恢复时间
        self.events.randomize_push_robot.interval_range_s = (0.0, 5.0)

        # 起身动作短，回合长度5s即可覆盖翻身到稳定站立过程
        self.episode_length_s = 5.0
