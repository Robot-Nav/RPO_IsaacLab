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

# RPO四足/双足机器人资产配置，定义URDF加载、初始关节位姿与PD执行器参数
# 执行器分组配置：legs(腿)/feet(脚)/shoulders(肩)/arms(臂)，均使用DelayedPDActuator模拟控制延迟
import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robolab.assets import ISAAC_DATA_DIR

RPO_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{ISAAC_DATA_DIR}/robots/roboparty/rpo/urdf/rpo.urdf",
        fix_base=False,
        activate_contact_sensors=True,
        # 用胶囊体替换圆柱体碰撞，提升PhysX碰撞检测稳定性
        replace_cylinders_with_capsules=True,
        # URDF转换层关节驱动增益置零，实际控制由isaaclab执行器接管
        joint_drive = sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
        articulation_props = sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        # 初始关节角为半蹲姿态，左右对称，手臂自然下垂微收
        joint_pos={
            "left_thigh_pitch_joint": -0.1,
            "left_knee_joint": 0.3,
            "left_ankle_pitch_joint": -0.2,
            "left_arm_pitch_joint": 0.18,
            "left_arm_roll_joint": 0.06,
            "left_elbow_pitch_joint": 0.78,
            "right_thigh_pitch_joint": -0.1,
            "right_knee_joint": 0.3,
            "right_ankle_pitch_joint": -0.2,
            "right_arm_pitch_joint": 0.18,
            "right_arm_roll_joint": -0.06,
            "right_elbow_pitch_joint": 0.78,
        },
        joint_vel={".*": 0.0},
    ),
    # 软关节限位缩放因子，将URDF限位收窄10%避免硬碰撞
    soft_joint_pos_limit_factor=0.90,
    actuators={
        # 腿部执行器：髋膝踝主关节，刚度较高支撑负重，膝/腰刚度更大
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_thigh_yaw_joint",
                ".*_thigh_roll_joint",
                ".*_thigh_pitch_joint",
                ".*_knee_joint",
                ".*torso.*",
            ],
            effort_limit_sim=120.0,
            velocity_limit_sim=25.0,
            stiffness={
                ".*_thigh_yaw_joint": 100.0,
                ".*_thigh_roll_joint": 100.0,
                ".*_thigh_pitch_joint": 100.0,
                ".*_knee_joint": 150.0,
                ".*torso.*": 150.0,
            },
            damping={
                ".*_thigh_yaw_joint": 3.3,
                ".*_thigh_roll_joint": 3.3,
                ".*_thigh_pitch_joint": 3.3,
                ".*_knee_joint": 5.0,
                ".*torso.*": 5.0,
            },
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        # 足部执行器：踝关节pitch/roll，扭矩较小
        "feet": DelayedPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=27.0,
            velocity_limit_sim=8.0,
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        # 肩部执行器：手臂pitch/roll/yaw三个自由度
        "shoulders": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_arm_pitch_joint",
                ".*_arm_roll_joint",
                ".*_arm_yaw_joint",
            ],
            effort_limit_sim=27.0,
            velocity_limit_sim=8.0,
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        # 肘部执行器：elbow pitch/yaw，pitch刚度高于yaw
        "arms": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_elbow_pitch_joint",
                ".*_elbow_yaw_joint",
            ],
            stiffness={
                ".*_elbow_pitch_joint": 30.0,
                ".*_elbow_yaw_joint": 20.0,
            },
            damping={
                ".*_elbow_pitch_joint": 1.5,
                ".*_elbow_yaw_joint": 1.0,
            },
            effort_limit_sim=27.0,
            velocity_limit_sim=8.0,
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
    },
)


# RPO机器人刚体link名称列表，顺序与URDF树一致
# 用于按名称索引各link的位姿、速度等张量
RPO_LINKS = [
    "base_link",
    "left_thigh_yaw_link",
    "left_thigh_roll_link",
    "left_thigh_pitch_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_thigh_yaw_link",
    "right_thigh_roll_link",
    "right_thigh_pitch_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_arm_pitch_link",
    "left_arm_roll_link",
    "left_arm_yaw_link",
    "left_elbow_pitch_link",
    "left_elbow_yaw_link",
    "right_arm_pitch_link",
    "right_arm_roll_link",
    "right_arm_yaw_link",
    "right_elbow_pitch_link",
    "right_elbow_yaw_link",
]
