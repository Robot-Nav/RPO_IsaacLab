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

# Direct基础环境配置集合，定义场景、机器人、奖励、指令、噪声、事件等数据类
# 子类通过@configclass继承并覆写字段，实现RPO-Flat/Rough等具体任务配置

import math
from dataclasses import MISSING

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (  # noqa:F401
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRndCfg,
    RslRlSymmetryCfg,
)
from isaaclab.sim import SimulationCfg, PhysxCfg
import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from .scene_cfg import SceneCfg

from . import mdp


@configclass
class RewardCfg:
    """奖励配置基类，子类覆写具体RewTerm字段。"""
    pass


@configclass
class HeightScannerCfg:
    """高度扫描RayCaster配置，机器人前方地形感知。"""
    enable_height_scan: bool = False
    enable_height_scan_actor: bool = False  # 是否将高度扫描接入actor（非特权）观测
    prim_body_name: str = MISSING  # 扫描挂载的刚体名
    resolution: float = 0.1  # 网格分辨率(m)
    size: tuple = (1.6, 1.0)  # 扫描区域尺寸(m)
    debug_vis: bool = False
    drift_range: tuple = (0.0, 0.0)  # 扫描位置随机漂移范围
    offset: tuple = (0.0, 0.0, 20.0)  # 相对挂载体的偏置，z方向向上保证射线向下命中


@configclass
class SceneContextCfg:
    """场景上下文配置：并行环境数、机器人、地形、高度扫描等。"""
    num_envs: int = 4096
    env_spacing: float = 2.5
    robot: ArticulationCfg = MISSING
    terrain_type: str = MISSING
    terrain_generator: TerrainGeneratorCfg = None
    max_init_terrain_level: int = 5  # 初始地形难度上限，用于课程
    height_scanner: HeightScannerCfg = HeightScannerCfg()


@configclass
class RobotCfg:
    """机器人本体配置：观测/动作历史长度、动作缩放、终止判定、足部刚体名。"""
    actor_obs_history_length: int = 10
    critic_obs_history_length: int = 10
    action_history_length: int = 3
    action_scale: float = 0.25  # 策略输出乘以此系数叠加默认关节角
    terminate_contacts_body_names: list = None  # 接触地面即终止的刚体名
    terminate_base_height: float = None  # 基座高度低于此值终止
    terminate_base_orientation: float = None  # 基座倾斜超过此角(弧度)终止
    feet_body_names: list = MISSING


@configclass
class ObsScalesCfg:
    """观测向量各分量的缩放系数，统一量纲便于网络训练。"""
    lin_vel: float = 1.0
    ang_vel: float = 1.0
    projected_gravity: float = 1.0
    commands: float = 1.0
    joint_pos: float = 1.0
    joint_vel: float = 1.0
    actions: float = 1.0
    height_scan: float = 1.0


@configclass
class NormalizationCfg:
    """观测归一化配置：缩放因子与截断阈值。"""
    obs_scales: ObsScalesCfg = ObsScalesCfg()
    clip_observations: float = 100.0
    clip_actions: float = 100.0
    height_scan_offset: float = 0.5  # 高度扫描零点偏置（机器人脚踝高度）


@configclass
class CommandRangesCfg:
    """速度指令采样范围。"""
    lin_vel_x: tuple = (-0.6, 1.0)
    lin_vel_y: tuple = (-0.5, 0.5)
    ang_vel_z: tuple = (-1.57, 1.57)
    heading: tuple = (-math.pi, math.pi)


@configclass
class CommandsCfg:
    """速度指令生成配置。"""
    resampling_time_range: tuple = (10.0, 10.0)  # 指令重采样周期(s)
    rel_standing_envs: float = 0.2  # 静止指令环境比例
    rel_heading_envs: float = 1.0  # 航向追踪环境比例
    heading_command: bool = True  # True=航向角追踪，False=角速度追踪
    heading_control_stiffness: float = 0.5  # 航向PD控制刚度
    debug_vis: bool = True
    ranges: CommandRangesCfg = CommandRangesCfg()


@configclass
class NoiseScalesCfg:
    """观测噪声尺度，乘以obs_scales后注入均匀噪声。"""
    ang_vel: float = 0.2
    projected_gravity: float = 0.05
    joint_pos: float = 0.01
    joint_vel: float = 1.5
    height_scan: float = 0.1


@configclass
class NoiseCfg:
    """观测噪声总开关与尺度配置。"""
    add_noise: bool = True
    noise_scales: NoiseScalesCfg = NoiseScalesCfg()


@configclass
class EventCfg:
    """域随机化与重置事件配置：物理材质、质量、质心、执行器增益、关节参数、复位扰动、随机推力。"""
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "mass_distribution_params": (-3.0, 3.0),
            "operation": "add",
        },
    )
    randomize_rigid_body_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.025, 0.025), "z": (-0.05, 0.05)},
        },
    )
    scale_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=MISSING
            ),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    scale_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=MISSING),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    scale_joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=MISSING),
            "friction_distribution_params": (1.0, 1.0),
            "armature_distribution_params": (0.5, 1.5),
            "operation": "scale",
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.2, 0.2),
                "roll": (-0.52, 0.52),
                "pitch": (-0.52, 0.52),
                "yaw": (-0.78, 0.78),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )
    push_robot = EventTerm(
        # 周期性随机推力，训练策略应对外部扰动
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.2, 0.2),
                "roll": (-0.52, 0.52),
                "pitch": (-0.52, 0.52),
                "yaw": (-0.78, 0.78),
            }
        },
    )


@configclass
class BaseEnvCfg(DirectRLEnvCfg):
    """Direct基础环境配置基类，子类（RPOFlat/RPORough）覆写字段实现具体任务。"""
    viewer: ViewerCfg = ViewerCfg()
    decimation: int = 4  # 仿真步:策略步 = 4:1，5ms*4=20ms控制周期
    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=decimation,
        physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            )
        )
    episode_length_s: float =20.0
    scene_context: SceneContextCfg = SceneContextCfg()
    scene: SceneCfg = MISSING
    robot: RobotCfg = RobotCfg()
    reward: RewardCfg = RewardCfg()
    normalization: NormalizationCfg = NormalizationCfg()
    commands: CommandsCfg = CommandsCfg()
    noise: NoiseCfg = NoiseCfg(
        add_noise=True,
        noise_scales=NoiseScalesCfg(),
    )
    events: EventCfg = EventCfg()

    def __post_init__(self):
        pass


@configclass
class BaseAgentCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO agent配置基类：网络结构、优化器、训练超参与日志后端。"""
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24  # 每环境每迭代步数
    max_iterations = 12001
    runner_class_name = "OnPolicyRunner"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",  # 自适应学习率基于KL散度
        gamma=0.994,
        lam=0.9,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,  # RslRlSymmetryCfg()
        rnd_cfg=None,  # RslRlRndCfg()
    )
    clip_actions = None
    save_interval = 500
    experiment_name = ""
    run_name = ""
    logger = "wandb"
    neptune_project = "robolab"
    wandb_project = "robolab"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    def __post_init__(self):
        pass
