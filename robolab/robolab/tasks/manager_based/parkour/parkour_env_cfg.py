# Parkour环境配置：场景（地形/传感器/相机）、观测（策略/评论员/判别器）、奖励、终止、事件、课程
# 基于AmpEnvCfg扩展，引入多奖励组MultiRewardCfg与噪声深度相机pipeline，适配RPO四足平台
import math
import os
from dataclasses import MISSING
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
from isaaclab.terrains import FlatPatchSamplingCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from robolab.tasks.manager_based.amp.amp_env_cfg import AmpEnvCfg, ObservationsCfg as AmpObservationsCfg
from robolab.tasks.manager_based.parkour.managers import MultiRewardCfg
import robolab.tasks.manager_based.parkour.mdp as mdp
import robolab.terrains as terrain_gen
from robolab.sensors import Grid3dPointsGeneratorCfg, NoisyGroupedRayCasterCameraCfg, VolumePointsCfg
from robolab.terrains import GreedyconcatEdgeCylinderCfg, TerrainImporterCfg
from robolab.utils.noise import (
    CropAndResizeCfg,
    DepthArtifactNoiseCfg,
    DepthNormalizationCfg,
    GaussianBlurNoiseCfg,
    PerlinNoiseCfg,
    PixelFailureNoiseCfg,
    RandomGaussianNoiseCfg,
    RandomConvNoiseCfg,
    RangeBasedGaussianNoiseCfg,
    ScaleRandomizationNoiseCfg,
    StereoFusionNoiseCfg,
)
from robolab.tasks.manager_based.parkour.terrain_generator_cfg import ROUGH_TERRAINS_CFG

__file_dir__ = os.path.dirname(os.path.realpath(__file__))

# NOTE: KEY_BODY_NAMES must match lab_key_body_names in robolab/scripts/tools/retarget/config/rpo.yaml
# 关键部位名（脚踝/膝盖/肘部），用于AMP判别器对比演示动作的关键体素位置，需与重定向配置保持一致
KEY_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_knee_link",
    "right_knee_link",
    "left_elbow_yaw_link",
    "right_elbow_yaw_link"
]

# 共享给leg_volume_points与volume_points_penetration奖励，使用同一对象保证脚部网格/配置修改保持同步
# 以下两组3D点网格用于足部/膝盖体积穿深检测：扫描体素网格是否进入地形，作为碰撞惩罚
LEG_VOLUME_POINTS_GRID = Grid3dPointsGeneratorCfg(
    x_min=-0.05,
    x_max=0.13,
    x_num=19,
    y_min=-0.03,
    y_max=0.03,
    y_num=7,
    z_min=-0.04,
    z_max=-0.02,
    z_num=3,
)
KNEE_VOLUME_POINTS_GRID = Grid3dPointsGeneratorCfg(
    x_min=-0.03,
    x_max=0.04,
    x_num=8,
    y_min=-0.03,
    y_max=0.03,
    y_num=7,
    z_min=-0.3,
    z_max=0.0,
    z_num=31,
)

@configclass
class SceneCfg(InteractiveSceneCfg):
    # 地面地形：使用ROUGH_TERRAINS_CFG生成器产生多类障碍地形，max_init_terrain_level限制初始难度上限
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
        virtual_obstacles={
            "edges": GreedyconcatEdgeCylinderCfg(
                cylinder_radius=0.03,
                min_points=2,
            ),
        },
    )
    # robots
    robot: ArticulationCfg = MISSING
    # 传感器：左右脚踝下方各一条射线扫描器，用于feet_at_plane奖励判断足底是否贴合地形
    left_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.04, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.12, size=[0.12, 0.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        update_period=0.02,
    )
    right_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.04, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.12, size=[0.12, 0.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        update_period=0.02,
    )
    # 接触传感器：history_length=3保留3帧历史，track_air_time用于步态/足部触地时长统计
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # 体积点云传感器：在脚踝/膝盖链接周围生成3D网格点，配合穿深奖励检测是否撞入地形
    leg_volume_points = VolumePointsCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link",
        points_generator=LEG_VOLUME_POINTS_GRID,
        debug_vis=False,
    )
    knee_volume_points = VolumePointsCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_knee_link",
        points_generator=KNEE_VOLUME_POINTS_GRID,
        debug_vis=False,
    )
    # 深度相机：基于RayCaster模拟Pinhole相机，叠加sim2real噪声pipeline模拟真实深度图
    camera = NoisyGroupedRayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        mesh_prim_paths=[
            "/World/ground",
            # NOTE: Don't forget to add the robot links in robot-specific configuration file.
        ],
        ray_alignment="yaw",
        pattern_cfg=PinholeCameraPatternCfg(
            focal_length=1.0,
            horizontal_aperture=2 * math.tan(math.radians(89.51) / 2),  # fovx
            vertical_aperture=2 * math.tan(math.radians(58.29) / 2),  # fovy
            width=64,
            height=36,
        ),
        debug_vis=False,
        data_types=["distance_to_image_plane"],
        update_period=0.02,
        depth_clipping_behavior="max",
        offset=NoisyGroupedRayCasterCameraCfg.OffsetCfg(
            pos=(
                0.0875,
                0.01,
                0.20568,
            ),
            rot=(
                0.866,
                0.0,
                0.5,
                0.0,
            ),
            convention="world",
        ),
        min_distance=0.1,
        # noise
        # 噪声pipeline顺序很关键：先在原始度量深度上做保守增强（缩放/空洞/卷积/柏林噪声/像素失效），
        # 最后做固定的预处理（裁剪/高斯模糊/归一化到[0,1]），尽量贴近真实深度图
        noise_pipeline={
            # --- conservative augmentations (applied on raw metric depth, before normalization) ---
            "scale_randomization": ScaleRandomizationNoiseCfg(
                apply_probability=0.5,
                scale_min=0.97,
                scale_max=1.03,
            ),
            "stereo_fusion": StereoFusionNoiseCfg(
                apply_probability=0.4,
                disparity_grad_threshold=0.10,
                texture_var_threshold=3e-4,
                hole_probability=0.02,
                hole_kernel_size=1,
                hole_value=2.5,  # treat holes as max-range (2.5 m) before normalization
            ),
            "random_conv": RandomConvNoiseCfg(
                apply_probability=0.3,
                kernel_std=0.05,
                center_weight=1.0,
            ),
            "perlin_noise": PerlinNoiseCfg(
                apply_probability=0.5,
                octaves=3,
                base_frequency=8.0,
                lacunarity=2.0,
                persistence=0.5,
                amplitude=1.0,
                noise_std=0.01,  # ~1 cm at 1 m, relative to 2.5 m range
            ),
            "pixel_failures": PixelFailureNoiseCfg(
                apply_probability=0.5,
                dead_pixel_prob=5e-4,
                saturated_pixel_prob=5e-4,
                dead_value=0.0,
                saturated_value=2.5,  # saturated = max-range before normalization
            ),
            # --- fixed preprocessing (keep last) ---
            "crop_and_resize": CropAndResizeCfg(crop_region=(18, 0, 16, 16)),
            "gaussian_blur": GaussianBlurNoiseCfg(kernel_size=3, sigma=1),
            "depth_normalization": DepthNormalizationCfg(
                depth_range=(0.0, 2.5),
                normalize=True,
                output_range=(0.0, 1.0),
            ),
        },
        data_histories={"distance_to_image_plane_noised": 37},
    )
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    # sensors
    # 全局高度扫描器：在躯干下方2x1m网格扫描地形高度，仅用于critic观测（policy不直接见高度图）
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 5.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[2.0, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # 策略观测：本体感知+深度图（不含高度扫描，强迫策略用视觉补偿地形不可见部分）
        # 每项均history_length=8保留8帧历史，flatten_history_dim=True将历史展平拼接到向量

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            history_length=8,
            flatten_history_dim=True,
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=8,
            flatten_history_dim=True,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            history_length=8,
            flatten_history_dim=True,
            params={"command_name": "base_velocity"},
            noise=None,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.03, n_max=0.03), history_length=8, flatten_history_dim=True
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-0.5, n_max=0.5),
            scale=0.05,
            history_length=8,
            flatten_history_dim=True,
        )
        actions = ObsTerm(func=mdp.last_action, history_length=8, flatten_history_dim=True, clip=(-10.0, 10.0))
        depth_image = ObsTerm(
            func=mdp.delayed_visualizable_image,
            params={
                "data_type": "distance_to_image_plane_noised_history",
                "sensor_cfg": SceneEntityCfg("camera"),
                "history_skip_frames": 5,
                "num_output_frames": 8,
                "delayed_frame_ranges": (0, 1),
                "debug_vis": False,
            },
            noise=None,
        )

        def __post_init__(self):
            self.enable_corruption = True
            # concatenate_terms=False：每项obs保持独立张量，由底层框架按各自形状打包
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""
        # 评论员观测：额外包含base_lin_vel和height_scan，提供训练时特权信息加速收敛
        # enable_corruption=False表示critic侧不加噪声，使用干净真值

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, history_length=8, flatten_history_dim=True)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            history_length=8,
            flatten_history_dim=True,
            scale=0.25,
        )
        projected_gravity = ObsTerm(func=mdp.projected_gravity, history_length=8, flatten_history_dim=True)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            history_length=8,
            flatten_history_dim=True,
            params={"command_name": "base_velocity"},
            noise=None,
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, history_length=8, flatten_history_dim=True)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, history_length=8, flatten_history_dim=True)
        actions = ObsTerm(func=mdp.last_action, history_length=8, flatten_history_dim=True, clip=(-10.0, 10.0))
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-5.0, 5.0),
            history_length=8,
            flatten_history_dim=True,
        )
        depth_image = ObsTerm(
            func=mdp.delayed_visualizable_image,
            params={
                "data_type": "distance_to_image_plane_noised_history",
                "sensor_cfg": SceneEntityCfg("camera"),
                "history_skip_frames": 5,
                "num_output_frames": 8,
                "delayed_frame_ranges": (0, 1),
                "debug_vis": False,
            },
            noise=None,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    critic: CriticCfg = CriticCfg()

    @configclass
    class DiscriminatorCfg(ObsGroup):
        # AMP判别器观测：用于区分策略动作与演示动作，root姿态/角速度/关节状态/关键体素位置
        # history_length=10覆盖一个步态周期，concatenate_terms=True按时间维拼接
        root_local_rot_tan_norm = ObsTerm(func=mdp.root_local_rot_tan_norm)
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        key_body_pos_b = ObsTerm(
            func=mdp.key_body_pos_b,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="robot", 
                    body_names=KEY_BODY_NAMES, 
                    preserve_order=True
                )
            },
        )
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.concatenate_dim = -1
            self.history_length = 10
            self.flatten_history_dim = False
            
    disc: DiscriminatorCfg = DiscriminatorCfg()
            
    @configclass
    class DiscriminatorDemoCfg(ObsGroup):
        # AMP演示侧观测：从motion数据集读取参考动作，与DiscriminatorCfg结构对应
        # flatten_steps_dim=False保留时间步维度，便于判别器按序列对比
        ref_root_local_rot_tan_norm = ObsTerm(
            func=mdp.ref_root_local_rot_tan_norm,
            params={
                "animation": "animation",
                "flatten_steps_dim": False,
            }
        )
        # ref_root_lin_vel_b = ObsTerm(
        #     func=mdp.ref_root_lin_vel_b,
        #     params={
        #         "animation": "animation",
        #         "flatten_steps_dim": False,
        #     }
        # )
        ref_root_ang_vel_b = ObsTerm(
            func=mdp.ref_root_ang_vel_b,
            params={
                "animation": "animation",
                "flatten_steps_dim": False,
            }
        )
        ref_joint_pos = ObsTerm(
            func=mdp.ref_joint_pos,
            params={
                "animation": "animation",
                "flatten_steps_dim": False,
            }
        )
        ref_joint_vel = ObsTerm(
            func=mdp.ref_joint_vel,
            params={
                "animation": "animation",
                "flatten_steps_dim": False,
            }
        )
        ref_key_body_pos_b = ObsTerm(
            func=mdp.ref_key_body_pos_b,
            params={
                "animation": "animation",
                "flatten_steps_dim": False,
            }
        )
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.concatenate_dim = -1
    
    disc_demo: DiscriminatorDemoCfg = DiscriminatorDemoCfg()


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    # 关节位置动作：输出目标关节偏移量，scale=0.25限制单步动作幅度保证平滑
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    # 速度/位姿命令：每8-12s重采样，根据当前地形类型从velocity_ranges选择对应速度范围
    # random_velocity_terrain指定地形随机化时使用stand模式（站立）作为默认
    # straight_target_prob=0.8强制80%概率y向目标速度为0，鼓励直线行走
    # only_positive_lin_vel_x=True限制只朝前走，避免后退穿越障碍

    base_velocity = mdp.PoseVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        debug_vis=False,
        velocity_control_stiffness=2.0,
        heading_control_stiffness=2.0,
        rel_standing_envs=0.05,
        straight_target_prob=0.8, # 80% chance to force the target y to 0 for straight walking.
        ranges=mdp.PoseVelocityCommandCfg.Ranges(lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(-1.0, 1.0)),
        random_velocity_terrain=["perlin_rough_stand"],
        velocity_ranges={
            "perlin_rough": {"lin_vel_x": (0.4, 1.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
            "perlin_rough_walk": {"lin_vel_x": (0.4, 1.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (0.0, 0.0)},
            "perlin_rough_trun": {"lin_vel_x": (0.0, 0.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
            "perlin_rough_stand": {"lin_vel_x": (0.0, 0.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (0.0, 0.0)},
            "square_gaps": {"lin_vel_x": (0.4, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
            "pyramid_stairs_32": {"lin_vel_x": (0.4, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
            "pyramid_stairs_30": {"lin_vel_x": (0.4, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
            "pyramid_stairs_28": {"lin_vel_x": (0.4, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
            "pyramid_stairs_inv_32": {"lin_vel_x": (0.4, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
            "pyramid_stairs_inv_30": {"lin_vel_x": (0.4, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
            "pyramid_stairs_inv_28": {"lin_vel_x": (0.4, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
            "hf_pyramid_slope_inv": {"lin_vel_x": (0.4, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
        },
        only_positive_lin_vel_x=True,
        lin_vel_threshold=0.0,
        ang_vel_threshold=0.0,
        target_dis_threshold=0.4,
    )


@configclass
class ParkourRewardsCfg(MultiRewardCfg):
    """Flat reward terms for parkour (single group ``rewards`` for MultiRewardManager)."""
    # Parkour奖励集合：跟踪/正则/安全三类，配合课程动态调整权重
    # Task rewards：跟踪线/角速度、航向误差、存活奖励等任务目标
    # Regularization rewards：穿深/滑动/姿态/能耗/动作平滑等正则项
    # Safety rewards：关节限位/扭矩/速度限制/非法接触等安全约束

    # Task rewards
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=5.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=5.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    heading_error = RewTerm(func=mdp.heading_error, weight=-1.0, params={"command_name": "base_velocity"})
    dont_wait = RewTerm(func=mdp.dont_wait, weight=-0.5, params={"command_name": "base_velocity"})
    is_alive = RewTerm(func=mdp.is_alive, weight=3.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-5.0)
    stand_still = RewTerm(func=mdp.stand_still, weight=-1.0)
    rpo_thigh_yaw_joint_sign_penalty = RewTerm(func=mdp.rpo_thigh_yaw_joint_sign_penalty, weight=-10.0)
    # Regularization rewards
    volume_points_penetration_feet = RewTerm(
        func=mdp.volume_points_penetration_feet,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("leg_volume_points"),
            # 台阶地形启用按地形权重的渐进惩罚：上层台阶穿深惩罚更轻，避免楼梯学习卡死
            "enable_terrain_foot_weights": True,
            "stairs_weight_min": 0.2,
            "stairs_weight_max": 1.0,
            "debug_print_terrain": False,
        },
    )
    volume_points_penetration_knee = RewTerm(
        func=mdp.volume_points_penetration,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("knee_volume_points"),
        },
    )
    feet_slide = RewTerm(
        func=mdp.contact_slide,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "threshold": 1.0,
        },
    )
    joint_deviation_upper_body = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_arm_.*_joint", ".*_elbow_.*_joint", "torso_joint"],
            )
        },
    )
    freeze_upper_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.8,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["torso_joint"]
            ),
        },
    )
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1)
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    dof_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    joint_regularization = RewTerm(func=mdp.joint_deviation_l1, weight=-1e-4)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-3.0)
    pelvis_orientation_l2 = RewTerm(
        func=mdp.link_orientation, weight=-3.0, params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
    )
    feet_flat_ori = RewTerm(
        func=mdp.feet_orientation_contact,
        weight=-0.4,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    feet_at_plane = RewTerm(
        func=mdp.feet_at_plane,
        weight=-0.1,
        params={
            "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "left_height_scanner_cfg": SceneEntityCfg("left_height_scanner"),
            "right_height_scanner_cfg": SceneEntityCfg("right_height_scanner"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            # height_offset为足底到脚踝链接的z偏移，使用左右脚扫描器测高判定足底是否水平贴地
            "height_offset": 0.035,
        },
    )
    sound_suppression = RewTerm(
        func=mdp.sound_suppression_acc_per_foot,
        weight=-5e-4,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=".*_ankle_roll_link",
            ),
        },
    )
    energy = RewTerm(
        func=mdp.motors_power_square,
        weight=-5e-5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "normalize_by_stiffness": True,
        },
    )

    # Safety rewards
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    dof_vel_limits = RewTerm(
        func=mdp.joint_vel_limits,
        weight=-1.0,
        params={"soft_ratio": 0.9, "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    torque_limits = RewTerm(
        func=mdp.applied_torque_limits_by_ratio,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "limit_ratio": 0.8,
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="(?!.*_ankle_roll_link).*"),
            "threshold": 1.0,
        },
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle_roll_link", ".*_knee_link"]),
        },
    )

@configclass
class RewardsCfg(MultiRewardCfg):
    rewards: ParkourRewardsCfg = ParkourRewardsCfg()
    
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    # 终止条件：超时/越界/躯干碰撞/姿态失控/跌落，跌落阈值0.5m避免穿越深沟时误判

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    terrain_out_bound = DoneTerm(func=mdp.terrain_out_of_bounds, time_out=True, params={"distance_buffer": 2.0})
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"),
            "threshold": 1.0,
        },
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})
    root_height = DoneTerm(func=mdp.root_height_below_env_origin_minimum, params={"minimum_height": 0.5})


@configclass
class EventCfg:
    """Configuration for events."""
    # 事件配置：startup模式做域随机化（摩擦/质量/质心/执行器增益/关节参数/相机外参）
    # reset模式做reset（基座位姿/关节偏移）和虚拟障碍注册

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.6),
            "restitution_range": (0.05, 0.5),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-1.0, 1.0),
            "operation": "add",
        },
    )
    
    randomize_rigid_body_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link", "base_link"]),
            "com_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.02, 0.02)},
        },
    )
    
    scale_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_.*_link", "right_.*_link"]),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    
    scale_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_joint"]),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    scale_joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_joint"]),
            "friction_distribution_params": (1.0, 1.0),
            "armature_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    
    # # reset
    
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.1, 0.1)},
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.2),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
        },
    )

    register_virtual_obstacles = EventTerm(
        func=mdp.register_virtual_obstacle_to_sensor,
        mode="startup",
        params={
            # 把地形边界圆柱虚拟障碍注册到体积点传感器，避免足部穿深误判边界护栏为软地面
            "sensor_cfgs": SceneEntityCfg("leg_volume_points"),
        },
    )
    
    register_virtual_obstacles_knee = EventTerm(
        func=mdp.register_virtual_obstacle_to_sensor,
        mode="startup",
        params={
            "sensor_cfgs": SceneEntityCfg("knee_volume_points"),
        },
    )
    
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.15, 0.15),
            "velocity_range": (0.0, 0.0),
        },
    )
    
    
    # reset_robot_joints=EventTerm(
    #     func=mdp.reset_joints_by_scale,
    #     mode="reset",
    #     params={
    #         "position_range": (0.8, 1.2),
    #         "velocity_range": (0.0, 0.0),
    #     },
    # )

        
    randomize_camera_offset = EventTerm(
        func=mdp.randomize_camera_offsets,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("camera"),
            "offset_pose_ranges": {
                "x": (-0.03, 0.03),
                "y": (-0.03, 0.03),
                "z": (-0.03, 0.03),
                "roll": (-math.radians(3), math.radians(3)),
                "pitch": (-math.radians(3), math.radians(3)),
                "yaw": (-math.radians(3), math.radians(3)),
            },
            "distribution": "uniform",
        },
    )
    
@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    # 课程学习：tracking_exp_vel按速度跟踪表现自动升降地形难度
    # modify_rewards_weight随难度提升把穿深/碰撞等安全奖励权重放大到final_weight
    # step_size=0.03控制单步权重变化幅度，避免训练抖动

    terrain_levels = CurrTerm(
        func=mdp.tracking_exp_vel,
        params={
            # 线速度阈值(0.7,0.9)区间触发地形难度升降，角速度阈值0表示纯前向运动
            "lin_vel_threshold": (0.7, 0.9),
            "ang_vel_threshold": (0.0, 0.0),
        },
    )
    volume_points_penetration_weight_feet = CurrTerm(
        func=mdp.modify_rewards_weight,
        params={
            "term_name": "volume_points_penetration_feet",
            # init→final把足部穿深惩罚从-1放大到-100，难度越高对穿深越严格
            "init_weight": -1.0,
            "final_weight": -100.0,
            "lin_vel_threshold": (0.7, 0.9),
            "ang_vel_threshold": (0.0, 0.0),
            "step_size": 0.03,
        },
    )
    volume_points_penetration_weight_knee = CurrTerm(
        func=mdp.modify_rewards_weight,
        params={
            "term_name": "volume_points_penetration_knee",
            "init_weight": -1.0,
            "final_weight": -100.0,
            "lin_vel_threshold": (0.7, 0.9),
            "ang_vel_threshold": (0.0, 0.0),
            "step_size": 0.03,
        },
    )
    feet_stumble_weight = CurrTerm(
        func=mdp.modify_rewards_weight,
        params={
            "term_name": "feet_stumble",
            "init_weight": -1.0,
            "final_weight": -10.0,
            "lin_vel_threshold": (0.7, 0.9),
            "ang_vel_threshold": (0.0, 0.0),
            "step_size": 0.03,
        },
    )
    undesired_contacts_weight = CurrTerm(
        func=mdp.modify_rewards_weight,
        params={
            "term_name": "undesired_contacts",
            "init_weight": -1.0,
            "final_weight": -10.0,
            "lin_vel_threshold": (0.7, 0.9),
            "ang_vel_threshold": (0.0, 0.0),
            "step_size": 0.03,
        },
    )

@configclass
class MonitorCfg:
    pass


##
# Environment configuration
##


@configclass
class ParkourEnvCfg(AmpEnvCfg):
    # Scene settings
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        # general settings
        self.decimation = 4  # 控制频率=200Hz/4=50Hz
        self.episode_length_s = 20.0
        # simulation settings
        # 仿真dt=5ms，decimation=4 → 控制周期20ms；physx参数提升GPU碰撞容量以适应复杂地形并行仿真
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.sim.physx.gpu_collision_stack_size = 2**29
        # update sensor update periods
        # 接触传感器update_period对齐仿真dt保证逐物理步采样
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
