# RPO四足机器人的Parkour环境特化配置：绑定RPO机器人资产、AMP动作数据集、鞋型脚配置
# 提供训练（ROUGH）与回放（PLAY）两套配置，PLAY版禁用难度梯度与墙障碍方便可视化
import copy
import os

from isaaclab.utils import configclass

from robolab import ROBOLAB_ROOT_DIR
from robolab.assets.robots.roboparty import RPO_CFG, RPO_LINKS
from robolab.sensors import get_link_prim_targets
from robolab.tasks.manager_based.parkour.parkour_env_cfg import ROUGH_TERRAINS_CFG, ParkourEnvCfg

# RPO出生高度0.85m（脚底到躯干原点），适配四足站立姿态
RPO_CFG.init_state.pos = (0.0, 0.0, 0.85)
AMP_NUM_STEPS = 3


# 回放地形：基于训练地形深拷贝后清零wall_prob，避免PLAY模式出现墙体干扰可视化
ROUGH_TERRAINS_CFG_PLAY = copy.deepcopy(ROUGH_TERRAINS_CFG)
for sub_terrain_name, sub_terrain_cfg in ROUGH_TERRAINS_CFG_PLAY.sub_terrains.items():
    sub_terrain_cfg.wall_prob = [0.0, 0.0, 0.0, 0.0]


@configclass
class RPOParkourRoughEnvCfg(ParkourEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG
        self.scene.robot = RPO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # 把RPO机器人各link prim加入相机mesh_prim_paths，使深度图能命中机器人自身（如手臂）
        self.scene.camera.mesh_prim_paths.extend(get_link_prim_targets(RPO_LINKS))
        # AMP动作数据集：rpo_lab目录下的多段动作，覆盖前进/转弯/站立等基础步态
        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            ROBOLAB_ROOT_DIR, "data", "motions", "rpo_lab"
        )
        # motion_data_weights按动作名称加权采样，避免长动作主导判别器
        self.motion_data.motion_dataset.motion_data_weights = {
            "36_01": 1,
            "36_11": 1,
            "114_08": 1,
            "114_09": 1,
            "A1-_Stand_stageii": 1,
            "B9_-__Walk_turn_left_90_stageii": 1,
            "B10_-__Walk_turn_left_45_stageii": 1,
            "B13_-__Walk_turn_right_90_stageii": 1,
            "B14_-__Walk_turn_right_45_t2_stageii": 1,
            "B15_-__Walk_turn_around_stageii": 1,
            "turn_l": 1,
            "turn_r": 1,
        }
        # AMP判别器每次取3步序列，与disc.history_length保持一致
        self.animation.animation.num_steps_to_use = AMP_NUM_STEPS
        self.observations.disc.history_length = AMP_NUM_STEPS


class ShoeConfigMixin:
    # 鞋型脚配置Mixin：RPO穿戴鞋型附件后足部几何变化，需同步体积点z范围与feet_at_plane高度偏移
    def apply_shoe_config(self):
        self.scene.robot = RPO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # 鞋型脚比裸足厚约2cm，向下扩展体积点z范围覆盖整个鞋底
        self.scene.leg_volume_points.points_generator.z_min = -0.063
        self.scene.leg_volume_points.points_generator.z_max = -0.023
        # 鞋底到脚踝链接的z偏移由0.035变为0.058，匹配鞋底实际厚度
        self.rewards.rewards.feet_at_plane.params["height_offset"] = 0.058



@configclass
class RPOParkourRoughEnvCfg_PLAY(RPOParkourRoughEnvCfg):
    # PLAY配置：用于训练后回放/可视化，关闭随机化与难度梯度，方便观察策略行为
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG_PLAY
        # make a smaller scene for play
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        self.episode_length_s = 10
        # 关闭root_height终止条件：回放时不希望因跌落被提前结束
        self.terminations.root_height = None

        # self.commands.base_velocity.velocity_ranges["pyramid_stairs"] = {"lin_vel_x": (1.0, 1.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)}
        # self.commands.base_velocity.velocity_ranges["pyramid_stairs_high"] = {"lin_vel_x": (1.0, 1.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)}
        # self.commands.base_velocity.velocity_ranges["pyramid_stairs_inv"] = {"lin_vel_x": (1.0, 1.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)}
        # self.commands.base_velocity.velocity_ranges["pyramid_stairs_inv_high"] = {"lin_vel_x": (1.0, 1.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)}
        # self.commands.base_velocity.velocity_ranges["pyramid_stairs_inv_high_ground_aligned"] = {"lin_vel_x": (1.0, 1.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)}
        # self.commands.base_velocity.velocity_ranges["hf_pyramid_slope_inv"] = {"lin_vel_x": (1.0, 1.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)}
        self.commands.base_velocity.resampling_time_range = (8.0, 12.0)
        # rel_standing_envs=0：PLAY时所有环境都必须行走，不设站立比例
        self.commands.base_velocity.rel_standing_envs = 0.0

        # spawn the robot randomly in the grid (instead of their terrain levels)
        # reduce the number of terrains to save memory
        # 把地形网格缩减为1x1，让回放固定在单一地形上节省显存
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 1
            self.scene.terrain.terrain_generator.num_cols = 1

        # 开启关键传感器与命令的可视化便于调试
        self.scene.leg_volume_points.debug_vis = True
        self.scene.knee_volume_points.debug_vis = True
        self.commands.base_velocity.debug_vis = True
        # 关闭物理材质随机化：PLAY时使用固定摩擦系数保证行为可复现
        self.events.physics_material = None
        # 重置时关节偏移清零：每次reset都从默认姿态开始
        self.events.reset_robot_joints.params = {
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        }


@configclass
class RPOParkourEnvCfg(RPOParkourRoughEnvCfg, ShoeConfigMixin):
    # 训练配置：组合Rough训练基线与鞋型脚配置
    def __post_init__(self):
        super().__post_init__()
        self.apply_shoe_config()


@configclass
class RPOParkourEnvCfg_PLAY(RPOParkourRoughEnvCfg_PLAY, ShoeConfigMixin):
    # 回放配置：组合PLAY基线与鞋型脚配置
    def __post_init__(self):
        super().__post_init__()
        self.apply_shoe_config()
