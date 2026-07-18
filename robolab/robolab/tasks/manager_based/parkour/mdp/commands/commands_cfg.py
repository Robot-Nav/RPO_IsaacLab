# 速度/位姿命令配置：基于地形flat_patch采样目标点，按位置/航向误差生成机体速度命令
# 支持按地形类型设置不同速度范围，随机速度地形覆盖默认命令
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

from .pose_velocity_command import PoseVelocityCommand


@configclass
class PoseVelocityCommandCfg(CommandTermCfg):
    """Configuration for the position command generator."""
    # 命令生成器配置：目标位姿由地形patch采样，速度命令由位置/航向误差经刚度比例转换得到

    class_type: type = PoseVelocityCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    velocity_control_stiffness: float = 1.0
    """Scale factor to convert the position error to linear velocity command. Defaults to 1.0."""
    # 位置误差到线速度命令的比例增益，越大追踪越激进

    heading_control_stiffness: float = 1.0
    """Scale factor to convert the heading error to angular velocity command. Defaults to 1.0."""
    # 航向误差到偏航角速度命令的比例增益

    only_positive_lin_vel_x: bool = False
    """Whether to only sample positive linear x velocity commands. Defaults to False."""
    # 仅前向行走场景下置True，禁止后退命令

    @configclass
    class Ranges:
        """Uniform distribution ranges for the velocity commands."""
        # 速度命令采样范围，用于random_velocity_terrain地形的随机速度覆盖

        lin_vel_x: tuple[float, float] = MISSING
        """Range for the linear-x velocity command (in m/s)."""

        lin_vel_y: tuple[float, float] = MISSING
        """Range for the linear-y velocity command (in m/s)."""

        ang_vel_z: tuple[float, float] = MISSING
        """Range for the angular-z velocity command (in rad/s)."""

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands. Only used in random_velocity_terrains."""

    random_velocity_terrain: list[str] = None
    """List of terrain types for which the velocity commands should be randomized."""
    # 列出的地形类型会忽略位置命令，改用ranges随机采样速度命令

    velocity_ranges: dict = None
    """Dictionary containing velocity ranges for different terrains. If not None, the velocity ranges will be set based on the terrain type."""
    # 按地形类型名映射速度范围，让简单地形允许高速、复杂地形限制速度

    lin_vel_threshold: float = 0.15
    """Minimal threshold for the linear velocity command (in m/s)."""
    # 线速度死区，低于阈值的命令归零避免微小抖动

    ang_vel_threshold: float = 0.15
    """Minimal threshold for the angular velocity command (in rad/s)."""
    # 角速度死区

    lin_vel_metrics_std: float = 0.5
    """Standard deviation for the linear velocity metrics."""
    # 评估指标用的exp核标准差，用于tracking_exp_vel_xy统计

    ang_vel_metrics_std: float = 0.5
    """Standard deviation for the angular velocity metrics."""

    rel_standing_envs: float = 0.0
    """The sampled probability of environments that should be standing still. Defaults to 0.0."""
    # 站立环境比例，重采样时按此概率把命令置零训练原地稳定

    straight_target_prob: float = 0.0
    """Probability of forcing the sampled target y to the robot's current y for straight walking."""
    # 强制目标y对齐当前y的概率，用于训练直线行走

    target_dis_threshold: float = 0.2
    """The distance threshold to the target position below which the command is set to zero. Defaults to 0.2."""
    # 到达目标距离阈值，小于阈值时命令归零避免目标点附近震荡

    flat_patch_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/TerrainFlatPatches",
        markers={
            "Goal": sim_utils.CylinderCfg(
                radius=0.15,
                height=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
            "Patches": sim_utils.CylinderCfg(
                radius=0.15,
                height=0.05,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            ),
        },
    )
    """The configuration for the goal pose visualization marker."""

    patch_vis = False
    """Whether to visualize the flat patches."""

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""
