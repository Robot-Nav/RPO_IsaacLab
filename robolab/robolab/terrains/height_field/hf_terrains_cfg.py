# 高度场地形配置类集合：基于 Perlin 噪声扩展 IsaacLab 原生高度场地形
# 所有配置混入 WallTerrainCfgMixin 提供边界墙生成参数，perlin_cfg 控制叠加的噪声地形参数
# 支持难度区间参数：标量为固定值，tuple/list 按难度线性插值，用于课程学习
from dataclasses import MISSING
from typing import List

from isaaclab.terrains.height_field import (
    HfDiscreteObstaclesTerrainCfg,
    HfInvertedPyramidSlopedTerrainCfg,
    HfInvertedPyramidStairsTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfPyramidStairsTerrainCfg,
    HfSteppingStonesTerrainCfg,
    HfTerrainBaseCfg,
    HfWaveTerrainCfg,
)
from isaaclab.utils import configclass

from . import hf_terrains


class WallTerrainCfgMixin:
    """边界墙配置混入类，提供四面墙的生成概率与几何参数。"""
    wall_prob: List[float] = [0.0, 0.0, 0.0, 0.0]  # [left, right, front, back] 四条边各自生成墙的概率
    wall_height: float = 5.0  # 墙高
    wall_thickness: float = 0.05  # 墙厚


@configclass
class PerlinPlaneTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """纯 Perlin 分形噪声平面地形配置，作为其他地形的噪声叠加基底。"""
    function = hf_terrains.perlin_plane_terrain

    noise_scale: float | List[float] = 0.05
    noise_frequency: int = 20

    fractal_octaves = 2
    fractal_lacunarity = 2.0
    fractal_gain = 0.25

    centering = False  # 为 True 时噪声将以 0 为中心


@configclass
class PerlinPyramidSlopedTerrainCfg(HfPyramidSlopedTerrainCfg, WallTerrainCfgMixin):
    """金字塔斜坡地形配置，中心保留平顶平台，可选叠加 Perlin 噪声。"""
    function = hf_terrains.perlin_pyramid_sloped_terrain
    perlin_cfg: PerlinPlaneTerrainCfg | None = None


@configclass
class PerlinInvertedPyramidSlopedTerrainCfg(HfInvertedPyramidSlopedTerrainCfg, WallTerrainCfgMixin):
    """倒置金字塔斜坡地形配置，平台位于底部。"""
    function = hf_terrains.perlin_pyramid_sloped_terrain
    perlin_cfg: PerlinPlaneTerrainCfg | None = None


@configclass
class PerlinPyramidStairsTerrainCfg(HfPyramidStairsTerrainCfg, WallTerrainCfgMixin):
    """金字塔阶梯地形配置，从外向内逐级升高至中心平台。"""
    function = hf_terrains.perlin_pyramid_stairs_terrain
    perlin_cfg: PerlinPlaneTerrainCfg | None = None


@configclass
class PerlinInvertedPyramidStairsTerrainCfg(HfInvertedPyramidStairsTerrainCfg, WallTerrainCfgMixin):
    """倒置金字塔阶梯地形配置，平台位于底部。"""
    function = hf_terrains.perlin_pyramid_stairs_terrain
    perlin_cfg: PerlinPlaneTerrainCfg | None = None


@configclass
class PerlinInvertedPyramidStairsGroundAlignedTerrainCfg(HfInvertedPyramidStairsTerrainCfg, WallTerrainCfgMixin):
    """Alias of :class:`PerlinInvertedPyramidStairsTerrainCfg` (same defaults since align_min_height defaults to True on the parent).

    地面对齐变体：生成后将整体高度场下移使最小值对齐地面，避免悬空。
    """
    function = hf_terrains.perlin_pyramid_stairs_ground_aligned_terrain
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

@configclass
class PerlinDiscreteObstaclesTerrainCfg(HfDiscreteObstaclesTerrainCfg, WallTerrainCfgMixin):
    """离散障碍物地形配置：在平面上随机放置不同高度的立方柱，中心保留平台。"""
    function = hf_terrains.perlin_discrete_obstacles_terrain
    perlin_cfg: PerlinPlaneTerrainCfg | None = None


@configclass
class PerlinWaveTerrainCfg(HfWaveTerrainCfg, WallTerrainCfgMixin):
    """正弦波形地形配置：基于 sin/cos 叠加生成波形，可选叠加 Perlin 噪声。"""
    function = hf_terrains.perlin_wave_terrain
    perlin_cfg: PerlinPlaneTerrainCfg | None = None


@configclass
class PerlinSteppingStonesTerrainCfg(HfSteppingStonesTerrainCfg, WallTerrainCfgMixin):
    """踏石地形配置：在深坑基底上随机放置凸起石块，中心保留平台供机器人生成。"""
    function = hf_terrains.perlin_stepping_stones_terrain
    perlin_cfg: PerlinPlaneTerrainCfg | None = None


# -- Newly added terrain configurations for parkour terrains-- #
@configclass
class PerlinParapetTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a parapet terrain, can be used for jump and hurdle tasks.

    跳栏/跨栏地形配置：中央凸起矩形障碍物，难度递增时高度与长度可变。
    """

    function = hf_terrains.perlin_parapet_terrain
    parapet_height: tuple[float, float] | float = (0.1, 0.3)
    parapet_length: tuple[float, float] | float = (0.1, 0.3)
    parapet_width: float | None = None
    curved_top_rate: float | None = None
    """The rate to generate curved top. If None, the top will be flat."""
    perlin_cfg: PerlinPlaneTerrainCfg | None = None


@configclass
class PerlinGutterTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a gutter parkour terrain.

    凹槽地形配置：中央下凹矩形沟槽，用于训练机器人跨越或绕行。
    """

    function = hf_terrains.perlin_gutter_terrain
    gutter_length: tuple[float, float] | float = (0.5, 1.5)  # 凹槽间距
    gutter_depth: tuple[float, float] | float = (0.1, 0.3)  # 凹槽深度
    gutter_width: float | None = None  # 凹槽长度方向尺寸
    perlin_cfg: PerlinPlaneTerrainCfg | None = None


@configclass
class PerlinStairsUpDownTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a stairs up and down parkour terrain.

    上行-平台-下行阶梯地形配置，中央保留平台；难度递增时步高、步长、步数可变。
    """

    function = hf_terrains.perlin_stairs_up_down_terrain
    per_step_height: tuple[float, float] | float = MISSING
    """The height of each step. Could be a fixed value or a range (min, max)."""
    per_step_width: float | None = None
    """The width of each step. If None, it will be equal to the width of the terrain."""
    per_step_length: tuple[float, float] | float = MISSING
    """The length of each step along the y-axis."""
    num_steps: tuple[int, int] | int = MISSING
    """The number of steps. Could be a fixed value or a range (min, max)."""

    platform_length: float = 1.0
    """The length of the platform at the bottom of the stairs."""

    perlin_cfg: PerlinPlaneTerrainCfg | None = None


@configclass
class PerlinStairsDownUpTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a stairs down and up parkour terrain.

    下行-平台-上行阶梯地形配置，与 PerlinStairsUpDownTerrainCfg 方向相反。
    """

    function = hf_terrains.perlin_stairs_down_up_terrain
    per_step_height: tuple[float, float] | float = MISSING
    """The height of each step. Could be a fixed value or a range (min, max)."""
    per_step_width: float | None = None
    """The width of each step. If None, it will be equal to the width of the terrain."""
    per_step_length: tuple[float, float] | float = MISSING
    """The length of each step along the y-axis."""
    num_steps: tuple[int, int] | int = MISSING
    """The number of steps. Could be a fixed value or a range (min, max)."""

    platform_length: float = 1.0
    """The length of the platform at the bottom of the stairs."""

    perlin_cfg: PerlinPlaneTerrainCfg | None = None


@configclass
class PerlinTiltTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a tilt terrain.

    墙体带开口地形配置：中央放置带门洞的墙体，难度递增时门洞宽度收窄，开口角度可变。
    """

    function = hf_terrains.perlin_tilt_terrain
    wall_height: tuple[float, float] | float = MISSING
    wall_width: float | None = None
    wall_length: tuple[float, float] | float = MISSING
    wall_opening_angle: tuple[float, float] | float = MISSING  # 单位：度
    wall_opening_width: tuple[float, float] | float = MISSING
    perlin_cfg: PerlinPlaneTerrainCfg | None = None


@configclass
class PerlinTiltedRampTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a tilted ramp terrain.

    倾斜斜坡地形配置：可两侧对称或左右交替布置斜坡（switch_spacing > 0 时交替），
    spacing_curriculum 控制间距是否随难度递增。
    """

    function = hf_terrains.perlin_tilted_ramp_terrain
    tilt_angle: tuple[float, float] | float = MISSING  # 单位：度
    tilt_height: tuple[float, float] | float = MISSING
    tilt_width: tuple[float, float] | float = MISSING
    tilt_length: tuple[float, float] | float = MISSING
    switch_spacing: tuple[float, float] | float = MISSING
    spacing_curriculum: bool | None = None
    overlap_size: float | None = None
    perlin_cfg: PerlinPlaneTerrainCfg | None = None


@configclass
class PerlinSlopeTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a slope up and down terrain with a flat ground in the middle.

    上坡-平台-下坡地形配置，中央保留平台；up_down 控制坡方向（True=先上后下）。
    """

    function = hf_terrains.perlin_slope_terrain
    slope_angle: tuple[float, float] | float = MISSING  # 单位：度
    per_slope_length: tuple[float, float] | float = MISSING
    platform_length: float = 1.0
    slope_width: float | None = None
    up_down: bool | None = None  # True 或 None 时先上后下，否则先下后上
    perlin_cfg: PerlinPlaneTerrainCfg | None = None


@configclass
class PerlinCrossStoneTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a cross stone terrain.

    十字踏石地形配置：从中央平台向四个方向延伸放置石块，地面下凹形成深坑，
    xy_random_ratio 控制石块位置的随机扰动幅度。
    """

    function = hf_terrains.perlin_cross_stone_terrain
    stone_size: tuple[float, float] = MISSING
    stone_height: tuple[float, float] | float = MISSING
    stone_spacing: tuple[float, float] | float = MISSING
    ground_depth: float = -0.5
    platform_width: float = 1.5
    xy_random_ratio: float = 0.2
    perlin_cfg: PerlinPlaneTerrainCfg | None = None


@configclass
class PerlinSquareGapTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """方形沟槽地形配置：在中央平台外圈环绕一道方形凹槽，难度递增时沟槽宽度增加。"""
    function = hf_terrains.perlin_square_gap_terrain

    gap_distance_range: tuple[float, float] = (0.1, 0.5)
    gap_depth: tuple[float, float] = (0.2, 0.5)
    platform_width: float = 1.5
    border_width: float = 0.0

    perlin_cfg: PerlinPlaneTerrainCfg | None = None
