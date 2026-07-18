# 三角网格地形配置类集合：动作匹配地形、浮箱地形、随机多盒地形
# 所有配置混入 WallTerrainCfgMixin 提供边界墙参数；区间参数按难度线性插值
from dataclasses import MISSING
from typing import List

from isaaclab.terrains.terrain_generator_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass

from ..height_field import PerlinPlaneTerrainCfg
from . import mesh_terrains


class WallTerrainCfgMixin:
    """边界墙配置混入类，提供四面墙的生成概率与几何参数。"""
    wall_prob: List[float] = [0.0, 0.0, 0.0, 0.0]  # [left, right, front, back] 四条边各自生成墙的概率
    wall_height: float = 5.0  # 墙高
    wall_thickness: float = 0.05  # 墙厚


@configclass
class MotionMatchedTerrainCfg(SubTerrainBaseCfg):
    """Configuration for motion-matched terrain generation.

    ## Terrain Mesh Requirements
    - All terrain meshes must have the a border at the bottom.
    - The terrain origin (0, 0, 0) is at the surface of the terrain center, which means that the point should
        be above the terrain at (0, 0, t) given any t > 0 and below the terrain at (0, 0, t) given any t < 0.
    - The USER should ensure that the non-flat part of the terrain is within the size of the terrain.

    动作匹配地形配置：从外部数据集按难度索引加载预生成网格，与运动捕捉数据一一对应。
    """

    function = mesh_terrains.motion_matched_terrain

    path: str = MISSING
    """Directory containing both terrains and the motions, so that these can be matched together.
    """

    metadata_yaml: str = MISSING
    """YAML file containing the motion matching configuration.
    This file should specify the motion matching parameters, such as the motion files to be used,
    the matching criteria, and any other relevant settings.

    You may use the `scripts/motion_matched_metadata_generator.py` to generate the metadata.yaml file if you arrange your
    dataset in the structure as described in `scripts/motion_matched_metadata_generator.py`.

    ## Typical yaml file structure

    ```yaml
    terrains:
        - terrain_id: "jumpbox1" # can be any string.
          terrain_file: "path/to/terrain.stl" # path to the terrain mesh file, relative to the datasetdir.
        - terrain_id: "jumpbox2"
          terrain_file: "path/to/another_terrain.stl"
    motion_files:
        - terrain_id: "jumpbox1" # should match the terrain_id above.
          motion_file: "path/to/motion1_poses.npz" # path to the motion file, relative to the datasetdir.
          weight: (optional) 1.0
        - terrain_id: "jumpbox2"
          motion_file: "path/to/motion2_retargetted.npz"
          weight: (optional) 1.0
    ```

    """


@configclass
class PerlinMeshFloatingBoxTerrainCfg(SubTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a floating box mesh terrain.

    浮箱地形配置：中央悬空立方体，下方可选 Perlin 噪声地面。
    区间参数反向插值（高难度对应小尺寸），no_perlin_at_obstacle 控制浮箱下方是否清零噪声。
    """

    function = mesh_terrains.floating_box_terrain
    floating_height: tuple[float, float] | float = MISSING
    """The height of the box above the ground. Could be a fixed value or a range (min, max)."""
    box_length: tuple[float, float] | float = MISSING
    """The length of the box along the y-axis. Could be a fixed value or a range (min, max)."""
    box_width: float | None = None
    """The width of the box along the x-axis. If None, it will be equal to the width of the terrain."""
    box_height: tuple[float, float] | float = MISSING
    """The height of the box along the z-axis."""
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    # values used for perlin noise generation
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    slope_threshold: float | None = None
    no_perlin_at_obstacle: bool = True
    """If True, no perlin noise will be generated exactly below the box."""


@configclass
class PerlinMeshRandomMultiBoxTerrainCfg(SubTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a sub terrain with multiple random boxes with perlin noise.

    随机多盒地形配置：均值±range 控制尺寸波动，generation_ratio 决定箱体密度，
    box_perlin_cfg 控制箱体顶部叠加的独立 Perlin 噪声。
    """

    function = mesh_terrains.random_multi_box_terrain
    box_height_mean: tuple[float, float] | float = MISSING
    box_height_range: float = MISSING
    box_length_mean: tuple[float, float] | float = MISSING
    box_length_range: float = MISSING
    box_width_mean: tuple[float, float] | float = MISSING
    box_width_range: float = MISSING
    platform_width: float = MISSING

    generation_ratio: float = MISSING

    perlin_cfg: PerlinPlaneTerrainCfg | None = None
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    slope_threshold: float | None = None
    no_perlin_at_obstacle: bool = False
    box_perlin_cfg: PerlinPlaneTerrainCfg | None = None
    """Used only when perlin_cfg is not None"""
