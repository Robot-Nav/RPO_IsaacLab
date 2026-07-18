# 边缘圆柱虚拟障碍物配置类集合：5 种边缘检测算法各有独立配置
# 共通参数：angle_threshold（尖锐边夹角阈值）、cylinder_radius（虚拟圆柱半径）、num_grid_cells（空间网格划分数）
# 算法分支：Plucker（Plücker 坐标合并共线）、Ransac（随机抽样一致性拟合）、Greedyconcat（贪心串联相邻边）、Ray（射线投射+Canny）、Feature（pyvista 特征边）
from __future__ import annotations

import math
from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.sensors import patterns
from isaaclab.utils import configclass

from .edge_cylinder import (
    FeatureEdgeCylinder,
    GreedyconcatEdgeCylinder,
    PluckerEdgeCylinder,
    RansacEdgeCylinder,
    RayEdgeCylinder,
)
from .virtual_obstacle_base import VirtualObstacleCfg


@configclass
class EdgeCylinderCfg(VirtualObstacleCfg):
    """The class to use for the edge cylinder detector.

    边缘圆柱检测器基类配置：基于网格面邻接角检测尖锐边并构造虚拟圆柱。
    """

    class_type: type = MISSING
    """The class to use for the edge detector."""
    angle_threshold: float = 70.0
    """The angle threshold to consider an edge as sharp."""

    cylinder_radius: float = 0.2
    """The radius of the edge cylinder, which is used to treat the edge cylinders as a virtual obstacle."""
    num_grid_cells: int = 64**3
    """The number of grid cells to use for spatial partitioning of the edge cylinders.
    Usually the power of 2, e.g., 64^3 = 262144.

    空间网格用于加速穿深查询：将圆柱按 AABB 分桶，查询时仅检查所在桶及邻桶。
    """

    visualizer: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/edgeMarkers",
        markers={
            "cylinder": sim_utils.CylinderCfg(
                radius=1,
                height=1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.9), opacity=0.2),
            )
        },
    )


@configclass
class PluckerEdgeCylinderCfg(EdgeCylinderCfg):
    """Configuration for the plucker edge cylinder generator.

    Plücker 坐标法：用方向向量与矩量构成 6D 表示，使共线但方向相反的边可归并；
    同组内按参数 t 排序合并重叠/相邻段，输出合并后的边段坐标。
    """

    class_type: type = PluckerEdgeCylinder
    """The class to use for the sharp edge detector."""


@configclass
class RansacEdgeCylinderCfg(EdgeCylinderCfg):
    """The class to use for the ransac edge cylinder generator.

    RANSAC 法：先用 DBSCAN 对端点聚类，再对每簇随机抽两点拟合直线，
    统计内点数最多者作为该簇的代表线段，循环直至剩余点不足。
    """

    class_type: type = RansacEdgeCylinder

    max_iter: int = 500
    """The maximum number of iterations."""

    point_distance_threshold: float = 0.04
    """The distance threshold to consider a point as an inlier."""

    min_points: int = 5
    """The minimum number of points required to fit."""

    cluster_eps: float = 0.08
    """The maximum distance between points in a cluster."""


@configclass
class GreedyconcatEdgeCylinderCfg(EdgeCylinderCfg):
    """The class to use for the greedy-concat edge cylinder generator.

    贪心串联法：从某顶点出发，按相邻角度阈值贪心延伸，将共顶点且方向接近的短边合并为长折线；
    再用最大垂距 < 0.05 的判据将折线简化为直线段。
    """

    class_type: type = GreedyconcatEdgeCylinder

    adjacent_angle_threshold: float = 30.0
    """The angle threshold to consider two edges as adjacent."""

    point_distance_threshold: float = 0.06
    """The distance threshold to consider a point as an inlier."""

    min_points: int = 5
    """The minimum number of points in one line."""


@configclass
class RayEdgeCylinderCfg(VirtualObstacleCfg):
    """The class to use for the ray-based edge cylinder generator.

    射线投射法：从多个相机位姿发射网格状射线，得到深度图与法向图；
    对两类图分别跑 Canny 边缘检测，按异或合并得候选边缘点，再经 DBSCAN + RANSAC 拟合为线段。
    """

    class_type: type = RayEdgeCylinder

    cylinder_radius: float = 0.2
    """The radius of the edge cylinder, which is used to treat the edge cylinders as a virtual obstacle."""
    num_grid_cells: int = 64**3
    """The number of grid cells to use for spatial partitioning of the edge cylinders.
    Usually the power of 2, e.g., 64^3 = 262144.
    """
    max_iter: int = 500
    """The maximum number of iterations."""

    point_distance_threshold: float = 0.005
    """The distance threshold to consider a point as an inlier."""

    min_points: int = 15
    """The minimum number of points required to fit."""

    cluster_eps: float = 0.08
    """The maximum distance between points in a cluster."""

    ray_pattern: patterns.GridPatternCfg = patterns.GridPatternCfg(
        resolution=0.01,
        size=[6, 6],
        direction=(0.0, 0.0, -1.0),
    )
    """The pattern to use for ray sampling."""

    ray_offset_pos: list[float] = [0.0, 0.0, 1.0]
    """The offset position of the rays."""

    ray_rotate_axes: list[list[float]] = [
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
        [1.0, -1.0, 0.0],
        [-1.0, -1.0, 0.0],
    ]

    ray_rotate_angle: list[float] = [math.pi * 0.25, math.pi * 0.25, math.pi * 0.25, math.pi * 0.25]
    """The axes and angles to rotate the rays.

    四组轴向+角度定义四个虚拟相机位姿，从对角方向覆盖地形。
    """

    max_ray_depth: float = 8.0
    """The maximum depth of the rays to sample."""

    depth_canny_thresholds: list[float] = [250, 300]
    """The thresholds for the Canny edge detector to detect edges in the depth image."""

    normal_canny_thresholds: list[float] = [80, 250]
    """The thresholds for the Canny edge detector to detect edges in the normal image."""

    cutoff_z_height: float = 0.1
    """The height threshold to filter out rays that are too close to the ground."""

    visualizer: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/edgeMarkers",
        markers={
            "cylinder": sim_utils.CylinderCfg(
                radius=1,
                height=1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.9), opacity=0.2),
            )
        },
    )

    points_visualizer: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/edgePoints",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 0.5)),
            ),
        },
    )


@configclass
class FeatureEdgeCylinderCfg(EdgeCylinderCfg):
    """The class to use for the feature-extracted edge cylinder generator.

    特征边法：借助 pyvista 的 extract_feature_edges 直接提取边界边、非流形边、特征边，
    feature_angle 控制特征边角度阈值，输出线段端点对。
    """

    class_type: type = FeatureEdgeCylinder

    cylinder_radius: float = 0.2
    """The radius of the edge cylinder, which is used to treat the edge cylinders as a virtual obstacle."""

    feature_angle: float = 15.0
    """The angle threshold to consider a feature as an edge feature."""
