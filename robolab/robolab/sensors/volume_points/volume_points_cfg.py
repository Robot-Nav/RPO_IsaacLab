# 体积点云传感器配置
# 定义点云生成器、刚体过滤路径与可视化标记样式（绿点=未穿透，红点=穿透）

from dataclasses import MISSING
from typing import Dict

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.sensors import SensorBaseCfg
from isaaclab.utils import configclass

from .points_generator_cfg import PointsGeneratorCfg
from .volume_points import VolumePoints

VOLUME_POINTS_VISUALIZER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/volumePoints",
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.01,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
        "sphere_penetrated": sim_utils.SphereCfg(
            radius=0.01,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    },
)


@configclass
class VolumePointsCfg(SensorBaseCfg):
    """Configuration for the volume points sensor.

    体积点云传感器配置：绑定 VolumePoints 运行时类，
    通过 points_generator 指定点云生成方式，filter_prim_paths_expr 过滤参与采样的刚体。
    """

    class_type: type = VolumePoints

    filter_prim_paths_expr: list[str] = list()
    """The list of primitive paths (or expressions) to filter volume points' body with. Defaults to an empty list,
    in which case
    no filtering is applied.

    .. note::
        The expression in the list can contain the environment namespace regex ``{ENV_REGEX_NS}`` which
        will be replaced with the environment namespace.

        Example: ``{ENV_REGEX_NS}/Object`` will be replaced with ``/World/envs/env_.*/Object``.

    """

    points_generator: PointsGeneratorCfg = MISSING
    """ The points generator configuration. The generator function should be callable and accept only its cfg.
    """

    visualizer_cfg: VisualizationMarkersCfg = VOLUME_POINTS_VISUALIZER_CFG
