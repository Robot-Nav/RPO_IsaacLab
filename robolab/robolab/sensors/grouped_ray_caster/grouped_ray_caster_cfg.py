# 分组射线投射器配置
# 提供 min_distance 字段过滤过近命中，并提供 get_link_prim_targets 工具函数快速构造机器人连杆射线目标

from dataclasses import MISSING

from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg
from isaaclab.utils import configclass

from .grouped_ray_caster import GroupedRayCaster


@configclass
class GroupedRayCasterCfg(MultiMeshRayCasterCfg):
    """Configuration for the GroupedRayCaster sensor.

    分组射线投射器配置，绑定 GroupedRayCaster 作为运行时类。
    """

    class_type: type = GroupedRayCaster

    min_distance: float = 0.0
    """The minimum distance from the sensor to ray cast to. aka ignore the hits closer than this distance."""


def get_link_prim_targets(
    links: list[str],
    prefix: str = "/World/envs/env_.*/Robot/",
    suffix: str = "/visuals",
    is_shared=True,  # whether the target prim is assumed to be the same mesh across all environments.
    **kwargs: dict,
) -> list[MultiMeshRayCasterCfg.RaycastTargetCfg]:
    """Build the raycast target given the list of links. It will combine and return a list of
    MultiMeshRayCasterCfg.RaycastTargetCfg.

    按机器人连杆名批量构造射线目标，默认前缀匹配多环境下的 Robot 路径，后缀取 visuals 子树。
    is_shared=True 表示网格在所有环境间共享（如地形），False 表示每环境独立网格（如机器人本体）。
    """
    return [
        MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr=f"{prefix}{link}{suffix}", is_shared=is_shared, **kwargs)
        for link in links
    ]
