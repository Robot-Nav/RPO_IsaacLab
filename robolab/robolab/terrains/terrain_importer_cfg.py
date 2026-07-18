# 地形导入器配置，扩展基类以支持虚拟障碍物字典与 hacked_generator 地形类型
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from isaaclab.terrains import TerrainImporterCfg as TerrainImporterCfgBase
from isaaclab.utils import configclass

from .terrain_importer import TerrainImporter

if TYPE_CHECKING:
    from .virtual_obstacle import VirtualObstacleCfg


@configclass
class TerrainImporterCfg(TerrainImporterCfgBase):
    class_type: type = TerrainImporter
    """The inherited class to use for the terrain importer."""

    virtual_obstacles: dict[str, VirtualObstacleCfg] = {}
    """The virtual obstacles to use for the terrain importer.

    虚拟障碍物配置字典，键为障碍物名称，值为对应配置；在 import_mesh 阶段基于网格生成实体障碍物。
    """

    terrain_type: Literal["generator", "plane", "usd", "hacked_generator"] = "generator"
    """The type of terrain to generate. Defaults to "generator".

    Available options are "plane", "usd", and "generator".

    ## NOTE
    The TerrainImporter of this package has some dedicated hack to fit the self-defined tasks.
    We add a "hacked_generator" option to hack and run our own terrain generator implementation.

    hacked_generator 类型借用 plane 入口跳转到自定义生成器实现，避免修改 IsaacLab 内部代码。
    """
