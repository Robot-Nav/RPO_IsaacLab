# 地形导入器，扩展 IsaacLab TerrainImporter 以集成虚拟障碍物生成与自定义生成器路径
# 关键扩展：1) import_mesh 时基于网格生成虚拟边界障碍物 2) "hacked_generator" 类型复用 plane 接口走自定义生成器
# 虚拟障碍物生成在网格导入前完成，因为部分障碍物算法会修改网格本身
from __future__ import annotations

import numpy as np
import torch
import trimesh
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGenerator
from isaaclab.terrains import TerrainImporter as TerrainImporterBase
from isaaclab.utils.timer import Timer

if TYPE_CHECKING:
    from .terrain_importer_cfg import TerrainImporterCfg
    from .virtual_obstacle import VirtualObstacleBase


class TerrainImporter(TerrainImporterBase):
    """扩展 IsaacLab 地形导入器，集成虚拟障碍物生成与 hacked_generator 控制流。"""

    def __init__(self, cfg: TerrainImporterCfg):
        # 先实例化所有虚拟障碍物配置，生成时机在 import_mesh 中触发
        self._virtual_obstacles = {}
        for name, virtual_obstacle_cfg in cfg.virtual_obstacles.items():
            if virtual_obstacle_cfg is None:
                continue
            virtual_obstacle = virtual_obstacle_cfg.class_type(virtual_obstacle_cfg)
            self._virtual_obstacles[name] = virtual_obstacle

        if cfg.terrain_type == "hacked_generator":
            # 借用 plane 接口进入 import_ground_plane 分支，避免改动 IsaacLab 内部代码
            self._hacked_terrain_type = "hacked_generator"
            cfg.terrain_type = "plane"
            super().__init__(cfg)
            return

        if cfg.terrain_type == "generator":
            # 自管 generator 路径：保留 terrain_generator 引用，避免基类导入后丢弃
            cfg.validate()
            self.cfg = cfg
            self.device = sim_utils.SimulationContext.instance().device  # type: ignore
            self.terrain_prim_paths = list()
            self.terrain_origins = None
            self.env_origins = None
            self._terrain_flat_patches = dict()
            if self.cfg.terrain_generator is None:
                raise ValueError("Input terrain type is 'generator' but no value provided for 'terrain_generator'.")
            self.terrain_generator = self.cfg.terrain_generator.class_type(
                cfg=self.cfg.terrain_generator, device=self.device
            )
            self.import_mesh("terrain", self.terrain_generator.terrain_mesh)
            if self.cfg.use_terrain_origins:
                self.configure_env_origins(self.terrain_generator.terrain_origins)
            else:
                self.configure_env_origins()
            self._terrain_flat_patches = self.terrain_generator.flat_patches
            self.set_debug_vis(self.cfg.debug_vis)
            return

        super().__init__(cfg)

    @property
    def virtual_obstacles(self) -> dict[str, VirtualObstacleBase]:
        """Get the virtual obstacles representing the edges.
        TODO: Make the returned value more general.

        返回内部虚拟障碍物字典的浅拷贝，键为障碍物名称，值为障碍物对象引用。
        """
        # 仍指向原 VirtualObstacleBase 对象，仅字典容器为拷贝
        return self._virtual_obstacles.copy()

    @property
    def subterrain_specific_cfgs(self) -> list[SubTerrainBaseCfg] | None:
        """Get the specific configurations for all subterrains.

        透传 FiledTerrainGenerator.subterrain_specific_cfgs；若未使用扩展生成器则返回 None。
        """
        # 占位实现，实际由绑定的 terrain_generator 提供具体配置
        return (
            self.terrain_generator.subterrain_specific_cfgs
            if hasattr(self, "terrain_generator") and hasattr(self.terrain_generator, "subterrain_specific_cfgs")
            else None
        )

    """
    Operations - Import.
    """

    def import_mesh(self, name: str, mesh: trimesh.Trimesh):
        """Import a mesh into the simulation.
        NOTE: By designing this interface, IsaacLab's terrain importer calls import_mesh only once when startup.

        Args:
            name: The name of the mesh.
            mesh: The trimesh object to import.

        导入流程：先清洗网格（合并顶点、去重面、删除孤立顶点），再触发虚拟障碍物生成，
        最后交给基类完成 USD 落地。虚拟障碍物算法可能修改 mesh，因此必须先于基类 import 执行。
        """
        mesh.merge_vertices()
        mesh.update_faces(mesh.unique_faces())  # 移除重复面
        mesh.remove_unreferenced_vertices()
        # 基于导入的网格生成虚拟障碍物；置于基类 import 之前，因部分算法会修改 mesh
        for name, virtual_obstacle in self._virtual_obstacles.items():
            with Timer(f"Generate virtual obstacle {name}"):
                virtual_obstacle.generate(mesh, device=self.device)

        super().import_mesh(name, mesh)

    def import_ground_plane(self, name: str, size: tuple[float, float] = (2.0e6, 2.0e6)):
        """
        ## NOTE
        This is a hack to fit the self-defined tasks and guide the control flow to a self-defined implementation.
        In this case, we won't need to change the code of IsaacLab.

        hacked_generator 模式下复用基类 plane 入口，但实际走自定义生成器路径：
        生成地形网格 -> 导入网格 -> 配置 env_origins -> 引用 flat_patches。
        其余情况回退到基类默认 plane 实现。
        """
        if getattr(self, "_hacked_terrain_type", None) == "hacked_generator":
            # 校验配置项
            if self.cfg.terrain_generator is None:
                raise ValueError("Input terrain type is 'generator' but no value provided for 'terrain_generator'.")
            # 走自定义生成器路径
            self.terrain_generator = getattr(
                self.cfg.terrain_generator,
                "class_type",
                TerrainGenerator,
            )(cfg=self.cfg.terrain_generator, device=self.device)
            self.import_mesh("terrain", self.terrain_generator.terrain_mesh)
            # 基于生成器产出的 origins 配置每个环境的生成位置
            self.configure_env_origins(self.terrain_generator.terrain_origins)
            # 引用 flat_patches 用于足端采样等场景
            self._terrain_flat_patches = self.terrain_generator.flat_patches
        else:
            # 回退基类默认实现
            super().import_ground_plane(name, size)

    def set_debug_vis(self, debug_vis: bool) -> bool:
        """Set the debug visualization flag.

        Args:
            vis: True to enable debug visualization, False to disable.

        同步控制基类与所有虚拟障碍物的可视化开关。
        """
        results = super().set_debug_vis(debug_vis)

        for name, virtual_obstacle in self._virtual_obstacles.items():
            if debug_vis:
                virtual_obstacle.visualize()
            else:
                virtual_obstacle.disable_visualizer()

        return results

    def configure_env_origins(self, origins: np.ndarray | torch.Tensor | None = None):
        """Configure the environment origins.

        Args:
            origins: The origins of the environments. Shape is (num_envs, 3).

        hacked_generator 模式下若未显式传入 origins，则跳过基类调用，
        留待 import_ground_plane 中按生成器 origins 二次调用时再配置。
        """
        if origins is None and getattr(self, "_hacked_terrain_type", None) == "hacked_generator":
            # None 覆写时跳过，等待后续显式传入
            pass
        else:
            return super().configure_env_origins(origins)
