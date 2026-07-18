# 虚拟障碍物抽象基类与配置，定义从地形网格生成虚拟障碍物的统一接口
# 子类需实现 generate/disable_visualizer/visualize/get_points_penetration_offset 四个方法
# 用途：基于地形网格边缘等几何特征生成虚拟圆柱边界，用于训练时计算机器人穿深并施加惩罚
from __future__ import annotations

import torch
import trimesh
from abc import ABC, abstractmethod
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass


@configclass
class VirtualObstacleCfg:
    """Configuration for a virtual obstacle.

    虚拟障碍物配置基类：class_type 绑定具体实现类，visualizer 提供调试可视化配置。
    """

    class_type: type = MISSING
    """The class to use for the virtual obstacle."""

    visualizer: VisualizationMarkersCfg = MISSING
    """The visualizer configuration for the virtual obstacle."""


class VirtualObstacleBase(ABC):
    """虚拟障碍物抽象基类，约定从地形网格生成虚拟障碍物的生命周期方法。"""

    def __init__(self, cfg: VirtualObstacleCfg):
        self.cfg = cfg

    @abstractmethod
    def generate(self, mesh: trimesh.Trimesh, device: torch.device | str = "cpu") -> None:
        """Generate the virtual obstacle mesh based on the provided terrain mesh.
        NOTE: This interface might be updated in the future to support more complex generation logic.

        Args:
            mesh (trimesh.Trimesh): The terrain mesh to generate the virtual obstacle from.

        子类实现：从地形网格提取几何特征（如尖锐边缘）并构造虚拟障碍物内部表示。
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def disable_visualizer(self) -> None:
        """Disable the visualizer for the virtual obstacle if there is one."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    """
    Operations only after being generated.
    If called before generation, it should skip and print a warning.
    """

    @abstractmethod
    def visualize(self):
        """Visualize the virtual obstacle."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def get_points_penetration_offset(self, points: torch.Tensor) -> torch.Tensor:
        """Get the penetration offset for the given points.

        Args:
            points (torch.Tensor): Shape (N, 3) The points to check for penetration.

        Returns:
            torch.Tensor: Shape (N, 3) The penetration offsets for the points. pointing from the surface to the point.

        关键接口：给定机器人足端等采样点，返回从障碍物表面指向该点的偏移向量，
        用于训练时计算穿深惩罚。无障碍物时返回零张量。
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
