# 体积点云生成器配置
# PointsGeneratorCfg 为抽象基类，子类指定具体生成函数与参数

import torch
from dataclasses import MISSING
from typing import Callable

from isaaclab.utils import configclass

from .points_generator import grid3d_points_generator


@configclass
class PointsGeneratorCfg:
    """Specifying how the volume points are generated.

    点云生成器配置基类，func 字段指向具体生成函数，由传感器在初始化时调用。
    """

    func: Callable = MISSING  # type: ignore


@configclass
class Grid3dPointsGeneratorCfg(PointsGeneratorCfg):
    # 3D 均匀网格采样配置，沿 x/y/z 三轴各生成 x_num*y_num*z_num 个点

    func: Callable = grid3d_points_generator

    x_min: float = -1.0
    """Minimum x coordinate of the grid."""
    x_max: float = 1.0
    """Maximum x coordinate of the grid."""
    x_num: int = 10
    """Number of points along the x axis."""
    y_min: float = -1.0
    """Minimum y coordinate of the grid."""
    y_max: float = 1.0
    """Maximum y coordinate of the grid."""
    y_num: int = 10
    """Number of points along the y axis."""
    z_min: float = -1.0
    """Minimum z coordinate of the grid."""
    z_max: float = 1.0
    """Maximum z coordinate of the grid."""
    z_num: int = 10
    """Number of points along the z axis."""
