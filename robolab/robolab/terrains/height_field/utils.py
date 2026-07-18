# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# 高度场地形装饰器：在地形网格四周按概率添加边界墙，用于约束机器人在子地形内的活动范围
# wall_prob 顺序为 [left, right, front, back]，对应 x_min / x_max / y_min / y_max 四条边
from __future__ import annotations

import functools
import numpy as np
import trimesh
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.terrains.height_field import HfTerrainBaseCfg


def generate_wall(func: Callable) -> Callable:
    """Wrapper to add walls to the generated terrain mesh.

    装饰器：包装高度场生成函数，根据 cfg.wall_prob 在四条边按概率生成边界墙，
    墙体为长方体 mesh，高度与厚度由 cfg.wall_height / cfg.wall_thickness 决定。
    """

    @functools.wraps(func)
    def wrapper(difficulty: float, cfg: HfTerrainBaseCfg):
        meshes, origin = func(difficulty, cfg)
        if cfg is None or not hasattr(cfg, "wall_prob"):
            return meshes, origin

        mesh = meshes[0]
        wall_height = cfg.wall_height
        wall_thickness = cfg.wall_thickness
        result_meshes = [mesh]

        # 取网格 AABB 用于确定四面墙的位置与尺寸
        bounds = mesh.bounds
        min_bound, max_bound = bounds[0], bounds[1]

        # 左侧墙（x_min 方向）
        if np.random.uniform() < cfg.wall_prob[0]:
            left_wall = trimesh.creation.box(
                extents=[wall_thickness, max_bound[1] - min_bound[1], wall_height],
                transform=trimesh.transformations.translation_matrix(
                    [min_bound[0] - wall_thickness / 2, (min_bound[1] + max_bound[1]) / 2, wall_height / 2]
                ),
            )
            result_meshes.append(left_wall)

        # 右侧墙（x_max 方向）
        if np.random.uniform() < cfg.wall_prob[1]:
            right_wall = trimesh.creation.box(
                extents=[wall_thickness, max_bound[1] - min_bound[1], wall_height],
                transform=trimesh.transformations.translation_matrix(
                    [max_bound[0] + wall_thickness / 2, (min_bound[1] + max_bound[1]) / 2, wall_height / 2]
                ),
            )
            result_meshes.append(right_wall)

        # 前侧墙（y_min 方向）
        if np.random.uniform() < cfg.wall_prob[2]:
            front_wall = trimesh.creation.box(
                extents=[max_bound[0] - min_bound[0], wall_thickness, wall_height],
                transform=trimesh.transformations.translation_matrix(
                    [(min_bound[0] + max_bound[0]) / 2, min_bound[1] - wall_thickness / 2, wall_height / 2]
                ),
            )
            result_meshes.append(front_wall)

        # 后侧墙（y_max 方向）
        if np.random.uniform() < cfg.wall_prob[3]:
            back_wall = trimesh.creation.box(
                extents=[max_bound[0] - min_bound[0], wall_thickness, wall_height],
                transform=trimesh.transformations.translation_matrix(
                    [(min_bound[0] + max_bound[0]) / 2, max_bound[1] + wall_thickness / 2, wall_height / 2]
                ),
            )
            result_meshes.append(back_wall)

        return result_meshes, origin

    return wrapper
