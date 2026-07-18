# 终止条件函数：子地形越界检测/根部高度跌落检测
# 越界检测防止机器人离开出生子地形后撞到边界墙；根部高度检测防止跌落深坑或台阶下台
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def sub_terrain_out_of_bounds(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), distance_buffer: float = 3.0
) -> torch.Tensor:
    """Terminate when the actor move too close to the edge of the sub terrain.

    If the actor moves too close to the edge of the sub terrain, the termination is activated. The distance
    to the edge of the sub terrain is calculated based on the size of the sub terrain and the distance buffer.
    """
    if env.scene.cfg.terrain.terrain_type == "plane":
        return False  # we have infinite terrain because it is a plane
    elif env.scene.cfg.terrain.terrain_type == "generator":
        # obtain the size of the sub-terrains
        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size
        # extract the used quantities (to enable type-hinting)
        asset: RigidObject = env.scene[asset_cfg.name]

        # check if the agent is out of bounds
        # 以出生点为中心，距子地形边界小于distance_buffer时触发终止，留出安全余量
        x_out_of_bounds = (
            torch.abs(asset.data.root_pos_w[:, 0] - env.scene.terrain.env_origins[:, 0])
            > 0.5 * grid_width - distance_buffer
        )
        y_out_of_bounds = (
            torch.abs(asset.data.root_pos_w[:, 1] - env.scene.terrain.env_origins[:, 1])
            > 0.5 * grid_length - distance_buffer
        )
        return torch.logical_or(x_out_of_bounds, y_out_of_bounds)
    else:
        raise ValueError("Received unsupported terrain type, must be either 'plane' or 'generator'.")


def root_height_below_env_origin_minimum(
    env: ManagerBasedRLEnv, minimum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's root height is below the minimum height."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # 用env_origins的z（出生点高度，clamp到0以下避免上凸地形误判）作为基准，避免下行台阶的合理下降被误终止
    terrain_base_height = torch.clamp(env.scene.env_origins[:, 2], max=0.0)
    return asset.data.root_pos_w[:, 2] - terrain_base_height < minimum_height
