# 三角网格地形生成函数集合：直接产出 trimesh 对象列表，用于复杂几何（浮箱、随机盒群、动作匹配网格）
# motion_matched_terrain 从外部数据集加载预生成网格，floating_box/random_multi_box 程序化生成并叠加 Perlin 噪声地面
from __future__ import annotations

import numpy as np
import os
import scipy.spatial.transform as tf
import torch
import trimesh
import yaml
from typing import TYPE_CHECKING

from isaaclab.terrains.height_field.utils import convert_height_field_to_mesh

from ..height_field.hf_terrains import generate_perlin_noise
from .utils import crop_terrain_mesh_aabb, generate_wall

if TYPE_CHECKING:
    from . import mesh_terrains_cfg


@generate_wall
def motion_matched_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MotionMatchedTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a motion-matched terrain based on the difficulty level and configuration.

    Args:
        difficulty (float): The difficulty level for the terrain.
        cfg (mesh_terrains_cfg.MotionMatchedTerrainCfg): Configuration for the motion-matched terrain.

    Returns:
        tuple: A tuple containing a list of trimesh objects and an array of poses.

    难度按数据集中地形数量线性映射到索引，difficulty=1 时取最末地形。
    加载后裁剪到 cfg.size 范围，并以边界顶点平均高度对齐到原点。
    """
    # Load the YAML file containing terrain and motion data
    with open(cfg.metadata_yaml) as file:
        data = yaml.safe_load(file)

    # Extract terrains and motions from the YAML data
    terrains = data["terrains"]  # 顺序敏感：索引对应难度递增

    # 难度映射为地形索引，clip 防止越界
    terrain_idx = int(np.clip(difficulty * len(terrains), 0, len(terrains) - 1))
    selected_terrain = terrains[terrain_idx]

    terrain_file = selected_terrain["terrain_file"]
    terrain_abspath = os.path.join(cfg.path, terrain_file)
    terrain_mesh = trimesh.load(terrain_abspath, force="mesh")

    # crop terrain mesh by cfg.size
    # This does not change the terrain origin.
    terrain_mesh = crop_terrain_mesh_aabb(
        terrain_mesh,
        x_max=cfg.size[0] / 2,
        x_min=-cfg.size[0] / 2,
        y_max=cfg.size[1] / 2,
        y_min=-cfg.size[1] / 2,
    )

    # Find border height offset w.r.t current center of this terrain mesh.
    # This is used to align the terrain mesh with the origin.
    # NOTE: Assuming the border is flat, we take the mean height of the vertices
    # 取边界附近顶点的平均高度作为对齐基准，使裁剪后的网格边界与地面齐平
    border_height = np.mean(
        terrain_mesh.vertices[
            np.logical_or(
                np.abs(terrain_mesh.vertices[:, 0]) > (cfg.size[0] / 2 - 0.05),
                np.abs(terrain_mesh.vertices[:, 1]) > (cfg.size[1] / 2 - 0.05),
            )
        ][:, 2]
    )
    if np.isnan(border_height):
        print(f"Warning: Terrain {terrain_file} does not have a valid border height. Using 0 as the border height.")
        border_height = 0.0

    # To follow the terrain_generator convention, we move the terrain mesh to (size[0]/2, size[1]/2, -border_height).
    # 平移网格使左下角对齐 (0,0)，并下移 border_height 使边界与地面齐平
    move_terrain_transform = np.eye(4)
    move_terrain_transform[:2, 3] = np.array(cfg.size) / 2
    move_terrain_transform[2, 3] = -border_height
    terrain_mesh.apply_transform(move_terrain_transform)
    origin = np.array([cfg.size[0] / 2, cfg.size[1] / 2, -border_height])
    return terrain_mesh, origin


@generate_wall
def floating_box_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.PerlinMeshFloatingBoxTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generates a floating box terrain.

    生成悬空立方体地形：中央放置浮箱，下方为可选 Perlin 噪声地面。
    难度递增时浮箱高度与长度递减（区间参数反向插值：高难度对应小尺寸，提升跳跃难度）。
    """

    # resolve the terrain configuration
    # height of the floating box above the ground
    # 区间参数反向插值：difficulty=1 时取下限，使浮箱更低、跳跃距离更短，难度更高
    if isinstance(cfg.floating_height, (tuple, list)):
        floating_height = cfg.floating_height[1] - difficulty * (cfg.floating_height[1] - cfg.floating_height[0])
    else:
        floating_height = cfg.floating_height

    # length of the floating box
    if isinstance(cfg.box_length, (tuple, list)):
        box_length = cfg.box_length[1] - difficulty * (cfg.box_length[1] - cfg.box_length[0])
    else:
        box_length = cfg.box_length

    # height of the floating box
    if isinstance(cfg.box_height, (tuple, list)):
        box_height = np.random.uniform(*cfg.box_height)
    else:
        box_height = cfg.box_height

    # width of the floating box
    if cfg.box_width is None:
        box_width = cfg.size[0]
    else:
        box_width = cfg.box_width

    # initialize the list of meshes
    meshes_list = list()

    # extract quantities
    total_height = floating_height + box_height
    # constants for terrain generation
    terrain_height = 0.0

    # generate the box mesh
    dim = (box_width, box_length, box_height)
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], floating_height + box_height / 2)
    box_mesh = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(box_mesh)

    # generate the ground

    if cfg.perlin_cfg is None:
        # 无噪声时直接生成平面地面
        dim = (cfg.size[0], cfg.size[1], terrain_height)
        pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)
        ground_mesh = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
        meshes_list.append(ground_mesh)
    else:
        # 生成 Perlin 噪声高度场并转换为 mesh 作为地面
        clean_ground_height_field = np.zeros(
            (int(cfg.size[0] / cfg.horizontal_scale) + 1, int(cfg.size[1] / cfg.horizontal_scale) + 1), dtype=np.int16
        )
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        perlin_noise = generate_perlin_noise(
            difficulty,
            perlin_cfg,  # type: ignore[arg-type]
        )
        # 噪声尺寸可能与目标尺寸差 1 像素，零值填充对齐到 ground_height_field 形状
        h, w = perlin_noise.shape
        ground_h, ground_w = clean_ground_height_field.shape
        pad_h_left = max(0, (ground_h - h) // 2)
        pad_h_right = max(0, ground_h - h - pad_h_left)
        pad_w_left = max(0, (ground_w - w) // 2)
        pad_w_right = max(0, ground_w - w - pad_w_left)
        pad_width = ((pad_h_left, pad_h_right), (pad_w_left, pad_w_right))
        perlin_noise = np.pad(perlin_noise, pad_width, mode="constant", constant_values=0)
        if cfg.no_perlin_at_obstacle is True:
            # 浮箱正下方不生成噪声，避免机器人着陆时 footing 不稳定
            box_width_px = int(box_width / cfg.horizontal_scale)
            box_length_px = int(box_length / cfg.horizontal_scale)
            box_width_start_px = int((cfg.size[0] - box_width) / 2 / cfg.horizontal_scale)
            box_length_start_px = int((cfg.size[1] - box_length) / 2 / cfg.horizontal_scale)
            perlin_noise[
                box_width_start_px : box_width_start_px + box_width_px,
                box_length_start_px : box_length_start_px + box_length_px,
            ] = 0
        ground_height_field = clean_ground_height_field + perlin_noise
        # convert to trimesh
        vertices, triangles = convert_height_field_to_mesh(
            ground_height_field, cfg.horizontal_scale, cfg.vertical_scale, cfg.slope_threshold
        )
        ground_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        meshes_list.append(ground_mesh)

    # specify the origin of the terrain
    origin = np.array([pos[0], pos[1], total_height])

    return meshes_list, origin


@generate_wall
def random_multi_box_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.PerlinMeshRandomMultiBoxTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generates a terrain containing multiple boxes with random size and orientation.

    在 Perlin 噪声地面上随机放置多个立方体，尺寸围绕均值 ± range 波动，姿态随机旋转。
    中央 platform_width 范围内禁止放置箱体，保证机器人生成区域无障碍。
    """

    box_height_range = cfg.box_height_range
    box_length_range = cfg.box_length_range
    box_width_range = cfg.box_width_range

    # 各维度均值：区间参数按难度线性插值，标量直接使用
    # 校验均值下限不小于 range，避免插值后出现负尺寸
    if isinstance(cfg.box_height_mean, (tuple, list)):
        if cfg.box_height_mean[0] < box_height_range:
            raise RuntimeError("The minimum box height mean is smaller than the box height half range.")
        box_height_mean = cfg.box_height_mean[0] + difficulty * (cfg.box_height_mean[1] - cfg.box_height_mean[0])
    else:
        box_height_mean = cfg.box_height_mean
        if box_height_mean < box_height_range:
            raise RuntimeError("The minimum box height mean is smaller than the box height half range.")

    if isinstance(cfg.box_length_mean, (tuple, list)):
        if cfg.box_length_mean[0] < box_length_range:
            raise RuntimeError("The minimum box length mean is smaller than the box length half range.")
        box_length_mean = cfg.box_length_mean[0] + difficulty * (cfg.box_length_mean[1] - cfg.box_length_mean[0])
    else:
        box_length_mean = cfg.box_length_mean
        if box_length_mean < box_length_range:
            raise RuntimeError("The minimum box length mean is smaller than the box length half range.")

    if isinstance(cfg.box_width_mean, (tuple, list)):
        if cfg.box_width_mean[0] < box_width_range:
            raise RuntimeError("The minimum box width mean is smaller than the box width half range.")
        box_width_mean = cfg.box_width_mean[0] + difficulty * (cfg.box_width_mean[1] - cfg.box_width_mean[0])
    else:
        box_width_mean = cfg.box_width_mean
        if box_width_mean < box_width_range:
            raise RuntimeError("The minimum box width mean is smaller than the box width half range.")

    generation_ratio = cfg.generation_ratio

    width = cfg.size[0]
    length = cfg.size[1]

    mesh_list = []

    # 按面积比例估算箱体数量：generation_ratio * 场地面积 / 单箱平均面积
    num_boxes = int(generation_ratio * (width * length) / (box_length_mean * box_width_mean))
    num_boxes = max(1, num_boxes)
    if cfg.perlin_cfg is None:
        dim = (cfg.size[0], cfg.size[1], 0.0)
        pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0)
        ground_mesh = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
        mesh_list.append(ground_mesh)
    else:
        clean_ground_height_field = np.zeros(
            (int(cfg.size[0] / cfg.horizontal_scale) + 1, int(cfg.size[1] / cfg.horizontal_scale) + 1), dtype=np.int16
        )
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        perlin_noise = generate_perlin_noise(
            difficulty,
            perlin_cfg,  # type: ignore[arg-type]
        )
        h, w = perlin_noise.shape
        ground_h, ground_w = clean_ground_height_field.shape
        pad_h_left = max(0, (ground_h - h) // 2)
        pad_h_right = max(0, ground_h - h - pad_h_left)
        pad_w_left = max(0, (ground_w - w) // 2)
        pad_w_right = max(0, ground_w - w - pad_w_left)
        pad_width = ((pad_h_left, pad_h_right), (pad_w_left, pad_w_right))
        perlin_noise = np.pad(perlin_noise, pad_width, mode="constant", constant_values=0)
        ground_height_field = clean_ground_height_field + perlin_noise
        # convert to trimesh
        vertices, triangles = convert_height_field_to_mesh(
            ground_height_field, cfg.horizontal_scale, cfg.vertical_scale, cfg.slope_threshold
        )
        ground_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        mesh_list.append(ground_mesh)

    if cfg.box_perlin_cfg is not None and cfg.no_perlin_at_obstacle is False:
        # 箱体顶部叠加独立 Perlin 噪声配置，未指定的 scale 参数回退到主配置
        box_perlin_cfg = cfg.box_perlin_cfg
        box_perlin_cfg.horizontal_scale = (
            cfg.horizontal_scale if box_perlin_cfg.horizontal_scale is None else box_perlin_cfg.horizontal_scale
        )
        box_perlin_cfg.vertical_scale = (
            cfg.vertical_scale if box_perlin_cfg.vertical_scale is None else box_perlin_cfg.vertical_scale
        )
        box_perlin_cfg.slope_threshold = (
            cfg.slope_threshold if box_perlin_cfg.slope_threshold is None else box_perlin_cfg.slope_threshold
        )

    platform_width = cfg.platform_width

    for i in range(num_boxes):
        # 尺寸围绕均值 ± range 均匀采样
        box_width = box_width_mean + np.random.uniform(-1, 1) * box_width_range
        box_length = box_length_mean + np.random.uniform(-1, 1) * box_length_range
        box_height = box_height_mean + np.random.uniform(-1, 1) * box_height_range
        dim = (box_width, box_length, box_height)
        x = np.random.uniform(box_width / 2, width - box_width / 2)
        y = np.random.uniform(box_length / 2, length - box_length / 2)
        # 跳过中央 platform 区域，保证机器人生成位置无障碍
        if (
            x > width / 2 - platform_width / 2 - box_width / 2 and x < width / 2 + platform_width / 2 + box_width / 2
        ) and (
            y > length / 2 - platform_width / 2 - box_length / 2
            and y < length / 2 + platform_width / 2 + box_length / 2
        ):
            continue
        pos = (x, y, box_height / 2)
        theta = np.random.uniform(0, 2 * np.pi)
        translation_matrix = trimesh.transformations.translation_matrix(pos)
        rotation_matrix = trimesh.transformations.rotation_matrix(theta, (0, 0, 1))
        transform = translation_matrix @ rotation_matrix
        box_mesh = trimesh.creation.box(extents=dim)
        # top_z=box_mesh.vertices[:, 2].max()
        # top_face_mask=np.all(box_mesh.vertices[box_mesh.faces][:,:,2] == top_z, axis=1)
        # box_mesh.update_faces(~top_face_mask)
        # box_mesh.remove_unreferenced_vertices()
        box_mesh.apply_transform(transform)
        mesh_list.append(box_mesh)
        if cfg.box_perlin_cfg is not None and cfg.no_perlin_at_obstacle is False:
            # 箱体顶面叠加 Perlin 噪声 mesh：先生成局部噪声，再平移到箱体顶部并应用同一旋转
            box_perlin_cfg.size = (box_width, box_length)
            perlin_noise = generate_perlin_noise(
                difficulty,
                box_perlin_cfg,  # type: ignore[arg-type]
            )
            vertices, triangles = convert_height_field_to_mesh(
                perlin_noise,
                box_perlin_cfg.horizontal_scale,
                box_perlin_cfg.vertical_scale,
                box_perlin_cfg.slope_threshold,
            )
            box_noise = trimesh.Trimesh(vertices=vertices, faces=triangles)
            # 局部坐标原点移到箱体左下角，对齐箱体局部坐标系
            center_offset = (-box_width / 2, -box_length / 2, 0)
            center_translation = trimesh.transformations.translation_matrix(center_offset)
            box_noise.apply_transform(center_translation)
            noise_pos = (x, y, box_height)
            translation_matrix = trimesh.transformations.translation_matrix(noise_pos)
            transform = translation_matrix @ rotation_matrix
            box_noise.apply_transform(transform)
            mesh_list.append(box_noise)

    origin = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0])

    return mesh_list, origin
