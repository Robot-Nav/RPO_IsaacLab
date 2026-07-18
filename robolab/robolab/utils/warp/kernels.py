# Warp GPU加速内核，包含点-圆柱体穿透计算与分组网格射线投射
# 用于机器人足端/身体与场景障碍物的碰撞检测，以及深度传感器射线投射
from typing import Any

import warp as wp


@wp.kernel(enable_backward=False)
def points_penetrate_cylinder_kernel(
    points: wp.array(dtype=wp.vec3),
    cylinder_start: wp.array(dtype=wp.vec3),
    cylinder_end: wp.array(dtype=wp.vec3),
    cylinder_thinkness: wp.array(dtype=wp.float32),
    cell_offsets: wp.array(dtype=wp.int32),
    cell_indices: wp.array(dtype=wp.int32),
    grid_res: wp.vec3i,
    bbox_min: wp.vec3,
    cell_size: wp.vec3,
    penetrate_offset: wp.array(dtype=wp.vec3),
):
    """Compute the penetration depth of points into cylinders in a grid. Return the maximum depth for each point if it
    penetrates any cylinder.
    Args:
        points: Array of points to check for penetration. shape (N, 3) where N is the number of points.
        cylinders: Array of cylinders defined by start and end points and radius. shape (M, 7) where M is the number of cylinders.
        cell_offsets: Offsets for each grid cell in the flattened grid. shape (grid_res^3 + 1,)
        cell_indices: Indices of cylinders in each grid cell. shape (N, 8)
        grid_res: Resolution of the grid.
        bbox_min: Minimum coordinates of the bounding box for the grid. shape (3,)
        cell_size: Size of each grid cell. shape (3,)
        penetrate_offset: Output array to store the penetration offset from the surface of the cylinder to each point.
    """
    # 每个线程处理一个点：先定位点所在网格单元，再遍历相邻3x3x3单元内的圆柱体计算穿透
    # 利用空间网格加速近邻查询，避免对所有圆柱体暴力遍历
    tid = wp.tid()
    p = points[tid]

    # 计算点所在的网格单元索引
    bbox_min_to_p = p - bbox_min
    ix = int(bbox_min_to_p[0] / cell_size[0])
    iy = int(bbox_min_to_p[1] / cell_size[1])
    iz = int(bbox_min_to_p[2] / cell_size[2])

    depth = float(0.0)
    penetrate_offset_ = wp.vec3(0.0, 0.0, 0.0)

    # 遍历点所在单元及其相邻共27个单元，覆盖圆柱体可能跨越边界的情况
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                x = ix + dx
                y = iy + dy
                z = iz + dz

                if x < 0 or x >= grid_res.x or y < 0 or y >= grid_res.y or z < 0 or z >= grid_res.z:
                    continue

                # 三维索引展平为一维，用于CSR格式的网格查表
                flat = x * grid_res.y * grid_res.z
                flat = flat + y * grid_res.z
                flat = flat + z

                start = cell_offsets[flat]
                end = cell_offsets[flat + 1]

                for i in range(start, end):
                    cid = cell_indices[i]
                    a = cylinder_start[cid]
                    b = cylinder_end[cid]
                    r = cylinder_thinkness[cid]

                    # 将点投影到圆柱轴线段上，t为投影参数
                    ab = b - a
                    ab_len = wp.length(ab)
                    ab_dir = ab / ab_len
                    ap = p - a
                    t = wp.dot(ap, ab_dir)

                    if t < 0.0 or t > ab_len:
                        # points outside the cylinder segment
                        # 投影落在线段外，不计入穿透
                        continue

                    # Project point onto the cylinder segment
                    proj = a + t * ab_dir
                    dist = wp.length(p - proj)

                    if dist < r:
                        # 穿透深度 = 半径 - 点到轴线距离，取所有圆柱体中的最大值
                        d = r - dist
                        if d > depth:
                            depth = d
                            # 穿透偏移方向为从点指向投影点，长度为穿透深度
                            offset_ = (proj - p) * (d / dist)
                            # direction from point to projected point
                            penetrate_offset_.x = offset_.x
                            penetrate_offset_.y = offset_.y
                            penetrate_offset_.z = offset_.z

    if depth > 0.0:
        penetrate_offset[tid] = penetrate_offset_


@wp.kernel(enable_backward=False)
def raycast_mesh_kernel_grouped_transformed(
    mesh_wp_ids: wp.array(dtype=wp.uint64),  # all meshes in the scene
    mesh_transforms: wp.array(dtype=wp.transform),  # transforms of the meshes
    mesh_inv_transforms: wp.array(dtype=wp.transform),  # inverse transforms of the meshes
    ray_collision_groups: wp.array(dtype=wp.int32),
    mesh_idxs_for_group: wp.array(dtype=wp.int32),
    meah_idxs_slice_for_group: wp.array(
        dtype=wp.int32
    ),  # Given the ray collision group (i), mesh_idxs_for_group[meah_idxs_slice_for_group[i]:meah_idxs_slice_for_group[i+1]] are the mesh ids within this group.
    ray_starts: wp.array(dtype=wp.vec3),
    ray_directions: wp.array(dtype=wp.vec3),
    ray_hits: wp.array(dtype=wp.vec3),
    ray_distance: wp.array(dtype=wp.float32),
    ray_normal: wp.array(dtype=wp.vec3),
    ray_face_id: wp.array(dtype=wp.int32),
    ray_mesh_id: wp.array(dtype=wp.int16),
    max_dist: float = 1e6,
    min_dist: float = 0.0,
    return_distance: int = False,
    return_normal: int = False,
    return_face_id: int = False,
    return_mesh_id: int = False,
):
    # 分组射线投射内核：每条射线只与所属碰撞组的网格求交，支持网格世界变换
    # 通过逆变换将射线转入网格局部空间求交，命中后再将结果转回世界空间
    tid = wp.tid()
    t = float(0.0)  # hit distance along ray
    u = float(0.0)  # hit face barycentric u
    v = float(0.0)  # hit face barycentric v
    sign = float(0.0)  # hit face sign
    n = wp.vec3()  # hit face normal
    f = int(0)  # hit face index

    ray_distance_buf = float(max_dist)
    ray_collision_group = int(ray_collision_groups[tid])
    start = ray_starts[tid]
    direction = ray_directions[tid]

    # 仅遍历该射线碰撞组对应的网格子集
    for idx in range(
        meah_idxs_slice_for_group[ray_collision_group], meah_idxs_slice_for_group[ray_collision_group + 1]
    ):
        mesh_idx = int(mesh_idxs_for_group[idx])
        mesh_wp_id = mesh_wp_ids[mesh_idx]

        # transform the ray start and direction to the mesh's local space
        # 射线起点用transform_point（含平移），方向用transform_vector（仅旋转）
        mesh_transform = mesh_transforms[mesh_idx]
        mesh_inv_transform = mesh_inv_transforms[mesh_idx]
        start_local = wp.transform_point(mesh_inv_transform, start)
        direction_local = wp.transform_vector(mesh_inv_transform, direction)

        # ray cast against the mesh and store the hit position
        query_returns = wp.mesh_query_ray(mesh_wp_id, start_local, direction_local, max_dist)
        # if the ray hit, store the hit data
        # 取最近命中，忽略小于min_dist的近距命中（如自交）
        if query_returns.result and query_returns.t < ray_distance_buf and query_returns.t > min_dist:
            ray_hits[tid] = start + direction * query_returns.t
            ray_distance_buf = query_returns.t
            if return_distance == 1:
                ray_distance[tid] = query_returns.t
            if return_normal == 1:
                # transform the normal back to world space
                # 命中法线需用正向变换转回世界空间
                n = wp.transform_vector(mesh_transform, query_returns.normal)
                ray_normal[tid] = n
            if return_face_id == 1:
                ray_face_id[tid] = query_returns.face
            if return_mesh_id == 1:
                ray_mesh_id[tid] = wp.int16(mesh_idx)
