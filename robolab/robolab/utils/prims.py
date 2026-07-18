# USD Prim访问工具，封装Isaac Sim场景中铰接体ArticulationView的创建
# 用于运动参考轨迹回放时获取机器人关节张量视图
""" Code snippet from accessing prims in isaac sim (scene) """

import omni.physics.tensors.impl.api as physx
from pxr import UsdPhysics

import isaaclab.sim as sim_utils


def get_articulation_view(
    prim_path: str,
    physics_sim_view: physx.SimulationView,
) -> physx.ArticulationView:
    """create simulation view and the prim view (partially copied from `ray_caster.py`)"""
    # 按prim路径查找并创建ArticulationView，要求prim挂载了ArticulationRootAPI
    found_supported_prim_class = False
    prim = sim_utils.find_first_matching_prim(prim_path)
    if prim is None:
        raise RuntimeError(f"Failed to find a prim at path expression: {prim_path}")
    # create view based on the type of prim
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        # 将正则通配符.*转为PhysX支持的*格式
        articulation_view: physx.ArticulationView = physics_sim_view.create_articulation_view(
            prim_path.replace(".*", "*")
        )
        found_supported_prim_class = True
    if not found_supported_prim_class:
        raise RuntimeError(
            f"Failed to find a valid prim view class for the prim paths: {prim_path}, For robot motion reference, only"
            " accept articulated prims for now."
        )

    return articulation_view
