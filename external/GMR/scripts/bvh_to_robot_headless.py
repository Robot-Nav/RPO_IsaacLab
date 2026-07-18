#!/usr/bin/env python3
"""无头版 BVH 到机器人运动数据重定向，仅保存 pkl，不启动可视化。"""

import argparse
import os
import pickle
import pathlib

import numpy as np
from rich import print
from tqdm import tqdm

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.lafan1 import load_bvh_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bvh_file", required=True, type=str, help="输入 BVH 文件路径")
    parser.add_argument("--format", choices=["lafan1", "nokov"], default="lafan1")
    parser.add_argument("--robot", default="rpo", type=str)
    parser.add_argument("--motion_fps", default=30, type=int)
    parser.add_argument("--save_path", required=True, type=str, help="输出 pkl 路径")
    args = parser.parse_args()

    # 加载 BVH
    print(f"Loading BVH: {args.bvh_file}")
    lafan1_data_frames, actual_human_height = load_bvh_file(args.bvh_file, format=args.format)
    print(f"Frames: {len(lafan1_data_frames)}, human height: {actual_human_height:.3f} m")

    # 初始化重定向器
    retargeter = GMR(
        src_human=f"bvh_{args.format}",
        tgt_robot=args.robot,
        actual_human_height=actual_human_height,
    )

    # 逐帧重定向
    qpos_list = []
    for smplx_data in tqdm(lafan1_data_frames, desc="Retargeting"):
        qpos = retargeter.retarget(smplx_data)
        qpos_list.append(qpos)

    # 拆解数据：root_pos (xyz), root_rot (xyzw), dof_pos
    root_pos = np.array([qpos[:3] for qpos in qpos_list])
    root_rot = np.array([qpos[3:7][[1, 2, 3, 0]] for qpos in qpos_list])  # wxyz -> xyzw
    dof_pos = np.array([qpos[7:] for qpos in qpos_list])

    motion_data = {
        "fps": args.motion_fps,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": None,
        "link_body_list": None,
    }

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    with open(args.save_path, "wb") as f:
        pickle.dump(motion_data, f)

    print(f"Saved to {args.save_path}")
    print(f"  fps={args.motion_fps}, frames={len(qpos_list)}")
    print(f"  root_pos shape={root_pos.shape}, root_rot shape={root_rot.shape}, dof_pos shape={dof_pos.shape}")
