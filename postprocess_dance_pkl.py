"""后处理 rpo_dance_lab pkl：clip dof_pos 到 URDF 软限位，用 MuJoCo 重算 key_body_pos

Isaac Lab 不可用，无法运行 dataset_retarget.py 重新生成 pkl。
本脚本读取已有 pkl，对 dof_pos 做软限位裁剪，并用 MuJoCo 前向运动学
重新计算 6 个关键体的世界坐标（相对环境原点，env_origins=0）。

输入 pkl 字段：fps, root_pos, root_rot(wxyz), dof_pos(lab顺序), loop_mode, key_body_pos
输出 pkl 字段：同上，dof_pos 已 clip，key_body_pos 已重算
"""
import os
import pickle
import numpy as np
import mujoco

# lab_dof_names：pkl 中 dof_pos 的列顺序（与 rpo.yaml 一致）
LAB_DOF_NAMES = [
    "left_thigh_yaw_joint", "right_thigh_yaw_joint", "torso_joint",
    "left_thigh_roll_joint", "right_thigh_roll_joint",
    "left_arm_pitch_joint", "right_arm_pitch_joint",
    "left_thigh_pitch_joint", "right_thigh_pitch_joint",
    "left_arm_roll_joint", "right_arm_roll_joint",
    "left_knee_joint", "right_knee_joint",
    "left_arm_yaw_joint", "right_arm_yaw_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_elbow_pitch_joint", "right_elbow_pitch_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_elbow_yaw_joint", "right_elbow_yaw_joint",
]

# MuJoCo XML 中关节顺序（即 qpos[7:] 的列顺序，与 rpo.yaml gmr_dof_names 一致）
MUJOCO_DOF_NAMES = [
    "left_thigh_yaw_joint", "left_thigh_roll_joint", "left_thigh_pitch_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_thigh_yaw_joint", "right_thigh_roll_joint", "right_thigh_pitch_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "torso_joint",
    "left_arm_pitch_joint", "left_arm_roll_joint", "left_arm_yaw_joint",
    "left_elbow_pitch_joint", "left_elbow_yaw_joint",
    "right_arm_pitch_joint", "right_arm_roll_joint", "right_arm_yaw_joint",
    "right_elbow_pitch_joint", "right_elbow_yaw_joint",
]

# URDF 软限位（0.95 倍硬限位），与 gmr_to_lab.py 中 URDF_SOFT_LIMITS 一致
URDF_SOFT_LIMITS = {
    "left_thigh_yaw_joint": (-0.95, 0.19),
    "right_thigh_yaw_joint": (-0.19, 0.95),
    "torso_joint": (-2.983, 2.983),
    "left_thigh_roll_joint": (-0.19, 0.95),
    "right_thigh_roll_joint": (-0.95, 0.19),
    "left_arm_pitch_joint": (-2.983, 1.4915),
    "right_arm_pitch_joint": (-2.983, 1.4915),
    "left_thigh_pitch_joint": (-1.9893, 0.74613),
    "right_thigh_pitch_joint": (-1.9893, 0.74613),
    "left_arm_roll_joint": (-0.2375, 2.983),
    "right_arm_roll_joint": (-2.983, 0.2375),
    "left_knee_joint": (-0.19, 2.375),
    "right_knee_joint": (-0.19, 2.375),
    "left_arm_yaw_joint": (-1.4915, 1.4915),
    "right_arm_yaw_joint": (-1.4915, 1.4915),
    "left_ankle_pitch_joint": (-0.57, 0.57),
    "right_ankle_pitch_joint": (-0.57, 0.57),
    "left_elbow_pitch_joint": (-0.57, 1.4915),
    "right_elbow_pitch_joint": (-0.57, 1.4915),
    "left_ankle_roll_joint": (-0.475, 0.475),
    "right_ankle_roll_joint": (-0.475, 0.475),
    "left_elbow_yaw_joint": (-1.4915, 1.4915),
    "right_elbow_yaw_joint": (-1.4915, 1.4915),
}

# 关键体名称（与 rpo.yaml lab_key_body_names 一致）
KEY_BODY_NAMES = [
    "left_ankle_roll_link", "right_ankle_roll_link",
    "left_knee_link", "right_knee_link",
    "left_elbow_yaw_link", "right_elbow_yaw_link",
]

DANCE_DIR = "/home/fatu08/roboparty_train/robolab/data/motions/rpo_dance_lab"
MJCF_PATH = "/home/fatu08/roboparty_train/robolab/data/robots/roboparty/rpo/mjcf/rpo.xml"


def build_lab_to_mujoco_index():
    """lab dof 列索引 → MuJoCo qpos[7:] 列索引"""
    return [MUJOCO_DOF_NAMES.index(n) for n in LAB_DOF_NAMES]


def clip_dof_pos(dof_pos_lab):
    """对 lab 顺序的 dof_pos 按软限位裁剪，elbow_yaw 置零"""
    out = dof_pos_lab.copy()
    for i, name in enumerate(LAB_DOF_NAMES):
        if name in URDF_SOFT_LIMITS:
            lo, hi = URDF_SOFT_LIMITS[name]
            out[:, i] = np.clip(out[:, i], lo, hi)
        if name.endswith("_elbow_yaw_joint"):
            out[:, i] = 0.0
    return out


def compute_key_body_pos(model, data, root_pos, root_rot_wxyz, dof_pos_lab,
                         lab_to_mujoco_idx, key_body_ids):
    """用 MuJoCo 前向运动学逐帧计算关键体世界坐标

    Args:
        root_pos: (N, 3) 根位置
        root_rot_wxyz: (N, 4) 根旋转四元数 (w,x,y,z)
        dof_pos_lab: (N, 23) lab 顺序关节角（已 clip）
        lab_to_mujoco_idx: lab 列 → MuJoCo qpos[7:] 列的索引
        key_body_ids: 关键体在 MuJoCo body 中的 id 列表

    Returns:
        key_body_pos: (N, len(key_body_ids), 3) 关键体世界坐标
    """
    n = root_pos.shape[0]
    nkey = len(key_body_ids)
    key_body_pos = np.zeros((n, nkey, 3), dtype=np.float64)

    # 把 lab 顺序 dof_pos 转成 MuJoCo 顺序
    dof_pos_mujoco = dof_pos_lab[:, lab_to_mujoco_idx]  # (N, 23)

    for i in range(n):
        # qpos 布局：[0:3]=root_pos, [3:7]=root_quat(wxyz), [7:30]=dof
        data.qpos[0:3] = root_pos[i]
        data.qpos[3:7] = root_rot_wxyz[i]
        data.qpos[7:30] = dof_pos_mujoco[i]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for k, bid in enumerate(key_body_ids):
            key_body_pos[i, k] = data.xpos[bid]
    return key_body_pos


def process_pkl(path, model, data, lab_to_mujoco_idx, key_body_ids):
    """处理单个 pkl：clip + 重算 key_body_pos，原地保存"""
    with open(path, "rb") as f:
        d = pickle.load(f)

    root_pos = np.asarray(d["root_pos"], dtype=np.float64)
    root_rot = np.asarray(d["root_rot"], dtype=np.float64)  # wxyz
    dof_pos_lab = np.asarray(d["dof_pos"], dtype=np.float64)

    n = dof_pos_lab.shape[0]
    print(f"  {os.path.basename(path)}: {n} 帧")

    # 统计 clip 前违规
    violations_before = 0
    for j, name in enumerate(LAB_DOF_NAMES):
        lo, hi = URDF_SOFT_LIMITS[name]
        violations_before += int(np.sum((dof_pos_lab[:, j] < lo) | (dof_pos_lab[:, j] > hi)))

    # clip
    dof_pos_clipped = clip_dof_pos(dof_pos_lab)

    # 统计 clip 后违规（应为 0）
    violations_after = 0
    for j, name in enumerate(LAB_DOF_NAMES):
        lo, hi = URDF_SOFT_LIMITS[name]
        violations_after += int(np.sum((dof_pos_clipped[:, j] < lo - 1e-9) |
                                       (dof_pos_clipped[:, j] > hi + 1e-9)))

    # 重算 key_body_pos
    key_body_pos = compute_key_body_pos(
        model, data, root_pos, root_rot, dof_pos_clipped,
        lab_to_mujoco_idx, key_body_ids
    )

    # 写回
    d["dof_pos"] = dof_pos_clipped.astype(np.float32)
    d["key_body_pos"] = key_body_pos.astype(np.float32)

    with open(path, "wb") as f:
        pickle.dump(d, f)

    print(f"    clip 前 soft-limit 违规帧: {violations_before}")
    print(f"    clip 后 soft-limit 违规帧: {violations_after}")
    print(f"    key_body_pos 范围: "
          f"x[{key_body_pos[:,:,0].min():.3f},{key_body_pos[:,:,0].max():.3f}] "
          f"y[{key_body_pos[:,:,1].min():.3f},{key_body_pos[:,:,1].max():.3f}] "
          f"z[{key_body_pos[:,:,2].min():.3f},{key_body_pos[:,:,2].max():.3f}]")


def main():
    print(f"加载 MuJoCo 模型: {MJCF_PATH}")
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)
    print(f"  nq={model.nq}, nv={model.nv}, nbody={model.nbody}")

    # 关键体 body id
    key_body_ids = []
    for name in KEY_BODY_NAMES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"body '{name}' 未在 MuJoCo 模型中找到")
        key_body_ids.append(bid)
    print(f"  关键体 id: {dict(zip(KEY_BODY_NAMES, key_body_ids))}")

    lab_to_mujoco_idx = build_lab_to_mujoco_index()

    files = sorted([f for f in os.listdir(DANCE_DIR) if f.endswith(".pkl")])
    print(f"\n处理 {len(files)} 个 pkl 文件: {files}")

    for f in files:
        process_pkl(os.path.join(DANCE_DIR, f), model, data,
                    lab_to_mujoco_idx, key_body_ids)

    print("\n完成。")


if __name__ == "__main__":
    main()
