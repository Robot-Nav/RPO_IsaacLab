"""验证 dance lab pkl 数据完整性，对比 normal pkl，检查关节限位"""
import pickle
import numpy as np
import os

# RPO 关节限位（按 rpo.yaml 中 lab_dof_names 顺序）
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

# URDF 关节限位 (lower, upper) — 从 rpo.urdf 读取的正确限位
JOINT_LIMITS = {
    "left_thigh_yaw_joint": (-1.0, 0.2),
    "right_thigh_yaw_joint": (-0.2, 1.0),
    "torso_joint": (-3.14, 3.14),
    "left_thigh_roll_joint": (-0.2, 1.0),
    "right_thigh_roll_joint": (-1.0, 0.2),
    "left_arm_pitch_joint": (-3.14, 1.57),
    "right_arm_pitch_joint": (-3.14, 1.57),
    "left_thigh_pitch_joint": (-2.094, 0.7854),
    "right_thigh_pitch_joint": (-2.094, 0.7854),
    "left_arm_roll_joint": (-0.25, 3.14),
    "right_arm_roll_joint": (-3.14, 0.25),
    "left_knee_joint": (-0.2, 2.5),
    "right_knee_joint": (-0.2, 2.5),
    "left_arm_yaw_joint": (-1.57, 1.57),
    "right_arm_yaw_joint": (-1.57, 1.57),
    "left_ankle_pitch_joint": (-0.6, 0.6),
    "right_ankle_pitch_joint": (-0.6, 0.6),
    "left_elbow_pitch_joint": (-0.6, 1.57),
    "right_elbow_pitch_joint": (-0.6, 1.57),
    "left_ankle_roll_joint": (-0.5, 0.5),
    "right_ankle_roll_joint": (-0.5, 0.5),
    "left_elbow_yaw_joint": (-1.57, 1.57),
    "right_elbow_yaw_joint": (-1.57, 1.57),
}

DANCE_DIR = "/home/fatu08/roboparty_train/robolab/data/motions/rpo_dance_lab"
NORMAL_DIR = "/home/fatu08/roboparty_train/robolab/data/motions/rpo_lab"

KEY_BODY_NAMES = [
    "left_ankle_roll_link", "right_ankle_roll_link",
    "left_knee_link", "right_knee_link",
    "left_elbow_yaw_link", "right_elbow_yaw_link",
]


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def analyze_pkl(path, label):
    """分析单个 pkl 文件的数据特征"""
    print(f"\n{'='*70}")
    print(f"[{label}] {os.path.basename(path)}")
    print('='*70)

    data = load_pkl(path)

    # 字段完整性
    keys = list(data.keys())
    print(f"字段: {keys}")

    fps = data.get('fps', None)
    root_pos = data.get('root_pos', None)
    root_rot = data.get('root_rot', None)
    dof_pos = data.get('dof_pos', None)
    key_body_pos = data.get('key_body_pos', None)
    loop_mode = data.get('loop_mode', None)

    print(f"fps: {fps}")
    print(f"root_pos shape: {None if root_pos is None else root_pos.shape}")
    print(f"root_rot shape: {None if root_rot is None else root_rot.shape}")
    print(f"dof_pos shape:  {None if dof_pos is None else dof_pos.shape}")
    print(f"key_body_pos shape: {None if key_body_pos is None else key_body_pos.shape}")
    print(f"loop_mode: {loop_mode}")

    if root_pos is None or dof_pos is None:
        print("!! 数据不完整")
        return None

    num_frames = dof_pos.shape[0]
    duration = num_frames / fps if fps else 0
    print(f"帧数: {num_frames}, 时长: {duration:.2f}s")

    # root_pos 范围
    print(f"\n-- root_pos 范围 --")
    print(f"  x: [{root_pos[:,0].min():.3f}, {root_pos[:,0].max():.3f}]  span={np.ptp(root_pos[:,0]):.3f}")
    print(f"  y: [{root_pos[:,1].min():.3f}, {root_pos[:,1].max():.3f}]  span={np.ptp(root_pos[:,1]):.3f}")
    print(f"  z: [{root_pos[:,2].min():.3f}, {root_pos[:,2].max():.3f}]  span={np.ptp(root_pos[:,2]):.3f}")

    # root_rot 四元数范数（应为1）
    rot_norms = np.linalg.norm(root_rot, axis=1)
    print(f"\n-- root_rot 四元数范数 --")
    print(f"  min={rot_norms.min():.6f}, max={rot_norms.max():.6f}, mean={rot_norms.mean():.6f}")

    # dof_pos 每个关节范围
    print(f"\n-- dof_pos 每关节范围 (对比限位) --")
    print(f"  {'joint_name':<28s} {'min':>8s} {'max':>8s} {'lower':>8s} {'upper':>8s} {'out_of_limit':>12s}")
    out_of_limit_count = 0
    for i, name in enumerate(LAB_DOF_NAMES):
        col = dof_pos[:, i]
        lo, hi = JOINT_LIMITS[name]
        # soft limit factor 0.90
        slo = lo * 0.90 if lo < 0 else lo * 1.10
        shi = hi * 0.90 if hi > 0 else hi * 1.10
        # 实际限位违规（用原始限位判断）
        violations = int(np.sum((col < lo - 1e-3) | (col > hi + 1e-3)))
        out_of_limit_count += violations
        flag = "VIOLATION" if violations > 0 else "ok"
        print(f"  {name:<28s} {col.min():>8.3f} {col.max():>8.3f} {lo:>8.3f} {hi:>8.3f} {flag:>12s} ({violations}帧)")

    print(f"\n关节限位违规总帧数: {out_of_limit_count}")

    # key_body_pos 检查
    if key_body_pos is not None:
        print(f"\n-- key_body_pos 范围 (相对root) --")
        for i, name in enumerate(KEY_BODY_NAMES):
            col = key_body_pos[:, i, :]
            print(f"  {name:<28s} x:[{col[:,0].min():.3f},{col[:,0].max():.3f}] "
                  f"y:[{col[:,1].min():.3f},{col[:,1].max():.3f}] "
                  f"z:[{col[:,2].min():.3f},{col[:,2].max():.3f}]")

        # 检查 elbow_yaw_link 是否有信息量（位置变化幅度）
        l_elbow = key_body_pos[:, 4, :]
        r_elbow = key_body_pos[:, 5, :]
        l_elbow_span = np.linalg.norm(l_elbow.max(axis=0) - l_elbow.min(axis=0))
        r_elbow_span = np.linalg.norm(r_elbow.max(axis=0) - r_elbow.min(axis=0))
        print(f"\n  elbow_yaw_link 位移幅度:")
        print(f"    left:  span={l_elbow_span:.4f}  (std={np.linalg.norm(l_elbow.std(axis=0)):.4f})")
        print(f"    right: span={r_elbow_span:.4f}  (std={np.linalg.norm(r_elbow.std(axis=0)):.4f})")

    return {
        'fps': fps,
        'num_frames': num_frames,
        'root_pos_span': (np.ptp(root_pos[:,0]), np.ptp(root_pos[:,1]), np.ptp(root_pos[:,2])),
        'out_of_limit': out_of_limit_count,
    }


def compare_summary(dance_stats, normal_stats):
    """对比 dance 和 normal 的总体统计"""
    print(f"\n\n{'#'*70}")
    print(f"# 总体对比")
    print(f"{'#'*70}")

    print(f"\nDance 动作 ({len(dance_stats)} 个):")
    for s in dance_stats:
        print(f"  {s['num_frames']:>5d}帧 {s['root_pos_span'][0]:>6.2f}m "
              f"违规帧={s['out_of_limit']}")

    print(f"\nNormal 动作 ({len(normal_stats)} 个):")
    for s in normal_stats:
        print(f"  {s['num_frames']:>5d}帧 {s['root_pos_span'][0]:>6.2f}m "
              f"违规帧={s['out_of_limit']}")


def main():
    # 检查 dance 文件
    dance_files = sorted([f for f in os.listdir(DANCE_DIR) if f.endswith('.pkl')])
    normal_files = sorted([f for f in os.listdir(NORMAL_DIR) if f.endswith('.pkl')])

    print(f"Dance pkl 数量: {len(dance_files)}")
    print(f"Normal pkl 数量: {len(normal_files)}")

    # 分析所有 dance 文件
    dance_stats = []
    for f in dance_files:
        stat = analyze_pkl(os.path.join(DANCE_DIR, f), "DANCE")
        if stat:
            stat['name'] = f
            dance_stats.append(stat)

    # 分析 normal 中的几个代表文件
    normal_sample = ["127_06.pkl", "move_l.pkl", "move_r.pkl", "C3_-_run_stageii.pkl"]
    normal_stats = []
    for f in normal_sample:
        path = os.path.join(NORMAL_DIR, f)
        if os.path.exists(path):
            stat = analyze_pkl(path, "NORMAL")
            if stat:
                stat['name'] = f
                normal_stats.append(stat)

    compare_summary(dance_stats, normal_stats)


if __name__ == "__main__":
    main()
