#!/usr/bin/env python3
"""无 Isaac Sim 依赖的 CSV→NPZ 转换，使用 MuJoCo 做 FK 与速度差分。

输入 CSV 列顺序（与原始 csv_to_npz.py 一致）：
    base_pos(3) + base_rot_xyzw(4) + dof_pos(23)
输出 npz 字段：
    fps, joint_pos, joint_vel, body_pos_w, body_quat_w(wxyz),
    body_lin_vel_w, body_ang_vel_w
"""

import argparse
import os

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R

# RPO link 顺序，与 Isaac Lab RPO_LINKS 一致（不含 world）
RPO_LINKS = [
    "base_link",
    "left_thigh_yaw_link",
    "left_thigh_roll_link",
    "left_thigh_pitch_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_thigh_yaw_link",
    "right_thigh_roll_link",
    "right_thigh_pitch_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_arm_pitch_link",
    "left_arm_roll_link",
    "left_arm_yaw_link",
    "left_elbow_pitch_link",
    "left_elbow_yaw_link",
    "right_arm_pitch_link",
    "right_arm_roll_link",
    "right_arm_yaw_link",
    "right_elbow_pitch_link",
    "right_elbow_yaw_link",
]


def axis_angle_from_quat(q, scalar_first=True):
    """四元数转轴角，默认 scalar_first(wxyz)。"""
    r = R.from_quat(q, scalar_first=scalar_first)
    return r.as_rotvec()


def quat_slerp_batch(q0, q1, t):
    """批量球面插值，输入输出均为 wxyz。"""
    out = np.zeros_like(q0)
    for i in range(q0.shape[0]):
        dot = np.dot(q0[i], q1[i])
        if dot < 0:
            q1_i = -q1[i]
            dot = -dot
        else:
            q1_i = q1[i]
        dot = np.clip(dot, -1.0, 1.0)
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        if sin_theta_0 < 1e-6:
            out[i] = q0[i]
        else:
            theta = theta_0 * t[i]
            sin_theta = np.sin(theta)
            s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
            s1 = sin_theta / sin_theta_0
            out[i] = s0 * q0[i] + s1 * q1_i
            out[i] /= np.linalg.norm(out[i])
    return out


def compute_frame_blend(times, duration, input_frames):
    phase = times / duration
    idx0 = np.floor(phase * (input_frames - 1)).astype(np.int64)
    idx1 = np.minimum(idx0 + 1, input_frames - 1)
    blend = phase * (input_frames - 1) - idx0
    return idx0, idx1, blend


def so3_derivative(rots, dt):
    """rots: (T,4) wxyz；返回 (T,3) 角速度。"""
    q_prev = rots[:-2]
    q_next = rots[2:]
    q_rel = np.zeros_like(q_prev)
    for i in range(q_prev.shape[0]):
        # q_next * q_prev^{-1}，wxyz 下四元数乘
        w1, x1, y1, z1 = q_next[i]
        w2, x2, y2, z2 = q_prev[i]
        # conj of q_prev
        w2, x2, y2, z2 = w2, -x2, -y2, -z2
        q_rel[i] = [
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ]
    omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
    omega = np.concatenate([omega[:1], omega, omega[-1:]], axis=0)
    return omega


def load_and_interpolate(csv_path, input_fps, output_fps):
    motion = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
    input_frames = motion.shape[0]
    duration = (input_frames - 1) / input_fps
    input_dt = 1.0 / input_fps
    output_dt = 1.0 / output_fps
    times = np.arange(0, duration, output_dt, dtype=np.float32)
    output_frames = times.shape[0]

    base_pos_in = motion[:, :3]
    base_rot_in = motion[:, 3:7][:, [3, 0, 1, 2]]  # xyzw -> wxyz
    dof_pos_in = motion[:, 7:]

    idx0, idx1, blend = compute_frame_blend(times, duration, input_frames)

    base_pos = base_pos_in[idx0] * (1 - blend)[:, None] + base_pos_in[idx1] * blend[:, None]
    base_rot = quat_slerp_batch(base_rot_in[idx0], base_rot_in[idx1], blend)
    dof_pos = dof_pos_in[idx0] * (1 - blend)[:, None] + dof_pos_in[idx1] * blend[:, None]

    base_lin_vel = np.gradient(base_pos, output_dt, axis=0)
    dof_vel = np.gradient(dof_pos, output_dt, axis=0)
    base_ang_vel = so3_derivative(base_rot, output_dt)

    return base_pos, base_rot, base_lin_vel, base_ang_vel, dof_pos, dof_vel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--input_file", required=True, type=str)
    parser.add_argument("--input_fps", type=int, default=30)
    parser.add_argument("--output_fps", type=int, default=50)
    parser.add_argument("--output_name", type=str, default=None)
    parser.add_argument("--mjcf", type=str,
                        default="/home/fatu08/roboparty_train/robolab/data/robots/roboparty/rpo/mjcf/rpo.xml")
    args = parser.parse_args()

    if args.output_name is None:
        args.output_name = args.input_file.replace(".csv", ".npz")

    # 加载 MuJoCo 模型
    model = mj.MjModel.from_xml_path(args.mjcf)
    data = mj.MjData(model)
    body_ids = {name: mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name) for name in RPO_LINKS}
    n_links = len(RPO_LINKS)
    n_joints = model.nq - 7  # 减去 freejoint

    # 重采样
    base_pos, base_rot, base_lin_vel, base_ang_vel, dof_pos, dof_vel = load_and_interpolate(
        args.input_file, args.input_fps, args.output_fps
    )
    output_frames = base_pos.shape[0]
    print(f"Interpolated {output_frames} frames @ {args.output_fps} Hz")

    # FK 循环
    joint_pos_log = np.zeros((output_frames, n_joints), dtype=np.float32)
    joint_vel_log = np.zeros((output_frames, n_joints), dtype=np.float32)
    body_pos_w = np.zeros((output_frames, n_links, 3), dtype=np.float32)
    body_quat_w = np.zeros((output_frames, n_links, 4), dtype=np.float32)
    body_lin_vel_w = np.zeros((output_frames, n_links, 3), dtype=np.float32)
    body_ang_vel_w = np.zeros((output_frames, n_links, 3), dtype=np.float32)

    for t in range(output_frames):
        # 设置 qpos: freejoint [x,y,z, qw,qx,qy,qz] + 关节角
        data.qpos[:3] = base_pos[t]
        data.qpos[3:7] = base_rot[t]  # wxyz
        data.qpos[7:] = dof_pos[t]
        data.qvel[:3] = base_lin_vel[t]
        data.qvel[3:6] = base_ang_vel[t]
        data.qvel[6:] = dof_vel[t]

        mj.mj_forward(model, data)

        joint_pos_log[t] = data.qpos[7:].copy()
        joint_vel_log[t] = data.qvel[6:].copy()

        for i, name in enumerate(RPO_LINKS):
            bid = body_ids[name]
            body_pos_w[t, i] = data.xpos[bid].copy()
            # MuJoCo xquat 是 wxyz
            body_quat_w[t, i] = data.xquat[bid].copy()
            # cvel 在质心局部系，转换到世界系：v_w = R * v_local, omega_w = R * omega_local
            cvel = data.cvel[bid]
            rot_mat = data.xmat[bid].reshape(3, 3)
            body_lin_vel_w[t, i] = rot_mat @ cvel[3:]
            body_ang_vel_w[t, i] = rot_mat @ cvel[:3]

    log = {
        "fps": [args.output_fps],
        "joint_pos": joint_pos_log,
        "joint_vel": joint_vel_log,
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
    }

    os.makedirs(os.path.dirname(args.output_name) or ".", exist_ok=True)
    np.savez(args.output_name, **log)
    print(f"[INFO] Saved to {args.output_name}")
    print(f"  fps={args.output_fps}, frames={output_frames}")
    for k, v in log.items():
        if k != "fps":
            print(f"  {k}: {v.shape}")


if __name__ == "__main__":
    main()
