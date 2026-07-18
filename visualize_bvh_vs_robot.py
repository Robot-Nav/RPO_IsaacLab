"""BVH人类动作 与 retarget后RPO机器人动作 对比可视化

生成左右分屏视频：左侧BVH人类骨架，右侧RPO机器人骨架（基于pkl的root+key_body_pos）

用法:
    python visualize_bvh_vs_robot.py \
        --bvh external/dance_data/dance1_subject2.bvh \
        --pkl robolab/data/motions/rpo_dance_lab/dance1_subject2.pkl \
        --output dance1_subject2_compare.mp4 \
        [--fps 30] [--skip 0] [--max_frames 0]
"""
import argparse
import io
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot

import imageio.v2 as imageio


# ---------------- BVH 解析 ----------------

class BVHJoint:
    def __init__(self, name, offset, channels=None):
        self.name = name
        self.offset = np.array(offset, dtype=np.float64)
        self.channels = channels or []
        self.parent = None
        self.children = []
        self.end_site = None


def parse_bvh(path):
    """解析BVH，返回 (root_joint, frames, frame_time, channel_order)"""
    with open(path, "r") as f:
        text = f.read()

    # 简单栈式解析 HIERARCHY 段
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    root = None
    stack = []
    channel_order = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if line.startswith("ROOT") or line.startswith("JOINT"):
            name = line.split()[1]
            i += 1
            # 跳到 OFFSET
            while i < n and not lines[i].startswith("OFFSET"):
                i += 1
            offset = list(map(float, lines[i].split()[1:4]))
            i += 1
            # CHANNELS
            chan_parts = lines[i].split()
            n_chan = int(chan_parts[1])
            channels = chan_parts[2:2 + n_chan]
            i += 1
            joint = BVHJoint(name, offset, channels)
            channel_order.extend(channels)
            if stack:
                joint.parent = stack[-1]
                stack[-1].children.append(joint)
            else:
                root = joint
            # 下一个 token 应该是 { 或子节点
            # 不消费 {，留给外层处理
            continue
        elif line.startswith("End Site"):
            i += 1
            while i < n and not lines[i].startswith("OFFSET"):
                i += 1
            offset = list(map(float, lines[i].split()[1:4]))
            i += 1
            while i < n and lines[i] != "}":
                i += 1
            if i < n:
                i += 1  # 消费 }
            if stack:
                stack[-1].end_site = np.array(offset, dtype=np.float64)
            continue
        elif line == "{":
            # 最近的 JOINT/ROOT 入栈
            # 找到上一个解析出的joint
            # 简化：栈顶即最近添加的joint
            if stack:
                pass
            # 把root或最近的joint入栈
            # 这里依赖 JOINT/ROOT 之后才出现 {
            # 我们在 JOINT 分支中已创建joint，这里把它入栈
            # 但需要引用——用一个trick：栈里存joint
            # 由于解析顺序：JOINT name -> { -> OFFSET -> CHANNELS -> {
            # 第二个 { 才是子块的开始
            # 改为：JOINT 分支已创建joint，遇到 { 时，如果是嵌套块，将当前joint入栈
            # 这里我们简单处理：用栈追踪当前活跃joint
            # 找当前活跃joint：最近创建的（root 或栈顶的最后一个子）
            pass
            i += 1
            continue
        elif line == "}":
            if stack:
                stack.pop()
            i += 1
            continue
        elif line.startswith("MOTION"):
            i += 1
            # Frames:
            n_frames = int(lines[i].split(":")[1].strip())
            i += 1
            # Frame Time:
            frame_time = float(lines[i].split(":")[1].strip())
            i += 1
            frames = []
            while i < n and len(frames) < n_frames:
                parts = lines[i].split()
                if len(parts) >= len(channel_order):
                    frames.append(list(map(float, parts)))
                i += 1
            return root, np.array(frames), frame_time, channel_order
        else:
            i += 1
    return root, None, None, channel_order


# 上面的简单解析对嵌套joint不可靠，重写一个基于token流的版本
def parse_bvh_robust(path):
    """健壮的BVH解析：递归下降"""
    with open(path, "r") as f:
        text = f.read()
    tokens = text.replace("{", "\n{\n").replace("}", "\n}\n").split()
    pos = [0]

    def peek():
        return tokens[pos[0]] if pos[0] < len(tokens) else None

    def consume():
        t = tokens[pos[0]]
        pos[0] += 1
        return t

    def expect(t):
        c = consume()
        assert c == t, f"expected {t}, got {c}"

    def parse_joint():
        kw = consume()  # ROOT or JOINT
        assert kw in ("ROOT", "JOINT")
        name = consume()
        expect("{")
        offset = None
        channels = None
        children = []
        end_site = None
        while peek() != "}":
            t = peek()
            if t == "OFFSET":
                consume()
                offset = np.array([float(consume()), float(consume()), float(consume())])
            elif t == "CHANNELS":
                consume()
                n = int(consume())
                channels = [consume() for _ in range(n)]
            elif t == "JOINT":
                children.append(parse_joint())
            elif t == "End":
                consume()  # End
                consume()  # Site
                expect("{")
                while peek() != "}":
                    if peek() == "OFFSET":
                        consume()
                        end_site = np.array([float(consume()), float(consume()), float(consume())])
                    else:
                        consume()
                expect("}")
            else:
                consume()
        expect("}")
        joint = BVHJoint(name, offset if offset is not None else [0, 0, 0], channels)
        joint.end_site = end_site
        for c in children:
            c.parent = joint
            joint.children.append(c)
        return joint

    # HIERARCHY
    expect("HIERARCHY")
    root = parse_joint()
    # MOTION
    expect("MOTION")
    # Frames: N
    expect("Frames:")
    n_frames = int(consume())
    expect("Frame")
    expect("Time:")
    frame_time = float(consume())
    # 通道顺序：按DFS收集
    channel_order = []

    def collect_chans(j):
        if j.channels:
            channel_order.extend(j.channels)
        for c in j.children:
            collect_chans(c)
    collect_chans(root)

    frames = []
    for _ in range(n_frames):
        row = [float(consume()) for _ in range(len(channel_order))]
        frames.append(row)
    return root, np.array(frames), frame_time, channel_order


def compute_bvh_frame_positions(root, frame_data, channel_order):
    """对一帧做前向运动学，返回 (positions, connections)

    positions: list of (name, np.array(3))
    connections: list of (parent_pos, child_pos)
    """
    chan_idx = {c: i for i, c in enumerate(channel_order)}

    positions = []
    connections = []

    def walk(joint, parent_world, parent_rot):
        local_pos = joint.offset.copy()
        # 位置channel（仅ROOT有）
        for c in joint.channels:
            if c == "Xposition":
                local_pos[0] = frame_data[chan_idx[c]]
            elif c == "Yposition":
                local_pos[1] = frame_data[chan_idx[c]]
            elif c == "Zposition":
                local_pos[2] = frame_data[chan_idx[c]]
        # 旋转channel：按出现顺序收集euler角（intrinsic）
        euler_angles = []
        euler_axes = []
        for c in joint.channels:
            if c.endswith("rotation"):
                angle = frame_data[chan_idx[c]]
                euler_angles.append(angle)
                euler_axes.append(c[0].lower())
        if euler_angles:
            local_rot = Rot.from_euler("".join(euler_axes), euler_angles, degrees=True).as_matrix()
        else:
            local_rot = np.eye(3)

        world_rot = parent_rot @ local_rot
        world_pos = parent_world + parent_rot @ local_pos
        positions.append((joint.name, world_pos))
        if joint.parent is not None:
            connections.append((parent_world, world_pos))
        for child in joint.children:
            walk(child, world_pos, world_rot)
        if joint.end_site is not None:
            end_pos = world_pos + world_rot @ joint.end_site
            connections.append((world_pos, end_pos))

    walk(root, np.zeros(3), np.eye(3))
    return positions, connections


# ---------------- RPO 机器人骨架可视化（基于pkl的key_body_pos） ----------------

def compute_robot_skeleton(root_pos, root_quat_wxyz, key_body_pos):
    """根据root和key_body_pos构造简化骨架连线"""
    q = np.array(root_quat_wxyz, dtype=np.float64)
    if np.linalg.norm(q) > 1e-6:
        q = q / np.linalg.norm(q)
    # wxyz -> xyzw
    root_rot = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

    # key_body_pos顺序：
    # 0: left_ankle_roll_link
    # 1: right_ankle_roll_link
    # 2: left_knee_link
    # 3: right_knee_link
    # 4: left_elbow_yaw_link
    # 5: right_elbow_yaw_link
    connections = []
    # 腿：root -> 膝 -> 踝
    connections.append((root_pos, key_body_pos[2]))   # root -> left_knee
    connections.append((key_body_pos[2], key_body_pos[0]))  # left_knee -> left_ankle
    connections.append((root_pos, key_body_pos[3]))   # root -> right_knee
    connections.append((key_body_pos[3], key_body_pos[1]))  # right_knee -> right_ankle
    # 手臂：root -> 肘（无肩关节key body，直接连root）
    connections.append((root_pos, key_body_pos[4]))    # root -> left_elbow
    connections.append((root_pos, key_body_pos[5]))   # root -> right_elbow

    points = [root_pos] + [key_body_pos[i] for i in range(6)]
    return points, connections


# ---------------- 视频渲染 ----------------

def render_compare_video(bvh_path, pkl_path, output_path, fps=30, skip=0, max_frames=0):
    print(f"[1/4] 解析BVH: {bvh_path}")
    bvh_root, bvh_frames, bvh_frame_time, channel_order = parse_bvh_robust(bvh_path)
    print(f"    帧数={len(bvh_frames)}, 帧时间={bvh_frame_time}, 根通道={channel_order[:6]}")

    print(f"[2/4] 加载retarget pkl: {pkl_path}")
    with open(pkl_path, "rb") as f:
        motion = pickle.load(f)
    print(f"    帧数={len(motion['root_pos'])}, fps={motion['fps']}, keys={list(motion.keys())}")

    n_bvh = len(bvh_frames)
    n_pkl = len(motion["root_pos"])
    n_frames = min(n_bvh, n_pkl)
    if max_frames > 0:
        n_frames = min(n_frames, max_frames)
    start = skip
    n_frames = max(1, n_frames - start)
    print(f"    对齐帧数: {n_frames} (skip={skip})")

    print("[3/4] 预计算BVH骨架...")
    bvh_all_positions = []
    bvh_all_connections = []
    for i in range(start, start + n_frames):
        pos, conn = compute_bvh_frame_positions(bvh_root, bvh_frames[i], channel_order)
        bvh_all_positions.append(pos)
        bvh_all_connections.append(conn)

    all_bvh_pts = np.array([p[1] for fp in bvh_all_positions for p in fp])
    bvh_min = all_bvh_pts.min(axis=0)
    bvh_max = all_bvh_pts.max(axis=0)
    bvh_center = (bvh_min + bvh_max) / 2
    bvh_range = (bvh_max - bvh_min).max() / 2 + 0.5

    pkl_root = np.array(motion["root_pos"][start:start + n_frames])
    pkl_key = np.array(motion["key_body_pos"][start:start + n_frames])
    all_pkl_pts = np.concatenate([pkl_root, pkl_key.reshape(-1, 3)])
    pkl_min = all_pkl_pts.min(axis=0)
    pkl_max = all_pkl_pts.max(axis=0)
    pkl_center = (pkl_min + pkl_max) / 2
    pkl_range = (pkl_max - pkl_min).max() / 2 + 0.5

    unified_range = max(bvh_range, pkl_range) * 1.2

    # 单位统一：BVH通常是cm，pkl是m，分别自适应即可（各自居中，范围按各自单位）
    # 为对比效果，各自独立缩放
    bvh_half = bvh_range * 1.1
    pkl_half = pkl_range * 1.1

    print(f"[4/4] 渲染视频到: {output_path}")

    fig = plt.figure(figsize=(14, 6))
    ax_bvh = fig.add_subplot(121, projection="3d")
    ax_pkl = fig.add_subplot(122, projection="3d")

    def setup_ax(ax, title, center, half):
        ax.set_title(title, fontsize=12)
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half * 0.8, center[2] + half * 1.2)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=15, azim=-90)

    setup_ax(ax_bvh, "BVH Human Motion", bvh_center, bvh_half)
    setup_ax(ax_pkl, "RPO Robot (retarget)", pkl_center, pkl_half)

    n_bvh_conn = len(bvh_all_connections[0])
    bvh_lines = [ax_bvh.plot([], [], [], "b-", linewidth=1.5)[0] for _ in range(n_bvh_conn)]
    bvh_pts, = ax_bvh.plot([], [], [], "ro", markersize=3)

    n_pkl_conn = 6
    pkl_lines = [ax_pkl.plot([], [], [], "g-", linewidth=2.5)[0] for _ in range(n_pkl_conn)]
    pkl_pts, = ax_pkl.plot([], [], [], "mo", markersize=5)

    time_text = fig.suptitle("", fontsize=10)

    def draw_frame(frame):
        pos = bvh_all_positions[frame]
        conns = bvh_all_connections[frame]
        pts = np.array([p[1] for p in pos])
        bvh_pts.set_data(pts[:, 0], pts[:, 1])
        bvh_pts.set_3d_properties(pts[:, 2])
        for i, (p0, p1) in enumerate(conns):
            bvh_lines[i].set_data([p0[0], p1[0]], [p0[1], p1[1]])
            bvh_lines[i].set_3d_properties([p0[2], p1[2]])

        root = pkl_root[frame]
        key = pkl_key[frame]
        pts_r, conns_r = compute_robot_skeleton(root, motion["root_rot"][frame], key)
        pts_arr = np.array(pts_r)
        pkl_pts.set_data(pts_arr[:, 0], pts_arr[:, 1])
        pkl_pts.set_3d_properties(pts_arr[:, 2])
        for i, (p0, p1) in enumerate(conns_r):
            pkl_lines[i].set_data([p0[0], p1[0]], [p0[1], p1[1]])
            pkl_lines[i].set_3d_properties([p0[2], p1[2]])

        t = frame / fps
        time_text.set_text(f"frame={frame + start}/{start + n_frames}  t={t:.2f}s")

    # 逐帧渲染到内存，用imageio写入视频（避免依赖ffmpeg rawvideo pipe）
    frames_buf = []
    for f in range(n_frames):
        if f % 50 == 0:
            print(f"    渲染 {f}/{n_frames}")
        draw_frame(f)
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        buf = buf.reshape(h, w, 4)
        # ARGB -> RGBA
        buf = buf[:, :, [1, 2, 3, 0]]
        frames_buf.append(buf[:, :, :3])
    plt.close(fig)

    print(f"    写入视频: {output_path}")
    imageio.mimwrite(output_path, frames_buf, fps=fps, codec="libx264",
                     quality=8, macro_block_size=1)
    print(f"完成: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="BVH vs Robot retarget compare video")
    parser.add_argument("--bvh", required=True, help="BVH文件路径")
    parser.add_argument("--pkl", required=True, help="retarget后的pkl文件路径")
    parser.add_argument("--output", default="compare.mp4", help="输出视频路径")
    parser.add_argument("--fps", type=int, default=30, help="输出视频fps")
    parser.add_argument("--skip", type=int, default=0, help="跳过开头N帧")
    parser.add_argument("--max_frames", type=int, default=0, help="最大帧数(0=全部)")
    args = parser.parse_args()

    render_compare_video(
        args.bvh, args.pkl, args.output,
        fps=args.fps, skip=args.skip, max_frames=args.max_frames
    )


if __name__ == "__main__":
    main()
