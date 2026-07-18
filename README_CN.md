# RPO_IsaacLab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-v2.3-silver)](https://isaac-sim.github.io/IsaacLab)
[![RSL_RL](https://img.shields.io/badge/RSL_RL-3.3.0-silver)](https://github.com/leggedrobotics/rsl_rl)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)

[English](README.md) | [中文](README_CN.md) | [技术文档](TECHNICAL_DOC_CN.md)

**RPO_IsaacLab** 是面向 RPO 双足机器人的强化学习运动控制训练平台，基于 NVIDIA Isaac Lab 和 RSL-RL 构建。提供从动作捕捉重定向到策略仿真验证、再到实机部署的完整管线——支持行走、舞蹈、跑酷、跌倒恢复等多种类人运动技能训练。

---

## 一、核心能力

- **AMP 对抗式动作先验** — 判别器引导的风格化运动学习（行走、舞蹈）
- **BeyondMimic** — 参考轨迹模仿 + 跌倒恢复
- **Parkour 跑酷** — 复杂地形穿越
- **Attention Encoder** — 基于注意力的自适应地形感知
- **Interrupt Recovery** — 外部扰动下的鲁棒恢复
- **GMR 动作重定向** — BVH/FBX 动作捕捉数据到机器人骨骼的全自动映射
- **MuJoCo Sim2Sim** — 策略在 MuJoCo 中的可迁移性验证
- **Atom01 实机部署** — ONNX 导出 + ROS2 推理节点 + 真机控制

---

## 二、支持的任务

| 任务 ID | 说明 |
|---------|------|
| `RPO-AMP` / `RPO-AMP-Play` | AMP 行走风格训练 / 推理 |
| `RPO-AMP-Dance` / `RPO-AMP-Dance-Play` | AMP 舞蹈风格训练（LAFAN1 数据集） |
| `RPO-BeyondMimic` | 参考轨迹模仿 + 跌倒恢复 |
| `RPO-Getup-Mimic` | 起身动作模仿训练 |
| `RPO-Parkour` / `RPO-Parkour-Play` | 复杂地形跑酷 |
| `RPO-Flat` | 平地运动训练 |
| `RPO-Rough` | 粗糙地形运动训练 |
| `RPO-AttnEnc` | 注意力编码器训练 |
| `RPO-Interrupt` | 中断恢复训练 |

---

## 三、环境搭建

### 3.1 依赖要求

- **Python** 3.11
- **Isaac Sim** 5.1.0
- **Isaac Lab** v2.3
- **RSL-RL** 3.3.0
- **CUDA** 12.8+（Blackwell GPU 需使用 cu128 版 PyTorch）
- **Ubuntu** 22.04 x64
- **NVIDIA 驱动** ≥ 535

### 3.2 快速安装

```bash
# 克隆仓库（含 submodule）
git clone --recursive https://github.com/<your-username>/RPO_IsaacLab.git
cd RPO_IsaacLab

# 创建虚拟环境
conda create -n rpo_isaaclab python=3.11 -y
conda activate rpo_isaaclab

# 安装 Isaac Sim 5.1
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

# 安装 Isaac Lab 扩展（在 IsaacLab v2.3 源码目录中）
cd /path/to/IsaacLab_RPO
./isaaclab.sh --install none

# 安装项目依赖
cd RPO_IsaacLab
pip install -e ./rsl_rl
pip install -e ./robolab
```

验证安装：

```bash
python robolab/scripts/tools/list_envs.py
```

---

## 四、使用指南

### 环境验证

```bash
python robolab/scripts/tools/list_envs.py
```

### 训练

统一入口为 `robolab/scripts/rsl_rl/train.py`。训练日志自动保存至 `logs/rsl_rl/<experiment_name>/<时间戳>/`，不同任务独立目录（如 `rpo_amp`、`rpo_amp_dance`）。

```bash
# AMP 行走（平地）
python robolab/scripts/rsl_rl/train.py --task=RPO-AMP --headless --num_envs=4096

# AMP 行走（粗糙地形）
python robolab/scripts/rsl_rl/train.py --task=RPO-AMP-Rough --headless --num_envs=4096

# AMP 舞蹈（LAFAN1）
python robolab/scripts/rsl_rl/train.py --task=RPO-AMP-Dance --headless --num_envs=2048

# AMP 舞蹈（单条动作）
python robolab/scripts/rsl_rl/train.py --task=RPO-AMP-Dance-Single --headless --num_envs=2048

# BeyondMimic 参考轨迹模仿 + 跌倒恢复
python robolab/scripts/rsl_rl/train.py --task=RPO-BeyondMimic --headless --num_envs=4096

# Getup-Mimic 起身模仿
python robolab/scripts/rsl_rl/train.py --task=RPO-Getup-Mimic --headless --num_envs=4096

# Parkour 跑酷
python robolab/scripts/rsl_rl/train.py --task=RPO-Parkour --headless --num_envs=4096

# Direct RL 平地
python robolab/scripts/rsl_rl/train.py --task=RPO-Flat --headless --num_envs=4096

# Direct RL 粗糙地形
python robolab/scripts/rsl_rl/train.py --task=RPO-Rough --headless --num_envs=4096

# 注意力编码器
python robolab/scripts/rsl_rl/train.py --task=RPO-AttnEnc --headless --num_envs=4096

# 中断恢复
python robolab/scripts/rsl_rl/train.py --task=RPO-Interrupt --headless --num_envs=4096
```

> 通用选项：`--max_iterations <N>` 覆盖默认迭代次数；`--resume --load_run=<目录名>` 断点续训；`--logger=tensorboard` 启用 TensorBoard；`--distributed` 多卡训练（或 `torchrun` 启动）。

### 测试 / 回放

```bash
# AMP 行走
python robolab/scripts/rsl_rl/play_amp.py --task=RPO-AMP-Play --num_envs=1

# AMP 粗糙地形
python robolab/scripts/rsl_rl/play_amp.py --task=RPO-AMP-Rough-Play --num_envs=1

# AMP 舞蹈
python robolab/scripts/rsl_rl/play_amp.py --task=RPO-AMP-Dance-Play --num_envs=1

# AMP 舞蹈（单条）
python robolab/scripts/rsl_rl/play_amp.py --task=RPO-AMP-Dance-Single-Play --num_envs=1

# BeyondMimic
python robolab/scripts/rsl_rl/play_bm.py --task=RPO-BeyondMimic --num_envs=1

# Getup-Mimic
python robolab/scripts/rsl_rl/play_bm.py --task=RPO-Getup-Mimic --num_envs=1

# Direct RL（Flat / Rough / AttnEnc / Interrupt）
python robolab/scripts/rsl_rl/play.py --task=RPO-Flat --num_envs=1
python robolab/scripts/rsl_rl/play.py --task=RPO-Rough --num_envs=1
python robolab/scripts/rsl_rl/play.py --task=RPO-AttnEnc --num_envs=1
python robolab/scripts/rsl_rl/play.py --task=RPO-Interrupt --num_envs=1

# Parkour（加 --exportonnx 导出 ONNX）
python robolab/scripts/rsl_rl/play_parkour.py --task=RPO-Parkour-Play --num_envs=1 --exportonnx
```

> `--load_run=<目录名>` 指定训练日志目录；`--checkpoint=<路径>` 直接指定模型文件。回放时自动导出 JIT/ONNX 到 `exported/`。

### Sim2Sim 验证

```bash
# Direct RL（Flat / Rough / AttnEnc / Interrupt）
python robolab/scripts/mujoco/sim2sim_rpo.py --load_model <exported/policy.pt>
python robolab/scripts/mujoco/sim2sim_rpo.py --load_model <路径> --terrain
python robolab/scripts/mujoco/sim2sim_rpo_attn_enc.py --load_model <路径>
python robolab/scripts/mujoco/sim2sim_rpo_interrupt.py --load_model <路径>

# AMP / AMP-Rough（需 --terrain 加载完整 MJCF）
python robolab/scripts/mujoco/sim2sim_rpo_amp.py --load_model <路径> --terrain

# BeyondMimic（加载参考动作 NPZ）
python robolab/scripts/mujoco/sim2sim_rpo_bm.py \
    --load_model <路径> --motion_file <motion.npz>

# Parkour（分离 ONNX：编码器 + actor）
python robolab/scripts/mujoco/sim2sim_rpo_parkour.py \
    --depth_encoder <0-depth_encoder.onnx> --actor <actor.onnx>

# 动作 CSV 可视化回放
python robolab/scripts/mujoco/play_motion_csv.py --motion_file <motion.csv>
```

> `--headless` 可关闭 GUI 并录制视频。`--load_model` 指向 `exported/policy.pt`（TorchScript），而非 `model_*.pt`。

### 动作数据准备

AMP 和 BeyondMimic 训练需要 `.pkl` 格式的动作数据。使用 [GMR](https://github.com/Roboparty/GMR) 工具将 BVH/FBX 动捕数据重定向到 RPO 机器人骨骼，再按 Isaac Lab 关节顺序重排：

```bash
# GMR → Isaac Lab 批量重定向
python robolab/scripts/tools/retarget/dataset_retarget.py \
    --robot rpo \
    --input_dir robolab/data/motions/rpo_gmr \
    --output_dir robolab/data/motions/rpo_lab \
    --config_file robolab/scripts/tools/retarget/config/rpo.yaml

# 单文件重定向（支持 --frame_range）
python robolab/scripts/tools/retarget/single_retarget.py \
    --robot rpo \
    --input_file <input.pkl> \
    --output_file <output.pkl> \
    --config_file robolab/scripts/tools/retarget/config/rpo.yaml

# BeyondMimic NPZ 动作回放（可视化）
python robolab/scripts/tools/beyondmimic/replay_npz.py -f <motion.npz>

# CSV → NPZ 转换
python robolab/scripts/tools/beyondmimic/csv_to_npz.py -f <input.csv> --input_fps 60
```

### 实机部署

详见 [atom01_deploy/README_CN.md](atom01_deploy/README_CN.md)，包括 ONNX 模型导出、ROS2 推理节点配置和真机部署步骤。

---

## 五、项目结构

```
RPO_IsaacLab/
├── robolab/                    # Isaac Lab 扩展（环境、资产、脚本）
│   ├── robolab/
│   │   ├── assets/robots/      # RPO 机器人定义
│   │   ├── tasks/              # 强化学习环境（AMP、BeyondMimic、Parkour 等）
│   │   ├── utils/              # 数学工具、缓冲区、噪声、warp
│   │   └── scripts/            # 训练、回放、重定向、MuJoCo
│   └── data/motions/           # 动作数据集（.pkl）
├── rsl_rl/                     # RSL-RL 3.3.0（PPO、AMP、对称性）
├── atom01_deploy/              # 实机部署（ROS2 + ONNX）
├── external/                   # GMR 重定向工具、舞蹈数据源
└── TECHNICAL_DOC_CN.md         # 完整技术文档（算法原理 + 配置详解）
```

---

## 六、常见问题

- **`robolab`/`rsl_rl` 为空目录**：执行 `git submodule update --init --recursive`
- **找不到 Isaac Lab import**：先激活正确的 Python 环境再运行
- **找不到 RPO task 名称**：运行 `python robolab/scripts/tools/list_envs.py` 查看实际注册的 task ID
- **Blackwell GPU (RTX 50 系列) CUDA 报错**：确认 PyTorch 为 cu128 编译：`python -c "import torch;print(torch.version.cuda)"`

---

## 七、参考与致谢

- [IsaacLab](https://github.com/isaac-sim/IsaacLab) — NVIDIA 机器人仿真框架
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl) — 足式机器人 RL 训练库
- [RoboParty](https://github.com/Roboparty) — 开源机器人学习项目
- [legged_gym](https://github.com/leggedrobotics/legged_gym) — 足式机器人训练环境
- [legged_lab](https://github.com/zitongbai/legged_lab) — Isaac Lab 足式训练扩展
- [robot_lab](https://github.com/fan-ziqi/robot_lab) — Isaac Lab 机器人训练框架
- [InstinctLab](https://github.com/project-instinct/InstinctLab) — 本能策略框架

AMP 算法源自 *Adversarial Motion Priors for Stylized Locomotion* (Peng et al., SIGGRAPH 2021)。

---

**维护者**: Robot-Nav &nbsp; | &nbsp; **支持**: GitHub Issues
