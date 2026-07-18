# RPO_IsaacLab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-v2.3-silver)](https://isaac-sim.github.io/IsaacLab)
[![RSL_RL](https://img.shields.io/badge/RSL_RL-3.3.0-silver)](https://github.com/leggedrobotics/rsl_rl)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)

[English](README.md) | [中文](README_CN.md) | [Technical Docs](TECHNICAL_DOC_CN.md)

**RPO_IsaacLab** is a reinforcement learning training workspace for the RPO bipedal robot locomotion, built on NVIDIA Isaac Lab and RSL-RL. It provides a complete pipeline from motion retargeting to sim-to-sim validation and real-robot deployment — supporting walking, dancing, parkour, fall recovery, and more human-style locomotion skills.

---

## 1. Features

- **AMP (Adversarial Motion Priors)** — Stylized locomotion learning with discriminator-guided style rewards (walking, dancing)
- **BeyondMimic** — Reference trajectory imitation with fall recovery
- **Parkour** — Complex terrain traversal
- **Attention Encoder** — Adaptive locomotion with terrain-aware attention
- **Interrupt Recovery** — Robust recovery from external perturbations
- **GMR Motion Retargeting** — BVH/FBX to robot-specific motion data pipeline
- **MuJoCo Sim2Sim** — Policy transfer validation in MuJoCo
- **Atom01 Deployment** — ONNX export and ROS2 inference for real robot deployment

---

## 2. Supported Tasks

| Task | Description |
|------|-------------|
| `RPO-AMP` / `RPO-AMP-Play` | AMP walking style training / inference |
| `RPO-AMP-Dance` / `RPO-AMP-Dance-Play` | AMP dance style training (LAFAN1 dataset) |
| `RPO-BeyondMimic` | Reference trajectory imitation + fall recovery |
| `RPO-Getup-Mimic` | Get-up motion learning |
| `RPO-Parkour` / `RPO-Parkour-Play` | Parkour over complex terrain |
| `RPO-Flat` | Flat terrain locomotion |
| `RPO-Rough` | Rough terrain locomotion |
| `RPO-AttnEnc` | Attention encoder training |
| `RPO-Interrupt` | Interrupt recovery training |

---

## 3. Installation

### 3.1 Prerequisites

- **Python** 3.11
- **Isaac Sim** 5.1.0
- **Isaac Lab** v2.3
- **RSL-RL** 3.3.0
- **CUDA** 12.8+ (Blackwell GPU requires cu128 torch build)
- **Ubuntu** 22.04 x64
- **NVIDIA Driver** >= 535

### 3.2 Quick Start

```bash
# Clone the repository
git clone --recursive https://github.com/<your-username>/RPO_IsaacLab.git
cd RPO_IsaacLab

# Create conda environment
conda create -n rpo_isaaclab python=3.11 -y
conda activate rpo_isaaclab

# Install Isaac Sim 5.1
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

# Install Isaac Lab extensions (from IsaacLab v2.3 source)
cd /path/to/IsaacLab_RPO
./isaaclab.sh --install none

# Install project dependencies
cd RPO_IsaacLab
pip install -e ./rsl_rl
pip install -e ./robolab
```

Verify installation:

```bash
python robolab/scripts/tools/list_envs.py
```

---

## 4. Usage

### 4.1 Environment Verification

```bash
python robolab/scripts/tools/list_envs.py
```

### 4.2 Training

All tasks share the same entry point `robolab/scripts/rsl_rl/train.py`. Logs are saved to `logs/rsl_rl/<experiment_name>/<timestamp>/`, with a separate directory per task (e.g., `rpo_amp`, `rpo_amp_dance`).

```bash
# AMP walking (flat terrain)
python robolab/scripts/rsl_rl/train.py --task=RPO-AMP --headless --num_envs=4096

# AMP walking (rough terrain)
python robolab/scripts/rsl_rl/train.py --task=RPO-AMP-Rough --headless --num_envs=4096

# AMP dancing (LAFAN1)
python robolab/scripts/rsl_rl/train.py --task=RPO-AMP-Dance --headless --num_envs=2048

# AMP dancing (single motion)
python robolab/scripts/rsl_rl/train.py --task=RPO-AMP-Dance-Single --headless --num_envs=2048

# BeyondMimic reference tracking + fall recovery
python robolab/scripts/rsl_rl/train.py --task=RPO-BeyondMimic --headless --num_envs=4096

# Getup-Mimic
python robolab/scripts/rsl_rl/train.py --task=RPO-Getup-Mimic --headless --num_envs=4096

# Parkour
python robolab/scripts/rsl_rl/train.py --task=RPO-Parkour --headless --num_envs=4096

# Direct RL flat terrain
python robolab/scripts/rsl_rl/train.py --task=RPO-Flat --headless --num_envs=4096

# Direct RL rough terrain
python robolab/scripts/rsl_rl/train.py --task=RPO-Rough --headless --num_envs=4096

# Attention encoder
python robolab/scripts/rsl_rl/train.py --task=RPO-AttnEnc --headless --num_envs=4096

# Interrupt recovery
python robolab/scripts/rsl_rl/train.py --task=RPO-Interrupt --headless --num_envs=4096
```

> Common options: `--max_iterations <N>` overrides the default iteration limit; `--resume --load_run=<dir>` resumes from a checkpoint; `--logger=tensorboard` enables TensorBoard; `--distributed` enables multi-GPU training (or launch via `torchrun`).

### 4.3 Testing / Playback

```bash
# AMP walking
python robolab/scripts/rsl_rl/play_amp.py --task=RPO-AMP-Play --num_envs=1

# AMP rough terrain
python robolab/scripts/rsl_rl/play_amp.py --task=RPO-AMP-Rough-Play --num_envs=1

# AMP dancing
python robolab/scripts/rsl_rl/play_amp.py --task=RPO-AMP-Dance-Play --num_envs=1

# AMP dancing (single motion)
python robolab/scripts/rsl_rl/play_amp.py --task=RPO-AMP-Dance-Single-Play --num_envs=1

# BeyondMimic
python robolab/scripts/rsl_rl/play_bm.py --task=RPO-BeyondMimic --num_envs=1

# Getup-Mimic
python robolab/scripts/rsl_rl/play_bm.py --task=RPO-Getup-Mimic --num_envs=1

# Direct RL (Flat / Rough / AttnEnc / Interrupt)
python robolab/scripts/rsl_rl/play.py --task=RPO-Flat --num_envs=1
python robolab/scripts/rsl_rl/play.py --task=RPO-Rough --num_envs=1
python robolab/scripts/rsl_rl/play.py --task=RPO-AttnEnc --num_envs=1
python robolab/scripts/rsl_rl/play.py --task=RPO-Interrupt --num_envs=1

# Parkour (add --exportonnx to export ONNX)
python robolab/scripts/rsl_rl/play_parkour.py --task=RPO-Parkour-Play --num_envs=1 --exportonnx
```

> Use `--load_run=<dir>` to specify the training log directory, or `--checkpoint=<path>` to load a specific checkpoint. Playback automatically exports JIT/ONNX models to the `exported/` directory.

### 4.4 Sim2Sim Validation

```bash
# Direct RL (Flat / Rough / AttnEnc / Interrupt)
python robolab/scripts/mujoco/sim2sim_rpo.py --load_model <exported/policy.pt>
python robolab/scripts/mujoco/sim2sim_rpo.py --load_model <path> --terrain
python robolab/scripts/mujoco/sim2sim_rpo_attn_enc.py --load_model <path>
python robolab/scripts/mujoco/sim2sim_rpo_interrupt.py --load_model <path>

# AMP / AMP-Rough (requires --terrain to load the full MJCF)
python robolab/scripts/mujoco/sim2sim_rpo_amp.py --load_model <path> --terrain

# BeyondMimic (load reference motion NPZ)
python robolab/scripts/mujoco/sim2sim_rpo_bm.py \
    --load_model <path> --motion_file <motion.npz>

# Parkour (separate ONNX graphs: encoder + actor)
python robolab/scripts/mujoco/sim2sim_rpo_parkour.py \
    --depth_encoder <0-depth_encoder.onnx> --actor <actor.onnx>

# Motion CSV visualization
python robolab/scripts/mujoco/play_motion_csv.py --motion_file <motion.csv>
```

> Add `--headless` to disable the GUI and record a video. `--load_model` should point to the TorchScript `exported/policy.pt`, not the training checkpoint `model_*.pt`.

### 4.5 Motion Data Preparation

AMP and BeyondMimic require motion data in `.pkl` format. Use the [GMR](https://github.com/Roboparty/GMR) tool to retarget BVH/FBX mocap data to the RPO robot skeleton, then reorder joints for Isaac Lab:

```bash
# Batch GMR -> Isaac Lab retargeting
python robolab/scripts/tools/retarget/dataset_retarget.py \
    --robot rpo \
    --input_dir robolab/data/motions/rpo_gmr \
    --output_dir robolab/data/motions/rpo_lab \
    --config_file robolab/scripts/tools/retarget/config/rpo.yaml

# Single-file retargeting (supports --frame_range)
python robolab/scripts/tools/retarget/single_retarget.py \
    --robot rpo \
    --input_file <input.pkl> \
    --output_file <output.pkl> \
    --config_file robolab/scripts/tools/retarget/config/rpo.yaml

# BeyondMimic NPZ motion replay (visualization)
python robolab/scripts/tools/beyondmimic/replay_npz.py -f <motion.npz>

# CSV -> NPZ conversion
python robolab/scripts/tools/beyondmimic/csv_to_npz.py -f <input.csv> --input_fps 60
```

### 4.6 Robot Deployment

See [atom01_deploy/README_CN.md](atom01_deploy/README_CN.md) for ONNX export, ROS2 inference node setup, and real robot deployment instructions.

---

## 5. Repository Structure

```
RPO_IsaacLab/
├── robolab/                    # Isaac Lab extensions (environments, assets, scripts)
│   ├── robolab/
│   │   ├── assets/robots/      # RPO robot definition
│   │   ├── tasks/              # RL environments (AMP, BeyondMimic, Parkour, etc.)
│   │   ├── utils/              # Math, buffers, noise, warp
│   │   └── scripts/            # Train, play, retarget, mujoco
│   └── data/motions/           # Motion datasets (.pkl)
├── rsl_rl/                     # RSL-RL 3.3.0 (PPO, AMP, symmetry)
├── atom01_deploy/              # Real robot deployment (ROS2, ONNX)
├── external/                   # GMR retargeting tool, dance data
└── TECHNICAL_DOC_CN.md         # Full technical documentation (Chinese)
```

---

## 6. FAQ

- **Empty `robolab`/`rsl_rl` directories**: Run `git submodule update --init --recursive`
- **Isaac Lab imports not found**: Activate the correct Python environment first
- **Task name not found**: Run `python robolab/scripts/tools/list_envs.py` for the actual task IDs
- **Blackwell GPU (RTX 50xx) CUDA errors**: Ensure torch is built for CUDA 12.8: `python -c "import torch;print(torch.version.cuda)"`

---

## 7. References

- [IsaacLab](https://github.com/isaac-sim/IsaacLab) — NVIDIA robot simulation framework
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl) — Legged robot RL library
- [RoboParty](https://github.com/Roboparty) — Open-source robot learning projects
- [legged_gym](https://github.com/leggedrobotics/legged_gym) — Legged robot training environments
- [legged_lab](https://github.com/zitongbai/legged_lab) — Isaac Lab legged training extension
- [robot_lab](https://github.com/fan-ziqi/robot_lab) — Isaac Lab robot training framework
- [InstinctLab](https://github.com/project-instinct/InstinctLab) — Instinctive policy framework

AMP algorithm from *Adversarial Motion Priors for Stylized Locomotion* (Peng et al., SIGGRAPH 2021).

---

**Maintainer**: Robot-Nav &nbsp; | &nbsp; **Support**: GitHub Issues
