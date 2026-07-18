# RPO 双足机器人运动控制——强化学习训练全解析（AMP / BeyondMimic / Parkour / Direct RL / AttnEnc / Interrupt / Distillation / MoE / RND）

## 一、项目概述

`RPO_IsaacLab` 是面向 RPO 双足机器人运动控制策略的强化学习训练工作区，在 Isaac Lab 源码树之外独立维护，将任务定义、动作重定向工具、MuJoCo Sim2Sim 脚本和 RSL-RL 训练入口聚合到一个仓库中。

核心目标：让 RPO 带臂双足机器人学会符合人类动作风格的行走、起步、停步、转向等运动技能，并通过 Sim2Sim 验证策略在 MuJoCo 下的可迁移性。

### 仓库结构

```
RPO_IsaacLab/
├── robolab/                              # Isaac Lab 扩展
│   ├── robolab/
│   │   ├── assets/robots/roboparty.py    # RPO 机器人 ArticulationCfg
│   │   ├── sensors/                      # grouped_ray_caster / noisy_camera / volume_points
│   │   ├── terrains/                     # height_field / trimesh / virtual_obstacle
│   │   ├── utils/                        # buffers / noise / warp / perlin
│   │   └── tasks/
│   │       ├── manager_based/
│   │       │   ├── amp/                  # AMP 环境（训练重点）
│   │       │   ├── beyondmimic/          # BeyondMimic 环境
│   │       │   └── parkour/              # Parkour 环境
│   │       └── direct/                   # base / attn_enc / interrupt 直接式环境
│   ├── scripts/
│   │   ├── rsl_rl/                       # train.py / play*.py
│   │   ├── mujoco/                       # sim2sim_rpo*.py
│   │   └── tools/                        # list_envs / retarget / beyondmimic
│   └── data/motions/                     # AMP/BeyondMimic 动作数据集
│       ├── rpo_lab/  rpo_gmr/  rpo_bm/
└── rsl_rl/                               # RSL-RL 3.3.0 源码（含 ppo_amp、amp_runner）
```

两个子目录的职责分工：

| 子目录 | 说明 |
|--------|------|
| `robolab` | Isaac Lab 扩展，包含 RPO 机器人资产、管理式(managed)环境、传感器、地形生成器与训练/部署脚本 |
| `rsl_rl` | RSL-RL 3.3.0 快照，含 AMP/BeyondMimic 等算法实现 |

### 训练任务一览

通过 `robolab/scripts/tools/list_envs.py` 列出所有 `RPO` 开头的 gym 任务：

| 任务 ID | 说明 |
|---------|------|
| `RPO-AMP` / `RPO-AMP-Play` | 对抗式动作先验行走风格训练（平地） |
| `RPO-AMP-Rough` / `RPO-AMP-Rough-Play` | 对抗式动作先验行走风格训练（粗糙地形，含 height_scan 特权 critic） |
| `RPO-AMP-Dance` / `RPO-AMP-Dance-Play` | 对抗式动作先验舞蹈风格训练（LAFAN1 数据集） |
| `RPO-BeyondMimic` | 参考轨迹模仿 + 跌倒恢复 |
| `RPO-Getup-Mimic` | 起身动作模仿训练 |
| `RPO-Parkour` / `RPO-Parkour-Play` | 复杂地形跑酷 |
| `RPO-Flat` | 平地运动训练（Direct RL + PPO） |
| `RPO-Rough` | 粗糙地形运动训练（Direct RL + PPO，含 height_scan 特权 critic） |
| `RPO-AttnEnc` | 注意力编码器训练 |
| `RPO-Interrupt` | 中断恢复训练 |

---

## 二、RPO 机器人

RPO 是一款带双臂的双足机器人，共 22 个关节（左右各 11），躯干 1 个，具备完整的下肢运动与上肢摆臂能力。

### 关节拓扑

```
base_link
├── 左腿: left_thigh_yaw → left_thigh_roll → left_thigh_pitch → left_knee → left_ankle_pitch → left_ankle_roll
├── 右腿: right_thigh_yaw → right_thigh_roll → right_thigh_pitch → right_knee → right_ankle_pitch → right_ankle_roll
├── 躯干: torso_joint
├── 左臂: left_arm_pitch → left_arm_roll → left_arm_yaw → left_elbow_pitch → left_elbow_yaw
└── 右臂: right_arm_pitch → right_arm_roll → right_arm_yaw → right_elbow_pitch → right_elbow_yaw
```

### 执行器分组与参数

执行器使用 `DelayedPDActuatorCfg`，支持 0~2 步延迟，模拟真实控制器的通信延迟：

| 执行器组 | 关节 | 刚度 | 阻尼 | 力矩限制 | 速度限制 |
|----------|------|------|------|----------|----------|
| `legs` | thigh_yaw/roll/pitch, knee, torso | 100~150 | 3.3~5.0 | 120 N·m | 25 rad/s |
| `feet` | ankle_pitch/roll | 40 | 2.0 | 27 N·m | 8 rad/s |
| `shoulders` | arm_pitch/roll/yaw | 40 | 2.0 | 27 N·m | 8 rad/s |
| `arms` | elbow_pitch/yaw | 30/20 | 1.5/1.0 | 27 N·m | 8 rad/s |

### 初始位姿

初始站立位姿 `z=0.75m`，关节初始角度模拟自然微屈站姿：

```python
joint_pos = {
    "left_thigh_pitch_joint": -0.1,   "right_thigh_pitch_joint": -0.1,
    "left_knee_joint": 0.3,           "right_knee_joint": 0.3,
    "left_ankle_pitch_joint": -0.2,   "right_ankle_pitch_joint": -0.2,
    "left_arm_pitch_joint": 0.18,     "right_arm_pitch_joint": 0.18,
    "left_arm_roll_joint": 0.06,      "right_arm_roll_joint": -0.06,
    "left_elbow_pitch_joint": 0.78,   "right_elbow_pitch_joint": 0.78,
}
```

### 物理属性

- 启用自碰撞（`enabled_self_collisions=True`）
- 接触传感器（`activate_contact_sensors=True`）
- 求解器：position iteration = 8, velocity iteration = 4
- 关节位置软限位因子：0.90
- 资产路径：`data/robots/roboparty/rpo/urdf/rpo.urdf`

---

## 三、算法原理：AMP 对抗式动作先验

训练流程的核心是 **AMP（Adversarial Motion Priors，对抗式动作先验）**。AMP 在 PPO 基础上引入判别器：人类参考动作作为真实分布，策略生成的运动状态作为伪样本，对抗训练驱使策略产出风格自然的运动，同时任务奖励驱动其完成速度跟随等目标。

### 3.1 AMP 与 PPO 的关系

AMP 的策略优化器就是 PPO。实现中算法类为 `PPOAmp`，运行器为 `AMPRunner`，继承 PPO 的 clipped surrogate 目标、GAE 优势估计、自适应学习率与对称性约束，额外增加 AMP 判别器及其损失、梯度惩罚与风格奖励。

简言之：**PPO 最大化任务回报，AMP 判别器提供风格奖励塑形策略行为**。

### 3.2 PPO 策略优化

PPO 采用截断重要性采样。设比率：

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$$

优势 $\hat{A}_t$ 由 GAE 计算：

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

截断目标：

$$L^{CLIP}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta),\,1-\epsilon,\,1+\epsilon)\hat{A}_t\right)\right]$$

总策略损失：

$$L = L^{CLIP} - c_v L^{VF} + c_h H[\pi]$$

其中价值损失使用截断形式，熵奖励鼓励探索。学习率按 KL 自适应调整：实际 KL 偏离 `desired_kl` 时调高/调低。

### 3.3 AMP 判别器

#### 网络结构

判别器 $D_\varphi(s)$ 为 MLP，trunk 隐藏层 $[1024, 512]$（ELU 激活），输出层为线性映射至标量。输入为连续若干步的运动状态观测，沿时间步展平后输入：

| 观测项 | 含义 |
|--------|------|
| `base_ang_vel` | 基座角速度 |
| `joint_pos` | 关节位置 |
| `joint_vel` | 关节速度 |
| `key_body_pos_b` | 关键本体相对基座位置（左右踝、左右膝、左右肘） |

观测先经 `EmpiricalNormalization` 归一化，再沿时间步展平为 $[\text{disc\_obs\_steps} \times \text{disc\_obs\_dim}]$ 的向量。

#### 判别器观测组

AMP 区分两组观测：

- **策略侧**（`disc`）：来自环境实际采样的机器人状态，历史长度 `disc_obs_steps=3`
- **示范侧**（`disc_demo`）：来自 `AnimationTerm` 插值得到的参考状态，与策略侧历史长度一致

#### 损失函数

采用 **LS-GAN** 形式（`loss_type="LSGAN"`），示范数据标签 $+1$，策略生成数据标签 $-1$：

$$L_D = \frac{1}{2}\,\mathbb{E}_{s^E\sim\text{demo}}\left[(D(s^E)-1)^2\right] + \frac{1}{2}\,\mathbb{E}_{s^G\sim\pi}\left[(D(s^G)+1)^2\right]$$

对示范数据施加 R1 梯度惩罚，约束判别器在示范数据上的梯度范数趋零：

$$L_{gp} = \lambda_{gp}\,\mathbb{E}_{s^E}\left[\|\nabla_{s^E} D(s^E)\|_2^2\right]$$

判别器总损失：

$$L_{disc} = L_D + L_{gp}$$

判别器使用独立的 Adam 优化器，与策略优化器分别更新。trunk 与线性层分别施加不同的权重衰减（`disc_trunk_weight_decay=1e-3`，`disc_linear_weight_decay=1e-1`）。

### 3.4 风格奖励与任务-风格混合

LS-GAN 下的风格奖励：

$$r_{style} = \Delta t \cdot w_s \cdot \max\left(0,\ 1 - \tfrac{1}{4}(D(s^G)-1)^2\right)$$

$D(s^G)\to 1$ 时奖励趋近 $\Delta t \cdot w_s$，偏离示范时截断为 0。与环境任务奖励 $r_{task}$ 线性混合：

$$r_{total} = \alpha\, r_{task} + (1-\alpha)\, r_{style}, \quad \alpha = \text{task\_style\_lerp}$$

当前配置 `task_style_lerp=0.6`，任务奖励占 60%，风格奖励占 40%。

### 3.5 对称性约束

双足左右对称结构，启用数据增强与镜像损失：

- **数据增强**（`use_data_augmentation=True`）：通过 `rpo.compute_symmetric_states` 生成镜像样本，将 batch 扩充为原始 + 镜像，参与 PPO 梯度更新
- **镜像损失**（`use_mirror_loss=True`）：对镜像观测下的动作均值与原始动作的镜像变换计算 MSE 损失，系数 `mirror_loss_coeff=0.2`

$$L_{mirror} = c_{mirror} \cdot \text{MSE}(\mu_\pi(s_{sym}),\ \text{sym}(\mu_\pi(s_{orig})))$$

### 3.6 数据流

```
MotionDataManager
    ↓ 加载 data/motions/rpo_lab/*.pkl（按权重采样）
    ↓ 每条动作含 root_pos/rot, dof_pos, key_body_pos 帧序列
AnimationTerm
    ↓ 按 step_dt 采样时刻，帧混合插值（线性 + 四元数 slerp）
    ↓ 输出参考状态 → disc_demo 观测组
AmpEnv.step
    ↓ 保留重置前观测 → disc 观测组
PPOAmp.process_env_step
    ↓ 计算 style_reward → lerp 混合 → 存入 CircularBuffer
PPOAmp.update
    ↓ 策略梯度 + 判别器梯度分别更新
```

**动作数据权重配置**：

| 数据集 | 权重 | 说明 |
|--------|------|------|
| `127_06` | 16 | 主行走动作 |
| `A1-_Stand_stageii` | 6.5 | 原地站立 |
| `run_start_180_R_001__A345_M` | 4 | 起步（镜像） |
| `run_start_180_R_001__A345` | 4 | 起步 |
| `move_l` | 4.5 | 左移 |
| `move_r` | 5 | 右移 |
| `run_stop_180_R_001__A345_M` | 3 | 停步（镜像） |
| `run_stop_180_R_001__A345` | 3 | 停步 |

**舞蹈动作数据权重**（`RPODanceAmpEnvCfg`）：参考 [LAFAN1 数据集](https://zhuanlan.zhihu.com/p/1932080562740398063)。

| 数据集 | 权重 | 说明 |
|--------|------|------|
| `dance1_subject1` | 1.0 | LAFAN1 舞蹈1-1 |
| `dance1_subject2` | 1.0 | LAFAN1 舞蹈1-2 |
| `dance1_subject3` | 1.0 | LAFAN1 舞蹈1-3 |
| `dance2_subject1` | 1.0 | LAFAN1 舞蹈2-1 |
| `dance2_subject2` | 1.0 | LAFAN1 舞蹈2-2 |
| `dance2_subject3` | 1.0 | LAFAN1 舞蹈2-3 |
| `dance2_subject4` | 1.0 | LAFAN1 舞蹈2-4 |
| `dance2_subject5` | 1.0 | LAFAN1 舞蹈2-5 |

### 3.7 BeyondMimic 超越模仿

#### 算法原理

BeyondMimic 是一种**参考轨迹模仿**算法,区别于 AMP 的对抗式风格学习:它直接以参考动作的每一帧作为跟踪目标,通过显式的位姿/速度误差奖励驱动策略逐帧复现参考运动,同时保留 RL 的任务奖励与域随机化能力,使策略在模仿的基础上具备鲁棒性。

核心机制:

1. **锚点对齐**:选择一个根锚点(如 `torso_link`),将参考动作的根部位位姿与机器人实际位姿对齐,消除全局漂移
2. **逐帧跟踪**:每步从参考动作采样当前帧,计算锚点与各身体部位的位姿/速度误差
3. **指数衰减奖励**:误差经 `exp(-error/std²)` 转换为 `[0,1]` 奖励,`std` 控制容差
4. **自适应起始帧**:重置时随机采样起始帧,使策略学习完整轨迹而非特定片段

#### 命令生成

`MotionCommand` 从参考动作文件采样帧并驱动锚点对齐,关键参数:

| 参数 | 值 | 说明 |
|------|-----|------|
| `resampling_time_range` | (1e9, 1e9) | 极大值,保证单回合不重采样 |
| `pose_range` | x/y ±0.05m, roll/pitch ±0.1rad, yaw ±0.2rad | 重置时位姿扰动 |
| `velocity_range` | x/y ±0.5m/s, z ±0.2m/s, roll/pitch ±0.52rad/s, yaw ±0.78rad/s | 重置时速度扰动 |
| `joint_position_range` | (-0.1, 0.1) | 关节初始位置扰动 |

#### 跟踪奖励函数

所有跟踪奖励采用指数衰减形式 `exp(-error/std²)`,误差越小奖励越接近 1:

| 奖励项 | 权重 | std | 误差定义 |
|--------|------|-----|----------|
| `motion_global_anchor_pos` | 0.5 | 0.3 | $\|\|p_{anchor}^{ref} - p_{anchor}^{robot}\|\|^2$ |
| `motion_global_anchor_ori` | 0.5 | 0.4 | $\text{quat\_err}(q_{anchor}^{ref}, q_{anchor}^{robot})^2$ |
| `motion_body_pos` | 1.0 | 0.3 | $\text{mean}_i\|\|p_{body,i}^{ref} - p_{body,i}^{robot}\|\|^2$ |
| `motion_body_ori` | 1.0 | 0.4 | $\text{mean}_i\,\text{quat\_err}(...)^2$ |
| `motion_body_lin_vel` | 1.0 | 1.0 | $\text{mean}_i\|\|v_{body,i}^{ref} - v_{body,i}^{robot}\|\|^2$ |
| `motion_body_ang_vel` | 1.0 | 3.14 | $\text{mean}_i\|\|\omega_{body,i}^{ref} - \omega_{body,i}^{robot}\|\|^2$ |

> `body_pos_relative_w` 已扣除锚点平移与偏航,对水平跟随不敏感,聚焦于姿态模仿。

**惩罚项**:

| 项 | 权重 | 说明 |
|----|------|------|
| `joint_acc_l2` | -2.5e-7 | 关节加速度 |
| `joint_torques_l2` | -1e-5 | 关节力矩 |
| `action_rate_l2` | -0.1 | 动作变化率 |
| `joint_pos_limits` | -10.0 | 关节限位 |

#### 终止条件

| 条件 | 阈值 | 说明 |
|------|------|------|
| `time_out` | 20s | 单回合超时 |
| `bad_anchor_pos_z_only` | 0.25m | 锚点 Z 方向偏离参考 |
| `bad_anchor_ori` | 0.8 | 锚点投影重力差异 |

#### 观测空间

**策略观测**(含噪声):

| 观测项 | 噪声 |
|--------|------|
| `command` (参考关节角/速度) | 无 |
| `base_ang_vel` | ±0.2 |
| `projected_gravity` | ±0.05 |
| `joint_pos` (相对) | ±0.01 |
| `joint_vel` (相对) | ±0.5 |
| `actions` (上一步) | 无 |

**Critic 特权观测**:额外包含 `motion_anchor_pos_b`、`motion_anchor_ori_b`、`body_pos`、`body_ori`、`base_lin_vel`。

#### Getup-Mimic 起身模仿变体

`RPOGetupMimicEnvCfg` 在 BeyondMimic 基础上特化:

| 参数 | BeyondMimic | Getup-Mimic |
|------|-------------|-------------|
| `motion_file` | `yundong1.npz` | `getup_supin2prone.npz` |
| `anchor_body_name` | `torso_link` | `base_link` |
| `episode_length_s` | 20.0 | 5.0 |
| `motion_body_pos.weight` | 1.0 | 2.0 |
| `reset_on_motion_end` | 默认 | False |
| `randomize_push_robot.interval` | (1.0, 3.0) | (0.0, 5.0) |

**起身任务特有奖励** `stand_still_after_motion`:动作播完后施加关节位姿/速度稳定惩罚,通过 `projected_gravity_b[:, 2]` 软掩码避免倒地时累积惩罚:

$$r_{stand} = (w_p \sum_i |q_i - q_i^{default}| + w_v \sum_i |\dot{q}_i|) \cdot \text{clamp}\left(\frac{-g_z^b}{0.7}, 0, 1\right) \cdot \mathbb{1}[\text{motion\_ended}]$$

### 3.8 Parkour 跑酷

#### 算法原理

Parkour 任务让机器人在复杂地形(台阶/斜坡/沟壑)中跑酷,采用 **PPO + AMP + MoE + 深度图编码器** 的组合策略:

- **深度图编码器**:CNN 处理第一人称深度图,提取地形几何特征
- **MoE 混合专家**:5 个专家网络,门控按地形场景切换,提升模型容量
- **AMP 风格奖励**:行走动作数据提供风格先验,避免跑酷姿态异常
- **课程学习**:基于速度跟踪表现动态调整地形难度与安全奖励权重

#### 策略网络(EncoderMoEActorCritic)

| 组件 | 配置 |
|------|------|
| 深度图编码器 | CNN: channels=[4], kernels=[3], strides=[1], hidden=[256,256], output=128, ReLU + MaxPool |
| MoE 专家数 | 5 (soft 混合,所有专家参与) |
| Actor MLP | [256, 128, 64] |
| Critic MLP | [256, 128, 64] |
| 激活函数 | ELU |
| `actor_obs_normalization` | **False** (深度图经编码器后是 latent,归一化会破坏分布导致部署崩溃) |

#### 观测空间

**策略观测**(8 帧历史):

| 观测组 | 内容 |
|--------|------|
| `policy` | 本体感知(角速度/投影重力/指令/关节状态)+ 深度图 latent |
| `depth_image` | 深度图(经 CNN 编码为 128 维 latent) |

**Critic 特权观测**:额外包含 `height_scan`(高度扫描,187 维点云)。

**判别器观测**:根姿/角速度/关节状态/关键体位置,3 步历史。

#### 深度相机配置

模拟 Pinhole 相机并叠加多种噪声以贴近真实相机:

| 噪声模块 | 说明 |
|----------|------|
| 缩放噪声 | 深度值随机缩放 |
| 卷积模糊 | 模拟镜头模糊 |
| 柏林噪声 | 模拟传感器随机噪声 |

#### 体积点传感器(穿深检测)

`VolumePoints` 在足部/膝部刚体表面采样 3D 点云,与虚拟障碍物求交输出穿透偏移量,用于 `feet_at_plane` 奖励约束足部落点。

#### 地形生成器

`ROUGH_TERRAINS_CFG` 组合多种子地形,10×20 网格,`curriculum=True` 按行难度递增:

| 子地形 | 比例 | 说明 |
|--------|------|------|
| `perlin_rough` | 0.05 | Perlin 噪声平面 |
| `perlin_rough_walk` | - | Perlin 噪声(前进模式) |
| `pyramid_stairs` | - | 金字塔台阶 |
| `pyramid_stairs_inv` | - | 反向台阶 |
| `hf_pyramid_slope_inv` | - | 高度场反向斜坡 |
| `gap_pit` | - | 方坑 |

`wall_prob` 控制四向墙障碍概率防止越界,`flat_patch_sampling` 提供安全出生点候选。

#### 课程学习

| 课程项 | 机制 |
|--------|------|
| 地形难度 | 基于速度跟踪表现调整子地形行号 |
| `feet_penetration` 权重 | 随训练进度放大足部穿深惩罚 |
| 步态稳定性/碰撞惩罚 | 动态调节权重 |

#### Parkour 训练超参数

| 参数 | 值 |
|------|-----|
| `num_steps_per_env` | 24 |
| `max_iterations` | 30000 |
| `save_interval` | 500 |
| `learning_rate` | 1e-4 (KL 自适应, desired_kl=0.01) |
| `entropy_coef` | 0.006 |
| `mirror_loss_coeff` | 0.15 |
| AMP `style_reward_scale` | 2.0 |
| AMP `task_style_lerp` | 0.3 (任务奖励占主导) |
| AMP `grad_penalty_scale` | 5.0 |

### 3.9 Direct RL 直接式强化学习

#### 架构对比

| 特性 | Manager-Based | Direct RL |
|------|---------------|-----------|
| 环境基类 | `ManagerBasedRLEnv` | `DirectRLEnv` |
| 观测/奖励/终止 | 配置驱动(Manager) | 代码驱动(直接在 `env` 类实现) |
| 灵活性 | 配置组合,易扩展 | 性能更高,逻辑紧凑 |
| 适用场景 | AMP/BeyondMimic/Parkour | Flat/Rough/AttnEnc/Interrupt |

Direct RL 的 `BaseEnv` 直接继承 `DirectRLEnv`,在 Python 类中实现观测构建、奖励计算、终止判定,通过 `RewardManager` 统一调度奖励项。

#### Base 环境观测空间

Actor 观测(78 维/帧,10 帧历史 = 780 维):

| 段 | 索引 | 内容 | 缩放 |
|----|------|------|------|
| 0:3 | `base_ang_vel` | 基座角速度 | 0.25 |
| 3:6 | `projected_gravity` | 投影重力 | 1.0 |
| 6:9 | `velocity_commands` | 速度指令 | 1.0 |
| 9:32 | `joint_pos` (相对默认) | 关节位置 | 1.0 |
| 32:55 | `joint_vel` (相对默认) | 关节速度 | 0.25 |
| 55:78 | `actions` | 上一步动作 | 1.0 |

Critic 观测(139 维/帧 Flat,326 维/帧 Rough):额外包含 `base_lin_vel`、`feet_contact`、`contact_force`、`air_time`、`feet_height`、`joint_acc`、`joint_torques`,Rough 额外含 187 维 `height_scan`。

#### Base 环境奖励函数

RPO Flat/Rough 的完整奖励组合:

**速度跟踪**:

| 项 | 权重 | 公式 |
|----|------|------|
| `track_lin_vel_xy_exp` | 1.0 | $\exp(-\|v_{xy}^{cmd} - v_{xy}^{robot}\|^2 / 0.5^2)$ |
| `track_ang_vel_z_exp` | 1.0 | $\exp(-\|\omega_z^{cmd} - \omega_z^{robot}\|^2 / 0.5^2)$ |

**步态奖励**:

| 项 | 权重 | 说明 |
|----|------|------|
| `feet_air_time` | 0.25 | 单足支撑时另一足腾空时长奖励(threshold=0.4s) |
| `feet_distance` | 0.1 | 双足 y 向距离 [0.16, 0.50] |
| `knee_distance` | 0.1 | 膝盖 y 向距离 [0.18, 0.35] |
| `feet_height` | 0.2 | 摆动足抬起高度奖励(threshold=0.02m) |
| `feet_contact_without_cmd` | 0.1 | 零指令时双足同时着地 |

**姿态/稳定**:

| 项 | 权重 | 说明 |
|----|------|------|
| `flat_orientation_l2` | -1.0 | 躯干姿态偏离竖直 |
| `upward` | 0.4 | 投影重力 z 正向奖励 |
| `stand_still` | -0.2 | 静止时关节位姿/速度惩罚 |
| `termination_penalty` | -200.0 | 终止大额惩罚 |

**惩罚项**:

| 项 | 权重 |
|----|------|
| `lin_vel_z_l2` | -0.2 |
| `ang_vel_xy_l2` | -0.1 |
| `energy` | -1e-4 |
| `joint_torques_l2` | -1e-5 |
| `joint_vel_l2` | -2e-4 |
| `dof_acc_l2` | -2.5e-7 |
| `action_rate_l2` | -2e-2 |
| `action_smoothness_l2` | -2e-2 |
| `undesired_contacts` | -1.0 |
| `feet_slide` | -0.3 |
| `feet_force` | -3e-3 (threshold=500N, max=400N) |
| `feet_stumble` | -1.0 (水平力 > 3×垂直力) |
| `dof_pos_limits` | -1.0 |
| `joint_deviation_hip` | -0.03 |
| `joint_deviation_torso` | -1.0 |
| `joint_deviation_arms` | -0.06 |
| `joint_deviation_legs` | -0.01 |

#### Flat vs Rough 配置差异

| 参数 | Flat | Rough |
|------|------|-------|
| `state_space` | 139 | 326 |
| `enable_height_scan` | False | True |
| `terrain_generator` | `GRAVEL_TERRAINS_CFG` | `ROUGH_TERRAINS_CFG` |
| `ang_vel_xy_l2` 权重 | -0.1 | -0.05 |
| `lin_vel_z_l2` 权重 | -0.2 | -0.05 |
| `gpu_collision_stack_size` | 默认 | 2²⁹ |

> Rough 地形放宽姿态惩罚权重,因地形起伏本身会引入角速度。

### 3.10 注意力编码器 AttnEnc

#### 算法原理

`ActorCriticAttnEnc` 在标准 Actor-Critic 基础上引入**多头自注意力编码器**处理感知观测(高度图),与本体观测融合后送入 MLP:

1. 高度图(17×11)经 `AttentionEncoder` 编码为低维嵌入
2. 嵌入与本体观测拼接,送入 Actor/Critic MLP
3. 可选 **Critic 估计辅助任务**:Critic 额外预测 `root_lin_vel`(特权信息),辅助损失约束
4. 可选 **观测编码器**:学习低维 latent 表示,提升感知表征质量

#### 网络结构

| 组件 | 配置 |
|------|------|
| Actor/Critic MLP | [512, 256, 128] |
| `embedding_dim` | 32 |
| `head_num` | 4 (多头注意力) |
| `map_size` | (17, 11) |
| `map_resolution` | 0.1 m/cell |
| `actor_history_length` | 5 |
| `critic_history_length` | 5 |
| `enable_critic_estimation` | True (估计 `root_lin_vel`) |
| `estimation_slice` | [78, 79, 80] |
| `enable_obs_encoder` | True |
| `latent_dim` | 32 |

#### 辅助损失

$$L_{aux} = c_{aux} \cdot \text{MSE}(\hat{v}_{root}, v_{root})$$

`aux_loss_coef=0.05`,小于主损失避免主导。

#### AttnEnc 环境特化

| 参数 | 值 |
|------|-----|
| `terrain_generator` | `ROUGH_HARD_TERRAINS_CFG` (高难度) |
| `enable_height_scan_actor` | True (actor 侧也启用) |
| `height_scanner.size` | (1.6, 1.0) m |
| `actor_obs_history_length` | 5 (而非 base 的 10) |
| `normalization.height_scan_offset` | 0.75 |
| `undesired_foothold` | -0.2 (落点不可行惩罚) |
| `mirror_loss_coeff` | 0.1 (低于 interrupt 的 0.2) |

### 3.11 中断恢复 Interrupt

#### 算法原理

Interrupt 任务训练机器人在**手臂关节被外部中断**(强制偏移)时恢复稳定运动。8 个手臂关节(左右各 4:arm pitch/roll/yaw + elbow pitch)按课程学习逐步增大中断幅度。

#### 中断配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `use_interrupt` | True | 启用中断 |
| `interrupt_ratio` | 0.5 | 50% 环境启用中断 |
| `interrupt_init_range` | 0.2 | 课程初始 clipping 范围 |
| `interrupt_update_step` | 30 | 中断目标重采样周期(步) |
| `switch_prob` | 0.005 | 随机切换中断态概率 |
| `max_curriculum` | 1.0 | 课程最大幅度 |

**中断关节范围**:

| 关节 | 下界 | 上界 |
|------|------|------|
| arm_pitch | -1.57 | 1.57 |
| arm_roll | -0.25 / -1.57 | 1.57 / 0.25 |
| arm_yaw | -1.57 | 1.57 |
| elbow_pitch | -0.5 | 1.57 |

#### 中断特有奖励

| 项 | 权重 | 说明 |
|----|------|------|
| `joint_deviation_interrupt` | -1.0 | 手臂关节偏离惩罚(分组加权) |
| `stand_still_interrupt` | -0.2 | 静止稳定性(含手臂归零) |
| `action_penalty_interrupt` | -0.1 | 中断态手臂动作惩罚 |

> Interrupt 使用 `GRAVEL_TERRAINS_CFG`(平缓地形),聚焦中断任务而非地形适应。

### 3.12 策略蒸馏 Distillation

#### 算法原理

`Distillation` 通过**行为克隆**将拥有特权信息的教师网络迁移到仅含本体感知的学生网络,消除部署阶段对特权观测的依赖:

$$L_{behavior} = \text{MSE}(a^{student}, a^{teacher})$$

支持 MSE 与 Huber 损失,采用 **BPTT 截断**(梯度累积长度 `gradient_length=15`)处理循环网络中的长序列梯度爆炸/消失。

#### 教师学生架构

| 网络 | 输入 | 输出 |
|------|------|------|
| Teacher | 全观测(含特权) | 特权动作(监督目标) |
| Student | 本体观测 | 部署动作 |

#### 蒸馏流程

```
1. Teacher 与 Student 同步前向,Teacher 输出 detach 作为监督目标
2. Student 前向(保留梯度),计算 behavior_loss
3. 累积 gradient_length 步损失后统一反向传播
4. 梯度裁剪 + 优化器更新 Student 参数
```

### 3.13 混合专家 MoE

#### 算法原理

`MoeLayer` 采用 **soft 混合**:门控网络输出经 softmax 得到各专家权重,所有专家前向后按权重加权求和:

$$y = \sum_{i=1}^{N} g_i(x) \cdot E_i(x), \quad g(x) = \text{softmax}(\text{gate}(x))$$

实现为 soft MoE(所有专家都参与计算),便于训练稳定,非稀疏路由。

#### MoE 统计监控

`gate_stats` 输出:

| 指标 | 说明 |
|------|------|
| `expert_i` | 各专家平均权重 |
| `gate_entropy` | 门控分布熵(越高越均衡) |
| `max_weight` | 最大权重均值 |

### 3.14 RND 随机网络蒸馏

PPO 支持可选的 RND 内在奖励模块:

- **目标网络** `target`:固定随机初始化的网络,输出目标嵌入
- **预测网络** `predictor`:可训练网络,逼近目标嵌入
- **内在奖励**:$r^{int} = \|\|f_{predictor}(s) - f_{target}(s)\|\|^2$

预测误差越大,状态越"新奇",鼓励探索未知状态。

---

## 四、环境设计

### 4.1 观测空间

AMP 环境包含四组观测，分别服务于策略、价值网络和判别器：

**策略观测组**（`policy`，历史长度 3，启用噪声与 corruption）：

| 观测项 | 噪声范围 |
|--------|----------|
| `base_ang_vel` | ±0.35 |
| `projected_gravity` | ±0.05 |
| `velocity_commands` | 无 |
| `joint_pos`（相对默认） | ±0.03 |
| `joint_vel`（相对默认） | ±1.75 |
| `actions`（上一步动作） | 无 |

**价值网络观测组**（`critic`，历史长度 3，无噪声，含特权信息）：

在策略观测基础上额外包含 `base_lin_vel`（基座线速度），且 `joint_pos`/`joint_vel` 使用绝对值而非相对值。

**判别器观测组**（`disc`，历史长度 3，无噪声）：

| 观测项 | 说明 |
|--------|------|
| `base_ang_vel` | 基座角速度 |
| `joint_pos` | 关节绝对位置 |
| `joint_vel` | 关节速度 |
| `key_body_pos_b` | 关键体相对基座位置（6 个体） |

**判别器示范观测组**（`disc_demo`，来自 `AnimationTerm` 插值参考）：

| 观测项 | 说明 |
|--------|------|
| `ref_root_ang_vel_b` | 参考基座角速度（基座坐标系） |
| `ref_joint_pos` | 参考关节位置 |
| `ref_joint_vel` | 参考关节速度 |
| `ref_key_body_pos_b` | 参考关键体位置 |

### 4.2 动作空间

关节位置目标控制，缩放因子 0.25，使用默认关节偏移：

```python
joint_pos = mdp.JointPositionActionCfg(
    asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
)
```

### 4.3 奖励函数

RPO-AMP 的奖励函数由任务奖励和惩罚项组成：

**任务奖励**：

| 奖励项 | 权重 | 说明 |
|--------|------|------|
| `track_lin_vel_xy_exp` | 1.25 | 线速度跟踪（高斯核，std=0.5） |
| `track_ang_vel_z_exp` | 1.25 | 角速度跟踪（高斯核，std=0.5） |
| `alive` | 0.15 | 存活奖励 |

**基座惩罚**：

| 惩罚项 | 权重 | 说明 |
|--------|------|------|
| `ang_vel_xy_l2` | -0.1 | 基座侧倾/俯仰角速度 |
| `flat_orientation_l2` | -1.2 | 基座姿态偏离竖直 |

**关节惩罚**：

| 惩罚项 | 权重 | 说明 |
|--------|------|------|
| `joint_vel_l2` | -2e-4 | 关节速度 |
| `joint_acc_l2` | -2.5e-7 | 关节加速度 |
| `action_rate_l2` | -0.01 | 动作变化率 |
| `joint_pos_limits` | -1.0 | 关节位置限位 |
| `joint_energy` | -1e-4 | 关节能量（力矩×速度） |
| `joint_torques_l2` | -1e-5 | 关节力矩 |
| `arm_pitch_mean_offset` | -0.1 | 手臂俯仰均值偏移 |

**足部奖惩**：

| 项 | 权重 | 说明 |
|----|------|------|
| `feet_slide` | -0.1 | 足部滑动 |
| `feet_distance_y` | 0.05 | 双足侧向距离（min=0.14, max=0.50） |
| `sound_suppression` | -5e-5 | 足部冲击加速度抑制 |

**其他惩罚**：

| 项 | 权重 | 说明 |
|----|------|------|
| `undesired_contacts` | -10.0 | 非足部接触（排除踝部） |

### 4.4 终止条件

| 条件 | 说明 |
|------|------|
| `time_out` | 单回合超时（20s） |
| `base_contact` | 非法接触：大腿/基座/手臂/肘部触碰地面 |
| `base_height` | 基座高度低于 0.2m |
| `bad_orientation` | 基座姿态倾角超过 60° |

### 4.5 域随机化

训练时施加以下随机化以提升策略鲁棒性：

**启动时随机化**（`mode="startup"`）：

| 随机化项 | 参数 |
|----------|------|
| 刚体材质 | 静摩擦 [0.3, 1.6]，动摩擦 [0.3, 1.2]，恢复系数 [0.0, 0.5] |
| 基座质量偏移 | ±3.0 kg（additive） |
| 躯干/基座质心偏移 | x/y/z 各 ±0.03m |
| 肢体质量缩放 | [0.8, 1.2]（scale） |
| 执行器增益缩放 | 刚度/阻尼 [0.8, 1.2]（scale） |
| 关节参数 | armature [0.8, 1.2]（scale） |

**重置时随机化**（`mode="reset"`）：

| 随机化项 | 参数 |
|----------|------|
| 基座位置 | x/y ∈ ±0.5m，yaw ∈ ±π |
| 基座速度 | 线速度/角速度各 ±0.2 |
| 关节位置 | 默认值 × [0.8, 1.2] |

**间隔随机化**（`mode="interval"`，每 5~10s）：

| 随机化项 | 参数 |
|----------|------|
| 推动机器人 | x/y ∈ ±0.5m/s，yaw ∈ ±1.0rad/s |

### 4.6 命令空间

速度命令由 `UniformVelocityCommand` 生成，重采样间隔 10s：

| 命令 | 范围 |
|------|------|
| `lin_vel_x` | [-0.5, 2.5] m/s |
| `lin_vel_y` | [-0.5, 0.5] m/s |
| `ang_vel_z` | [-1.5, 1.5] rad/s |

2% 的环境为站立环境（命令速度接近零），100% 为航向命令模式。

---

## 五、训练配置

### 5.1 PPO 超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `clip_param` | 0.2 | 截断范围 ε |
| `gamma` | 0.99 | 折扣因子 |
| `lam` | 0.95 | GAE 系数 |
| `learning_rate` | 1e-4 | 自适应（desired_kl=0.01） |
| `num_learning_epochs` | 5 | 每轮 epoch 数 |
| `num_mini_batches` | 4 | mini-batch 数 |
| `entropy_coef` | 0.01 | 熵奖励系数 |
| `value_loss_coef` | 1.0 | 价值损失系数 |
| `max_grad_norm` | 1.0 | 梯度裁剪范数 |
| `num_steps_per_env` | 24 | 每环境每轮步数 |
| `max_iterations` | 5000 | 最大迭代数 |
| `save_interval` | 100 | 保存间隔 |

### 5.2 策略网络

| 参数 | 值 |
|------|-----|
| Actor 隐藏层 | [512, 256, 128] |
| Critic 隐藏层 | [512, 256, 128] |
| 激活函数 | ELU |
| 初始噪声标准差 | 1.0 |
| Actor 观测归一化 | 启用 |
| Critic 观测归一化 | 启用 |

### 5.3 AMP 判别器超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `hidden_dims` | [1024, 512] | 判别器 trunk 维度 |
| `activation` | ELU | 激活函数 |
| `disc_learning_rate` | 1e-4 | 判别器学习率 |
| `grad_penalty_scale` | 10.0 | λ_gp |
| `disc_trunk_weight_decay` | 1e-3 | trunk 权重衰减 |
| `disc_linear_weight_decay` | 1e-1 | 输出层权重衰减 |
| `disc_max_grad_norm` | 1.0 | 判别器梯度裁剪 |
| `disc_obs_buffer_size` | 100 | 回放缓冲大小 |
| `style_reward_scale` | 1.5 | w_s |
| `task_style_lerp` | 0.6 | 任务/风格混合 α |
| `loss_type` | LSGAN | 判别器损失类型 |

### 5.4 对称性配置

| 参数 | 值 |
|------|-----|
| `use_data_augmentation` | True |
| `use_mirror_loss` | True |
| `mirror_loss_coeff` | 0.2 |
| `data_augmentation_func` | `rpo.compute_symmetric_states` |

### 5.5 环境参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_envs` | 8192 | 并行环境数 |
| `decimation` | 4 | 控制步降采样 |
| `sim.dt` | 0.005s | 物理仿真步长 |
| `step_dt` | 0.02s | 控制频率 50Hz |
| `episode_length_s` | 20s | 单回合时长 |
| `env_spacing` | 2.5m | 环境间距 |

### 5.6 训练流程

`AMPRunner.learn` 的单次迭代流程：

1. **Rollout**：采集 `num_steps_per_env=24` 步数据
   - 策略采样动作 → 环境步进 → `PPOAmp.process_env_step` 计算 style_reward 与 lerp 混合奖励 → 存入 RolloutStorage 与 CircularBuffer
2. **计算回报**：`PPOAmp.compute_returns`，使用 GAE 计算优势与回报
3. **策略更新**：`PPOAmp.update`
   - PPO 策略梯度 + 对称性数据增强/镜像损失
   - AMP 判别器梯度（LS-GAN 损失 + R1 梯度惩罚）
   - 两个优化器分别更新
4. **日志记录**：TensorBoard 记录 policy loss、disc loss、disc score、style reward 等
5. **保存模型**：每 100 次迭代保存 checkpoint，包含策略、判别器、归一化器与优化器状态

### 5.7 模型保存内容

```python
saved_dict = {
    "model_state_dict": ...,                # ActorCritic 策略
    "optimizer_state_dict": ...,            # PPO 优化器
    "amp_discriminator_state_dict": ...,    # AMP 判别器
    "amp_discriminator_normalizer_state_dict": ...,  # 判别器观测归一化器
    "amp_discriminator_optimizer_state_dict": ...,   # 判别器优化器
    "iter": ...,                            # 当前迭代数
    "infos": ...,                           # 额外信息
}
```

---

## 六、动作数据与重定向

### 6.1 动作数据集

AMP 动作数据位于 `robolab/data/motions/rpo_lab/`，为 pkl 格式，由 [GMR](https://github.com/Roboparty/GMR) 生成后经重定向处理。每条动作包含以下帧序列：

- `root_pos_w` / `root_quat`：根节点世界位置/姿态
- `root_vel_w` / `root_ang_vel_w`：根节点线速度/角速度
- `dof_pos` / `dof_vel`：关节位置/速度
- `key_body_pos_b`：关键体相对基座位置

### 6.2 关节重定向

GMR 生成的关节顺序与 URDF 一致（先左腿后右腿），但 Isaac Lab 内部按深度优先排序（左右交替），必须通过 `retarget/config/rpo.yaml` 重排：

**GMR 关节顺序**（先左后右）：

```
left_thigh_yaw → left_thigh_roll → left_thigh_pitch → left_knee →
left_ankle_pitch → left_ankle_roll → right_thigh_yaw → ... →
torso → left_arm_pitch → ...
```

**Isaac Lab 关节顺序**（深度优先交替）：

```
left_thigh_yaw → right_thigh_yaw → torso → left_thigh_roll →
right_thigh_roll → left_arm_pitch → right_arm_pitch → ...
```

重定向命令：

```bash
python robolab/scripts/tools/retarget/dataset_retarget.py
```

> GMR 生成的关节顺序与 URDF 一致，但与 Isaac Lab 不同，须通过 `retarget/config/rpo.yaml` 重排关节序列后再训练。

### 6.3 舞蹈动作数据集（LAFAN1）

舞蹈训练使用 Ubisoft LAFAN1 数据集中的舞蹈子集，经 GMR 重定向后存放于 `data/motions/rpo_dance_lab/`。

**数据来源**：[Ubisoft LaForge Animation Dataset](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)

**舞蹈文件**（8 段，50 FPS，23 DOF）：

| 文件名 | 说明 |
|--------|------|
| `dance1_subject1` ~ `dance1_subject3` | 舞蹈风格 1（3 个受试者） |
| `dance2_subject1` ~ `dance2_subject5` | 舞蹈风格 2（5 个受试者） |

**重定向流程**：

```
LAFAN1 BVH (external/dance_data/lafan1/)
    ↓ GMR 工具重定向 (bvh_lafan1_to_rpo.json IK 配置)
GMR 格式 (data/motions/rpo_dance_gmr_only/)
    ↓ dataset_retarget.py (rpo.yaml 关节重排)
Isaac Lab 格式 (data/motions/rpo_dance_lab/)
```

**GMR 工具配置**：

- IK 配置：`external/GMR/general_motion_retargeting/ik_configs/bvh_lafan1_to_rpo.json`
- 机器人注册：`external/GMR/general_motion_retargeting/params.py` 中 `rpo` 条目
- BVH 批量重定向脚本：`external/GMR/scripts/bvh_to_robot_dataset.py`

**舞蹈 vs 行走配置差异**：

| 参数 | 行走 (`RPOAmpEnvCfg`) | 舞蹈 (`RPODanceAmpEnvCfg`) |
|------|----------------------|---------------------------|
| `motion_data_dir` | `rpo_lab` | `rpo_dance_lab` |
| `track_lin_vel_xy_exp` 权重 | 1.25 | 0.3 |
| `track_ang_vel_z_exp` 权重 | 1.25 | 0.3 |
| `alive` 权重 | 0.15 | 0.3 |
| `lin_vel_x` 范围 | (-0.5, 2.5) | (-0.5, 0.5) |
| `lin_vel_y` 范围 | (-0.5, 0.5) | (-0.3, 0.3) |
| `ang_vel_z` 范围 | (-1.5, 1.5) | (-1.0, 1.0) |
| `max_iterations` | 5000 | 10000 |
| `experiment_name` | `rpo_amp` | `rpo_amp_dance` |

> 舞蹈训练降低速度跟踪权重、缩小速度范围，以原地动作为主，强调 AMP 风格奖励学习舞蹈风格。

---

## 七、环境搭建与运行

### 7.1 环境要求

| 依赖 | 版本 |
|------|------|
| Python | 3.11 |
| Isaac Sim | 5.1.0 |
| Isaac Lab | v2.3（兼容 4.5/5.0/5.1） |
| RSL-RL | 3.3.0 |
| CUDA | 12.8+（Blackwell GPU 需 cu128 版 torch） |
| OS | Ubuntu 22.04 x64 |
| NVIDIA 驱动 | ≥ 535 |

### 7.2 克隆与安装

```bash
git clone --recursive https://github.com/<your-username>/RPO_IsaacLab.git
cd RPO_IsaacLab
# 子目录为空时执行：
git submodule update --init --recursive
```

创建 conda 环境并安装 Isaac Sim 5.1（pip 方式）：

```bash
conda create -n rpo_isaaclab python=3.11 -y
conda activate rpo_isaaclab
pip install --upgrade pip
# Blackwell GPU 用 cu128 版 torch
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```

安装 Isaac Lab 源码扩展（指向 IsaacLab v2.3 源码目录）：

```bash
cd /path/to/IsaacLab_RPO
./isaaclab.sh --install none          # none: 跳过 RL 框架，rsl_rl 单独装
```

安装项目依赖：

```bash
cd RPO_IsaacLab
pip install -e ./rsl_rl               # rsl-rl-lib 3.3.0
pip install -e ./robolab              # robolab 扩展（含 mujoco、onnxruntime-gpu 等）
```

验证：

```bash
python robolab/scripts/tools/list_envs.py   # 列出 RPO-AMP 等任务
```

### 7.3 训练

所有任务均使用统一训练入口 `train.py`。日志输出至 `logs/rsl_rl/<experiment_name>/<时间戳>/`，含 `params/`、模型 `model_*.pt` 与 TensorBoard 事件。

**AMP 行走风格训练（平地）**：

```bash
python robolab/scripts/rsl_rl/train.py \
    --task=RPO-AMP --headless --num_envs=4096 --logger=tensorboard
```
日志输出至 `logs/rsl_rl/rpo_amp/`。默认 `max_iterations=5000`。

**AMP 行走风格训练（粗糙地形）**：

```bash
python robolab/scripts/rsl_rl/train.py \
    --task=RPO-AMP-Rough --headless --num_envs=4096 --logger=tensorboard
```
日志输出至 `logs/rsl_rl/rpo_amp_rough/`。默认 `max_iterations=40000`，critic 含 height_scan 特权信息。

**AMP 舞蹈风格训练**（LAFAN1 数据集，30000 轮）：

```bash
python robolab/scripts/rsl_rl/train.py \
    --task=RPO-AMP-Dance --headless --num_envs=2048 --logger=tensorboard
```
日志输出至 `logs/rsl_rl/rpo_amp_dance/`。RTX 5080 16GB 建议 `--num_envs=2048`。

**AMP 舞蹈单条动作训练**：

```bash
python robolab/scripts/rsl_rl/train.py \
    --task=RPO-AMP-Dance-Single --headless --num_envs=2048 --logger=tensorboard
```
日志输出至 `logs/rsl_rl/rpo_amp_dance_single/`。

**Direct RL 平地训练**：

```bash
python robolab/scripts/rsl_rl/train.py \
    --task=RPO-Flat --headless --num_envs=4096 --logger=tensorboard
```
日志输出至 `logs/rsl_rl/rpo_flat/`。默认 `max_iterations=9001`。

**Direct RL 粗糙地形训练**：

```bash
python robolab/scripts/rsl_rl/train.py \
    --task=RPO-Rough --headless --num_envs=4096 --logger=tensorboard
```
日志输出至 `logs/rsl_rl/rpo_rough/`。默认 `max_iterations=9001`，critic 含 height_scan 特权信息。

**BeyondMimic 训练**：

```bash
python robolab/scripts/rsl_rl/train.py \
    --task=RPO-BeyondMimic --headless --num_envs=4096 --logger=tensorboard
```

**Parkour 训练**：

```bash
python robolab/scripts/rsl_rl/train.py \
    --task=RPO-Parkour --headless --num_envs=4096 --logger=tensorboard
```

**Getup-Mimic 训练**：

```bash
python robolab/scripts/rsl_rl/train.py \
    --task=RPO-Getup-Mimic --headless --num_envs=4096 --logger=tensorboard
```

**注意力编码器（AttnEnc）训练**：

```bash
python robolab/scripts/rsl_rl/train.py \
    --task=RPO-AttnEnc --headless --num_envs=4096 --logger=tensorboard
```

**中断恢复（Interrupt）训练**：

```bash
python robolab/scripts/rsl_rl/train.py \
    --task=RPO-Interrupt --headless --num_envs=4096 --logger=tensorboard
```

> 覆盖默认迭代数：添加 `--max_iterations=40000`。续训已有模型：添加 `--resume --load_run=<run名称>`。

### 7.4 测试与回放

Play 命令使用单环境、关闭随机化和噪声，在 Isaac Sim 中以固定速度指令可视化策略表现。`--load_run` 指定训练日志目录名，`--checkpoint` 指定具体模型文件（可选，默认自动匹配最新模型）。

**AMP 行走（平地）**：

```bash
python robolab/scripts/rsl_rl/play_amp.py \
    --task=RPO-AMP-Play --load_run=<run名称> --num_envs=1
```

**AMP 行走（粗糙地形）**：

```bash
python robolab/scripts/rsl_rl/play_amp.py \
    --task=RPO-AMP-Rough-Play --load_run=<run名称> --num_envs=1
```

**AMP 舞蹈**：

```bash
python robolab/scripts/rsl_rl/play_amp.py \
    --task=RPO-AMP-Dance-Play --load_run=<run名称> --num_envs=1
```

**AMP 舞蹈（单条动作）**：

```bash
python robolab/scripts/rsl_rl/play_amp.py \
    --task=RPO-AMP-Dance-Single-Play --load_run=<run名称> --num_envs=1
```

**Direct RL 平地/粗糙/注意力编码器/中断恢复**：

```bash
python robolab/scripts/rsl_rl/play.py \
    --task=RPO-Flat --load_run=<run名称> --num_envs=1

python robolab/scripts/rsl_rl/play.py \
    --task=RPO-Rough --load_run=<run名称> --num_envs=1

python robolab/scripts/rsl_rl/play.py \
    --task=RPO-AttnEnc --load_run=<run名称> --num_envs=1

python robolab/scripts/rsl_rl/play.py \
    --task=RPO-Interrupt --load_run=<run名称> --num_envs=1
```

**BeyondMimic**：

```bash
python robolab/scripts/rsl_rl/play_bm.py \
    --task=RPO-BeyondMimic --load_run=<run名称> --num_envs=1
```

**Getup-Mimic**：

```bash
python robolab/scripts/rsl_rl/play_bm.py \
    --task=RPO-Getup-Mimic --load_run=<run名称> --num_envs=1
```

**Parkour**（导出 ONNX 时加 `--exportonnx`）：

```bash
python robolab/scripts/rsl_rl/play_parkour.py \
    --task=RPO-Parkour-Play --load_run=<run名称> --num_envs=1 --exportonnx
```

> Play 启动时自动将策略导出为 JIT/ONNX 格式至 `exported/` 目录。若使用绝对路径指定模型，用 `--checkpoint=<完整路径>` 替代 `--load_run`。

### 7.5 Sim2Sim 迁移验证

将训练导出的 JIT 策略（`exported/policy.pt`）载入 MuJoCo 进行 Sim2Sim 部署验证。

**Direct RL（Flat / Rough / AttnEnc / Interrupt）**：

```bash
# 平地
python robolab/scripts/mujoco/sim2sim_rpo.py \
    --load_model=<exported/policy.pt 路径>

# 粗糙地形（带 terrain MJCF 模型）
python robolab/scripts/mujoco/sim2sim_rpo.py \
    --load_model=<路径> --terrain

# 头模式（录制视频）
python robolab/scripts/mujoco/sim2sim_rpo.py \
    --load_model=<路径> --terrain --headless

# 注意力编码器策略
python robolab/scripts/mujoco/sim2sim_rpo_attn_enc.py \
    --load_model=<路径> [--terrain]

# 中断恢复策略
python robolab/scripts/mujoco/sim2sim_rpo_interrupt.py \
    --load_model=<路径> [--terrain]
```

**AMP / AMP-Rough**（含判别器策略需要 `--terrain` 加载完整 MJCF）：

```bash
# AMP 平地
python robolab/scripts/mujoco/sim2sim_rpo_amp.py \
    --load_model=<exported/policy.pt 路径> --terrain

# AMP 粗糙地形
python robolab/scripts/mujoco/sim2sim_rpo_amp.py \
    --load_model=<路径> --terrain --headless
```

**键盘控制**（MuJoCo GUI 模式）：

| 按键 | 功能 |
|------|------|
| `8` / `2` | 前后移动 |
| `4` / `6` | 左右平移 |
| `7` / `9` | 左右转向 |
| `F` | 切换相机跟随 |
| `0` | 重置机器人位姿 |

> `--load_model` 必须指向 Play 运行时自动导出的 `exported/policy.pt`（TorchScript 格式），而非训练保存的 `model_*.pt`。若尚未执行 Play，先跑一次 Play 命令即可触发导出。

### 7.6 Sim2Sim 实现细节

#### 控制循环

所有 `sim2sim_rpo*.py` 脚本遵循统一控制循环:

```
物理仿真 1kHz (dt=0.001)
    ↓ 每 decimation=20 步执行一次策略推理 (50Hz)
    ↓ 构建观测(78维/帧 × frame_stack帧)
    ↓ 策略前向 → 23维动作
    ↓ target_q = action × action_scale + default_pos
    ↓ usd2urdf 关节顺序重排
    ↓ PD 控制: tau = (target_q - q)·kp + (target_dq - dq)·kd
    ↓ 力矩限幅: tau = clamp(tau, -tau_limit, tau_limit)
    ↓ mujoco.mj_step()
```

#### PD 控制器

```python
def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd
```

- `target_dq = 0`(零速度目标)
- `kp`/`kd` 与训练时执行器刚度/阻尼一致
- 力矩限幅 `tau_limit` 与 URDF 一致

#### 观测构建

Actor 观测(78 维/帧,`frame_stack=10` 帧历史堆叠):

| 段 | 索引 | 内容 |
|----|------|------|
| 0:3 | `omega` | 基座角速度(基座坐标系) |
| 3:6 | `gvec` | 投影重力 |
| 6:9 | `cmd.vx/vy/dyaw` | 速度指令 |
| 9:32 | `q_obs` | 关节位置(相对默认,经 usd2urdf 重排) |
| 32:55 | `dq_obs` | 关节速度(经 usd2urdf 重排) |
| 55:78 | `action` | 上一步动作 |

历史堆叠通过 `hist_obs` 滑动窗口实现,首帧用当前观测填充全部历史。

#### usd2urdf 关节重排

Isaac Lab(USD)与 MuJoCo(URDF)关节顺序不同,通过 `cfg.robot_config.usd2urdf` 索引数组重排:

```python
for i in range(len(cfg.robot_config.usd2urdf)):
    q_obs[i] = q_[cfg.robot_config.usd2urdf[i]]  # USD → URDF
    target_pos[cfg.robot_config.usd2urdf[i]] = target_q[i]  # URDF → USD
```

#### 各脚本差异

| 脚本 | 适用任务 | 特殊处理 |
|------|----------|----------|
| `sim2sim_rpo.py` | Flat/Rough | 基础 PD 控制,可选 `--terrain` 加载地形 MJCF |
| `sim2sim_rpo_amp.py` | AMP/AMP-Rough | 需 `--terrain` 加载完整 MJCF(含判别器依赖) |
| `sim2sim_rpo_bm.py` | BeyondMimic | 加载参考动作,逐帧跟踪 |
| `sim2sim_rpo_parkour.py` | Parkour | 深度图渲染 + 编码器前向 + MoE 推理 |
| `play_motion_csv.py` | 动作回放 | 从 CSV 读取关节序列逐帧驱动 |

#### 键盘控制

| 按键 | 功能 |
|------|------|
| `8`/`2` | vx ±0.1 m/s |
| `4`/`6` | vy ±0.1 m/s |
| `7`/`9` | dyaw ±0.1 rad/s |
| `F` | 切换相机跟随 |
| `0` | 重置速度指令 |

#### 视频录制

`--headless` 模式使用离屏渲染器录制 MP4,结束后自动生成关节位置/力矩/速度的 matplotlib 对比图。

---

## 八、RSL-RL 算法库实现

`rsl_rl/` 是 RSL-RL 3.3.0 快照,包含 PPO/AMP/蒸馏等算法实现。

### 8.1 PPO 算法

#### update() 流程

`PPO.update()` 单次更新流程([ppo.py:204-440](file:///home/fatu08/roboparty_train/rsl_rl/rsl_rl/algorithms/ppo.py#L204-L440)):

1. **Mini-batch 生成**:从 RolloutStorage 采样 `num_learning_epochs × num_mini_batches` 个 batch
2. **对称性数据增强**(可选):`data_augmentation_func` 将 batch 扩充为原始 + 镜像,等价 2× 样本
3. **策略重评估**:`policy.act` + `policy.evaluate` 计算当前 log_prob、value、entropy
4. **KL 自适应学习率**:
   - $KL = \sum_i \log(\sigma/\sigma_{old}) + \frac{\sigma_{old}^2 + (\mu - \mu_{old})^2}{2\sigma^2} - 0.5$
   - $KL > 2 \cdot desired\_kl$: $lr \leftarrow \max(1e-5, lr/1.5)$
   - $KL < desired\_kl/2$: $lr \leftarrow \min(1e-2, lr \times 1.5)$
5. **截断策略损失**:
   - $ratio = \exp(\log\pi_\theta - \log\pi_{\theta_{old}})$
   - $L^{CLIP} = \max(-\hat{A} \cdot ratio, -\hat{A} \cdot \text{clip}(ratio, 1-\epsilon, 1+\epsilon))$
6. **价值损失**(截断形式):
   - $V_{clipped} = V_{target} + \text{clip}(V - V_{target}, -\epsilon, \epsilon)$
   - $L^{VF} = \max((V - R)^2, (V_{clipped} - R)^2)$
7. **辅助损失**(可选,AttnEnc):`policy.get_aux_loss()` (Critic 估计)
8. **对称性镜像损失**(可选):
   - 对镜像观测计算动作均值 $\mu_\pi(s_{sym})$
   - 与原始动作均值的镜像变换 $\text{sym}(\mu_\pi(s_{orig}))$ 计算 MSE
   - $L_{mirror} = c_{mirror} \cdot \text{MSE}(\mu_\pi(s_{sym}), \text{sym}(\mu_\pi(s_{orig})))$
9. **RND 损失**(可选):预测网络逼近目标网络嵌入
10. **梯度裁剪 + 优化器更新**

#### 总损失

$$L = L^{CLIP} + c_v L^{VF} - c_h H[\pi] + c_{aux} L^{aux} + c_{mirror} L^{mirror}$$

### 8.2 AMP 算法

`PPOAmp` 继承 PPO,额外维护判别器与风格奖励([ppo_amp.py](file:///home/fatu08/roboparty_train/rsl_rl/rsl_rl/algorithms/ppo_amp.py)):

- `process_env_step`:计算 style_reward → task_style_lerp 混合 → 存入 CircularBuffer
- `update`:PPO 策略更新 + 判别器梯度(LS-GAN + R1 梯度惩罚)分别更新
- 判别器使用独立 Adam 优化器,trunk 与线性层分别施加不同权重衰减

### 8.3 蒸馏算法

`Distillation` 实现策略蒸馏([distillation.py](file:///home/fatu08/roboparty_train/rsl_rl/rsl_rl/algorithms/distillation.py)):

- **act**:Student 输出部署动作,Teacher 输出特权动作(detach 作为监督目标)
- **update**:行为克隆损失 `MSE(student_actions, teacher_actions)`,BPTT 截断累积 `gradient_length=15` 步后统一反向传播
- **compute_returns**:无操作(蒸馏不用回报)

### 8.4 Runner 流程

#### OnPolicyRunner

`OnPolicyRunner.learn()` 单次迭代:

1. `agent.act(obs)`:策略采样动作
2. `env.step(actions)`:环境步进
3. `agent.process_env_step`:处理奖励、dones,存入 storage
4. `agent.compute_returns`:GAE 计算优势与回报
5. `agent.update`:策略/判别器更新
6. 日志记录 + 模型保存

#### AMPRunner

继承 OnPolicyRunner,额外处理:
- `disc`/`disc_demo` 观测组分别填入 CircularBuffer
- 风格奖励计算与 task_style_lerp 混合

### 8.5 存储模块

#### RolloutStorage

存储训练数据:obs、actions、rewards、dones、values、log_prob、mu、sigma、hidden_states。提供 `mini_batch_generator` 与 `recurrent_mini_batch_generator`。

#### CircularBuffer

AMP 判别器专用回放缓冲:
- `disc_obs_buffer_size=100`:存储最近 100 步判别器观测
- 滑动窗口覆盖,支持随机采样

### 8.6 网络模块

| 模块 | 用途 |
|------|------|
| `MLP` | 基础多层感知机 |
| `CNN` | 卷积网络(深度图编码) |
| `Memory` | LSTM/GRU 循环网络 |
| `AttentionEncoder` | 多头自注意力编码器 |
| `EmpiricalNormalization` | 在线经验归一化(均值/方差累积) |

### 8.7 Actor-Critic 变体

| 类名 | 特性 |
|------|------|
| `ActorCritic` | 基础 Actor-Critic |
| `ActorCriticRecurrent` | LSTM/GRU 循环 |
| `ActorCriticCnn` | CNN 编码(视觉观测) |
| `ActorCriticAttnEnc` | 注意力编码器 + 辅助任务 |
| `ActorCriticMoE` | 混合专家 |
| `ActorCriticEncoderMoE` | 编码器 + MoE(Parkour) |
| `ActorCriticEncoder` | 编码器(蒸馏) |
| `StudentTeacher` | 师生网络(蒸馏) |

---

## 九、传感器系统

### 9.1 分组射线投射器 GroupedRayCaster

`grouped_ray_caster/` 支持多组网格合并投射射线,用于高效地形/自碰撞感知:

- **射线碰撞组**:按环境分组的批量射线投射,避免不同环境间误命中
- **网格索引切片**:支持对特定网格子集投射
- **应用**:高度扫描(`height_scan`)、足底扫描(`feet_scanner`)

### 9.2 带噪声相机 NoisyCamera

`noisy_camera/` 为各类 IsaacLab 相机叠加图像噪声,支持 rgb/depth 通道分别加噪:

| 相机类型 | 说明 |
|----------|------|
| `NoisyRaycasterCamera` | 射线投射相机 + 噪声 |
| `NoisyGroupedRaycasterCamera` | 分组射线投射相机 + 噪声 |
| `NoisyMultiMeshRaycasterCamera` | 多网格射线投射相机 + 噪声 |
| `NoisyTiledCamera` | 平铺相机 + 噪声 |

噪声模块包括:缩放噪声、卷积模糊、柏林噪声、高斯噪声、椒盐噪声等。维护异步环形历史缓冲,便于策略网络堆叠多帧观测。

### 9.3 体积点云 VolumePoints

`volume_points/` 在刚体表面/体积内采样 3D 点云,用于碰撞检测与穿透量计算:

- `PointsGenerator`:在体坐标系生成点云 pattern(z_min/z_max 控制高度范围)
- `VolumePoints`:与虚拟障碍物求交输出穿透偏移量
- **应用**:Parkour 足部/膝部穿深检测、`feet_at_plane` 奖励

---

## 十、地形系统

### 10.1 高度场地形 HeightField

`height_field/hf_terrains.py` 包含多种地形生成函数:

| 地形类型 | 说明 |
|----------|------|
| Perlin 噪声平面 | 自然起伏地面 |
| 金字塔斜坡 | 台阶式上坡/下坡 |
| 楼梯 | 规则阶梯 |
| 障碍物 | 散布方块 |
| 波浪 | 正弦波形地面 |
| 踏石 | 离散落脚点 |

支持插值难度、Perlin 噪声增强和平台区域保留。

### 10.2 三角网格地形 Trimesh

`trimesh/mesh_terrains.py` 生成三角网格地形:

| 地形类型 | 说明 |
|----------|------|
| 运动匹配地形 | 与参考动作轨迹匹配的地形 |
| 浮箱地形 | 离散浮动平台 |
| 随机多箱地形 | 随机分布的箱体障碍 |

支持 Perlin 噪声地面与程序化几何结构生成。

### 10.3 虚拟障碍物 VirtualObstacle

`virtual_obstacle/edge_cylinder.py` 基于网格面邻接角检测尖锐边,通过 Plucker 坐标、RANSAC、贪心串联、射线投射和特征提取等算法合并/拟合边段,输出用于穿深计算的圆柱。

### 10.4 地形生成器与导入器

- `TerrainGenerator`:扩展 IsaacLab,记录每格子地形配置与索引,支持课程模式与随机模式排布
- `TerrainImporter`:扩展 IsaacLab,集成虚拟障碍物生成,在网格导入前基于网格生成虚拟边界障碍物

---

## 十一、工具模块

### 11.1 异步缓冲 Buffers

| 类 | 用途 |
|----|------|
| `AsyncCircularBuffer` | 异步环形缓冲,管理传感器/控制器历史观测,支持并发写入与非阻塞读取 |
| `AsyncDelayBuffer` | 异步延迟缓冲,带延迟的数据缓存与读取,模拟传感器/控制器延迟 |

### 11.2 噪声模型 Noise

- `noise_model.py`:高斯噪声、椒盐噪声等预设噪声函数,支持图像与系统层随机化
- `noise_cfg.py`:配置驱动噪声流水线,定义噪声类型、强度、适用通道

### 11.3 Warp GPU 加速

`warp/` 模块使用 NVIDIA Warp 框架实现 GPU 加速:

| 文件 | 用途 |
|------|------|
| `cylinder.py` | 圆柱空间网格,AABB 分桶加速点云-圆柱穿透量计算 |
| `kernels.py` | 核心 CUDA 内核,射线-三角形交点计算 |
| `raycast.py` | 异步 GPU 射线投射接口,用于高度场与虚拟障碍物检测 |

### 11.4 其他工具

| 文件 | 用途 |
|------|------|
| `perlin.py` | 柏林噪声生成(分形噪声),用于地形与图像噪声 |
| `math.py` | 向量/四元数操作(旋转、插值、几何变换) |
| `keyboard.py` | 键盘输入监听(Sim2Sim 交互控制) |
| `prims.py` | Omniverse Prim 操作工具 |

---

## 十二、实物部署 atom01_deploy

`atom01_deploy/` 是面向 RPO 机器人的 ROS2 实物部署框架,基于 [roboparty_deploy](https://github.com/Roboparty/roboparty_deploy)。

### 12.1 部署架构

| 组件 | 说明 |
|------|------|
| 主控 | Orange Pi 5 Plus (Ubuntu 22.04, 内核 5.10) 或 RDK X5 (Ubuntu 22.04, 内核 6.1.83) |
| 框架 | ROS2 Humble |
| 语言 | C++17 / Python 3 |
| 实时性 | 实时内核 + RTPRIO 98 |

### 12.2 模块结构

```
atom01_deploy/
├── src/
│   ├── imu/           # IMU 数据处理与姿态估计
│   ├── inference/     # 策略推理引擎(加载 JIT/ONNX 模型)
│   └── motors/        # 电机控制(PD 控制 + 多通道通信)
├── scripts/
│   ├── motion_player.py     # 动作 CSV 回放
│   ├── imu_py_example.py    # IMU 示例
│   └── set_zero.py          # 电机零位标定
└── tools/
    ├── start_robot.sh       # 一键启动
    └── create_ap/           # WiFi 热点(脱离网线调试)
```

### 12.3 环境配置

1. 安装 ROS2 Humble
2. 安装依赖:`ccache`、`fmt`、`spdlog`、`eigen3`、`screen`
3. (可选)手柄控制:`ros-humble-joy`
4. (可选)Python 脚本:`python3-yaml`、`python3-numpy`
5. Orange Pi 5 Plus 安装 5.10 实时内核:`cd assets && sudo apt install ./*.deb`
6. 配置实时优先级(`/etc/security/limits.conf`):
   ```
   <username>  -  rtprio  98
   <username>  -  memlock unlimited
   ```
7. 重启后验证:`ulimit -r` 应输出 98

### 12.4 部署流程

1. **训练 → Sim2Sim 验证 → 实物部署**:
   - Isaac Lab 训练策略 → Play 导出 JIT/ONNX
   - MuJoCo Sim2Sim 验证可迁移性
   - `inference` 模块加载模型,实时推理
2. **控制循环**:
   - IMU 读取基座姿态/角速度
   - 电机编码器读取关节位置/速度
   - 构建观测 → 策略推理 → 关节位置目标
   - 电机 PD 控制器驱动关节
3. **通信**:ROS2 话题/服务,主控与电机板通过总线通信

### 12.5 辅助工具

| 工具 | 用途 |
|------|------|
| `set_zero.py` | 电机零位标定,记录初始状态 |
| `motion_player.py` | 从 CSV 读取动作序列,实物复现仿真动作 |
| `imu_py_example.py` | IMU 数据读取与可视化示例 |
| `create_ap` | WiFi 热点,脱离网线调试 |
| `start_robot.sh` | 一键启动(传感器、电机、控制流) |

---

## 十三、常见问题

| 问题 | 解决方案 |
|------|----------|
| `robolab`/`rsl_rl` 为空目录 | 执行 `git submodule update --init --recursive` |
| 找不到 Isaac Lab import | 先激活 Isaac Sim 对应的 Python 环境再安装/运行 |
| 找不到 RPO task 名 | 运行 `list_envs.py`，复制实际输出的 task id |
| Blackwell GPU（RTX 50 系列）训练报 CUDA 错误 | 确认 torch 为 cu128 构建，`python -c "import torch;print(torch.version.cuda)"` 应为 12.8+ |
| AMP 训练发散/步态抖动 | 降低 `style_reward_scale` 或 `task_style_lerp`，检查动作数据关节顺序是否已重定向 |

---

## 十四、参考与致谢

基于以下开源工作构建：

- [IsaacLab](https://github.com/isaac-sim/IsaacLab) — NVIDIA 机器人仿真框架
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl) — 足式机器人 RL 训练库
- [legged_gym](https://github.com/leggedrobotics/legged_gym) — 足式机器人训练环境
- [legged_lab](https://github.com/zitongbai/legged_lab) — Isaac Lab 足式训练扩展
- [robot_lab](https://github.com/fan-ziqi/robot_lab) — Isaac Lab 机器人训练框架
- [InstinctLab](https://github.com/project-instinct/InstinctLab) — Isaac Lab 本能策略框架

AMP 算法源自 *Adversarial Motion Priors for Stylized Locomotion* (Peng et al., SIGGRAPH 2021)。

---

**致谢**：本项目框架与基础训练管线源自 [RoboParty](https://github.com/Roboparty) 开源项目，在此表示诚挚感谢。

---

## 十五、命令索引

本章汇总项目常用命令，作为快速参考。详细参数与原理见前文对应章节。

### 15.1 环境验证

```bash
python robolab/scripts/tools/list_envs.py
```

### 15.2 训练

| 任务 | 命令 |
|------|------|
| AMP 平地 | `python robolab/scripts/rsl_rl/train.py --task=RPO-AMP --headless --num_envs=4096 --logger=tensorboard` |
| AMP 粗糙地形 | `python robolab/scripts/rsl_rl/train.py --task=RPO-AMP-Rough --headless --num_envs=4096 --logger=tensorboard` |
| AMP 舞蹈 | `python robolab/scripts/rsl_rl/train.py --task=RPO-AMP-Dance --headless --num_envs=2048 --logger=tensorboard` |
| AMP 舞蹈单条 | `python robolab/scripts/rsl_rl/train.py --task=RPO-AMP-Dance-Single --headless --num_envs=2048 --logger=tensorboard` |
| BeyondMimic | `python robolab/scripts/rsl_rl/train.py --task=RPO-BeyondMimic --headless --num_envs=4096 --logger=tensorboard` |
| Getup-Mimic | `python robolab/scripts/rsl_rl/train.py --task=RPO-Getup-Mimic --headless --num_envs=4096 --logger=tensorboard` |
| Parkour | `python robolab/scripts/rsl_rl/train.py --task=RPO-Parkour --headless --num_envs=4096 --logger=tensorboard` |
| Direct RL 平地 | `python robolab/scripts/rsl_rl/train.py --task=RPO-Flat --headless --num_envs=4096 --logger=tensorboard` |
| Direct RL 粗糙地形 | `python robolab/scripts/rsl_rl/train.py --task=RPO-Rough --headless --num_envs=4096 --logger=tensorboard` |
| Attention Encoder | `python robolab/scripts/rsl_rl/train.py --task=RPO-AttnEnc --headless --num_envs=4096 --logger=tensorboard` |
| Interrupt Recovery | `python robolab/scripts/rsl_rl/train.py --task=RPO-Interrupt --headless --num_envs=4096 --logger=tensorboard` |

通用选项：`--max_iterations <N>`、`--resume --load_run=<目录>`、`--distributed`。

### 15.3 测试与回放

| 任务 | 脚本/命令 |
|------|-----------|
| AMP 平地 | `python robolab/scripts/rsl_rl/play_amp.py --task=RPO-AMP-Play --num_envs=1` |
| AMP 粗糙地形 | `python robolab/scripts/rsl_rl/play_amp.py --task=RPO-AMP-Rough-Play --num_envs=1` |
| AMP 舞蹈 | `python robolab/scripts/rsl_rl/play_amp.py --task=RPO-AMP-Dance-Play --num_envs=1` |
| AMP 舞蹈单条 | `python robolab/scripts/rsl_rl/play_amp.py --task=RPO-AMP-Dance-Single-Play --num_envs=1` |
| BeyondMimic | `python robolab/scripts/rsl_rl/play_bm.py --task=RPO-BeyondMimic --num_envs=1` |
| Getup-Mimic | `python robolab/scripts/rsl_rl/play_bm.py --task=RPO-Getup-Mimic --num_envs=1` |
| Direct RL 平地/粗糙/AttnEnc/Interrupt | `python robolab/scripts/rsl_rl/play.py --task=RPO-Flat/Rough/AttnEnc/Interrupt --num_envs=1` |
| Parkour | `python robolab/scripts/rsl_rl/play_parkour.py --task=RPO-Parkour-Play --num_envs=1 --exportonnx` |

通用选项：`--load_run=<目录>`、`--checkpoint=<路径>`。

### 15.4 Sim2Sim 验证

| 策略类型 | 命令 |
|----------|------|
| Direct RL 平地 | `python robolab/scripts/mujoco/sim2sim_rpo.py --load_model=<exported/policy.pt>` |
| Direct RL 粗糙地形 | `python robolab/scripts/mujoco/sim2sim_rpo.py --load_model=<路径> --terrain` |
| AttnEnc | `python robolab/scripts/mujoco/sim2sim_rpo_attn_enc.py --load_model=<路径> [--terrain]` |
| Interrupt | `python robolab/scripts/mujoco/sim2sim_rpo_interrupt.py --load_model=<路径> [--terrain]` |
| AMP / AMP-Rough | `python robolab/scripts/mujoco/sim2sim_rpo_amp.py --load_model=<路径> --terrain` |
| BeyondMimic | `python robolab/scripts/mujoco/sim2sim_rpo_bm.py --load_model=<路径> --motion_file=<motion.npz>` |
| Parkour | `python robolab/scripts/mujoco/sim2sim_rpo_parkour.py --depth_encoder=<0-depth_encoder.onnx> --actor=<actor.onnx>` |
| 动作 CSV 回放 | `python robolab/scripts/mujoco/play_motion_csv.py --motion_file=<motion.csv>` |

通用选项：`--headless`（关闭 GUI 并录屏）。

### 15.5 动作数据准备

```bash
# 批量重定向 GMR → Isaac Lab
python robolab/scripts/tools/retarget/dataset_retarget.py \
    --robot rpo --input_dir robolab/data/motions/rpo_gmr \
    --output_dir robolab/data/motions/rpo_lab \
    --config_file robolab/scripts/tools/retarget/config/rpo.yaml

# 单文件重定向
python robolab/scripts/tools/retarget/single_retarget.py \
    --robot rpo --input_file <input.pkl> --output_file <output.pkl> \
    --config_file robolab/scripts/tools/retarget/config/rpo.yaml

# BeyondMimic NPZ 可视化
python robolab/scripts/tools/beyondmimic/replay_npz.py -f <motion.npz>

# CSV → NPZ 转换
python robolab/scripts/tools/beyondmimic/csv_to_npz.py -f <input.csv> --input_fps 60
```

### 15.6 实机部署

```bash
# 模型导出与 ROS2 推理节点配置，详见 atom01_deploy 文档
cd atom01_deploy
# 一键启动实机控制流
./start_robot.sh
```
