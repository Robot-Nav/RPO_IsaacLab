# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


"""
Configuration classes defining the different terrains available. Each configuration class must
inherit from ``isaaclab.terrains.terrains_cfg.TerrainConfig`` and define the following attributes:

- ``name``: Name of the terrain. This is used for the prim name in the USD stage.
- ``function``: Function to generate the terrain. This function must take as input the terrain difficulty
  and the configuration parameters and return a `tuple with the `trimesh`` mesh object and terrain origin.
"""

# 地形生成器配置：定义GRAVEL/ROUGH/ROUGH_HARD三套地形，按子地形比例混合生成训练场地
# GRAVEL：平地为主+少量随机起伏，用于Flat任务；ROUGH：楼梯/坡/网格等中等难度+课程；ROUGH_HARD：高难度含深坑/星形/间隙

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

# 平地+少量碎石：70%平面+30%随机起伏，无课程（curriculum=False），用于RPO-Flat
GRAVEL_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=False,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.3, noise_range=(-0.02, 0.04), noise_step=0.02, border_width=0.25
        ),
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.7
        ),
    },
)

# Rough地形：含金字塔/倒金字塔楼梯、坡、网格、随机起伏、平面，启用课程学习
ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    use_cache=False,
    sub_terrains={
        # 倒金字塔楼梯：step_width决定踏面宽度，step_height_range随课程线性增加
        "inv_pyramid_stairs_25": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.05, 0.15),
            step_width=0.25,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "inv_pyramid_stairs_35": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.05, 0.15),
            step_width=0.35,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        # 正向金字塔楼梯：上行台阶
        "pyramid_stairs_25": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.05, 0.15),
            step_width=0.25,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_35": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.05, 0.15),
            step_width=0.35,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        # 斜坡与反向斜坡
        "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.1, 0.3), border_width=1.0, platform_width=2.0
        ),
        "inv_slope": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.1, 0.3), border_width=1.0, platform_width=2.0, inverted=True
        ),
        # 网格地形：离散障碍物阵列
        "grid": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.1, grid_width=0.45, grid_height_range=(0.05, 0.15), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1, noise_range=(-0.02, 0.04), noise_step=0.02, border_width=1.0
        ),
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.4
        ),
    },
)

# Rough Hard高难度地形：在Rough基础上增大阶梯高度（0.2m上限），增加深坑/星形/间隙/踏石
ROUGH_HARD_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    use_cache=False,
    sub_terrains={
        "inv_pyramid_stairs_25": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.05, 0.2),
            step_width=0.25,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "inv_pyramid_stairs_35": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.05, 0.2),
            step_width=0.35,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_25": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.05, 0.2),
            step_width=0.25,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_35": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.05, 0.2),
            step_width=0.35,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.05, slope_range=(0.1, 0.3), border_width=1.0, platform_width=2.0
        ),
        "inv_slope": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.05, slope_range=(0.1, 0.3), border_width=1.0, platform_width=2.0, inverted=True
        ),
        "grid": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.1, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1, noise_range=(-0.02, 0.04), noise_step=0.02, border_width=1.0
        ),
        # 深坑地形：double_pit=True增加难度
        "high_platform": terrain_gen.MeshPitTerrainCfg(
            proportion=0.1, pit_depth_range=(0.1, 0.3), platform_width=2.0, double_pit=True
        ),
        # 星形障碍：多放射状栏杆，bar_height_range=10为大高度差
        "star": terrain_gen.MeshStarTerrainCfg(
            proportion=0.1, num_bars=12, bar_width_range=(0.25, 0.4), bar_height_range=(10.0, 10.0), platform_width=2.0
        ),
        # 间隙地形：需跨越的水平空隙，命中点为nan需在obs中处理
        "gap": terrain_gen.MeshGapTerrainCfg(
            proportion=0.15, gap_width_range=(0.1, 0.3), platform_width=2.0
        ), # points in gap are nan
        # 踏石：离散可落脚点，stone_height_max=0表示与地面齐平
        "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=0.15,
            stone_height_max=0.0,
            stone_width_range=(0.3, 0.4),
            stone_distance_range=(0.1, 0.2),
            platform_width=2.0,
            border_width=1.0,
        ),
    },
)
