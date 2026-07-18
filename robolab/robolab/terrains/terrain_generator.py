# 程序化地形生成器，扩展 IsaacLab TerrainGenerator 以记录每格子地形配置与索引
# 用于课程学习：按行递增难度，按列固定子地形类型；支持随机模式与课程模式两种排布
# 关键产出：subterrain_index_grid (num_rows, num_cols) 与扁平化子地形配置列表，供训练时查询机器人生成位置对应的子地形参数
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGenerator

if TYPE_CHECKING:
    from .terrain_generator_cfg import FiledTerrainGeneratorCfg


class FiledTerrainGenerator(TerrainGenerator):
    """Terrain generator that records per-cell sub-terrain configs and indices.

    扩展基类以保留每格子地形的实际配置（含 difficulty、seed）和子地形类型索引网格，
    便于训练阶段根据机器人所在的 (row, col) 反查地形参数。
    """

    def __init__(self, cfg: FiledTerrainGeneratorCfg, device: str = "cpu"):
        # 子地形类型索引网格，按 (row, col) 索引；在生成过程中填充
        self.subterrain_index_grid: np.ndarray | None = None
        # 扁平化子地形配置列表：self._subterrain_specific_cfgs[row * num_cols + col]
        self._subterrain_specific_cfgs: list[SubTerrainBaseCfg] = []
        super().__init__(cfg, device)

    def _get_terrain_mesh(self, difficulty: float, cfg: SubTerrainBaseCfg):
        """Record the specific config for each sub-terrain mesh generation.

        覆写基类钩子：在生成每格子地形网格后，将本次使用的具体 difficulty 与 seed
        固化到配置副本中，避免后续查询时丢失实际生效参数。
        """
        mesh, origin = super()._get_terrain_mesh(difficulty, cfg)
        cfg = cfg.copy()
        cfg.difficulty = float(difficulty)
        cfg.seed = self.cfg.seed
        self._subterrain_specific_cfgs.append(cfg)
        return mesh, origin

    def _generate_random_terrains(self):
        """Add terrains with random sub-terrain type per grid cell.

        随机模式：每个网格按子地形 proportion 比例抽样类型，difficulty 在 difficulty_range 内均匀采样。
        适用于无课程学习的多样性训练。
        """
        proportions = np.array([sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()])
        proportions /= np.sum(proportions)
        sub_terrains_cfgs = list(self.cfg.sub_terrains.values())

        self.subterrain_index_grid = np.zeros((self.cfg.num_rows, self.cfg.num_cols), dtype=np.int32)
        for index in range(self.cfg.num_rows * self.cfg.num_cols):
            sub_row, sub_col = np.unravel_index(index, (self.cfg.num_rows, self.cfg.num_cols))
            # 按 proportion 概率抽取子地形类型，保证整体分布与配置比例一致
            sub_index = int(self.np_rng.choice(len(proportions), p=proportions))
            self.subterrain_index_grid[sub_row, sub_col] = sub_index
            difficulty = self.np_rng.uniform(*self.cfg.difficulty_range)
            mesh, origin = self._get_terrain_mesh(difficulty, sub_terrains_cfgs[sub_index])
            self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_terrains_cfgs[sub_index])

    def _generate_curriculum_terrains(self):
        """Add terrains with sub-terrain type fixed per column and difficulty along rows.

        课程模式：列方向按 proportion 切分固定子地形类型，行方向 difficulty 由低到高递增。
        difficulty = lower + (upper - lower) * (sub_row + uniform()) / num_rows
        保证训练初期落在低难度行，逐步推进至高难度。
        """
        proportions = np.array([sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()])
        proportions /= np.sum(proportions)

        sub_indices = []
        for index in range(self.cfg.num_cols):
            # 按累积比例确定该列所属的子地形类型，列方向上同类型连续分布
            sub_index = np.min(np.where(index / self.cfg.num_cols + 0.001 < np.cumsum(proportions))[0])
            sub_indices.append(sub_index)
        sub_indices = np.array(sub_indices, dtype=np.int32)
        sub_terrains_cfgs = list(self.cfg.sub_terrains.values())

        self.subterrain_index_grid = np.zeros((self.cfg.num_rows, self.cfg.num_cols), dtype=np.int32)
        for sub_col in range(self.cfg.num_cols):
            for sub_row in range(self.cfg.num_rows):
                sub_index = int(sub_indices[sub_col])
                self.subterrain_index_grid[sub_row, sub_col] = sub_index
                lower, upper = self.cfg.difficulty_range
                # 行号映射为难度系数，叠加均匀噪声避免难度边界过于整齐
                difficulty = (sub_row + self.np_rng.uniform()) / self.cfg.num_rows
                difficulty = lower + (upper - lower) * difficulty
                mesh, origin = self._get_terrain_mesh(difficulty, sub_terrains_cfgs[sub_index])
                self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_terrains_cfgs[sub_index])

    def get_subterrain_indices(
        self, row_ids: torch.Tensor | int, col_ids: torch.Tensor | int, device: str | torch.device | None = None
    ) -> torch.Tensor:
        """Return sub-terrain dict indices for grid cells (row, col), aligned with terrain_levels/types.

        与 terrain_levels/terrain_types 对齐，按 (row, col) 返回子地形类型索引张量，
        用于训练时根据机器人当前位置查询所属子地形类型。
        """
        if self.subterrain_index_grid is None:
            raise RuntimeError("subterrain_index_grid is not initialized. Terrain generation may have failed.")
        grid = torch.as_tensor(self.subterrain_index_grid, device=device, dtype=torch.long)
        return grid[row_ids, col_ids]

    @property
    def subterrain_specific_cfgs(self) -> list[SubTerrainBaseCfg]:
        """Get the specific configurations for all sub-terrains."""
        return self._subterrain_specific_cfgs.copy()

    def get_subterrain_cfg(
        self, row_ids: int | torch.Tensor, col_ids: int | torch.Tensor
    ) -> list[SubTerrainBaseCfg] | SubTerrainBaseCfg | None:
        """Get the specific configuration for a sub-terrain by its row and column index.

        按 row * num_cols + col 一维索引查询子地形实际生效配置；支持单点或批量查询。
        """
        num_cols = self.cfg.num_cols
        if isinstance(row_ids, torch.Tensor):
            row_ids = row_ids.cpu().numpy()
        if isinstance(col_ids, torch.Tensor):
            col_ids = col_ids.cpu().numpy()
        idx = row_ids * num_cols + col_ids
        if isinstance(idx, np.ndarray):
            return [
                self._subterrain_specific_cfgs[i] if 0 <= i < len(self._subterrain_specific_cfgs) else None for i in idx
            ]
        if isinstance(idx, (int, np.integer)):
            i = int(idx)
            return self._subterrain_specific_cfgs[i] if 0 <= i < len(self._subterrain_specific_cfgs) else None
        return None
