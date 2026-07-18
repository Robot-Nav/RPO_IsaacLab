# 带噪声多网格射线投射相机，在 MultiMeshRayCasterCamera 基础上叠加噪声
# 用于多网格场景的深度相机噪声仿真
# 通过 NoisyCameraMixin 复用噪声流水线与历史缓冲

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.sensors.ray_caster.multi_mesh_ray_caster_camera import MultiMeshRayCasterCamera

from .noisy_camera import NoisyCameraMixin

if TYPE_CHECKING:
    from .noisy_multi_mesh_ray_caster_camera_cfg import NoisyMultiMeshRayCasterCameraCfg


class NoisyMultiMeshRayCasterCamera(NoisyCameraMixin, MultiMeshRayCasterCamera):
    cfg: NoisyMultiMeshRayCasterCameraCfg

    def _initialize_impl(self):
        super()._initialize_impl()  # type: ignore
        # 多网格射线投射本体初始化完成后构建噪声流水线与历史缓冲
        self.build_noise_pipeline()
        self.build_history_buffers()

    """
    Operations
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset the sensor and noise pipeline."""
        # 重置时同步清理噪声状态与历史缓冲，避免跨 episode 数据污染
        super().reset(env_ids)
        self.reset_noise_pipeline(env_ids)
        self.reset_history_buffers(env_ids)

    """
    Implementation
    """

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data."""

        super()._update_buffers_impl(env_ids)
        # 父类完成多网格射线投射后，施加噪声并更新历史缓冲
        self.apply_noise_pipeline_to_all_data_types(env_ids)
        self.update_history_buffers(env_ids)
