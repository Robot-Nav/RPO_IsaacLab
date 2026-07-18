# 带噪声射线投射相机，在 RayCasterCamera 输出基础上叠加噪声用于 sim2real
# 通过混入 NoisyCameraMixin 复用噪声流水线与历史缓冲逻辑
# 适用于单网格场景的深度相机噪声仿真

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar, Literal

import isaacsim.core.utils.stage as stage_utils
import omni.physics.tensors.impl.api as physx
from isaacsim.core.prims import XFormPrim

import isaaclab.utils.math as math_utils
from isaaclab.sensors.camera import CameraData
from isaaclab.sensors.ray_caster import RayCasterCamera
from isaaclab.utils.warp import raycast_mesh

from .noisy_camera import NoisyCameraMixin

if TYPE_CHECKING:
    from .noisy_raycaster_camera_cfg import NoisyRayCasterCameraCfg


class NoisyRayCasterCamera(NoisyCameraMixin, RayCasterCamera):
    cfg: NoisyRayCasterCameraCfg

    def _initialize_impl(self):
        super()._initialize_impl()  # type: ignore
        # 相机本体初始化后构建噪声流水线与历史缓冲
        self.build_noise_pipeline()
        self.build_history_buffers()

    """
    Operations
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset the sensor and noise pipeline."""
        # 重置时同步重置带状态噪声算子与历史缓冲，避免跨 episode 状态泄漏
        super().reset(env_ids)
        self.reset_noise_pipeline(env_ids)
        self.reset_history_buffers(env_ids)

    """
    Implementation
    """

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data."""

        super()._update_buffers_impl(env_ids)
        # 先由父类完成射线投射得到原始数据，再施加噪声并更新历史
        self.apply_noise_pipeline_to_all_data_types(env_ids)
        self.update_history_buffers(env_ids)
