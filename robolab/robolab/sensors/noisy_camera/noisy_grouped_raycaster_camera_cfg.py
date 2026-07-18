# 带噪声分组射线投射相机配置
# 混入 NoisyCameraCfgMixin 提供 noise_pipeline 与 data_histories 字段

from isaaclab.utils import configclass

from ..grouped_ray_caster import GroupedRayCasterCameraCfg
from .noisy_camera_cfg import NoisyCameraCfgMixin
from .noisy_grouped_raycaster_camera import NoisyGroupedRayCasterCamera


@configclass
class NoisyGroupedRayCasterCameraCfg(NoisyCameraCfgMixin, GroupedRayCasterCameraCfg):
    """
    Configuration class for the NoisyGroupedRayCasterCamera sensor and manages image transforms and their parameters.

    继承 GroupedRayCasterCameraCfg 并混入噪声配置，绑定 NoisyGroupedRayCasterCamera 作为运行时类。
    """

    class_type: type = NoisyGroupedRayCasterCamera
