# 带噪声射线投射相机配置
# 混入 NoisyCameraCfgMixin 提供 noise_pipeline 与 data_histories 字段

from isaaclab.sensors.ray_caster import RayCasterCamera, RayCasterCameraCfg, RayCasterCfg
from isaaclab.utils import configclass

from .noisy_camera_cfg import NoisyCameraCfgMixin
from .noisy_raycaster_camera import NoisyRayCasterCamera


@configclass
class NoisyRayCasterCameraCfg(NoisyCameraCfgMixin, RayCasterCameraCfg):
    """
    Configuration class for the NoisyRayCasterCamera sensor and manages image transforms and their parameters.

    继承 RayCasterCameraCfg 并混入噪声配置，绑定 NoisyRayCasterCamera 作为运行时类。
    """

    class_type: type = NoisyRayCasterCamera
