# 带噪声多网格射线投射相机配置
# 混入 NoisyCameraCfgMixin 提供 noise_pipeline 与 data_histories 字段

from isaaclab.sensors.ray_caster import MultiMeshRayCasterCameraCfg
from isaaclab.utils import configclass

from .noisy_camera_cfg import NoisyCameraCfgMixin
from .noisy_multi_mesh_ray_caster_camera import NoisyMultiMeshRayCasterCamera


@configclass
class NoisyMultiMeshRayCasterCameraCfg(NoisyCameraCfgMixin, MultiMeshRayCasterCameraCfg):
    """
    Configuration class for the NoisyMultiMeshRayCasterCamera sensor and manages image transforms and their parameters.

    继承 MultiMeshRayCasterCameraCfg 并混入噪声配置，绑定 NoisyMultiMeshRayCasterCamera 作为运行时类。
    """

    class_type: type = NoisyMultiMeshRayCasterCamera
