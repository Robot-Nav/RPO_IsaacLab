# 带噪声相机配置 Mixin，定义噪声流水线与历史缓冲参数
# 噪声流水线按字典插入顺序执行，历史缓冲以 data_type 为键指定长度

from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseCfg


@configclass
class NoisyCameraCfgMixin:
    """
    Configuration class for the NoisyCamera sensor and manages image transforms and their parameters.

    噪声相机配置混入类，作为各带噪声相机 Cfg 的公共基类。
    """

    noise_pipeline: dict[str, NoiseCfg] = {}
    """Configurations for the noise pipeline. The keys can be self-defined names.
    NOTE: All enabled items in cfg.data_types will be applied with the noise pipeline.
    NOTE: If you want to add history to the noised output, you need to specify the data_type as the one with _noised postfix.
    NOTE: After python 3.8, the dict is ordered by insertion order.
    """

    data_histories: dict[str, int] = {}
    """ Configurations for adding history to specified data_types. Please specify which `data_type`
    you want to add history and the history length. The stacked historical history observation will
    be placed in sensor.data[f"{data_type}_history"]
    NOTE: If you want to add history to the noised output, you need to specify the data_type as the one with _noised postfix.
    """
