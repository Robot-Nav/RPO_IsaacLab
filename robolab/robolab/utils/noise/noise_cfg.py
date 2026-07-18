# 图像噪声配置类集合，与noise_model中的噪声函数/模型一一对应
# 通过configclass机制声明各噪声参数，供环境配置按需组合启用
import torch
from typing import Callable, Optional

from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseCfg

from .noise_model import (
    ImageNoiseModel,
    LatencyNoiseModel,
    SensorDeadNoiseModel,
    blind_spot_noise,
    crop_and_resize,
    depth_artifact_noise,
    depth_contour_noise,
    depth_normalization,
    depth_sky_artifact_noise,
    depth_stero_noise,
    gaussian_blur_noise,
    perlin_noise,
    pixel_failure_noise,
    random_gaussian_noise,
    random_conv_noise,
    range_based_gaussian_noise,
    scale_randomization_noise,
    stereo_too_close_noise,
    stereo_fusion_noise,
)


@configclass
class ImageNoiseCfg(NoiseCfg):
    func: Callable[[torch.Tensor, NoiseCfg, torch.Tensor], torch.Tensor] | type[ImageNoiseModel] = ImageNoiseModel
    """ The callable function to apply noise to the image.
    The function should take two arguments:
        - the image in shape (N_, H, W, C) where N_ = len(env_ids)
        - the cfg object (as this configclass's object).
        - the env_ids tensor for specifying the environment ids
    """
    device: str | torch.device = "cpu"


# -- below are configuration classes to different types of noise applied to depth image


@configclass
class DepthContourNoiseCfg(ImageNoiseCfg):
    # 深度轮廓噪声配置，检测深度跳变边缘并置零
    contour_threshold: float = 2.0
    maxpool_kernel_size: int = 1
    func = depth_contour_noise
    """ The noise model class to apply depth contour noise. """


@configclass
class DepthArtifactNoiseCfg(ImageNoiseCfg):
    # 深度伪影配置，随机生成矩形条带状无效区域
    artifacts_prob: float = 0.0001  # should be very low
    artifacts_height_mean_std: list[float] = [2, 0.5]
    artifacts_width_mean_std: list[float] = [2, 0.5]
    noise_value: float = 0.0
    func = depth_artifact_noise


@configclass
class DepthSteroNoiseCfg(ImageNoiseCfg):
    # 立体相机深度噪声配置，模拟D435i的远近噪声与过近伪影
    stero_far_distance: float = 2.0
    stero_min_distance: float = 0.12  # when using (240,424) resolution
    stero_far_noise_std: float = 0.08  # the noise std of pixels that are greater than stero_far_noise_distance
    stero_near_noise_std: float = 0.02  # the noise std of pixels that are less than stero_far_noise_distance

    stero_full_block_artifacts_prob: float = (
        0.001  # the probability of adding artifacts to pixels that are less than stero_min_distance
    )
    stero_full_block_values: list = [0.0, 0.25, 0.5, 1.0, 3.0]
    stero_full_block_height_mean_std: list = [62, 1.5]
    stero_full_block_width_mean_std: list = [3, 0.01]

    stero_half_block_spark_prob: float = 0.02
    stero_half_block_value: int = 3000  # to the maximum value directly

    func = depth_stero_noise


@configclass
class DepthSkyArtifactNoiseCfg(ImageNoiseCfg):
    # 天空区域伪影配置，模拟相机指向天空/天花板时的条带失效
    sky_artifacts_prob: float = 0.0001
    sky_artifacts_far_distance: float = 2.0  # pixels greater than this distance will be viewed as sky
    sky_artifacts_values: list = [0.6, 1.0, 1.2, 1.5, 1.8]
    sky_artifacts_height_mean_std: list = [2, 3.2]
    sky_artifacts_width_mean_std: list = [2, 3.2]

    func = depth_sky_artifact_noise


@configclass
class LatencyNoiseCfg(ImageNoiseCfg):
    # 传感器延迟配置，支持四种延迟分布与两种重采样频率
    history_length: int = 5

    # sample frequency related settings

    sample_frequency: Optional[str] = None
    """Optional frequency setting for resampling delays
        - None (default): no resampling
        - "every_n_steps": resample every n steps, with n specified by `sample_frequency_steps`
        - "random_with_probability": resample with a certain probability, specified by `sample_probability`
    """
    sample_frequency_steps: int = 50  # used when sample_frequency is "every_n_steps"
    sample_frequency_steps_offset: int = 5  # the offset for the sample frequency steps

    sample_probability: float = 0.1  # used when sample_frequency is "random_with_probability"

    # sample distribution related settings

    latency_distribution: Optional[str] = "constant"
    """Optional distribution for sampling latency steps
        - "uniform": (uniform distribution), with range specified by 'latency_range'
        - "normal": normal distribution, with mean and std specified by 'latency_mean_std', and range specified by 'latency_range'
        - "choice": choose from a predefined list of latency steps, specified by 'latency_choices'
        - "constant" (default): use a fixed number of steps, specified by 'latency_steps'
    """
    latency_range: tuple[int, int] = (1, history_length)

    latency_mean_std: tuple[float, float] = (3, 1)  # used when latency_distribution is "normal"

    latency_choices: list[int] = [1, 2, 3, 4, 5]  # used when latency_distribution is "choice"
    latency_choices_probabilities: Optional[list[float]] = (
        None  # probabilities for each choice, default to None (uniform distribution)
    )
    # The number of probabilities must match the number of choices.

    latency_steps: int = 5  # used when latency_distribution is "constant"

    func: type[LatencyNoiseModel] = LatencyNoiseModel


@configclass
class DepthNormalizationCfg(ImageNoiseCfg):
    """Configuration for normalizing depth values to a specific range."""

    depth_range: tuple[float, float] = (0.0, 10.0)
    """Depth value range for normalization."""

    normalize: bool = True
    """Whether to normalize depth values to the range [0, 1] after clipping."""

    output_range: tuple[float, float] = (0.0, 1.0)
    """Range to normalize depth values to."""

    func = depth_normalization
    """The noise model class to apply depth normalization."""


@configclass
class CropAndResizeCfg(ImageNoiseCfg):
    """Configuration for cropping and resizing images."""

    crop_region: tuple[int, int, int, int] = (0, 0, 0, 0)
    """The size to be cropped, corresponding to up, down, left, right, respectively."""

    resize_shape: tuple[int, int] = None
    """The size to be reshape to, corresponding to height, width, respectively."""

    func = crop_and_resize


@configclass
class BlindSpotNoiseCfg(ImageNoiseCfg):
    """Configuration for adding blind spot noise (zeroing out regions of the image)."""

    crop_region: tuple[int, int, int, int] = (0, 0, 0, 0)
    """The size to be blind spotted, corresponding to up, down, left, right, respectively."""

    func = blind_spot_noise


@configclass
class GaussianBlurNoiseCfg(ImageNoiseCfg):
    """Configuration for adding Gaussian blur noise to images."""

    kernel_size: int = 3
    """The size of the Gaussian kernel. It should be an odd number."""

    sigma: float = 1.0
    """The standard deviation of the Gaussian kernel."""

    func = gaussian_blur_noise


@configclass
class RandomGaussianNoiseCfg(ImageNoiseCfg):
    """Configuration for adding random Gaussian noise to images."""

    probability: float = 0.1
    """The probability of applying the Gaussian noise."""

    noise_mean: float = 0.0
    """The mean of the Gaussian noise."""

    noise_std: float = 1.0
    """The standard deviation of the Gaussian noise."""

    func = random_gaussian_noise


@configclass
class StereoFusionNoiseCfg(ImageNoiseCfg):
    """Configuration for stereo fusion consistency-check hole simulation."""
    # 立体一致性检查空洞配置，遮挡边缘与弱纹理区域生成空洞

    apply_probability: float = 0.5
    disparity_grad_threshold: float = 0.08
    texture_var_threshold: float = 0.0005
    hole_probability: float = 0.08
    hole_kernel_size: int = 3
    hole_value: float = 0.0

    func = stereo_fusion_noise


@configclass
class RandomConvNoiseCfg(ImageNoiseCfg):
    """Configuration for random 3x3 convolution distortion."""
    # 随机卷积畸变配置，模拟局部光学畸变

    apply_probability: float = 0.5
    kernel_std: float = 0.12
    center_weight: float = 1.0

    func = random_conv_noise


@configclass
class PerlinNoiseCfg(ImageNoiseCfg):
    """Configuration for multi-octave spatially-correlated Perlin-style noise."""
    # 分形柏林噪声配置，生成空间相关的连续噪声叠加到图像

    apply_probability: float = 0.5
    octaves: int = 4
    base_frequency: float = 8.0
    lacunarity: float = 2.0
    persistence: float = 0.5
    amplitude: float = 1.0
    noise_std: float = 0.02

    func = perlin_noise


@configclass
class ScaleRandomizationNoiseCfg(ImageNoiseCfg):
    """Configuration for random depth scale perturbation."""
    # 深度尺度随机化配置，模拟标定漂移导致的全局尺度偏差

    apply_probability: float = 1.0
    scale_min: float = 0.90
    scale_max: float = 1.10

    func = scale_randomization_noise


@configclass
class PixelFailureNoiseCfg(ImageNoiseCfg):
    """Configuration for dead/saturated pixel simulation."""
    # 坏点/饱和像素配置，模拟传感器死像素与过曝像素

    apply_probability: float = 0.7
    dead_pixel_prob: float = 0.001
    saturated_pixel_prob: float = 0.001
    dead_value: float = 0.0
    saturated_value: float = 1.0

    func = pixel_failure_noise


@configclass
class RangeBasedGaussianNoiseCfg(ImageNoiseCfg):
    """Configuration for adding range-based Gaussian noise to images."""
    # 范围受限高斯噪声配置，仅在数值落在[min_value, max_value]区间的像素上施加噪声

    min_value: float | None = None
    """The minimum value of the range."""

    max_value: float | None = None
    """The maximum value of the range."""

    noise_std: float = 1.0
    """The standard deviation of the Gaussian noise."""

    func = range_based_gaussian_noise


@configclass
class StereoTooCloseNoiseCfg(ImageNoiseCfg):
    """Configuration for adding stereo too close noise to images."""
    # 立体相机过近噪声配置，过近区域分full_block与half_block两种伪影模式

    close_threshold: float = 0.12
    """The threshold of the too close distance."""

    # full block related settings
    full_block_height_mean_std: tuple[float, float] = (62, 1.5)
    full_block_width_mean_std: tuple[float, float] = (3, 0.01)
    full_block_values: list[float] = [0.0, 0.25, 0.5, 1.0, 3.0]
    full_block_artifacts_prob: float = 0.008

    # half block related settings
    half_block_height_mean_std: tuple[float, float] = (2, 3.2)
    half_block_width_mean_std: tuple[float, float] = (2, 3.2)
    half_block_value: float = 30  # to the maximum value directly
    half_block_spark_prob: float = 0.02

    func = stereo_too_close_noise


@configclass
class SensorDeadNoiseCfg(ImageNoiseCfg):
    """Configuration for adding sensor dead behavior, which might be autonomous restarted.
    Thus causing some frames of non-refreshed data.
    """
    # 传感器掉线配置，按概率触发掉线并持续若干帧输出冻结数据

    dead_probability: float = 0.01
    """The probability of the sensor dead."""

    dead_frames: int | list[int] = 90  # 1.5 second at 60Hz
    """The number of frames to be non-refreshed (before the sensor is restarted).
    Can be a single number or a list of numbers to be uniformly selected from.
    """

    func = SensorDeadNoiseModel
