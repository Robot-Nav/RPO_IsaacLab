# 二维柏林噪声与分形噪声生成，用于构建自然地形高度图
# 经典Perlin算法：网格顶点随机梯度 + 缓和插值，分形版叠加多octave增强细节
import numpy as np
from typing import Sequence


def generate_perlin_noise_2d(shape: Sequence[int], res: Sequence[int]) -> np.ndarray:
    # 标准二维柏林噪声：在res网格顶点生成随机梯度，网格内用缓和函数插值
    # 输出范围[0,1]，shape必须能被res整除
    def f(t):
        # 五次缓和函数，保证C2连续避免低频伪影
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    # 每个网格顶点分配一个单位圆上的随机梯度向量
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    # 各顶点梯度与到点的方向向量点积，得到四个角的贡献
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    # Interpolation
    # 用缓和函数t对四个角的贡献做双线性插值
    t = f(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1) * 0.5 + 0.5


def generate_fractal_noise_2d(
    xSize=20,
    ySize=20,
    xSamples=1600,
    ySamples=1600,
    frequency=10,
    fractalOctaves=2,
    fractalLacunarity=2.0,
    fractalGain=0.25,
    zScale=0.23,
    centering=False,  # If True, the noise will be centered around 0
) -> np.ndarray:
    # 分形布朗运动：叠加多octave的柏林噪声，频率倍增、幅度衰减，生成多尺度地形细节
    xScale = int(frequency * xSize)
    yScale = int(frequency * ySize)
    amplitude = 1

    # check to make sure the sample shape is the multiple of scale shape
    # 采样数需能容纳最高octave的网格分辨率，避免插值越界
    expected_xSamples = int(xScale * (fractalLacunarity**fractalOctaves))
    expected_ySamples = int(yScale * (fractalLacunarity**fractalOctaves))

    if xSamples > expected_xSamples or ySamples > expected_ySamples:
        raise RuntimeError(
            "Situation not checked, using expected_*Samples is in case the *Samples is not the multiple of *Size"
        )

    noise = np.zeros((expected_xSamples, expected_ySamples))
    for _ in range(fractalOctaves):
        noise += amplitude * generate_perlin_noise_2d(noise.shape, (xScale, yScale)) * zScale
        amplitude *= fractalGain
        xScale, yScale = int(fractalLacunarity * xScale), int(fractalLacunarity * yScale)

    if centering:
        noise -= np.mean(noise)

    return noise[:xSamples, :ySamples].copy()
