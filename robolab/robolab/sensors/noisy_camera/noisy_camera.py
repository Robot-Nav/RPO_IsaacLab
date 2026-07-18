# 带噪声相机 Mixin 基类，叠加图像噪声并维护时序历史缓冲，用于 sim2real 域随机化
# 通过流水线串联多个 ImageNoiseCfg 算子，支持对 rgb/depth 等数据通道分别加噪
# 提供异步环形历史缓冲（AsyncCircularBuffer），便于策略网络堆叠多帧观测

from __future__ import annotations

import inspect
import torch
from prettytable import PrettyTable
from typing import Sequence

from isaaclab.utils import string_to_callable

from robolab.utils.buffers import AsyncCircularBuffer
from robolab.utils.noise import ImageNoiseCfg


class NoisyCameraMixin:  # as a subclass of SensorBase
    """
    This mixin class adds noise to the camera data.

    噪声相机混入基类，以多继承方式为各类 IsaacLab 相机叠加图像噪声。
    维护两条独立流水线：
      - noise_pipeline：按配置顺序串联的噪声算子，支持无状态函数与带状态类
      - data_histories：基于异步环形缓冲的多帧历史堆叠，输出 {data_type}_history
    """

    def __str__(self) -> str:
        return_ = super().__str__()
        noise_info_table = PrettyTable()
        noise_info_table.field_names = ["Noise Name", "Noise Cfg Name"]
        for noise_name, noise_cfg in self.cfg.noise_pipeline:  # type: ignore
            noise_info_table.add_row([noise_name, noise_cfg.__name__])
        return_ += "\n" + str(noise_info_table)
        history_info_table = PrettyTable()
        history_info_table.field_names = ["History Name", "History Length"]
        for history_name, history_length in self.cfg.data_histories.items():  # type: ignore
            history_info_table.add_row([history_name, history_length])
        return_ += "\n" + str(history_info_table)
        return return_

    """
    Noise Pipeline
    """

    def build_noise_pipeline(self):
        self.noise_pipeline: Sequence[ImageNoiseCfg] | list[ImageNoiseCfg] = []
        """Build the noise pipeline based on the configuration."""
        # 根据配置中的 noise_pipeline 字典逐项实例化噪声算子。
        # func 既可以是可调用对象，也可以是字符串形式的导入路径，还可以是带状态的类（用于维持跨帧噪声状态）。

        for noise_name, noise_cfg in self.cfg.noise_pipeline.items():  # type: ignore
            # Check if the noise configuration is valid
            if not isinstance(noise_cfg, ImageNoiseCfg):
                raise ValueError(f"Invalid noise configuration for {noise_name}: {noise_cfg}")

            noise_cfg.device = self.device
            # Ensure the device is set correctly if the function is not a class

            if isinstance(noise_cfg.func, str):
                noise_cfg.func = string_to_callable(noise_cfg.func)

            if inspect.isclass(noise_cfg.func):
                # If the function is a class, instantiate it
                # 带状态的噪声（如时域相关噪声）需要保留实例，因此对类对象做一次性实例化
                noise_cfg.func = noise_cfg.func(noise_cfg, num_envs=self.num_instances, device=self.device)

            # Add the noise configuration to the pipeline
            self.noise_pipeline.append(noise_cfg)

        # apply the noise pipeline to the initialized output buffers for noised output
        # 在初始化阶段对全量数据跑一遍噪声流水线，生成 {data_type}_noised 输出键
        for data_type in self.cfg.data_types:
            self._data.output[f"{data_type}_noised"] = self.apply_noise_pipeline(
                self._data.output[data_type], env_ids=self._ALL_INDICES
            )

    def apply_noise_pipeline(self, data: torch.Tensor, env_ids: torch.Tensor | Sequence[int]) -> torch.Tensor:
        """Apply noise to the data(image).
        ## NOTE: The input data is only for selected envs (by env_ids).
        Args:
            data: The data to which noise will be applied. if Image, the shape should be (N_, H, W, C) for all environments.
        """
        # 输入会被原地修改，因此先 clone 一份避免污染上游数据
        # Check if the noise sequence is built
        if self.noise_pipeline is None:
            raise RuntimeError("Noise sequence not built. Call build_noise_pipeline() first.")

        # Apply noise to the image by calling the noise pipeline one by one.
        # 噪声算子按配置顺序串联执行，前一个的输出作为后一个的输入
        data = data.clone()
        for noise_cfg in self.noise_pipeline:
            data = noise_cfg.func(data, noise_cfg, env_ids)  # type: ignore

        return data

    def apply_noise_pipeline_to_all_data_types(self, env_ids: torch.Tensor | Sequence[int]):
        """Apply the noise pipeline to all data types."""
        # 每帧更新时按 env_ids 增量应用噪声，避免对未变更的环境重复计算
        for data_type in self.cfg.data_types:
            self._data.output[f"{data_type}_noised"][env_ids] = self.apply_noise_pipeline(
                self._data.output[data_type][env_ids], env_ids=env_ids
            )

    def reset_noise_pipeline(self, env_ids: Sequence[int] | None = None):
        """Reset the noise pipeline for the specified environment IDs."""
        # 只重置具备 reset 方法的噪声算子（带状态噪声），无状态噪声无需处理
        if self.noise_pipeline is None:
            raise RuntimeError("Noise sequence not built. Call build_noise_pipeline() first.")

        for noise_cfg in self.noise_pipeline:
            if hasattr(noise_cfg.func, "reset"):
                noise_cfg.func.reset(env_ids)

    """
    History Buffers
    """

    def build_history_buffers(self):
        """Build the history buffers for the specified data types."""
        # 为每个需要历史的数据类型创建异步环形缓冲，并在输出字典中预留 {data_type}_history 张量
        self.output_history_buffers: dict[str, AsyncCircularBuffer] = dict()

        for data_type, history_length in self.cfg.data_histories.items():
            self.output_history_buffers[data_type] = AsyncCircularBuffer(
                history_length, self.num_instances, self.device
            )
            data_shape = self._data.output[data_type].shape
            self._data.output[f"{data_type}_history"] = torch.zeros(
                (data_shape[0], history_length, *data_shape[1:]), device=self.device
            )

    def update_history_buffers(self, env_ids: torch.Tensor | Sequence[int]):
        """Append the history buffers for the specified data types and update the result in self._data.output.
        Only configured data types will be appended, so only env_ids are needed. Please call this function after all
        outputs are computed.
        """
        # 必须在所有输出计算完成后调用，否则历史中会写入未更新的数据
        for data_type in self.cfg.data_histories.keys():
            self.output_history_buffers[data_type].append(self._data.output[data_type][env_ids], env_ids)
            self._data.output[f"{data_type}_history"][env_ids] = self.output_history_buffers[data_type].__getitem__(
                batch_ids=env_ids
            )

    def reset_history_buffers(self, env_ids: torch.Tensor | Sequence[int]):
        """Reset the history buffers for the specified data types."""
        for data_type in self.cfg.data_histories.keys():
            self.output_history_buffers[data_type].reset(env_ids)
