# 异步延迟缓冲区，按各环境独立的延迟步数回溯历史数据
# 用于模拟传感器传输延迟，支撑sim2real迁移中观测时序对齐
import torch
from collections.abc import Sequence
from typing import Union

from isaaclab.utils.buffers import DelayBuffer

from .async_circular_buffer import AsyncCircularBuffer


class AsyncDelayBuffer(DelayBuffer):
    """Asynchronous delay buffer that allows retrieving stored data with delays asynchronously for each batch index."""
    # 异步延迟缓冲，每个batch维度可配置独立的时间滞后步数
    # 相比基类DelayBuffer，支持按batch_ids子集计算延迟，避免全环境同步刷新

    def __init__(self, history_length: int, batch_size: int, device: str):
        """Initialize the asynchronous delay buffer.

        Args:
            history_length: The history of the buffer, i.e., the number of time steps in the past that the data
                will be buffered. It is recommended to set this value equal to the maximum time-step lag that
                is expected. The minimum acceptable value is zero, which means only the latest data is stored.
            batch_size: The batch dimension of the data.
            device: The device used for processing.
        """
        super().__init__(history_length, batch_size, device)
        # 环形缓冲容量 = 历史长度 + 1，多出的一位用于存放当前最新数据
        self._circular_buffer = AsyncCircularBuffer(self._history_length + 1, batch_size, device)

    def compute(self, data: torch.Tensor, batch_ids: Sequence[int] | None = None) -> torch.Tensor:
        # 写入新数据后按各batch的time_lag回溯对应历史帧，模拟传感器延迟
        if batch_ids is None:
            return super().compute(data)
        else:
            if len(batch_ids) != data.shape[0]:
                raise ValueError(f"Batch IDs length {len(batch_ids)} does not match data shape {data.shape[0]}.")

        # add the new data to the last layer
        self._circular_buffer.append(data, batch_ids)
        # return the output
        # 每个batch按自身time_lag取出对应延迟步数的旧数据
        delayed_data = self._circular_buffer.__getitem__(self._time_lags[batch_ids], batch_ids)
        return delayed_data.clone()
