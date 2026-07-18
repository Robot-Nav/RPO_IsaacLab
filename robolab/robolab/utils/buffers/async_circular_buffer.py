# 异步环形缓冲区，支持按批次索引独立读写历史数据
# 用于异步延迟场景下按各环境独立维护历史观测，支持sim2real的传感器延迟建模
import torch
from collections.abc import Sequence
from typing import Union

from isaaclab.utils.buffers import CircularBuffer


class AsyncCircularBuffer(CircularBuffer):
    # 异步环形缓冲区，每个batch维度的写入指针与历史长度独立维护
    # 相比基类CircularBuffer，支持按batch_ids子集进行读写，避免全量同步开销

    def __init__(self, max_len: int, batch_size: int, device: str):
        super().__init__(max_len, batch_size, device)

    @property
    def buffer(self) -> torch.Tensor:
        # 返回按时间顺序排列的完整历史缓冲，要求所有环境至少写入过一次
        if any(self._num_pushes == 0):
            raise RuntimeError("Attempting to access a buffer that is not fully initialized.")
        return self.get_by_batch_ids()

    def get_by_batch_ids(self, batch_ids: Sequence[int] | None = None) -> torch.Tensor:
        # 按批次索引取出时间顺序的历史序列
        # 核心思路：根据当前写入指针计算每个batch需要回溯的位移，再用torch.gather一次性取数
        # Index seems too large, potentially needing speed optimization. But we may wait and see.
        batch_ids = self._ALL_INDICES if batch_ids is None else torch.as_tensor(batch_ids, device=self._device)
        # 指针指向最新写入位置，回溯位移 = max_length - pointer - 1，确保index=0对应最新数据
        shifts = self.max_length - self._pointer - 1
        selected_shifts = shifts if batch_ids is None else shifts[batch_ids]
        selected_buf = self._buffer.clone() if batch_ids is None else self._buffer[:, batch_ids, ...].clone()
        selected_batch_size = self._batch_size if batch_ids is None else batch_ids.size(0)
        T = self.max_length
        arange = torch.arange(T, device=self._device)  # (T,)
        # 时间维减去位移后取模，得到该batch在环形缓冲中的实际读取位置
        index = ((arange[:, None] - selected_shifts[None, :]) % T).long()  # (T, 1) - (1, selected_B) -> (T, selected_B)
        extra_shape = selected_buf.shape[2:]  # (*D)
        index = index.view(T, selected_batch_size, *([1] * len(extra_shape)))  # (T, selected_B, 1....)
        index = index.expand(T, selected_batch_size, *extra_shape)  # (T, selected_B, *D)
        buf = torch.gather(selected_buf, dim=0, index=index)
        # 将时间维换到第二维，输出形状为 (selected_B, T, *D)
        return torch.transpose(buf, dim0=0, dim1=1)

    def append(self, data: torch.Tensor, batch_ids: Sequence[int] | None = None):
        # 向指定batch_ids追加新数据，每个batch独立推进写入指针
        if batch_ids is None:
            return super().append(data)
        else:
            if data.shape[0] != len(batch_ids):
                raise ValueError(f"Data shape {data.shape[0]} does not match batch_ids length {len(batch_ids)}.")

        data = data.to(self._device)

        if self._buffer is None:
            self._pointer = -torch.ones(self._batch_size, dtype=torch.int, device=self._device)
            self._buffer = torch.empty(
                (self.max_length, self._batch_size) + data.shape[1:], device=self._device, dtype=data.dtype
            )

        # 指针先加1再取模，实现环形写入
        self._pointer[batch_ids] = (self._pointer[batch_ids] + 1) % self.max_length
        self._buffer[self._pointer[batch_ids], batch_ids] = data
        is_first_push = self._num_pushes[batch_ids] == 0

        if torch.any(is_first_push):
            # 首次写入时用同一份数据填充整个历史窗口，避免读取到未初始化值
            batch_ids = torch.as_tensor(batch_ids, device=self._device)
            first_push_batch_ids = batch_ids[is_first_push]
            self._buffer[:, first_push_batch_ids] = data[is_first_push]

        self._num_pushes[batch_ids] += 1

    def __getitem__(self, key: torch.Tensor | None = None, batch_ids: Sequence[int] | None = None) -> torch.Tensor:
        # 按时间步key回溯历史数据，key=0表示最新，key越大越久远
        if batch_ids is None:
            return super().__getitem__(key)
        elif key is None:
            return self.get_by_batch_ids(batch_ids)
        else:
            if len(batch_ids) != key.shape[0]:
                raise ValueError(f"Batch IDs length {len(batch_ids)} does not match key shape {key.shape[0]}.")

        if torch.any(self._num_pushes[batch_ids] == 0) or self._buffer is None:
            raise RuntimeError("Attempting to retrieve data on an empty circular buffer. Please append data first.")

        current_pointers = self._pointer[batch_ids]

        # 限制回溯步数不超过已有写入次数，防止历史不足时越界
        valid_keys = torch.minimum(key, self._num_pushes[batch_ids] - 1)
        # 从当前指针向前回溯valid_keys步，并取模映射到环形缓冲实际索引
        index_in_buffer = torch.remainder(current_pointers - valid_keys, self.max_length)
        return self._buffer[index_in_buffer, batch_ids]
