from __future__ import annotations
# 外感知观测函数：从相机/射线传感器读取图像并预处理，支持历史帧延迟采样用于sim2real
# 包含可视化调试工具与延迟帧采样类，深度图像噪声pipeline的核心组件
import numpy as np
import torch
from typing import TYPE_CHECKING, Literal

import cv2

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.events import (  # This could be dangerous for code maintainability. Maybe optimize this import later.
    _randomize_prop_by_op,
)
from isaaclab.managers import ManagerTermBase, ManagerTermBaseCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from isaaclab.sensors import Camera, RayCasterCamera, TiledCamera

    from robolab.sensors import GroupedRayCasterCamera, NoisyGroupedRayCasterCamera


def _debug_visualize_image(
    image: torch.Tensor,
    scale_up_vis: int = 5,
    window_name: str = "vis_image",
) -> None:
    """Visualize images in a cv2 window for debugging purposes.

    This function normalizes images to [0, 255], handles different channel configurations,
    scales them up for better visualization, and displays them in an OpenCV window.

    Args:
        images: Image tensor in shape (H, W)
        scale_up_vis: The factor to scale up the image for better visualization if the
            resolution is too low. Defaults to 5.
        window_name: The name of the OpenCV window. Defaults to "vis_image".
    """
    # 调试可视化：归一化到0-255后放大显示，便于观察低分辨率深度图细节
    # automatically normalize images to [0, 255]
    img = (image * 255.0 / image.max()).cpu().numpy().astype("uint8")  # (H, W)
    # Scale up the image for better visualization
    img = cv2.resize(img, (img.shape[1] * scale_up_vis, img.shape[0] * scale_up_vis), interpolation=cv2.INTER_AREA)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img)
    cv2.waitKey(1)


def visualizable_image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
    data_type: str = "rgb",
    debug_vis: bool = False,
    scale_up_vis: int = 5,
    history_skip_frames: int = 0,
) -> torch.Tensor:
    """Images of a specific datatype from the camera sensor."""
    # 从传感器读取图像并预处理：history类型按帧跳采，非history类型转NCHW
    # extract the used quantities (to enable type-hinting)
    sensor: TiledCamera | Camera | RayCasterCamera | GroupedRayCasterCamera | NoisyGroupedRayCasterCamera = (
        env.scene.sensors[sensor_cfg.name]
    )

    # obtain the input image
    images = sensor.data.output[data_type].clone()  # (N, H, W, C) or (N, history, H, W, C)
    if "history" in data_type:
        # NOTE: Only depth-related data types with history are supported. where C = 1.
        # 历史帧 squeeze 掉单通道维度，按history_skip_frames跳采降采样
        images = images.squeeze(
            -1
        )  # (N, history, H, W, C) -> (N, history, H, W), images[:, -1] shall be the latest frame.
        if history_skip_frames > 0:
            images = images[:, ::history_skip_frames, :, :]
    else:
        images = images.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

    # rgb/depth image normalization
    # NOTE: rgb image is not tested yet.

    if debug_vis:
        # (N, C, H, W) -> (C, H, N, W) -> (C*H, N*W)
        _debug_visualize_image(
            images.permute(1, 2, 0, 3).flatten(start_dim=0, end_dim=1).flatten(start_dim=1, end_dim=2), scale_up_vis
        )

    return images


class delayed_visualizable_image(ManagerTermBase):
    """A callable class that could sample delayed images from camera sensor that has history data. This is initially
    designed to use NoisyGroupedRayCasterCamera. The output shape will always be (N, num_output_frames, H, W) for now.
    """
    # 延迟图像采样类：从历史帧中按随机延迟采样，模拟真实相机传输延迟提升sim2real
    # 每个env独立维护延迟帧数，reset时重新采样

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.sensor_cfg = cfg.params.get("sensor_cfg", SceneEntityCfg("camera"))
        self.data_type = cfg.params["data_type"]  # must provide the data_type, must have "history" in the data_type
        assert "history" in self.data_type, "data_type must have 'history' in it"
        self.sensor: NoisyGroupedRayCasterCamera = env.scene.sensors[self.sensor_cfg.name]
        self.delayed_frame_ranges = cfg.params.get("delayed_frame_ranges", (0, 0))  # (min_delay, max_delay)
        # not recommended for gaussian distribution, but it is supported.
        self.delayed_frame_distribution: Literal["uniform", "log_uniform"] = cfg.params.get(
            "delayed_frame_distribution", "uniform"
        )
        self._num_delayed_frames = torch.zeros(env.num_envs, device=env.device)  # depending on the sensor update period
        self.history_skip_frames = max(cfg.params.get("history_skip_frames", 1), 1)
        # if greater than 0, the output data from this observation term will have history dimension, else no history dimension.
        self.num_output_frames = max(cfg.params.get("num_output_frames", 0), 1)
        assert len(self.sensor.data.output[self.data_type].shape) >= 5, (
            f"sensor data of type {self.data_type} should have (N, history, H, W, C) shape, but got"
            f" {self.sensor.data.output[self.data_type].shape}"
        )
        self.sensor_history_length = self.sensor.data.output[self.data_type].shape[1]

        # build frame offset based on num_output_frames and history_skip_frames
        # use reverse order because [:, -1] gets the latest frame in sensor data. frame_offset[0] should be the largest
        # to return the oldest frame in the output.
        # frame_offset反序排列：索引0对应最大偏移（最旧帧），保证输出[-1]为最新帧
        self.frame_offset = torch.flip(
            torch.arange(
                0,
                self.num_output_frames * self.history_skip_frames,
                self.history_skip_frames,
                device=env.device,
            ),
            dims=(0,),
        )  # (num_output_frames,)

        self.check_delay_bounds()

    def check_delay_bounds(self) -> None:
        """
        Check if the delayed frame ranges are within the bounds of the sensor history length.
        If not, raise an error.
        """
        # 校验：所需帧数（输出帧+延迟）不能超过传感器历史长度
        max_delayed_frames = self.delayed_frame_ranges[1]
        frames_needed_if_no_delay = (self.num_output_frames - 1) * self.history_skip_frames + 1
        if (frames_needed_if_no_delay + max_delayed_frames) > self.sensor_history_length:
            raise ValueError(
                "The delayed frame ranges are too large for the sensor history length. The maximum delayed frames is"
                f" {max_delayed_frames}, but the frames needed if no delay is {frames_needed_if_no_delay}, which is"
                f" {frames_needed_if_no_delay + max_delayed_frames}."
            )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """
        Reset the delayed frame ranges.
        """
        # reset时按delayed_frame_ranges重新采样每个env的延迟帧数
        if env_ids is None:
            env_ids = slice(None)
        self._num_delayed_frames[env_ids] = _randomize_prop_by_op(
            self._num_delayed_frames[env_ids].unsqueeze(-1),
            self.delayed_frame_ranges,
            None,
            slice(None),
            operation="abs",
            distribution=self.delayed_frame_distribution,
        ).squeeze(
            -1
        )  # (N,)

    def __call__(
        self,
        env: ManagerBasedEnv,
        data_type: str,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
        history_skip_frames: int = 0,
        num_output_frames: int = 0,
        delayed_frame_ranges: tuple[int, int] = (0, 0),
        delayed_frame_distribution: Literal["uniform", "log_uniform"] = "uniform",
        debug_vis: bool = False,
        scale_up_vis: int = 5,
    ) -> torch.Tensor:
        """
        Get the delayed frames from the sensor data.
        """
        # 从历史帧按frame_offset和延迟帧数采样，模拟相机传输延迟
        # obtain the input image
        images = self.sensor.data.output[self.data_type].clone()  # (N, history, H, W, C)
        # NOTE: Only depth-related data types with history are supported for now. where C = 1.
        images = images.squeeze(
            -1
        )  # (N, history, H, W, C) -> (N, history, H, W), images[:, -1] shall be the latest frame.
        # get the delayed frames
        # 帧索引 = 历史长度 - 帧偏移 - 延迟帧数 - 1，保证最新帧索引为history_length-1
        frame_indices = (
            self.sensor_history_length - self.frame_offset.unsqueeze(0) - self._num_delayed_frames.unsqueeze(1) - 1
        )  # (N, num_output_frames)
        frame_indices = frame_indices.to(torch.long)
        # final safety check to avoid frame_indices being out of bounds
        assert (frame_indices >= 0).all(), f"frame_indices should be non-negative, but got {frame_indices}"
        assert (
            frame_indices < self.sensor_history_length
        ).all(), f"frame_indices should be less than the sensor history length {self.sensor_history_length}"
        # Use advanced indexing: create batch indices and use them together with frame_indices
        # 高级索引：batch_indices与frame_indices配合实现按env独立采样不同延迟帧
        batch_indices = (
            torch.arange(images.shape[0], device=images.device)
            .unsqueeze(1)
            .expand(-1, frame_indices.shape[1])
            .to(torch.long)
        )
        delayed_frames = images[batch_indices, frame_indices]  # (N, num_output_frames, H, W)
        if debug_vis:
            # (N, num_output_frames, H, W) -> (num_output_frames, H, N, W) -> (num_output_frames * H, N * W)
            _debug_visualize_image(
                delayed_frames.permute(1, 2, 0, 3).flatten(start_dim=0, end_dim=1).flatten(start_dim=1, end_dim=2),
                scale_up_vis,
            )
        return delayed_frames  # still delayed_frames[:, -1] shall be the latest frame.
