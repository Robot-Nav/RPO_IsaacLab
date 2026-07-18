# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# 基于图像编码器的Actor-Critic网络（parkour风格）
# Conv2dHeadModel组合Conv2d编码器与MLP头；Actor/Critic各自配置编码器
# 支持Critic与Actor共享编码器（"shared"），便于ONNX部署时分块导出

from __future__ import annotations

import os
from typing import Any, NoReturn

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.networks import EmpiricalNormalization, MLP


class Conv2dHeadModel(nn.Module):
    """Conv2d encoder with an MLP head, matching the structure used by parkour policies.

    前段为多卷积层（可选MaxPool降采样），后段接MLP头，整体对2D图像编码为低维向量。
    """

    def __init__(
        self,
        image_shape: tuple[int, int, int],
        channels: list[int],
        kernel_sizes: list[int],
        strides: list[int],
        hidden_sizes: list[int],
        output_size: int,
        paddings: list[int] | None = None,
        nonlinearity: str = "ReLU",
        use_maxpool: bool = False,
        last_activation: str | None = None,
    ) -> None:
        super().__init__()
        if paddings is None:
            paddings = [0 for _ in channels]
        if not (len(channels) == len(kernel_sizes) == len(strides) == len(paddings)):
            raise ValueError("Conv2dHeadModel channel, kernel, stride, and padding lists must have the same length.")

        activation = getattr(nn, nonlinearity)
        in_channels = [image_shape[0], *channels[:-1]]
        conv_layers: list[nn.Module] = []
        for in_ch, out_ch, kernel_size, stride, padding in zip(in_channels, channels, kernel_sizes, strides, paddings):
            conv_stride = 1 if use_maxpool else stride
            conv_layers.extend(
                [
                    nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=conv_stride, padding=padding),
                    activation(),
                ]
            )
            if use_maxpool and stride > 1:
                conv_layers.append(nn.MaxPool2d(stride))
        self.conv = nn.Sequential(*conv_layers)

        with torch.no_grad():
            probe = torch.zeros(1, *image_shape)
            conv_out_size = int(self.conv(probe).numel())
        self.head = MLP(conv_out_size, output_size, hidden_sizes, nonlinearity.lower(), last_activation=last_activation)
        self.out_features = output_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x).flatten(start_dim=1))


class EncoderActorCritic(nn.Module):
    """Actor-critic with image encoders that feed a single actor and critic MLP.

    每个图像观测组件由独立Conv2dHeadModel编码为低维向量，与1D本体观测拼接后送入共享Actor/Critic MLP。
    支持Critic与Actor共享编码器分支，便于ONNX部署时按子图分块导出。
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = (256, 256, 256),
        critic_hidden_dims: tuple[int] | list[int] = (256, 256, 256),
        actor_encoder_obs_groups: tuple[str, ...] | list[str] = ("depth_image",),
        critic_encoder_obs_groups: tuple[str, ...] | list[str] | str | None = "shared",
        encoder_cfg: dict[str, Any] | None = None,
        critic_encoder_cfg: dict[str, Any] | None = None,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        encoder_onnx_stems: dict[str, str] | None = None,
        encoder_onnx_sequential_idx: int = 0,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "EncoderActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()
        self.encoder_onnx_stems = encoder_onnx_stems
        self.encoder_onnx_sequential_idx = encoder_onnx_sequential_idx
        self.obs_groups = obs_groups
        self.actor_encoder_obs_groups = list(actor_encoder_obs_groups)
        self.critic_encoder_obs_groups = critic_encoder_obs_groups
        self.actor_obs_group = obs_groups["policy"][0]
        self.critic_obs_group = obs_groups["critic"][0]

        encoder_cfg = encoder_cfg or {}
        critic_encoder_cfg = critic_encoder_cfg or encoder_cfg
        self.actor_encoders = self._build_encoders(
            obs, self.actor_obs_group, self.actor_encoder_obs_groups, encoder_cfg
        )

        if critic_encoder_obs_groups == "shared":
            self.critic_encoders = self.actor_encoders
            critic_encoder_groups = self.actor_encoder_obs_groups
        elif critic_encoder_obs_groups is None:
            self.critic_encoders = None
            critic_encoder_groups = []
        else:
            critic_encoder_groups = list(critic_encoder_obs_groups)
            self.critic_encoders = self._build_encoders(
                obs, self.critic_obs_group, critic_encoder_groups, critic_encoder_cfg
            )
        self._critic_encoder_obs_groups_resolved = critic_encoder_groups

        print("Encoder networks (Conv2d branches):")
        for name, enc in self.actor_encoders.items():
            print(f"  actor encoder '{name}': {enc}")
        if critic_encoder_obs_groups == "shared":
            print("  critic encoders: shared with actor")
        elif self.critic_encoders is not None:
            for name, enc in self.critic_encoders.items():
                print(f"  critic encoder '{name}': {enc}")
        else:
            print("  critic encoders: none (MLP-only critic observations)")

        num_actor_obs = self._num_1d_obs(
            obs, self.actor_obs_group, self.actor_encoder_obs_groups
        ) + self._encoder_output_size(
            self.actor_encoders, self.actor_encoder_obs_groups
        )
        num_critic_obs = self._num_1d_obs(
            obs, self.critic_obs_group, critic_encoder_groups
        ) + self._encoder_output_size(
            self.critic_encoders, critic_encoder_groups
        )

        self.state_dependent_std = state_dependent_std
        self.actor = self._build_actor(num_actor_obs, num_actions, actor_hidden_dims, activation)
        print(f"Actor network: {self.actor}")

        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        self.critic = self._build_critic(num_critic_obs, critic_hidden_dims, activation)
        print(f"Critic network: {self.critic}")

        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        self.noise_std_type = noise_std_type
        if self.state_dependent_std:
            torch.nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if self.noise_std_type == "scalar":
                torch.nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
            elif self.noise_std_type == "log":
                torch.nn.init.constant_(
                    self.actor[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                )
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = None
        Normal.set_default_validate_args(False)

    def _build_actor(
        self,
        num_actor_obs: int,
        num_actions: int,
        actor_hidden_dims: tuple[int] | list[int],
        activation: str,
    ) -> nn.Module:
        if self.state_dependent_std:
            return MLP(num_actor_obs, [2, num_actions], actor_hidden_dims, activation)
        return MLP(num_actor_obs, num_actions, actor_hidden_dims, activation)

    def _build_critic(
        self, num_critic_obs: int, critic_hidden_dims: tuple[int] | list[int], activation: str
    ) -> nn.Module:
        return MLP(num_critic_obs, 1, critic_hidden_dims, activation)

    def _build_encoders(
        self, obs: TensorDict, obs_group: str, component_names: list[str], cfg: dict[str, Any]
    ) -> nn.ModuleDict:
        encoders = nn.ModuleDict()
        group_obs = obs[obs_group]
        for component_name in component_names:
            if len(group_obs[component_name].shape) != 4:
                raise ValueError(f"Encoder observation '{obs_group}.{component_name}' must have shape (N, C, H, W).")
            encoders[component_name] = Conv2dHeadModel(tuple(group_obs[component_name].shape[1:]), **cfg)
        return encoders

    def _num_1d_obs(self, obs: TensorDict, obs_group: str, encoder_component_names: list[str]) -> int:
        num_obs = 0
        group_obs = obs[obs_group]
        for component_name, component_obs in group_obs.items():
            if component_name in encoder_component_names:
                continue
            if len(component_obs.shape) != 2:
                raise ValueError(f"Observation component '{obs_group}.{component_name}' must have shape (N, D).")
            num_obs += component_obs.shape[-1]
        return num_obs

    def _encoder_output_size(self, encoders: nn.ModuleDict | None, obs_groups: list[str]) -> int:
        if encoders is None:
            return 0
        return sum(encoders[obs_group].out_features for obs_group in obs_groups)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def _update_distribution(self, obs: torch.Tensor) -> None:
        if self.state_dependent_std:
            mean_and_std = self.actor(obs)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            mean = self.actor(obs)
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        self._update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        if self.state_dependent_std:
            return self.actor(obs)[..., 0, :]
        return self.actor(obs)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        obs = self.critic_obs_normalizer(self.get_critic_obs(obs))
        return self.critic(obs)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        group_obs = obs[self.actor_obs_group]
        obs_list = [
            component_obs
            for component_name, component_obs in group_obs.items()
            if component_name not in self.actor_encoder_obs_groups
        ]
        obs_list.extend(
            self.actor_encoders[component_name](group_obs[component_name])
            for component_name in self.actor_encoder_obs_groups
        )
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        group_obs = obs[self.critic_obs_group]
        obs_list = [
            component_obs
            for component_name, component_obs in group_obs.items()
            if component_name not in self._critic_encoder_obs_groups_resolved
        ]
        if self.critic_encoders is not None:
            obs_list.extend(
                self.critic_encoders[component_name](group_obs[component_name])
                for component_name in self._critic_encoder_obs_groups_resolved
            )
        return torch.cat(obs_list, dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True

    def export_as_onnx(self, obs: TensorDict, filedir: str) -> None:
        """Export depth (or image) encoders as separate ONNX files and actor as ``actor.onnx``.

        Encoder files are named ``{encoder_onnx_sequential_idx}-{stem}.onnx`` where ``stem`` comes from
        ``encoder_onnx_stems[component_name]`` when set, otherwise ``component_name``. The actor subgraph
        includes ``actor_obs_normalizer`` when enabled so deployment can feed raw concatenated features.
        """
        if self.state_dependent_std:
            raise NotImplementedError(
                "export_as_onnx does not support state_dependent_std=True; use a policy without this flag."
            )
        self.eval()
        stems = self.encoder_onnx_stems or {}
        seq = self.encoder_onnx_sequential_idx
        with torch.no_grad():
            group_obs = obs[self.actor_obs_group]
            for component_name in self.actor_encoder_obs_groups:
                stem = stems.get(component_name, component_name)
                enc_in = group_obs[component_name]
                enc = self.actor_encoders[component_name]
                out_path = os.path.join(filedir, f"{seq}-{stem}.onnx")
                torch.onnx.export(
                    enc,
                    enc_in,
                    out_path,
                    input_names=["input"],
                    output_names=["output"],
                    opset_version=12,
                )
                print(f"Exported encoder '{component_name}' to {out_path}")

            actor_in = self.get_actor_obs(obs)
            if self.actor_obs_normalization:
                actor_module = nn.Sequential(self.actor_obs_normalizer, self.actor)
            else:
                actor_module = self.actor
            actor_path = os.path.join(filedir, "actor.onnx")
            torch.onnx.export(
                actor_module,
                actor_in,
                actor_path,
                input_names=["input"],
                output_names=["output"],
                opset_version=12,
            )
            print(f"Exported actor MLP to {actor_path}")
