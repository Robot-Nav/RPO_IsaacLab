"""Configuration terms for different managers."""
# 管理器项配置：定义多奖励组MultiRewardCfg基类与占位DummyRewardCfg
# MultiRewardCfg供ParkourRewardsCfg继承，DummyRewardCfg用于parkour_env.load_managers()占位绕过父类校验

from __future__ import annotations

import torch
from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any

from isaaclab.utils import configclass


@configclass
class MultiRewardCfg:
    """Configuration for a reward group. Please inherit it if you want to define
    your own reward group so that the manager can recognize it.
    """
    # 多奖励组配置基类：自定义奖励组需继承此类，MultiRewardManager通过类型识别装配

    pass


@configclass
class DummyRewardCfg:
    """A placeholder for reward cfg."""
    # 空占位配置：parkour_env.load_managers先用它替换rewards走父类装配，避免父类按MultiRewardCfg做额外处理

    pass
