
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# BeyondMimic超越模仿任务包入口，注册RPO四足机器人模仿与起身训练的Gym环境
# 暴露 RPO-BeyondMimic（全身动作模仿）与 RPO-Getup-Mimic（跌倒起身）两个环境

import gymnasium as gym
from . import agents

##
# Register Gym environments.
##

# 全身动作模仿环境：基于参考动作帧复现，叠加自适应起始帧采样提升训练效率
gym.register(
    id="RPO-BeyondMimic",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rpo_beyondmimic_env_cfg:RPOBeyondMimicEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rpo_beyondmimic_agent_cfg:RPOBeyondMimicPPORunnerCfg",
    },
)

# 起身模仿环境：参考动作为仰卧/俯卧到站立过程，动作结束后需保持稳定站立
gym.register(
    id="RPO-Getup-Mimic",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rpo_getup_mimic_env_cfg:RPOGetupMimicEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rpo_getup_mimic_agent_cfg:RPOGetupMimicPPORunnerCfg",
    },
)
