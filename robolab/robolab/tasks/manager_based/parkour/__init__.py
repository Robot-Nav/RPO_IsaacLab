# Parkour任务注册入口：向gym注册RPO-Parkour训练与回放环境
# 训练用RPOParkourEnvCfg（含随机化/课程），回放用RPOParkourEnvCfg_PLAY（关闭随机化）
import gymnasium as gym

from . import agents

gym.register(
    id="RPO-Parkour",
    entry_point="robolab.tasks.manager_based.parkour.parkour_env:ParkourEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rpo_parkour_env_cfg:RPOParkourEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rpo_parkour_agent_cfg:RPOParkourAmpRunnerCfg",
    },
)

gym.register(
    id="RPO-Parkour-Play",
    entry_point="robolab.tasks.manager_based.parkour.parkour_env:ParkourEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rpo_parkour_env_cfg:RPOParkourEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rpo_parkour_agent_cfg:RPOParkourAmpRunnerCfg",
    },
)
