# RPO Parkour训练agent配置：PPO+AMP+MoE+对称性约束的组合策略
# 含深度图编码器、5专家MoE门控、AMP判别器、左右镜像数据增强
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlSymmetryCfg

from robolab.tasks.manager_based.amp.agents.rpo_amp_agent_cfg import (
    RslRlAmpCfg,
    RslRlPpoAmpAlgorithmCfg,
)
from robolab.tasks.manager_based.parkour.mdp.symmetry import rpo


@configclass
class RslRlPpoEncoderMoEActorCriticCfg:
    # 编码器+MoE混合Actor-Critic：深度图经CNN编码后与本体观测拼接，MoE门控按场景切换专家
    class_name: str = "EncoderMoEActorCritic"
    init_noise_std: float = 1.0
    noise_std_type: str = "log"
    num_moe_experts: int = 5
    moe_gate_hidden_dims: list[int] = []
    actor_hidden_dims: list[int] = [256, 128, 64]
    critic_hidden_dims: list[int] = [256, 128, 64]
    actor_obs_normalization: bool = False # NOTE!: DO NOT SET TO TRUE, OR IT WILL CAUSE THE ROBOT TO CRASH IN REAL-WORLD DEPLOYMENT CAUSE WE USE THE LATENT ENCODER FOR DEPTH OBSERVATIONS!
    # 关闭观测归一化：深度图经编码器后是latent，归一化会破坏编码器输出分布导致部署崩溃
    critic_obs_normalization: bool = False
    activation: str = "elu"
    actor_encoder_obs_groups: list[str] = ["depth_image"]
    critic_encoder_obs_groups: list[str] = ["depth_image"]
    encoder_cfg: dict = {
        "channels": [4],
        "kernel_sizes": [3],
        "strides": [1],
        "hidden_sizes": [256, 256],
        "output_size": 128,
        "paddings": [1],
        "nonlinearity": "ReLU",
        "use_maxpool": True,
        "last_activation": "ReLU",
    }
    # onnx导出用：编码器作为独立stem，部署时与策略解耦
    encoder_onnx_stems: dict[str, str] = {"depth_image": "depth_encoder"}
    encoder_onnx_sequential_idx: int = 0


@configclass
class RPOParkourAmpRunnerCfg(RslRlOnPolicyRunnerCfg):
    # RPO Parkour训练runner：AMP+对称性+MoE的完整训练配置
    class_name = "AMPRunner"
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "rpo_parkour"
    wandb_project = "rpo_parkour"
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
        "discriminator": ["disc"],
        "discriminator_demonstration": ["disc_demo"],
    }
    policy = RslRlPpoEncoderMoEActorCriticCfg()
    algorithm = RslRlPpoAmpAlgorithmCfg(
        class_name="PPOAMP",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            mirror_loss_coeff=0.15,
            data_augmentation_func=rpo.compute_symmetric_states
        ),
        # 对称性：数据增强+镜像loss双约束，mirror_loss_coeff=0.15平衡约束强度
        amp_cfg=RslRlAmpCfg(
            disc_obs_buffer_size=100,
            grad_penalty_scale=5.0,
            disc_trunk_weight_decay=1.0e-4,
            disc_linear_weight_decay=1.0e-2,
            disc_learning_rate=1.0e-4,
            disc_max_grad_norm=1.0,
            amp_discriminator=RslRlAmpCfg.AMPDiscriminatorCfg(
                hidden_dims=[1024, 512],
                activation="ReLU",
                style_reward_scale=2.0,
                task_style_lerp=0.3,
            ),
            # AMP判别器：style_reward_scale=2.0放大风格奖励，task_style_lerp=0.3让任务奖励占主导
            loss_type="LSGAN",
        ),
    )
