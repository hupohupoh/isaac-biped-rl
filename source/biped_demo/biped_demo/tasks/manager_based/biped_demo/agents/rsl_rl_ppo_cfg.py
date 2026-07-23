"""PPO configuration for the bipedal robot velocity tracking task.

The observation and action dimensions scale with the number of joints (12 DOF).
Critic is deeper than the Go2 config for better value estimation on the harder bipedal task.
"""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for bipedal locomotion."""

    num_steps_per_env = 24          # H1 official: 24 for 4096 envs (was 124 for 64 envs)
    max_iterations = 15000          # more iters to compensate for shorter steps
    save_interval = 100
    experiment_name = "biped_demo"
    empirical_normalization = False

    # Actor network: matches H1 official rough-terrain config
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.0,
            std_range=(0.15, 1.0),     # floor sigma at 0.15 — prevents entropy collapse
        ),
    )

    # Critic network: aligned with H1 official (was [256,128,64] — too weak for 4096 envs)
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )

    # PPO hyperparameters — aligned with H1 official 4096-env recipe
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,             # H1 official: 0.01 for 4096 envs (was 0.005)
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
