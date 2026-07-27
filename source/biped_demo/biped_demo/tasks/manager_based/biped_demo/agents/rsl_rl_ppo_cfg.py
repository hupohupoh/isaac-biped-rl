"""PPO configuration for the bipedal robot velocity tracking task.

Flat-terrain locomotion for a 12-DOF biped (45 cm, 3.4 kg).
Restored to larger network after [256,128,64] proved too small —
error_vel_xy stuck at 0.45 regardless of reward tuning, suggesting
the actor lacked capacity to represent coordinated 12-joint walking.

Reference: livelybot Pi (12-DOF, same scale) uses [512,256,128]/[768,256,128]
and trains for 10k+ iterations. H1 official flat uses [128,128,128] for a
much easier 19-DOF task (arms can counterbalance, no risk of falling).
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

    num_steps_per_env = 24
    max_iterations = 10000          # 12-DOF walking needs more training
    save_interval = 100
    experiment_name = "biped_demo"
    empirical_normalization = False

    # Actor: [512, 256, 128] — matches H1 rough + livelybot Pi (both 12+ DOF)
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.5,               # lower — action_std was hitting 1.69
            std_range=(0.05, 10.0),
        ),
    )

    # Critic: [512, 256, 128] — matches H1 official
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )

    # PPO hyperparameters — standard Isaac Lab recipe for 4096 envs
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,            # reduced — 750-dim obs needs stable updates
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
