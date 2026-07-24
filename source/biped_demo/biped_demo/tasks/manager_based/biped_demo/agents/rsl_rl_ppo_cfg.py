"""PPO configuration for the bipedal robot velocity tracking task.

Flat-terrain locomotion for a small 12-DOF biped (45 cm, 3.4 kg).
Smaller network than H1 rough-terrain config — flat terrain + few DOFs
don't need the capacity, and a smaller net converges more gradually.

Reference: IsaacLab official H1FlatPPORunnerCfg uses [128, 128, 128]
for 4096-env flat-terrain training.
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
    max_iterations = 5000           # flat terrain converges faster — enough headroom
    save_interval = 100
    experiment_name = "biped_demo"
    empirical_normalization = False

    # Actor network: [256, 128, 64] — smaller than H1 rough (512/256/128)
    # because flat terrain + 12 DOF is an easier exploration problem.
    # H1 official flat: [128, 128, 128] for reference.
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.0,
            std_range=(0.05, 10.0),
        ),
    )

    # Critic network: matches actor capacity
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
    )

    # PPO hyperparameters — standard Isaac Lab recipe for 4096 envs
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
