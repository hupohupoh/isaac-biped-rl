"""Custom MDP terms for the bipedal robot."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


def foot_contact_state(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Binary contact state per foot: 1 when contact force > threshold.

    Returns:
        Tensor of shape [num_envs, num_feet] with 0/1 values.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    # net_forces_w: [num_envs, ALL_bodies * 3] — must index with body_ids
    net_forces = contact_sensor.data.net_forces_w
    net_forces = net_forces.reshape(env.num_envs, -1, 3)  # [num_envs, all_bodies, 3]
    net_forces = net_forces[:, sensor_cfg.body_ids, :]     # [num_envs, num_feet, 3]
    force_mag = torch.norm(net_forces, dim=-1)
    return (force_mag > threshold).float()


def foot_grf(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Ground reaction forces per foot (XYZ).

    Returns:
        Tensor of shape [num_envs, num_feet * 3], flattened XYZ per foot.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w
    net_forces = net_forces.reshape(env.num_envs, -1, 3)  # [num_envs, all_bodies, 3]
    net_forces = net_forces[:, sensor_cfg.body_ids, :]     # [num_envs, num_feet, 3]
    num_feet = len(sensor_cfg.body_ids)
    return net_forces.reshape(env.num_envs, num_feet * 3)


def leg_roll_angle_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    soft_limit: float = 0.087,    # 5°
    hard_limit: float = 0.175,    # 10°
    soft_weight: float = 2.0,
    hard_weight: float = 10.0,
) -> torch.Tensor:
    """Escalating penalty for leg_roll joints exceeding tight angle limits.

    |pos| < soft_limit (5°):  0 penalty
    soft < |pos| < hard (10°): linear ramp × soft_weight
    |pos| > hard_limit:       flat fine + exponential beyond × hard_weight

    Designed to prevent the "splayed legs" crawling gait.
    """
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]  # [num_envs, num_joints]
    abs_pos = torch.abs(joint_pos)

    penalty = torch.zeros_like(abs_pos)

    # ramp zone: 5° ~ 10°
    ramp = (abs_pos > soft_limit) & (abs_pos <= hard_limit)
    penalty[ramp] = (abs_pos[ramp] - soft_limit) * soft_weight

    # violation zone: > 10°
    hard = abs_pos > hard_limit
    ramp_portion = (hard_limit - soft_limit) * soft_weight   # penalty already accrued at hard_limit
    penalty[hard] = ramp_portion + (abs_pos[hard] - hard_limit) * hard_weight

    return torch.sum(penalty, dim=-1)


