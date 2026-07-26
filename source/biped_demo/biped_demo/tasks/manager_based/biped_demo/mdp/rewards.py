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


def joint_angle_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    soft_limit: float,
    hard_limit: float,
    soft_weight: float = 2.0,
    hard_weight: float = 10.0,
) -> torch.Tensor:
    """Escalating penalty for joints exceeding angle limits.

    |pos| < soft_limit:  0 penalty
    soft_limit < |pos| < hard_limit: linear ramp × soft_weight
    |pos| > hard_limit:  ramp_portion + excess × hard_weight

    Args:
        asset_cfg: Must specify joint_names to target specific joints.
        soft_limit: Angle (rad) where penalty starts ramping.
        hard_limit: Angle (rad) where heavy penalty kicks in.
        soft_weight: Ramp multiplier.
        hard_weight: Violation multiplier.
    """
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]  # [num_envs, num_joints]
    abs_pos = torch.abs(joint_pos)

    penalty = torch.zeros_like(abs_pos)

    ramp = (abs_pos > soft_limit) & (abs_pos <= hard_limit)
    penalty[ramp] = (abs_pos[ramp] - soft_limit) * soft_weight

    hard = abs_pos > hard_limit
    ramp_portion = (hard_limit - soft_limit) * soft_weight
    penalty[hard] = ramp_portion + (abs_pos[hard] - hard_limit) * hard_weight

    return torch.sum(penalty, dim=-1)


def foot_swing_forward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Reward feet that swing forward relative to the body while in the air.

    For each foot not in ground contact, compute its forward velocity
    (in the robot's body frame) relative to the body.  Only positive
    relative velocity (foot swinging forward faster than body) is rewarded.

    Why this can't be exploited:
    - Holding a foot up stationary:   rel_vel ≈ 0  → no reward
    - Shuffling on ground:            foot in contact → excluded
    - Alternating steps:              each swing phase → rewarded

    Total reward sums over both feet, so alternating steps get roughly
    twice the reward of a single-leg-dragging gait.
    """
    asset = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors[sensor_cfg.name]

    # Foot world velocities  [num_envs, num_feet, 3]
    foot_vel_w = asset.data.body_lin_vel_w[:, sensor_cfg.body_ids, :]

    # Robot body world velocity  [num_envs, 3]
    body_vel_w = asset.data.root_lin_vel_w

    # Relative velocity (world frame)  [num_envs, num_feet, 3]
    rel_vel_w = foot_vel_w - body_vel_w.unsqueeze(1)

    # Forward direction = body-frame X axis rotated into world
    q = asset.data.root_quat_w  # [num_envs, 4]  (x, y, z, w)
    qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    fx = 1.0 - 2.0 * (qy * qy + qz * qz)
    fy = 2.0 * (qx * qy + qw * qz)
    fz = 2.0 * (qx * qz - qw * qy)
    forward_w = torch.stack([fx, fy, fz], dim=-1)  # [num_envs, 3]

    # Project relative velocity onto forward direction
    fwd_component = (rel_vel_w * forward_w.unsqueeze(1)).sum(dim=-1)  # [num_envs, num_feet]

    # Only reward when the foot is in the air
    net_forces = contact_sensor.data.net_forces_w_history.torch
    foot_forces = net_forces[:, :, sensor_cfg.body_ids, :]          # [N, hist, feet, 3]
    force_mag = torch.norm(foot_forces, dim=-1).max(dim=1)[0]       # [num_envs, num_feet]
    in_air = (force_mag <= threshold).float()

    # Positive forward swing × in_air
    return torch.sum(torch.relu(fwd_component) * in_air, dim=-1)


def foot_tilt_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize feet that are not flat on the ground while in contact.

    Transforms world gravity into each foot's body frame.  A flat foot has
    gravity purely along -Z (XY ≈ 0).  Tilting the foot puts gravity into
    XY, which is penalized.

    Only active when the foot is in ground contact (force > threshold),
    so swing-phase foot tilt is ignored.
    """
    from isaaclab.utils.math import quat_rotate_inverse

    asset = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors[sensor_cfg.name]

    # Foot body quaternion [num_envs, num_feet, 4]  (x, y, z, w)
    foot_quat_w = asset.data.body_quat_w[:, sensor_cfg.body_ids, :]

    # World gravity (0, 0, -1) transformed into each foot frame
    n_envs = env.num_envs
    n_feet = len(sensor_cfg.body_ids)
    gravity_w = torch.tensor([0.0, 0.0, -1.0], device=env.device, dtype=foot_quat_w.dtype)
    gravity_w = gravity_w.expand(n_envs, n_feet, 3)
    gravity_foot = quat_rotate_inverse(foot_quat_w, gravity_w)  # [N, feet, 3]

    # XY norm of gravity in foot frame = tilt angle proxy
    tilt = torch.norm(gravity_foot[..., :2], dim=-1)  # [num_envs, num_feet]

    # Only when foot is in contact
    net_forces = contact_sensor.data.net_forces_w_history.torch
    foot_forces = net_forces[:, :, sensor_cfg.body_ids, :]
    force_mag = torch.norm(foot_forces, dim=-1).max(dim=1)[0]       # [num_envs, num_feet]
    in_contact = (force_mag > threshold).float()

    return torch.sum(tilt * in_contact, dim=-1)


