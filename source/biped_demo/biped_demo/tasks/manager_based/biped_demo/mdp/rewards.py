"""Custom MDP terms for the bipedal robot."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


def gait_phase(env: ManagerBasedRLEnv, period: float = 1.0) -> torch.Tensor:
    """Cyclic gait clock as (sin(2πφ), cos(2πφ)) where φ = (t % period) / period.

    Adding this to policy observations lets the RL agent discover alternating
    gait rhythm naturally, without needing a gait-phase gate inside reward terms.
    """
    episode_time_s = env.episode_length_buf.float() * env.step_dt
    phase = (episode_time_s % period) / period
    two_pi = 2.0 * math.pi
    return torch.stack([torch.sin(two_pi * phase), torch.cos(two_pi * phase)], dim=-1)


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
    """Reward feet swinging forward relative to the body while in the air.

    Simplified — no gait-phase or body-speed gates.  The gait rhythm is
    learned from the gait_phase observation.  Standing-still wiggling is
    naturally suppressed by the track_lin_vel reward (the body must move
    forward to score there).
    """
    asset = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors[sensor_cfg.name]

    foot_vel_w = asset.data.body_lin_vel_w[:, sensor_cfg.body_ids, :]
    body_vel_w = asset.data.root_lin_vel_w
    rel_vel_w = foot_vel_w - body_vel_w.unsqueeze(1)

    # Forward direction (body X axis in world)
    q = asset.data.root_quat_w  # [N, 4] (x, y, z, w)
    qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    fx = 1.0 - 2.0 * (qy * qy + qz * qz)
    fy = 2.0 * (qx * qy + qw * qz)
    fz = 2.0 * (qx * qz - qw * qy)
    forward_w = torch.stack([fx, fy, fz], dim=-1)  # [N, 3]

    fwd_component = (rel_vel_w * forward_w.unsqueeze(1)).sum(dim=-1)  # [N, num_feet]

    # In-air detection
    net_forces = contact_sensor.data.net_forces_w_history.torch
    foot_forces = net_forces[:, :, sensor_cfg.body_ids, :]       # [N, hist, feet, 3]
    force_mag = torch.norm(foot_forces, dim=-1).max(dim=1)[0]    # [N, num_feet]
    in_air = (force_mag <= threshold).float()

    # Soft body-speed gate — kills stationary wiggling without hurting walking
    body_fwd_speed = (body_vel_w * forward_w).sum(dim=-1).unsqueeze(1)
    speed_gate = torch.sigmoid((body_fwd_speed - 0.1) * 50.0)

    return torch.sum(torch.relu(fwd_component) * in_air * speed_gate, dim=-1)


def foot_tilt_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize feet that are not flat — always on, both stance and swing.

    Transforms world gravity into each foot's body frame.  A flat foot has
    gravity purely along -Z (XY ≈ 0).  Tilting the foot puts gravity into
    XY, which is penalized regardless of ground contact.
    """
    from isaaclab.utils.math import quat_apply_inverse

    asset = env.scene[asset_cfg.name]

    # Foot body quaternion [num_envs, num_feet, 4]  (x, y, z, w)
    foot_quat_w = asset.data.body_quat_w[:, sensor_cfg.body_ids, :]

    # World gravity (0, 0, -1) transformed into each foot frame
    n_envs = env.num_envs
    n_feet = len(sensor_cfg.body_ids)
    gravity_w = torch.tensor([0.0, 0.0, -1.0], device=env.device, dtype=foot_quat_w.dtype)
    gravity_w = gravity_w.expand(n_envs, n_feet, 3)
    gravity_foot = quat_apply_inverse(foot_quat_w, gravity_w)  # [N, feet, 3]

    # XY norm of gravity in foot frame = tilt angle proxy
    tilt = torch.norm(gravity_foot[..., :2], dim=-1)  # [num_envs, num_feet]

    return torch.sum(tilt, dim=-1)


def hip_swing_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    period: float = 0.5,
    amplitude: float = 0.3,
    sigma: float = 0.2,
    phase_sign: float = 1.0,
) -> torch.Tensor:
    """Encourage a hip_pitch joint to follow a sinusoidal stepping rhythm.

    Target = amplitude × phase_sign × sin(2πt / period)

    Use two separate reward terms — one per leg with opposite phase_sign
    (+1 for left, -1 for right) — to create alternating anti-phase motion.

    Reward: exp(-|actual - target| / sigma).
    Once the hips start swinging, feet lift naturally and foot_swing_forward
    / air_time rewards take over.
    """
    import math

    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]  # [N, num_joints]

    episode_time = env.episode_length_buf.float() * env.step_dt  # [N]
    phase = 2.0 * math.pi * episode_time / period
    target = amplitude * phase_sign * torch.sin(phase)  # [N]

    error = torch.abs(joint_pos - target.unsqueeze(-1))
    return torch.sum(torch.exp(-error / sigma), dim=-1)


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name: str | None = None,
) -> torch.Tensor:
    """Reward feet for matching a periodic gait contact pattern.

    Each foot has a target stance phase defined by (global_phase + offset[i]) % 1.
    When the phase < threshold, the foot should be in contact (stance).
    When the phase >= threshold, the foot should be in the air (swing).

    This directly rewards alternating stepping without being exploitable
    by holding a single leg up — both feet must follow the pattern.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for off in offset:
        phases.append((global_phase + off) % 1.0)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
        reward *= cmd_norm > 0.1
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward swinging feet for reaching target height with flat soles.

    Gaussian reward: exp(-sum(height_err^2 × tanh(vel)) / std),
    scaled by foot flatness.  A tilted foot gets partial reward even
    at the right height — the policy must lift flat to score fully.
    """
    from isaaclab.utils.math import quat_apply_inverse

    asset = env.scene[asset_cfg.name]
    n_feet = len(asset_cfg.body_ids)
    n_envs = env.num_envs

    # --- Height reward (original logic) ---
    foot_z_target_error = torch.square(
        asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height
    )
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    height_reward = torch.exp(-torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1) / std)

    # --- Flatness bonus ---
    foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids, :]          # [N, feet, 4]
    gravity_w = torch.tensor([0.0, 0.0, -1.0], device=env.device, dtype=foot_quat_w.dtype)
    gravity_w = gravity_w.expand(n_envs, n_feet, 3)
    gravity_foot = quat_apply_inverse(foot_quat_w, gravity_w)                # [N, feet, 3]
    tilt = torch.norm(gravity_foot[..., :2], dim=-1)                         # [N, feet]
    flatness = torch.exp(-torch.sum(tilt, dim=-1) / 0.1)                     # 1 when flat, <1 when tilted

    # --- Forward velocity gate — clearance only counts when moving forward ---
    q = asset.data.root_quat_w  # [N, 4] (x, y, z, w)
    qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    fx = 1.0 - 2.0 * (qy * qy + qz * qz)
    fy = 2.0 * (qx * qy + qw * qz)
    fz = 2.0 * (qx * qz - qw * qy)
    forward_w = torch.stack([fx, fy, fz], dim=-1)
    body_fwd_speed = (asset.data.root_lin_vel_w * forward_w).sum(dim=-1)
    fwd_gate = torch.sigmoid((body_fwd_speed - 0.05) * 30.0)  # 0 at rest, 1 when walking

    return height_reward * flatness * fwd_gate


def feet_air_time_gated(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    sigma: float = 0.5,
) -> torch.Tensor:
    """feet_air_time gated by velocity tracking quality.

    Standard air_time reward multiplied by exp(-tracking_error / sigma).
    When the robot tracks velocity commands poorly, air_time reward is
    suppressed → must walk forward to earn foot-lift credit.
    Prevents "marching in place" exploitation.
    """
    from isaaclab_tasks.core.velocity.mdp.rewards import feet_air_time

    reward = feet_air_time(env, command_name, sensor_cfg, threshold)

    cmd = env.command_manager.get_command(command_name)
    asset = env.scene["robot"]
    tracking_error = torch.norm(asset.data.root_lin_vel_w[:, :2] - cmd[:, :2], dim=-1)
    gate = torch.exp(-tracking_error / sigma).clamp(min=0.3)  # floor — never kills signal entirely
    return reward * gate


