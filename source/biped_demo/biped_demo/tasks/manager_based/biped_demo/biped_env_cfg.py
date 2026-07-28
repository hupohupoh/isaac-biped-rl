"""Environment configuration for the custom 12-DOF bipedal robot.

Velocity tracking on flat terrain with Manager-based RL workflow.

Rewards adapted from Isaac Lab's official H1 bipedal locomotion config:
    IsaacLab/source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/h1/
"""

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

# Isaac Lab renamed UniformNoiseCfg → AdditiveUniformNoiseCfg across versions.
# Both classes share the same API (n_min, n_max).
try:
    from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
except ImportError:
    from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from .assets.robot.biped import BIPED_CFG as RobotCFG
from . import mdp


# ── Scene ────────────────────────────────────────────────────────────────────


@configclass
class BipedSceneCfg(InteractiveSceneCfg):
    """Flat terrain + robot + contact sensor (for foot detection)."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=os.path.join(os.path.dirname(__file__), "flat_ground.usda"),
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = RobotCFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


# ── Actions ───────────────────────────────────────────────────────────────────


@configclass
class ActionsCfg:
    """Position targets for all 12 leg joints. scale=0.25 → ±0.25 rad range."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
    )


# ── Observations ──────────────────────────────────────────────────────────────


@configclass
class ObservationsCfg:
    """Policy (48-dim) + Privileged (11-dim) observation groups."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Student observations — what the real robot can measure.

        Observation noise simulates real sensor errors and prevents
        the policy from overfitting to perfect simulator readings.
        """

        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        actions = ObsTerm(func=mdp.last_action)
        gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.5})

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Teacher-only observations — simulator ground truth."""
        # Clean base velocity (no estimation noise)
        base_lin_vel_truth = ObsTerm(func=mdp.base_lin_vel)
        # Foot contact: binary + GRF per foot [num_envs, 2 + 6]
        foot_contact = ObsTerm(
            func=mdp.foot_contact_state,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle_roll_link"]),
                "threshold": 2.0,
            },
        )
        foot_grf = ObsTerm(
            func=mdp.foot_grf,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle_roll_link"]),
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


# ── Commands ──────────────────────────────────────────────────────────────────


@configclass
class CommandsCfg:
    """Velocity commands — forward-only, no yaw (phase 1: just learn to walk)."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.2),     # forward only — slight speed boost
            lin_vel_y=(0.0, 0.0),     # no lateral
            ang_vel_z=(0.0, 0.0),     # no yaw — force learning translation
            heading=(0.0, 0.0),       # go straight
        ),
    )


# ── Rewards ───────────────────────────────────────────────────────────────────
# Phase 1.5: forward walking works, now polish posture + gait.
#
# v2 HPC results: survival=100%, time_out=1.0, error_vel_yaw=0.17 (good),
# but error_vel_xy=0.46 (stuck), action_rate=-0.375 (jittering),
# entropy=25.98 (not converging), dof_pos_limits=-0.076 (joint abuse).
#
# Phase 1 (just 5 terms) fixed the "can't survive" problem.
# Now add back ONE gentle term (feet_air_time) to nudge toward lifting feet,
# and tighten posture/action penalties slightly.


@configclass
class RewardsCfg:
    """Phase 5 rewards — tight joint limits + alternating step pressure."""

    # ── 1. Velocity tracking ───────────────────────────────────────────
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=4.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,                    # anti-spin — turning costs a full point
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # ── 2. Survival ────────────────────────────────────────────────────
    alive = RewTerm(func=mdp.is_alive, weight=0.1)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-1.0)   # very light — let it try

    # ── 3. Posture ─────────────────────────────────────────────────────
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-1.0,
        params={"target_height": 0.35},
    )

    # ── 4. Joint angle limits (soft→hard escalating penalty) ──────────
    # Pitch: leg_pitch + knee_pitch.  Soft=60°, Hard=90°.
    joint_angle_pitch = RewTerm(
        func=mdp.joint_angle_penalty,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_leg_pitch_joint", ".*_knee_pitch_joint"],
            ),
            "soft_limit": 1.047,    # 60°
            "hard_limit": 1.571,    # 90°
            "soft_weight": 2.0,
            "hard_weight": 10.0,
        },
    )
    # Yaw: leg_yaw.  Soft=20°, Hard=45°.
    joint_angle_yaw = RewTerm(
        func=mdp.joint_angle_penalty,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_leg_yaw_joint"]),
            "soft_limit": 0.349,    # 20°
            "hard_limit": 0.785,    # 45°
            "soft_weight": 2.0,
            "hard_weight": 10.0,
        },
    )
    # Roll: leg_roll.  Soft=5°, Hard=10°.
    joint_angle_roll = RewTerm(
        func=mdp.joint_angle_penalty,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_leg_roll_joint"]),
            "soft_limit": 0.087,    # 5°
            "hard_limit": 0.175,    # 10°
            "soft_weight": 2.0,
            "hard_weight": 10.0,
        },
    )
    # Ankle pitch: soft=15°, hard=30°.
    joint_angle_ankle_pitch = RewTerm(
        func=mdp.joint_angle_penalty,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint"]),
            "soft_limit": 0.262,    # 15°
            "hard_limit": 0.524,    # 30°
            "soft_weight": 2.0,
            "hard_weight": 10.0,
        },
    )
    # Ankle roll: no penalty <10°, instant penalty >10°.
    joint_angle_ankle_roll = RewTerm(
        func=mdp.joint_angle_penalty,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_roll_joint"]),
            "soft_limit": 0.175,    # 10° — same as hard → no ramp, instant
            "hard_limit": 0.175,    # 10°
            "soft_weight": 0.0,     # no ramp
            "hard_weight": 10.0,
        },
    )

    # ── 5. Hip rhythm (light — feet_gait does the heavy lifting) ─────────
    hip_swing_left = RewTerm(
        func=mdp.hip_swing_reward,
        weight=0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["l_leg_pitch_joint"]),
            "period": 0.5,
            "amplitude": 0.3,
            "sigma": 0.2,
            "phase_sign": 1.0,
        },
    )
    hip_swing_right = RewTerm(
        func=mdp.hip_swing_reward,
        weight=0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["r_leg_pitch_joint"]),
            "period": 0.5,
            "amplitude": 0.3,
            "sigma": 0.2,
            "phase_sign": -1.0,
        },
    )

    # ── 6. Gait system — H1-style foot contact pattern + clearance ──────
    feet_gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.3,
        params={
            "period": 0.5,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[".*_ankle_roll_link"],
            ),
        },
    )
    foot_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=2.0,                    # step 1: when stable, drop to 1.0, then 0.5
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[".*_ankle_roll_link"],
            ),
            "target_height": 0.06,     # 6cm — tight gradient between shuffle and lift
            "std": 0.01,               # sharp — small errors matter
            "tanh_mult": 2.0,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,                  # aligned with H1 — shuffle hurts
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[".*_ankle_roll_link"],
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[".*_ankle_roll_link"],
            ),
        },
    )

    # ── 7. Forward swing + lift-off ─────────────────────────────────────
    foot_swing_forward = RewTerm(
        func=mdp.foot_swing_forward,
        weight=0.4,                    # reduced — clearance does the heavy lifting
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[".*_ankle_roll_link"],
            ),
            "threshold": 1.0,
        },
    )
    # ── 8. Foot tilt — always on, stance + swing ──────────────────────
    foot_tilt = RewTerm(
        func=mdp.foot_tilt_penalty,
        weight=-0.3,                   # reduced — clearance now carries flatness
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[".*_ankle_roll_link"],
            ),
            "threshold": 1.0,
        },
    )

    # ── 9. Smoothness ───────────────────────────────────────────────────
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # ── 10. Safety — knees/hips on ground = crawling ───────────────────
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["(?!.*ankle_roll_link).*"],
            ),
            "threshold": 1.0,
        },
    )


# ── Terminations ──────────────────────────────────────────────────────────────


@configclass
class TerminationsCfg:
    """Kill episodes that go bad before they produce NaN."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.7},   # ~40° tilt → dead (prevents crawling posture)
    )
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["base_link"],
            ),
            "threshold": 1.0,
        },
    )


# ── Events ────────────────────────────────────────────────────────────────────


@configclass
class EventCfg:
    """Domain randomisation for robust sim-to-real transfer."""

    # Reset base pose with small random offset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "yaw": (-0.5, 0.5),
            },
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        },
    )

    # Randomise foot friction at startup — wide range prevents overfitting
    randomize_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.1, 2.0),   # wider — high friction forces foot lift
            "dynamic_friction_range": (0.1, 2.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # Randomise base mass — simulates payload/battery variance
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-0.2, 0.5),  # -0.2~+0.5 kg for 3.4 kg robot
            "operation": "add",
        },
    )


# ── Curriculum ─────────────────────────────────────────────────────────────────


@configclass
class CurriculumCfg:
    """Gradually reduce clearance weight as walking improves."""

    clearance_step1 = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "foot_clearance",
            "weight": 1.0,                      # reduce from 2.0
            "num_steps": 1000 * 4096 * 24,      # after ~1000 iterations
        },
    )
    clearance_step2 = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "foot_clearance",
            "weight": 0.5,                      # reduce from 1.0
            "num_steps": 2000 * 4096 * 24,      # after ~2000 iterations
        },
    )


# ── Full env ──────────────────────────────────────────────────────────────────


@configclass
class BipedEnvCfg(ManagerBasedRLEnvCfg):
    """Complete bipedal velocity-tracking environment."""

    scene: BipedSceneCfg = BipedSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
