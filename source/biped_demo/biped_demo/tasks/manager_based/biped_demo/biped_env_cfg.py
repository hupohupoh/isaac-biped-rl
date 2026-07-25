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
    """Velocity commands.  Biped: forward-only + heading, no lateral."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0),    # backward 0.5 → forward 1.0 m/s
            lin_vel_y=(-0.3, 0.3),    # small lateral
            ang_vel_z=(-1.0, 1.0),    # yaw rate
            heading=(-3.14, 3.14),
        ),
    )


# ── Rewards ───────────────────────────────────────────────────────────────────
# Balanced between exploration drive and stability constraints.
# After two rounds of tuning (v1 too aggressive, v2 too punitive), v3 finds
# the middle ground — penalties scaled down for the small 3.4 kg biped.
#
#   Positive sum ≈ 1.5 + 0.8 + 0.1 + 0.5 = 2.9
#   Penalty sum ≈ -100(term) -1.0 -0.5 -0.02 -2.0 -0.5 -0.1 -0.005 -0.5 ≈ -106
#
# When walking well: net ≈ +1.5~2.0 / step (strong enough to learn)
# When flailing:    net ≈ -1.0~-2.0 / step (clear penalty gradient)


@configclass
class RewardsCfg:
    """Bipedal locomotion rewards — v3: calibrated for 45 cm / 3.4 kg biped."""

    # ── 1. Velocity tracking ───────────────────────────────────────────
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.8,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # ── 2. Survival ────────────────────────────────────────────────────
    alive = RewTerm(func=mdp.is_alive, weight=0.1)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-100.0)

    # ── 3. Posture ─────────────────────────────────────────────────────
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.02)
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-2.0,
        params={"target_height": 0.35},
    )

    # ── 4. Joint regularization ────────────────────────────────────────
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    joint_deviation_hips = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_leg_roll_joint", ".*_leg_yaw_joint"],
            ),
        },
    )

    # ── 5. Feet — gait ─────────────────────────────────────────────────
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[".*_ankle_roll_link"],
            ),
            "threshold": 0.3,
        },
    )

    # ── 6. Smoothness ──────────────────────────────────────────────────
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)

    # ── 7. Safety — body parts shouldn't touch ground ──────────────────
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
        params={"limit_angle": 1.0},   # ~57° tilt → dead (wider for small robot)
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
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
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


# ── Full env ──────────────────────────────────────────────────────────────────


@configclass
class BipedEnvCfg(ManagerBasedRLEnvCfg):
    """Complete bipedal velocity-tracking environment."""

    scene: BipedSceneCfg = BipedSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
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
