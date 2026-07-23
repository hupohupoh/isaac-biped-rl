"""Environment configuration for the custom 12-DOF bipedal robot.

Velocity tracking on flat terrain with Manager-based RL workflow.

Rewards adapted from Isaac Lab's official H1 bipedal locomotion config:
    IsaacLab/source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/h1/
"""

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
    """Position targets for all 12 leg joints.  scale=0.5 → ±0.5 rad range."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
        use_default_offset=True,
    )


# ── Observations ──────────────────────────────────────────────────────────────


@configclass
class ObservationsCfg:
    """Policy (48-dim) + Privileged (11-dim) observation groups."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Student observations — what the real robot can measure."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
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
# Keep it simple: velocity tracking + don't fall + don't shake.
# Too many penalty terms paralyse the policy — add them back one at a time
# once the basic gait converges.


@configclass
class RewardsCfg:
    """Bipedal locomotion rewards — simplified for reliable convergence."""

    # ── 1. Velocity tracking (the ONLY positive rewards) ──────────────────
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # ── 2. Survival ───────────────────────────────────────────────────────
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # ── 3. Posture ────────────────────────────────────────────────────────
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.0,
    )

    # ── 4. Gait — alternating steps, symmetric stride ─────────────────────
    # +reward for exactly-one-foot-on-ground time → natural alternating gait
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[".*_ankle_roll_link"],
            ),
            "threshold": 0.4,    # lower for small/light robots (H1=0.6 for 50kg)
        },
    )
    # Penalise both feet airborne
    desired_contacts = RewTerm(
        func=mdp.desired_contacts,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[".*_ankle_roll_link"],
            ),
            "threshold": 2.0,
        },
    )

    # ── 5. Smoothness ─────────────────────────────────────────────────────
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )


# ── Terminations ──────────────────────────────────────────────────────────────


@configclass
class TerminationsCfg:
    """Kill episodes that go bad before they produce NaN."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.7},   # ~40° tilt → dead
    )


# ── Events ────────────────────────────────────────────────────────────────────


@configclass
class EventCfg:
    """Domain randomisation."""

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

    # Randomise foot friction at startup: 80% grippy 0.8-0.95, 20% slippery 0.65-0.8
    randomize_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
            "static_friction_range": (0.65, 0.95),
            "dynamic_friction_range": (0.65, 0.95),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 256,
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
