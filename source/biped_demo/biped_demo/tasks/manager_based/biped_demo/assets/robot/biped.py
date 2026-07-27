"""Configuration for the custom 12-DOF bipedal robot (v2.4.1).

Robot structure (per leg, 6 DOF):
    base_link
      ├── {l,r}_leg_pitch_joint  → hip pitch   (axis Y, ±1.57 rad)
      ├── {l,r}_leg_roll_joint   → hip roll    (axis X, -1.57~0.5 / -0.5~1.57)
      ├── {l,r}_leg_yaw_joint    → hip yaw     (axis Z, ±1.57 rad)
      ├── {l,r}_knee_pitch_joint → knee pitch  (axis Y, ±1.57 rad)
      ├── {l,r}_ankle_pitch_joint → ankle pitch (axis Y, ±0.5 rad)
      └── {l,r}_ankle_roll_joint → ankle roll  (axis X, ±0.5 rad)

Total mass: ~3.4 kg
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

# ---- Path to USD model ----
# After URDF → USD conversion, place the USD file here and update this path.
# Expected path: <this_dir>/usd/biped.usd
BIPED_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "usd")

# ---- Biped robot articulation config ----
BIPED_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{BIPED_MODEL_DIR}/biped_clean.usda",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            fix_root_link=False,      # explicitly free-floating base
        ),
    ),
    # Initial standing pose: legs slightly bent for stability
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),  # base height ~0.35m (foot-to-base ~0.276m + clearance)
        joint_pos={
            # Hip joints — neutral (legs straight down)
            ".*_leg_pitch_joint": 0.0,
            ".*_leg_roll_joint": 0.0,
            ".*_leg_yaw_joint": 0.0,
            # Knee — slight bend for stable standing
            ".*_knee_pitch_joint": -0.5,
            # Ankle — compensate knee bend to keep feet flat
            ".*_ankle_pitch_joint": 0.25,
            ".*_ankle_roll_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    # Actuators: Isaac Lab auto-filters fixed joints like root_joint.
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[".*"],
            effort_limit=5.0,        # from URDF: 5 Nm per joint
            saturation_effort=5.0,
            velocity_limit=10.0,     # from URDF: 10 rad/s
            stiffness=40.0,          # higher stiffness = tighter position tracking
            damping=2.0,             # higher damping = absorbs oscillations
            friction=0.01,           # joint friction (small)
            armature=0.0,            # motor armature inertia (set by env cfg)
        ),
    },
)
