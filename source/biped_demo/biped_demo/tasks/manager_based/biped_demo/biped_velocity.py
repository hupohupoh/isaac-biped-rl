"""Velocity-tracking environment for the custom 12-DOF bipedal robot.

Uses the full BipedEnvCfg with flat terrain and velocity commands.
"""

from isaaclab.utils.configclass import configclass

from .biped_env_cfg import BipedEnvCfg


@configclass
class BipedVelocityEnvCfg(BipedEnvCfg):
    """Velocity tracking on flat terrain for the custom bipedal robot."""

    def __post_init__(self):
        # Post-init from parent
        super().__post_init__()

        # Disable debug visualization by default (can be re-enabled for debugging)
        self.commands.base_velocity.debug_vis = False
