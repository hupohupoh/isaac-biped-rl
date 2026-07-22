"""This sub-module contains functions specific to the biped environment.

Uses Isaac Lab's lazy_export to auto-expose all built-in MDP terms.
Custom reward/termination functions can be added in rewards.py.
"""

from isaaclab.utils.module import lazy_export

lazy_export()
