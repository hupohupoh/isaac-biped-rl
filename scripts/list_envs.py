# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to print all available environments registered in the biped_demo extension.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import contextlib

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="List Isaac Lab environments.")
parser.add_argument("--keyword", type=str, default=None, help="Keyword to filter environments.")
args_cli = parser.parse_args()

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
from prettytable import PrettyTable

import biped_demo.tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks  # noqa: F401
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


def main():
    """Print all environments registered in the biped_demo extension."""
    task_specs = [
        spec
        for spec in gym.registry.values()
        if "Biped-" in spec.id and (args_cli.keyword is None or args_cli.keyword in spec.id)
    ]

    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Available Biped Environments"
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"

    for index, spec in enumerate(task_specs):
        table.add_row([
            index + 1,
            spec.id,
            spec.entry_point,
            spec.kwargs.get("env_cfg_entry_point", ""),
        ])

    print(table)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise e
    finally:
        simulation_app.close()
