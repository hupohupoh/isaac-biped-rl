"""Debug script to find which observation term produces NaN."""
import argparse
import contextlib
import sys

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Biped-velocity-v0")
parser.add_argument("--num_envs", type=int, default=1)
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

import biped_demo.tasks  # noqa: F401


def main():
    torch.manual_seed(42)

    env_cfg, _ = resolve_task_config(args_cli.task, "")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device else "cuda:0"

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg)

        print(f"\n=== Observation space ===")
        print(f"Shape: {env.observation_space.shape}")
        print(f"Space type: {type(env.observation_space).__name__}")

        print(f"\n=== Action space ===")
        print(f"Shape: {env.action_space.shape}")

        print(f"\n=== Robot articulation ===")
        robot = env.unwrapped.scene["robot"]
        print(f"Number of joints: {robot.num_joints}")
        print(f"Number of bodies: {robot.num_bodies}")
        print(f"Joint names: {robot.joint_names}")
        print(f"Body names: {robot.body_names}")

        print(f"\n=== Taking one step ===")
        obs, _ = env.reset()
        print(f"Obs shape: {obs['policy'].shape}")
        print(f"Obs min/max: {obs['policy'].min().item():.4f} / {obs['policy'].max().item():.4f}")
        has_nan = torch.isnan(obs['policy']).any()
        print(f"Has NaN: {has_nan}")
        if has_nan:
            nan_count = torch.isnan(obs['policy']).sum().item()
            print(f"NaN count: {nan_count} / {obs['policy'].numel()}")
            # Print first 20 elements to see patterns
            flat = obs['policy'][0]
            for i in range(min(20, len(flat))):
                print(f"  obs[{i}]: {flat[i].item():.6f}")

        action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        step_result = env.step(action)
        obs, rew, terminated, truncated, info = step_result if len(step_result) == 5 else (*step_result, {})
        print(f"\nAfter step:")
        print(f"Obs shape: {obs['policy'].shape}")
        has_nan2 = torch.isnan(obs['policy']).any()
        print(f"Has NaN: {has_nan2}")
        print(f"Reward: {rew}")
        if has_nan2:
            flat = obs['policy'][0]
            for i in range(len(flat)):
                v = flat[i].item()
                if v != v:  # NaN check
                    print(f"  obs[{i}] = NaN")
        else:
            # Run a few more steps
            for step_i in range(5):
                action = 0.1 * torch.randn(env.action_space.shape, device=env.unwrapped.device)
                step_result = env.step(action)
                obs = step_result[0]
                if torch.isnan(obs['policy']).any():
                    print(f"NaN appeared at step {step_i + 2}")
                    break
            else:
                print("No NaN after 5 more steps")

        env.close()


if __name__ == "__main__":
    main()
