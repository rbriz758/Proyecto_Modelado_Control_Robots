"""Evaluate the trained TurtleBot3 PPO policy and save CSV metrics."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CHECKPOINT = "/home/rbriram/IsaacLab/logs/rsl_rl/turtlebot3_follow/model_999.pt"

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate TurtleBot3 PPO following policy")
parser.add_argument(
    "--phase",
    type=str,
    default="1.5",
    choices=["1", "1.5", "2"],
    help="Fase a evaluar (1, 1.5 o 2)",
)
parser.add_argument(
    "--scenario",
    type=str,
    default="mixto",
    choices=["mixto", "circular", "slalom", "stop_and_go", "acelerones", "erratico"],
    help="Escenario de comportamiento del objetivo movil",
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to PPO model checkpoint")
parser.add_argument("--num-envs", type=int, default=16, help="Number of vectorized environments")
parser.add_argument("--steps", type=int, default=300, help="Number of evaluation steps")
parser.add_argument("--output", type=str, default=None, help="Output CSV path")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    check_rsl_rl_version,
    create_rsl_rl_runner,
    handle_deprecated_rsl_rl_cfg,
)

try:
    from rl.ppo_cfg import TurtleBot3FollowPPORunnerCfg
    from rl.turtlebot3_follow_env import TurtleBot3FollowEnv, TurtleBot3FollowEnvCfg, configure_phase
except ImportError:
    from ppo_cfg import TurtleBot3FollowPPORunnerCfg
    from turtlebot3_follow_env import TurtleBot3FollowEnv, TurtleBot3FollowEnvCfg, configure_phase


def main() -> None:
    phase_key = args_cli.phase.replace(".", "_")
    if args_cli.checkpoint:
        checkpoint_path = Path(args_cli.checkpoint).expanduser().resolve()
    else:
        checkpoint_path = (PROJECT_ROOT / f"checkpoints/fase{phase_key}/model_fase{phase_key}.pt").resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint no encontrado para Fase {args_cli.phase}: {checkpoint_path}")

    if args_cli.output:
        output_path = Path(args_cli.output).resolve()
    else:
        output_path = (PROJECT_ROOT / f"results/fase{phase_key}/ppo.csv").resolve()

    env_cfg = TurtleBot3FollowEnvCfg()
    env_cfg = configure_phase(env_cfg, args_cli.phase)
    env_cfg.scenario = args_cli.scenario
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 42
    env = TurtleBot3FollowEnv(env_cfg)
    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=1.0)

    agent_cfg = TurtleBot3FollowPPORunnerCfg()
    handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    runner = create_rsl_rl_runner(wrapped_env, agent_cfg)
    runner.load(str(checkpoint_path), load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": False})
    policy = runner.get_inference_policy(device=wrapped_env.device)

    env.reset()
    obs = wrapped_env.get_observations()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "step",
        "mean_horizontal_error",
        "mean_area",
        "detection_rate",
        "mean_distance",
        "collision_rate",
        "success_rate",
    ]
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        with torch.inference_mode():
            for step in range(args_cli.steps):
                features = wrapped_env.unwrapped.observation_features
                actions = policy(obs)
                obs, _, dones, _ = wrapped_env.step(actions)
                robot_pos = wrapped_env.unwrapped.robot.data.root_pos_w.torch
                target_pos = wrapped_env.unwrapped.target.data.root_pos_w.torch
                distance = torch.linalg.vector_norm(robot_pos[:, :2] - target_pos[:, :2], dim=1)
                writer.writerow(
                    {
                        "step": step,
                        "mean_horizontal_error": float(torch.abs(features[:, 1]).mean().item()),
                        "mean_area": float(features[:, 3].mean().item()),
                        "detection_rate": float(features[:, 0].mean().item()),
                        "mean_distance": float(distance.mean().item()),
                        "collision_rate": float(wrapped_env.unwrapped.collision.float().mean().item()),
                        "success_rate": float(wrapped_env.unwrapped.success.float().mean().item()),
                    }
                )
                policy.reset(dones)
    wrapped_env.close()
    print(f"Resultados de evaluacion guardados en {output_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
