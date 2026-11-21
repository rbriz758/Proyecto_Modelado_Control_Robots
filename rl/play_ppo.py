"""Play a trained TurtleBot3 visual policy in the Isaac Sim GUI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained TurtleBot3 PPO policy")
parser.add_argument(
    "--phase",
    type=str,
    default="1.5",
    choices=["1", "1.5", "2"],
    help="Fase del proyecto a ejecutar (1, 1.5 o 2)",
)
parser.add_argument(
    "--scenario",
    type=str,
    default="mixto",
    choices=["mixto", "circular", "slalom", "stop_and_go", "acelerones", "erratico"],
    help="Comportamiento del objetivo movil en Fase 1.5",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Ruta a un checkpoint .pt específico (si no se indica, usa el de la fase seleccionada)",
)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=0, help="0 keeps playing until the GUI is closed")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = False
args_cli.enable_cameras = True
args_cli.visualizer = "kit"
args_cli.kit_args = (
    "--/app/window/enabled=true "
    "--/app/window/hide=false "
    "--/app/window/width=1280 "
    "--/app/window/height=720"
)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab.visualizers import VisualizerCfg
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
        checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    else:
        checkpoint = (PROJECT_ROOT / f"checkpoints/fase{phase_key}/model_fase{phase_key}.pt").resolve()
        if not checkpoint.is_file() and args_cli.phase == "2":
            checkpoint = (PROJECT_ROOT / "checkpoints/fase1_5/model_fase1_5.pt").resolve()

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint no encontrado para la Fase {args_cli.phase}: {checkpoint}")

    env_cfg = TurtleBot3FollowEnvCfg()
    env_cfg = configure_phase(env_cfg, args_cli.phase)
    env_cfg.scenario = args_cli.scenario
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 42
    if args_cli.phase == "2":
        cam_eye = (0.0, -2.7, 3.4)
        cam_lookat = (0.0, 0.0, 0.0)
    else:
        cam_eye = (2.8, -2.8, 2.0)
        cam_lookat = (0.8, 0.0, 0.2)

    env_cfg.sim.default_visualizer_cfg = VisualizerCfg(
        eye=cam_eye,
        lookat=cam_lookat,
    )

    agent_cfg = TurtleBot3FollowPPORunnerCfg()
    handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())

    env = TurtleBot3FollowEnv(env_cfg, render_mode="human")
    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = create_rsl_rl_runner(wrapped_env, agent_cfg)
    runner.load(str(checkpoint), load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": False})
    policy = runner.get_inference_policy(device=wrapped_env.device)
    env.reset()
    obs = wrapped_env.get_observations()

    print(f"[PLAY] Fase seleccionada: {args_cli.phase}")
    print(f"[PLAY] Escenario seleccionado: {args_cli.scenario}")
    print(f"[PLAY] Checkpoint cargado: {checkpoint}")
    print("[PLAY] Cierra la ventana de Isaac Sim o pulsa Ctrl+C para terminar.")
    step = 0
    try:
        with torch.inference_mode():
            while simulation_app.is_running():
                actions = policy(obs)
                obs, _, dones, _ = wrapped_env.step(actions)
                policy.reset(dones)
                step += 1
                if args_cli.steps > 0 and step >= args_cli.steps:
                    break
    finally:
        wrapped_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
