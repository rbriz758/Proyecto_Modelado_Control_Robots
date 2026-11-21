"""Train the TurtleBot3 visual-feature policy with PPO and rsl_rl."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train TurtleBot3 visual-feature following with PPO")
parser.add_argument(
    "--phase",
    type=str,
    default="1.5",
    choices=["1", "1.5", "2"],
    help="Fase del proyecto a entrenar (1, 1.5 o 2)",
)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--max-iterations", type=int, default=1000)
parser.add_argument("--log-dir", type=str, default=None)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Ruta a un checkpoint previo (.pt) para continuar acumulando aprendizaje",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import shutil
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
    env_cfg = TurtleBot3FollowEnvCfg()
    env_cfg = configure_phase(env_cfg, args_cli.phase)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 42

    agent_cfg = TurtleBot3FollowPPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iterations
    handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())

    if args_cli.log_dir:
        log_dir = Path(args_cli.log_dir).resolve()
    else:
        log_dir = (PROJECT_ROOT / f"checkpoints/fase{phase_key}/logs").resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    # Selección automática de checkpoint base si no se especifica
    checkpoint_to_load = None
    if args_cli.checkpoint:
        checkpoint_to_load = Path(args_cli.checkpoint).expanduser().resolve()
    elif args_cli.phase == "1.5":
        base_fase1 = (PROJECT_ROOT / "checkpoints/fase1/model_fase1.pt").resolve()
        if base_fase1.is_file():
            checkpoint_to_load = base_fase1

    print(f"[TRAIN] Fase={args_cli.phase} envs={env_cfg.scene.num_envs} iterations={agent_cfg.max_iterations}", flush=True)
    print(f"[TRAIN] Log dir: {log_dir}", flush=True)
    if checkpoint_to_load:
        print(f"[TRAIN] Checkpoint base cargado: {checkpoint_to_load}", flush=True)

    env = TurtleBot3FollowEnv(env_cfg)
    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = create_rsl_rl_runner(wrapped_env, agent_cfg, log_dir=str(log_dir))
    if checkpoint_to_load:
        if not checkpoint_to_load.is_file():
            raise FileNotFoundError(f"Checkpoint no encontrado: {checkpoint_to_load}")
        runner.load(str(checkpoint_to_load))
        for net in (runner.alg.actor, runner.alg.critic):
                if hasattr(net, "obs_normalizer") and hasattr(net.obs_normalizer, "count"):
                    net.obs_normalizer.count.clamp_(max=2048)

    print("[TRAIN] runner PPO listo", flush=True)
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=agent_cfg.init_at_random_ep_len,
    )
    print("[TRAIN] entrenamiento finalizado", flush=True)
    wrapped_env.close()

    # Copiar el modelo final entrenado a la ruta estándar de la fase
    generated_pts = sorted(log_dir.glob("model_*.pt"), key=lambda p: p.stat().st_mtime)
    if generated_pts:
        latest_pt = generated_pts[-1]
        target_model = PROJECT_ROOT / f"checkpoints/fase{phase_key}/model_fase{phase_key}.pt"
        shutil.copy(str(latest_pt), str(target_model))
        print(f"[TRAIN] Modelo de Fase {args_cli.phase} consolidado en: {target_model}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
