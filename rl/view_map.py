"""Visualiza solo el mapa (recinto cerrado de la Fase 2), sin politica ni entrenamiento.

El robot permanece quieto; se ven el recinto, los obstaculos, las cajas moviles y la
bola recorriendo su ruta. Cada episodio la bola arranca en un punto distinto de la ruta.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualizar el mapa de la Fase 2")
parser.add_argument("--steps", type=int, default=0, help="0 = hasta cerrar la ventana")
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

try:
    from rl.turtlebot3_follow_env import TurtleBot3FollowEnv, TurtleBot3FollowEnvCfg, configure_phase
except ImportError:
    from turtlebot3_follow_env import TurtleBot3FollowEnv, TurtleBot3FollowEnvCfg, configure_phase


def main() -> None:
    env_cfg = TurtleBot3FollowEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg = configure_phase(env_cfg, "2")
    env_cfg.scene.num_envs = 1
    env_cfg.seed = 42
    env_cfg.sim.default_visualizer_cfg = VisualizerCfg(eye=(0.0, -2.7, 3.4), lookat=(0.0, 0.0, 0.0))

    env = TurtleBot3FollowEnv(env_cfg, render_mode="human")
    env.reset()
    zero_action = torch.zeros((1, 2), device=env.device)
    print("[MAPA] Recinto 5.6x3.6 m: A slalom | B pasillo | C tunel | D cajas moviles")
    print("[MAPA] Cierra la ventana de Isaac Sim o pulsa Ctrl+C para terminar.")
    step = 0
    try:
        while simulation_app.is_running():
            env.step(zero_action)
            step += 1
            if args_cli.steps > 0 and step >= args_cli.steps:
                break
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

