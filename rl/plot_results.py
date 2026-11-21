"""Plot evaluation metrics for TurtleBot3 PPO policy."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

METRICS = [
    ("mean_horizontal_error", "Error horizontal medio (|e_x|)", "blue"),
    ("mean_area", "Area aparente media del objetivo", "green"),
    ("detection_rate", "Tasa de deteccion visual", "purple"),
    ("mean_distance", "Distancia media real al objetivo (m)", "orange"),
    ("collision_rate", "Tasa de colisiones con obstaculo", "red"),
    ("success_rate", "Tasa de seguimiento exitoso", "teal"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot TurtleBot3 evaluation CSV metrics")
    parser.add_argument("csv_files", nargs="+", type=Path, help="CSV file(s) with evaluation metrics")
    parser.add_argument("--output", type=Path, default=Path("results/rendimiento_ppo.png"), help="Output image path")
    args = parser.parse_args()

    fig, axes = plt.subplots(3, 2, figsize=(11, 8.5), sharex=True)

    for csv_path in args.csv_files:
        with csv_path.open(newline="") as csv_file:
            rows = list(csv.DictReader(csv_file))
        label = "Agente PPO" if csv_path.stem == "ppo" else csv_path.stem
        steps = [int(row["step"]) for row in rows]
        for axis, (key, title, color) in zip(axes.flat, METRICS):
            values = [float(row[key]) for row in rows]
            axis.plot(steps, values, label=label, color=color, linewidth=1.8)
            axis.set_title(title, fontsize=11, fontweight="bold")
            axis.grid(True, linestyle="--", alpha=0.4)
            axis.set_ylabel("Valor")

    axes[-1, 0].set_xlabel("Paso de evaluacion", fontsize=10)
    axes[-1, 1].set_xlabel("Paso de evaluacion", fontsize=10)
    fig.suptitle("Rendimiento Dinamico del Agente de Reinforcement Learning (PPO)", fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(f"Grafica guardada en {args.output}")


if __name__ == "__main__":
    main()
