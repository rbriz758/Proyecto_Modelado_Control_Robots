"""Controlador cinemático diferencial para TurtleBot3 Burger.

Este módulo implementa la cinemática directa e inversa del modelo diferencial,
así como el mapeo de consignas de velocidad (v, w) a consignas angulares
de los joints de las ruedas en Isaac Sim e Isaac Lab.
"""

from typing import Tuple, Union
import numpy as np


class TurtleBot3Kinematics:
    """Modelo cinemático diferencial del TurtleBot3 Burger.

    Parámetros físicos del TurtleBot3 Burger:
        - Radio de rueda (r): 0.033 m (diámetro 66 mm)
        - Distancia entre ruedas / track width (L): 0.160 m
        - Velocidad lineal máxima recomendada: 0.22 m/s
        - Velocidad angular máxima recomendada: 2.84 rad/s
    """

    def __init__(
        self,
        wheel_radius: float = 0.033,
        wheel_base: float = 0.160,
        max_linear_velocity: float = 0.22,
        max_angular_velocity: float = 2.84,
        max_wheel_speed: float = None,
    ):
        self.r = float(wheel_radius)
        self.L = float(wheel_base)
        self.max_linear_velocity = float(max_linear_velocity)
        self.max_angular_velocity = float(max_angular_velocity)
        
        # Si no se especifica, se deduce del límite lineal y angular combinado
        if max_wheel_speed is None:
            self.max_wheel_speed = (self.max_linear_velocity + (self.max_angular_velocity * self.L / 2.0)) / self.r
        else:
            self.max_wheel_speed = float(max_wheel_speed)

    def inverse_kinematics(self, v: float, omega: float) -> Tuple[float, float]:
        """Calcula las velocidades angulares de las ruedas [w_left, w_right] a partir de (v, omega).

        Args:
            v: Velocidad lineal longitudinal (m/s).
            omega: Velocidad angular en torno al eje Z (rad/s, positivo hacia la izquierda).

        Returns:
            Tuple[float, float]: (w_left, w_right) en radianes por segundo (rad/s).
        """
        # Saturación de consignas globales
        v_clamped = np.clip(v, -self.max_linear_velocity, self.max_linear_velocity)
        omega_clamped = np.clip(omega, -self.max_angular_velocity, self.max_angular_velocity)

        # Cinemática inversa:
        # w_L = (v - omega * L / 2) / r
        # w_R = (v + omega * L / 2) / r
        w_left = (v_clamped - (omega_clamped * self.L / 2.0)) / self.r
        w_right = (v_clamped + (omega_clamped * self.L / 2.0)) / self.r

        # Si alguna rueda excede el límite máximo físico, escalar proporcionalmente
        # para conservar el radio de curvatura (dirección) del movimiento
        max_calc = max(abs(w_left), abs(w_right))
        if max_calc > self.max_wheel_speed and max_calc > 1e-6:
            scale = self.max_wheel_speed / max_calc
            w_left *= scale
            w_right *= scale

        return float(w_left), float(w_right)

    def forward_kinematics(self, w_left: float, w_right: float) -> Tuple[float, float]:
        """Calcula las velocidades del robot (v, omega) a partir de las velocidades de rueda.

        Args:
            w_left: Velocidad angular rueda izquierda (rad/s).
            w_right: Velocidad angular rueda derecha (rad/s).

        Returns:
            Tuple[float, float]: (v, omega) en (m/s, rad/s).
        """
        v = self.r * (w_right + w_left) / 2.0
        omega = self.r * (w_right - w_left) / self.L
        return float(v), float(omega)

    def inverse_kinematics_torch(self, v, omega):
        """Vectorized inverse kinematics for Isaac Lab tensors."""
        import torch

        v_clamped = torch.clamp(v, -self.max_linear_velocity, self.max_linear_velocity)
        omega_clamped = torch.clamp(omega, -self.max_angular_velocity, self.max_angular_velocity)
        w_left = (v_clamped - omega_clamped * self.L / 2.0) / self.r
        w_right = (v_clamped + omega_clamped * self.L / 2.0) / self.r
        max_calc = torch.maximum(torch.abs(w_left), torch.abs(w_right))
        scale = torch.clamp(self.max_wheel_speed / torch.clamp(max_calc, min=1e-6), max=1.0)
        return torch.stack((w_left * scale, w_right * scale), dim=-1)


try:
    from isaacsim.core.utils.types import ArticulationAction

    class TurtleBot3DifferentialController:
        """Controlador de alto nivel para aplicar velocidades (v, omega) sobre un WheeledRobot o Robot."""

        def __init__(
            self,
            wheel_radius: float = 0.033,
            wheel_base: float = 0.160,
            max_linear_velocity: float = 0.22,
            max_angular_velocity: float = 2.84,
        ):
            self.kinematics = TurtleBot3Kinematics(
                wheel_radius=wheel_radius,
                wheel_base=wheel_base,
                max_linear_velocity=max_linear_velocity,
                max_angular_velocity=max_angular_velocity,
            )

        def forward(self, command: Union[np.ndarray, list, Tuple[float, float]]) -> ArticulationAction:
            """Convierte un vector de velocidad [v, omega] en una ArticulationAction para las ruedas.

            Args:
                command: Array o tupla con [v, omega].

            Returns:
                ArticulationAction con joint_velocities=[w_left, w_right].
            """
            v, omega = float(command[0]), float(command[1])
            w_left, w_right = self.kinematics.inverse_kinematics(v, omega)
            return ArticulationAction(joint_velocities=np.array([w_left, w_right]))

except ImportError:
    # Isaac Sim no está cargado en este contexto (ej. script de utilidad estándar)
    pass

