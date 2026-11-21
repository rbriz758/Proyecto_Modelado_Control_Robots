"""Direct Isaac Lab environment for compact visual TurtleBot3 following.

The policy receives visual features rather than simulator state: target detection,
normalized horizontal and vertical image errors, and apparent target area. The
features are extracted from the batched RGB camera with the same red-segmentation
thresholds as the standalone TurtleBot3Camera sensor.

Fase 2 (recinto cerrado): se anaden 5 sectores de obstaculo (segmentacion azul de la
misma camara RGB). El robot solo sigue a la bola; si no la ve, se queda quieto.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from pathlib import Path

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
try:
    from scripts.controlador_diferencial import TurtleBot3Kinematics
except ImportError:
    from controlador_diferencial import TurtleBot3Kinematics
try:
    from rl.enclosure_map import EnclosureMap, MOVING_BOXES, build_assets
except ImportError:
    from enclosure_map import EnclosureMap, MOVING_BOXES, build_assets


USD_PATH = str(PROJECT_ROOT / "assets/turtlebot3_usd/turtlebot3_burger/turtlebot3_burger.usda")


@configclass
class TurtleBot3FollowSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=USD_PATH,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.01)),
        actuators={
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=["wheel_.*_joint"],
                stiffness=0.0,
                damping=100.0,
                effort_limit=50.0,
                velocity_limit=20.0,
            )
        },
    )
    target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        spawn=sim_utils.SphereCfg(
            radius=0.15,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.05, 0.05)),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.5, 0.0, 0.2)),
    )
    obstacle_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle1",
        spawn=sim_utils.CylinderCfg(
            radius=0.16,
            height=0.6,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.20, 0.25)),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.8, 0.65, 0.3)),
    )
    obstacle = obstacle_1
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_footprint/base_link/rgb_camera",
        spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.05, 10.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.05, 0.0, 0.12), convention="world"),
        data_types=["rgb"],
        width=128,
        height=128,
    )

@configclass
class TurtleBot3FollowEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 12.0
    action_space = 2
    observation_space = 4
    state_space = 0
    sim = SimulationCfg(dt=1.0 / 120.0, render_interval=decimation)
    scene = TurtleBot3FollowSceneCfg(num_envs=16, env_spacing=4.0, replicate_physics=True, clone_in_fabric=False)

    phase: str = "1.5"
    scenario: str = "mixto"  # mixto | circular | slalom | stop_and_go | acelerones | erratico

    max_linear_velocity = 0.22
    max_angular_velocity = 2.84
    target_area_reference = 0.20
    target_area_scale = 0.06
    target_radius = 0.15
    robot_radius = 0.12
    obstacle_radius = 0.25
    collision_distance = 0.37
    target_collision_distance = 0.30
    max_target_distance = 4.0
    success_distance = 0.9
    success_center_error = 0.15

    target_speed_linear = 0.12
    target_speed_angular = 0.30
    target_start_min_dist = 0.70
    target_start_max_dist = 1.10

    reward_center = 1.0
    reward_area = 0.8
    reward_detected = 0.2
    reward_stable = 0.15
    penalty_lost = 0.5
    penalty_action_change = 0.04
    penalty_collision = 8.0
    reward_success = 2.0

    # --- Fase 2 (mapa del recinto con obstaculos) ---
    robot_clearance = 0.12
    max_lost_seconds = 6.0
    action_smoothing = 1.0  # 1.0 = sin filtro; <1 suaviza las ordenes a las ruedas
    penalty_angular = 0.0
    max_dv_step = 2.0  # cambio maximo de la accion lineal por paso (2.0 = sin limite)
    max_dw_step = 2.0  # idem para la accion angular


def configure_phase(env_cfg: TurtleBot3FollowEnvCfg, phase: str | float) -> TurtleBot3FollowEnvCfg:
    """Configura los hiperparametros del entorno segun la fase del proyecto."""
    phase_str = str(phase)
    env_cfg.phase = phase_str
    if phase_str == "1":
        env_cfg.target_area_reference = 0.06
        env_cfg.target_area_scale = 0.04
        env_cfg.target_speed_linear = 0.25
        env_cfg.target_speed_angular = 0.45
        env_cfg.target_start_min_dist = 1.2
        env_cfg.target_start_max_dist = 2.0
        env_cfg.penalty_lost = 0.5
        env_cfg.penalty_collision = 8.0
    elif phase_str == "1.5":
        env_cfg.max_linear_velocity = 0.35
        env_cfg.target_radius = 0.08
        env_cfg.target_collision_distance = 0.23
        env_cfg.target_area_reference = 0.06
        env_cfg.target_area_scale = 0.025
        env_cfg.target_speed_linear = 0.22
        env_cfg.target_speed_angular = 0.35
        env_cfg.target_start_min_dist = 0.75
        env_cfg.target_start_max_dist = 1.15
        env_cfg.max_target_distance = 5.0
        env_cfg.episode_length_s = 20.0
        env_cfg.penalty_lost = 0.5
        env_cfg.penalty_action_change = 0.20
        env_cfg.penalty_collision = 12.0
        env_cfg.scene.target.spawn.radius = 0.08
        env_cfg.scene.obstacle_1 = None
        env_cfg.scene.obstacle = None
    elif phase_str == "2":
        # Solo carga el mapa del recinto con los obstaculos (sin planteamiento RL adicional)
        env_cfg.max_linear_velocity = 0.35
        env_cfg.target_radius = 0.08
        env_cfg.target_collision_distance = 0.23
        env_cfg.target_area_reference = 0.06
        env_cfg.target_area_scale = 0.025
        env_cfg.target_speed_linear = 0.18
        env_cfg.target_start_min_dist = 0.45
        env_cfg.target_start_max_dist = 0.65
        env_cfg.max_target_distance = 4.0
        env_cfg.penalty_lost = 0.5
        env_cfg.penalty_action_change = 0.20
        env_cfg.penalty_collision = 12.0
        env_cfg.episode_length_s = 30.0
        scene = TurtleBot3FollowSceneCfg(
            num_envs=env_cfg.scene.num_envs, env_spacing=5.5, replicate_physics=True, clone_in_fabric=False
        )
        scene.obstacle_1 = None
        scene.obstacle = None
        scene.target.spawn.radius = 0.08
        scene.robot.init_state.pos = (-1.45, -0.58, 0.01)
        scene.target.init_state.pos = (-0.85, -0.58, 0.14)
        for name, asset_cfg in build_assets().items():
            setattr(scene, name, asset_cfg)
        env_cfg.scene = scene
    return env_cfg


class TurtleBot3FollowEnv(DirectRLEnv):
    """Vectorized visual-feature following task for TurtleBot3 Burger."""

    cfg: TurtleBot3FollowEnvCfg

    def __init__(self, cfg: TurtleBot3FollowEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.enclosure = cfg.phase == "2"
        self.robot = self.scene["robot"]
        self.target = self.scene["target"]
        self.camera = self.scene["camera"]
        wheel_ids, _ = self.robot.find_joints(["wheel_left_joint", "wheel_right_joint"], preserve_order=True)
        self._wheel_ids = torch.tensor(wheel_ids, device=self.device, dtype=torch.long)
        self.kinematics = TurtleBot3Kinematics(
            max_linear_velocity=cfg.max_linear_velocity,
            max_angular_velocity=cfg.max_angular_velocity,
        )
        self.obs_dim = int(cfg.observation_space)
        self.actions = torch.zeros((self.num_envs, 2), device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)
        self.trajectory_type = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.trajectory_speed = torch.ones(self.num_envs, device=self.device)
        self.direction = torch.ones(self.num_envs, device=self.device)
        self.stop_offset = torch.zeros(self.num_envs, device=self.device)
        self.speed_mode = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.target_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self.target_heading = torch.zeros(self.num_envs, device=self.device)
        self.turn_w1 = torch.ones(self.num_envs, device=self.device)
        self.turn_w2 = torch.ones(self.num_envs, device=self.device)
        self.turn_f1 = torch.ones(self.num_envs, device=self.device)
        self.turn_f2 = torch.ones(self.num_envs, device=self.device)
        self.turn_p1 = torch.zeros(self.num_envs, device=self.device)
        self.turn_p2 = torch.zeros(self.num_envs, device=self.device)
        self.last_side = torch.zeros(self.num_envs, device=self.device)
        self.target_start = torch.zeros((self.num_envs, 2), device=self.device)
        self.target_world = torch.zeros((self.num_envs, 3), device=self.device)
        self.observation_features = torch.zeros((self.num_envs, self.obs_dim), device=self.device)
        self.collision = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.success = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.enclosure:
            self.map = EnclosureMap(self.device)
            self.boxes = [self.scene[name] for name, *_ in MOVING_BOXES]
            self.path_s0 = torch.zeros(self.num_envs, device=self.device)
            self.box_phase = torch.zeros((self.num_envs, len(self.boxes)), device=self.device)
            self.wobble_phase = torch.zeros(self.num_envs, device=self.device)
            self.wobble_amp = torch.full((self.num_envs,), 0.12, device=self.device)
            self.wobble_freq = torch.full((self.num_envs,), 0.7, device=self.device)
            self.direction = torch.ones(self.num_envs, device=self.device)
            self.stop_offset = torch.zeros(self.num_envs, device=self.device)
            self.lost_steps = torch.zeros(self.num_envs, device=self.device)
            self.clearance = torch.full((self.num_envs,), 10.0, device=self.device)
            self.obstacle_1 = None
            self.obstacle = None
        elif "obstacle_1" in self.scene.keys():
            self.obstacle_1 = self.scene["obstacle_1"]
            self.obstacle = self.obstacle_1
        else:
            self.obstacle_1 = None
            self.obstacle = None

    def _setup_scene(self):
        pass

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        raw = torch.clamp(actions, -1.0, 1.0)
        self._raw_actions = raw
        # Dinamica del motor: filtro de primer orden (1.0 = sin suavizado)
        alpha = self.cfg.action_smoothing
        desired = alpha * raw + (1.0 - alpha) * self.previous_actions
        max_step = torch.tensor([self.cfg.max_dv_step, self.cfg.max_dw_step], device=self.device)
        delta = torch.clamp(desired - self.previous_actions, -max_step, max_step)
        self.actions = self.previous_actions + delta
        self._update_target()
        wheel_velocities = self.kinematics.inverse_kinematics_torch(
            self.actions[:, 0] * self.cfg.max_linear_velocity,
            self.actions[:, 1] * self.cfg.max_angular_velocity,
        )
        self.robot.set_joint_velocity_target_index(target=wheel_velocities, joint_ids=self._wheel_ids)

    def _apply_action(self) -> None:
        pass

    def _get_observations(self) -> dict:
        self.observation_features = self._extract_visual_features_from_rgb()
        return {"policy": self.observation_features}

    def _get_rewards(self) -> torch.Tensor:
        features = self.observation_features
        detected = features[:, 0]
        center_error = torch.abs(features[:, 1])
        area_error = torch.abs(features[:, 3] - self.cfg.target_area_reference)
        area_scale = -18.0 if self.cfg.target_radius < 0.12 else -6.0
        area_kernel = torch.exp(area_scale * area_error)
        action_change = torch.sum(torch.square(self._raw_actions - self.previous_actions), dim=1)
        stable = torch.exp(-4.0 * center_error) * area_kernel
        success = (
            (detected > 0.5)
            & (center_error < self.cfg.success_center_error)
            & (features[:, 3] > self.cfg.target_area_reference - self.cfg.target_area_scale)
            & (features[:, 3] < self.cfg.target_area_reference + self.cfg.target_area_scale)
        )
        reward = (
            self.cfg.reward_center * detected * (1.0 - center_error)
            + self.cfg.reward_area * detected * area_kernel
            + self.cfg.reward_detected * detected
            + self.cfg.reward_stable * stable
            - self.cfg.penalty_lost * (1.0 - detected)
            - self.cfg.penalty_action_change * action_change
            - self.cfg.penalty_collision * self.collision.float()
            + self.cfg.reward_success * success.float()
        )
        if self.cfg.target_radius < 0.12:
            # Freno suave si invade la distancia de seguridad de la bola
            too_close = torch.clamp(features[:, 3] - self.cfg.target_area_reference * 1.35, min=0.0)
            reward -= detected * too_close * 15.0
            # Evitar micro-oscilaciones de giro cuando la bola ya esta centrada
            centered = (center_error < 0.08).float()
            reward -= detected * centered * 0.25 * torch.square(self.actions[:, 1])

        self.success = success
        jitter = torch.abs(self.actions - self.previous_actions).mean()
        self.previous_actions.copy_(self.actions)
        self.extras.setdefault("log", {})["Metrics/mean_center_error"] = center_error.mean()
        self.extras["log"]["Metrics/mean_area"] = features[:, 3].mean()
        self.extras["log"]["Metrics/action_jitter"] = jitter
        self.extras["log"]["Metrics/area_error_detected"] = (area_error * detected).sum() / detected.sum().clamp(min=1.0)
        self.extras["log"]["Metrics/detection_rate"] = detected.mean()
        self.extras["log"]["Metrics/collision_rate"] = self.collision.float().mean()
        self.extras["log"]["Metrics/success_rate"] = success.float().mean()
        return reward * self.step_dt

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        robot_pos = self.robot.data.root_pos_w.torch
        target_pos = self.target.data.root_pos_w.torch
        robot_target_distance = torch.linalg.vector_norm(robot_pos[:, :2] - target_pos[:, :2], dim=1)

        if self.enclosure:
            local = robot_pos[:, :2] - self.scene.env_origins[:, :2]
            self.clearance = self.map.clearance(local, self._moving_centers())
            obstacle_collision = self.clearance < self.cfg.robot_clearance
            self.lost_steps = torch.where(
                self.observation_features[:, 0] > 0.5, torch.zeros_like(self.lost_steps), self.lost_steps + 1.0
            )
            lost_too_long = self.lost_steps * self.step_dt > self.cfg.max_lost_seconds
        elif self.obstacle_1 is not None:
            obs1_pos = self.obstacle_1.data.root_pos_w.torch
            d1 = torch.linalg.vector_norm(robot_pos[:, :2] - obs1_pos[:, :2], dim=1)
            obstacle_collision = (d1 < self.cfg.collision_distance) & (self.cfg.phase == "1")
            lost_too_long = torch.zeros_like(obstacle_collision)
        else:
            obstacle_collision = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            lost_too_long = obstacle_collision

        target_collision = robot_target_distance < self.cfg.target_collision_distance
        self.collision = obstacle_collision | target_collision
        terminated = self.collision | (robot_target_distance > self.cfg.max_target_distance) | lost_too_long
        timeout = self.episode_length_buf >= self.max_episode_length
        return terminated, timeout

    # ------------------------------------------------------------------
    # Fase 2: recinto cerrado
    # ------------------------------------------------------------------
    def _moving_centers(self) -> torch.Tensor:
        time = self.episode_length_buf.float() * self.step_dt
        return self.map.moving_centers(time, self.box_phase)

    def _write_moving_boxes(self, env_ids=None) -> None:
        centers = self._moving_centers()
        for k, box in enumerate(self.boxes):
            pose = torch.zeros((self.num_envs, 7), device=self.device)
            pose[:, 6] = 1.0
            pose[:, 0:2] = centers[:, k] + self.scene.env_origins[:, :2]
            pose[:, 2] = 0.15
            if env_ids is None:
                box.write_root_pose_to_sim_index(root_pose=pose, env_ids=None)
            else:
                box.write_root_pose_to_sim_index(root_pose=pose[env_ids], env_ids=env_ids)

    def _enclosure_target_xy(self) -> torch.Tensor:
        """Posicion de la bola sobre la ruta segun su comportamiento.

        Tipos: 0 velocidad constante, 1 velocidad variable (frena/acelera),
        2 stop-and-go con pausas reales, 3 zigzag lateral amplio.
        El sentido de la vuelta (horario/antihorario) es aleatorio por episodio.
        """
        time = self.episode_length_buf.float() * self.step_dt
        speed = self.trajectory_speed
        v = self.cfg.target_speed_linear
        kind = self.trajectory_type

        # Tipo 1: velocidad modulada (derivada en [0.75, 1.25])
        t_var = time - (0.25 / 1.2) * torch.sin(1.2 * time)
        # Tipo 2: ciclos de 6 s con 4 s de marcha y 2 s de parada (a 1.25x para igualar la media)
        cycle, moving = 6.0, 4.0

        def eff(x: torch.Tensor) -> torch.Tensor:
            return torch.floor(x / cycle) * moving + torch.clamp(torch.remainder(x, cycle), max=moving)

        t_stop = 1.25 * (eff(time + self.stop_offset) - eff(self.stop_offset))
        t_eff = torch.where(kind == 1, t_var, torch.where(kind == 2, t_stop, time))
        s = self.path_s0 + self.direction * v * speed * t_eff
        pos, _, normal = self.map.path_at(s)
        wobble = self.wobble_amp * torch.sin(self.wobble_freq * time + self.wobble_phase)
        pos = pos + normal * wobble.unsqueeze(1)
        safe_ball_clearance = max(self.cfg.target_radius, self.cfg.robot_clearance + 0.06)
        return self.map.resolve(pos, safe_ball_clearance, self._moving_centers())

    def _reset_enclosure(self, env_ids) -> None:
        count = len(env_ids)
        origin = self.scene.env_origins[env_ids]
        self.path_s0[env_ids] = torch.rand(count, device=self.device) * self.map.length
        kind = torch.randint(0, 4, (count,), device=self.device)
        self.trajectory_type[env_ids] = kind
        self.trajectory_speed[env_ids] = torch.empty(count, device=self.device).uniform_(0.85, 1.15)
        self.direction[env_ids] = torch.where(torch.rand(count, device=self.device) < 0.5, 1.0, -1.0)
        self.stop_offset[env_ids] = torch.rand(count, device=self.device) * 6.0
        # vaiven lateral: pequeno por defecto, amplio en el tipo 3 (zigzag)
        amp = torch.empty(count, device=self.device).uniform_(0.03, 0.06)
        amp = torch.where(kind == 3, torch.empty(count, device=self.device).uniform_(0.08, 0.12), amp)
        self.wobble_amp[env_ids] = amp
        self.wobble_freq[env_ids] = torch.empty(count, device=self.device).uniform_(0.5, 1.3)
        self.wobble_phase[env_ids] = torch.zeros(count, device=self.device)
        self.box_phase[env_ids] = torch.rand((count, len(self.boxes)), device=self.device) * 2.0 * math.pi
        self.lost_steps[env_ids] = 0.0
        self.clearance[env_ids] = 10.0

        # Bola sobre la ruta; el robot detras, mirando hacia ella (siempre a la vista)
        ball_xy, _, _ = self.map.path_at(self.path_s0[env_ids])
        dist = torch.empty(count, device=self.device).uniform_(
            self.cfg.target_start_min_dist, self.cfg.target_start_max_dist
        )
        robot_xy, _, _ = self.map.path_at(self.path_s0[env_ids] - self.direction[env_ids] * dist)

        # Evitar que una caja movil o columna quede pegada al robot en t=0
        mov0 = self.map.moving_centers(torch.zeros(count, device=self.device), self.box_phase[env_ids])
        if self.map.clearance(robot_xy, mov0).min() < 0.26:
            too_close = self.map.clearance(robot_xy, mov0) < 0.26
            self.box_phase[env_ids] = torch.where(
                too_close.unsqueeze(1), self.box_phase[env_ids] + math.pi, self.box_phase[env_ids]
            )
            mov0 = self.map.moving_centers(torch.zeros(count, device=self.device), self.box_phase[env_ids])
        robot_xy = self.map.resolve(robot_xy, self.cfg.robot_clearance + 0.06, mov0)
        ball_xy = self.map.resolve(ball_xy, self.cfg.target_radius + 0.02, mov0)
        heading = torch.atan2(ball_xy[:, 1] - robot_xy[:, 1], ball_xy[:, 0] - robot_xy[:, 0])
        yaw = heading + torch.empty(count, device=self.device).uniform_(-0.25, 0.25)

        robot_pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        robot_pose[:, 0:2] = robot_xy + origin[:, :2]
        robot_pose[:, 2] = 0.01
        robot_pose[:, 3] = 0.0
        robot_pose[:, 4] = 0.0
        robot_pose[:, 5] = torch.sin(yaw / 2.0)
        robot_pose[:, 6] = torch.cos(yaw / 2.0)
        self.robot.write_root_pose_to_sim_index(root_pose=robot_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((count, 6), device=self.device), env_ids=env_ids
        )
        joint_pos = self.robot.data.default_joint_pos.torch[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        target_pose = torch.zeros((count, 7), device=self.device)
        target_pose[:, 6] = 1.0
        target_pose[:, 0:2] = ball_xy + origin[:, :2]
        target_pose[:, 2] = 0.14
        self.target.write_root_pose_to_sim_index(root_pose=target_pose, env_ids=env_ids)
        self.target.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((count, 6), device=self.device), env_ids=env_ids
        )
        self._write_moving_boxes(env_ids)

        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
        self.last_side[env_ids] = 0.0
        self.collision[env_ids] = False
        self.success[env_ids] = False

    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        if self.enclosure:
            self._reset_enclosure(env_ids)
            return
        count = len(env_ids)
        origin = self.scene.env_origins[env_ids]
        self.target_start[env_ids] = torch.stack(
            (
                torch.empty(count, device=self.device).uniform_(
                    self.cfg.target_start_min_dist, self.cfg.target_start_max_dist
                ),
                torch.empty(count, device=self.device).uniform_(-0.45, 0.45),
            ),
            dim=1,
        )
        robot_pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        robot_pose[:, :2] += torch.empty((count, 2), device=self.device).uniform_(-0.15, 0.15)
        heading = torch.atan2(
            self.target_start[env_ids, 1] - robot_pose[:, 1],
            self.target_start[env_ids, 0] - robot_pose[:, 0],
        )
        yaw = heading + torch.empty(count, device=self.device).uniform_(-0.35, 0.35)
        robot_pose[:, 3] = 0.0
        robot_pose[:, 4] = 0.0
        robot_pose[:, 5] = torch.sin(yaw / 2.0)
        robot_pose[:, 6] = torch.cos(yaw / 2.0)
        robot_pose[:, :3] += origin
        self.robot.write_root_pose_to_sim_index(root_pose=robot_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((count, 6), device=self.device), env_ids=env_ids
        )
        joint_pos = self.robot.data.default_joint_pos.torch[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        scenario_map = {"circular": 0, "slalom": 1, "stop_and_go": 2, "acelerones": 3, "erratico": 4}
        if self.cfg.scenario in scenario_map:
            self.trajectory_type[env_ids] = scenario_map[self.cfg.scenario]
        else:
            self.trajectory_type[env_ids] = torch.randint(0, 5, (count,), device=self.device)
        self.speed_mode[env_ids] = torch.randint(0, 3, (count,), device=self.device)
        self.trajectory_speed[env_ids] = torch.empty(count, device=self.device).uniform_(0.90, 1.10)
        self.direction[env_ids] = torch.where(torch.rand(count, device=self.device) < 0.5, 1.0, -1.0)
        self.stop_offset[env_ids] = (
            torch.zeros(count, device=self.device)
            if self.cfg.scenario == "stop_and_go"
            else torch.rand(count, device=self.device) * 5.5
        )
        self.target_xy[env_ids] = self.target_start[env_ids]
        init_h_jitter = 0.15 if self.cfg.scenario in ("circular", "slalom", "stop_and_go") else 0.45
        self.target_heading[env_ids] = heading + torch.empty(count, device=self.device).uniform_(
            -init_h_jitter, init_h_jitter
        )
        self.turn_w1[env_ids] = torch.empty(count, device=self.device).uniform_(0.65, 1.35)
        self.turn_w2[env_ids] = torch.empty(count, device=self.device).uniform_(0.40, 0.90)
        self.turn_f1[env_ids] = torch.empty(count, device=self.device).uniform_(0.50, 1.15)
        self.turn_f2[env_ids] = torch.empty(count, device=self.device).uniform_(1.10, 1.90)
        self.turn_p1[env_ids] = torch.rand(count, device=self.device) * 2.0 * math.pi
        self.turn_p2[env_ids] = torch.rand(count, device=self.device) * 2.0 * math.pi

        ball_z = 0.14 if self.cfg.target_radius < 0.12 else 0.20
        target_pose = torch.zeros((count, 7), device=self.device)
        target_pose[:, 6] = 1.0
        target_pose[:, :2] = self.target_start[env_ids]
        target_pose[:, 2] = ball_z
        target_pose[:, :3] += origin
        self.target.write_root_pose_to_sim_index(root_pose=target_pose, env_ids=env_ids)
        self.target.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((count, 6), device=self.device), env_ids=env_ids
        )

        if self.obstacle_1 is not None:
            obs1_pos = torch.stack(
                (
                    torch.full((count,), 0.8, device=self.device),
                    torch.full((count,), 0.7, device=self.device),
                    torch.full((count,), 0.30, device=self.device),
                ),
                dim=1,
            )
            pose1 = torch.zeros((count, 7), device=self.device)
            pose1[:, 6] = 1.0
            pose1[:, :3] = obs1_pos + origin
            self.obstacle_1.write_root_pose_to_sim_index(root_pose=pose1, env_ids=env_ids)
            self.obstacle_1.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros((count, 6), device=self.device), env_ids=env_ids
            )

        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
        self.last_side[env_ids] = 0.0
        self.collision[env_ids] = False
        self.success[env_ids] = False

    def _update_target(self) -> None:
        if self.enclosure:
            xy = self._enclosure_target_xy()
            self.target_world[:, :2] = xy + self.scene.env_origins[:, :2]
            self.target_world[:, 2] = 0.0
            pose = torch.zeros((self.num_envs, 7), device=self.device)
            pose[:, 6] = 1.0
            pose[:, :3] = self.target_world
            self.target.write_root_pose_to_sim_index(root_pose=pose, env_ids=None)
            self._write_moving_boxes(None)
            return

        time = self.episode_length_buf.float() * self.step_dt
        speed = self.trajectory_speed
        sgn = self.direction
        x0 = self.target_start[:, 0]
        y0 = self.target_start[:, 1]
        v_lin = self.cfg.target_speed_linear
        w_ang = self.cfg.target_speed_angular
        kind = self.trajectory_type

        if self.cfg.phase == "1.5":
            # 5 escenarios diferenciados de comportamiento del objetivo movil (kind in 0..4):
            # 0: circular    -> Orbita circular de radio fijo a velocidad constante (0.20 m/s)
            # 1: slalom      -> Zig-zag sinusoidal rapido en S (izquierda <-> derecha) a velocidad constante
            # 2: stop_and_go -> Avanza 3.0 s, se detiene en seco (0 m/s) durante 2.5 s y gira al arrancar
            # 3: acelerones  -> Figura en 8 alternando paso muy lento (0.07 m/s) y sprints rapidos (0.31 m/s)
            # 4: erratico    -> Quiebros bruscos e impredecibles 2D con cambios de ritmo

            # Retorno suave hacia (0.8, 0.0) cuando se aleja para mantener la accion en escena
            dx = 0.8 - self.target_xy[:, 0]
            dy = -self.target_xy[:, 1]
            dist_c = torch.hypot(dx, dy)
            ang_c = torch.atan2(dy, dx)
            diff_c = torch.atan2(torch.sin(ang_c - self.target_heading), torch.cos(ang_c - self.target_heading))
            pull = torch.clamp((dist_c - 1.8) / 0.7, min=0.0, max=1.0)

            # 0) Circular: circunferencia regular sin tirones
            w_circ = self.direction * 0.26
            v_circ = torch.full_like(time, 0.23)

            # 1) Slalom: eslalon en S marcado de lado a lado
            w_slalom = self.direction * 1.15 * torch.cos(1.45 * time) + pull * 1.5 * diff_c
            v_slalom = torch.full_like(time, 0.24)

            # 2) Stop-and-Go: 3.0 s de marcha, 2.5 s clavada a 0.0 m/s, giro limpio al reanudar
            cyc_t = torch.remainder(time + self.stop_offset, 5.5)
            in_move_sg = (cyc_t < 3.0).float()
            turning_sg = (cyc_t < 0.75).float()
            w_sg = self.direction * 1.25 * turning_sg + pull * 1.8 * diff_c * in_move_sg
            v_sg = 0.26 * in_move_sg

            # 3) Acelerones: figura en 8 amplia con alternancia entre 0.09 m/s (lento) y 0.32 m/s (sprint)
            w_acel = self.direction * 0.70 * torch.sin(0.55 * time) + pull * 1.5 * diff_c
            v_acel = 0.205 - 0.115 * torch.cos(1.35 * time)

            # 4) Erratico: quiebros 2D bruscos multifrecuencia + velocidad cambiante
            w_err = (
                self.direction * 0.55 * torch.sin(0.40 * time + self.turn_p2)
                + self.direction * (self.turn_w1 * 1.45) * torch.sin(self.turn_f1 * time + self.turn_p1)
                + (self.turn_w2 * 1.25) * torch.cos(self.turn_f2 * time + self.turn_p2)
                + pull * 2.2 * diff_c
            )
            v_err = (v_lin * speed) * (1.0 - 0.45 * torch.sin(1.6 * time + self.turn_p1))

            omega_now = torch.where(
                kind == 0,
                w_circ,
                torch.where(kind == 1, w_slalom, torch.where(kind == 2, w_sg, torch.where(kind == 3, w_acel, w_err))),
            )
            v_now = torch.where(
                kind == 0,
                v_circ,
                torch.where(kind == 1, v_slalom, torch.where(kind == 2, v_sg, torch.where(kind == 3, v_acel, v_err))),
            )

            self.target_heading = self.target_heading + omega_now * self.step_dt
            step_xy = torch.stack((torch.cos(self.target_heading), torch.sin(self.target_heading)), dim=1)
            self.target_xy = self.target_xy + step_xy * (v_now * self.step_dt).unsqueeze(1)
            x, y = self.target_xy[:, 0], self.target_xy[:, 1]
        else:
            linear_x = x0 + v_lin * speed * time
            linear_y = y0 + 0.35 * torch.sin(0.4 * speed * time)
            circular_x = 1.0 + 0.55 * torch.cos(w_ang * speed * time)
            circular_y = 0.55 * torch.sin(w_ang * speed * time)
            sinusoidal_x = x0 + (v_lin * 0.8) * speed * time
            sinusoidal_y = y0 + 0.60 * torch.sin(w_ang * speed * time)
            x = torch.where(kind == 0, linear_x, torch.where(kind == 1, circular_x, sinusoidal_x))
            y = torch.where(kind == 0, linear_y, torch.where(kind == 1, circular_y, sinusoidal_y))

        self.target_world[:, :2] = torch.stack((x, y), dim=1) + self.scene.env_origins[:, :2]
        self.target_world[:, 2] = 0.14 if self.cfg.target_radius < 0.12 else 0.20
        pose = torch.zeros((self.num_envs, 7), device=self.device)
        pose[:, 6] = 1.0
        pose[:, :3] = self.target_world
        self.target.write_root_pose_to_sim_index(root_pose=pose, env_ids=None)

    def _extract_visual_features_from_rgb(self) -> torch.Tensor:
        """Segmentacion roja (bola) y, en la Fase 2, azul (obstaculos) sobre el RGB por lotes."""
        rgb = self.camera.data.output["rgb"].torch[..., :3].float()
        red = rgb[..., 0]
        green = rgb[..., 1]
        blue = rgb[..., 2]
        red_mask = (red > 110.0) & (red > green + 30.0) & (red > blue + 30.0)
        pixel_count = red_mask.sum(dim=(1, 2)).float()
        height, width = red_mask.shape[1:]
        min_pixels = 15.0
        detected = pixel_count >= min_pixels

        x_coords = torch.arange(width, device=self.device, dtype=torch.float32).view(1, 1, width)
        y_coords = torch.arange(height, device=self.device, dtype=torch.float32).view(1, height, 1)
        safe_pixel_count = torch.clamp(pixel_count, min=1.0)
        center_x = (red_mask.float() * x_coords).sum(dim=(1, 2)) / safe_pixel_count
        center_y = (red_mask.float() * y_coords).sum(dim=(1, 2)) / safe_pixel_count
        center_x = (center_x - width / 2.0) / (width / 2.0)
        center_y = (center_y - height / 2.0) / (height / 2.0)
        area_ratio = pixel_count / float(height * width)
        self.last_side = torch.where(detected, torch.where(center_x >= 0.0, 1.0, -1.0), self.last_side)
        lost_features = torch.stack(
            (torch.zeros_like(self.last_side), self.last_side, torch.zeros_like(self.last_side), torch.zeros_like(self.last_side)),
            dim=1,
        )
        features = torch.stack((detected.float(), center_x, center_y, area_ratio), dim=1)
        features = torch.where(detected.unsqueeze(1), features, lost_features)
        return features
