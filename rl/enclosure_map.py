"""Recinto cerrado de la Fase 2 (3.8 x 2.4 m): circuito compacto con columnas de slalom y cajas moviles.

Todas las coordenadas son locales al entorno (origen en el centro del recinto).
Recinto interior: x in [-1.9, 1.9], y in [-1.2, 1.2] (escala reducida un tercio).

Distribucion (vista cenital, circuito de slalom con 11 columnas estaticas):

    y=+1.2 +----------------------------------------------+
           | (col_11) (col_6)      (col_5)     (col_4) (col_10) |
           |  <---------------------------------------- ^ |
    y= 0   |                   (col_7)                    |
           | v ---------------------------------------->  |
           | (col_8)  (col_1)      (col_2)     (col_3) (col_9)  |
    y=-1.2 +----------------------------------------------+
"""

from __future__ import annotations

import math

import numpy as np
import torch

# Color azul para los obstaculos que el robot debe esquivar visualmente (columnas).
OBSTACLE_COLOR = (0.15, 0.35, 0.95)
# Color gris neutro para los muros perimetrales exteriores para no saturar el fondo de azul.
WALL_COLOR = (0.52, 0.54, 0.58)
WALL_HEIGHT = 0.35

# Muros perimetrales exteriores (nombre, cx, cy, sx, sy, altura) — recinto 3.8 x 2.4 m
STATIC_BOXES = [
    ("wall_n", 0.0, 1.25, 4.0, 0.1, WALL_HEIGHT),
    ("wall_s", 0.0, -1.25, 4.0, 0.1, WALL_HEIGHT),
    ("wall_e", 1.95, 0.0, 0.1, 2.6, WALL_HEIGHT),
    ("wall_w", -1.95, 0.0, 0.1, 2.6, WALL_HEIGHT),
]

# Columnas azules distribuidas por el recorrido (nombre, cx, cy, radio, altura)
COLUMNS = [
    # 6 columnas del slalom principal
    ("column_1", -0.95, -0.38, 0.11, 0.40),
    ("column_2", 0.00, -0.78, 0.11, 0.40),
    ("column_3", 0.95, -0.38, 0.11, 0.40),
    ("column_4", 0.95, 0.38, 0.11, 0.40),
    ("column_5", 0.00, 0.78, 0.11, 0.40),
    ("column_6", -0.95, 0.38, 0.11, 0.40),
    # 1 pilar central + 4 pilares en las esquinas
    ("column_7", 0.00, 0.00, 0.11, 0.40),
    ("column_8", -1.60, -0.88, 0.11, 0.40),
    ("column_9", 1.60, -0.88, 0.11, 0.40),
    ("column_10", 1.60, 0.88, 0.11, 0.40),
    ("column_11", -1.60, 0.88, 0.11, 0.40),
]

# Sin cajas moviles: todos los obstaculos interiores son columnas como las demas
MOVING_BOXES: list = []

# Ruta suave que serpentea entre las 6 columnas (escala 2/3)
PATH_WAYPOINTS = [
    (-1.45, -0.58),
    (-0.95, -0.74),
    (-0.48, -0.54),
    (0.00, -0.40),
    (0.48, -0.54),
    (0.95, -0.74),
    (1.45, -0.52),
    (1.52, 0.00),
    (1.45, 0.52),
    (0.95, 0.74),
    (0.48, 0.54),
    (0.00, 0.40),
    (-0.48, 0.54),
    (-0.95, 0.74),
    (-1.45, 0.58),
    (-1.52, 0.00),
]
PATH_SPACING = 0.02


def build_assets() -> dict:
    """Devuelve las configuraciones de escena (estaticas y moviles) del recinto."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

    assets: dict = {}
    wall_material = sim_utils.PreviewSurfaceCfg(diffuse_color=WALL_COLOR)
    obstacle_material = sim_utils.PreviewSurfaceCfg(diffuse_color=OBSTACLE_COLOR)
    collision = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
    for name, cx, cy, sx, sy, height in STATIC_BOXES:
        assets[name] = AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            spawn=sim_utils.CuboidCfg(
                size=(sx, sy, height), visual_material=wall_material, collision_props=collision
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(cx, cy, height / 2.0)),
        )
    for name, cx, cy, radius, height in COLUMNS:
        assets[name] = AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            spawn=sim_utils.CylinderCfg(
                radius=radius, height=height, visual_material=obstacle_material, collision_props=collision
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(cx, cy, height / 2.0)),
        )
    for name, cx, cy, amp, period, half in MOVING_BOXES:
        assets[name] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            spawn=sim_utils.CuboidCfg(
                size=(2 * half, 2 * half, 0.35),
                visual_material=obstacle_material,
                rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
                collision_props=collision,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(cx, cy, 0.175)),
        )
    return assets


def _densify(points: np.ndarray, max_step: float) -> np.ndarray:
    out = []
    n = len(points)
    for i in range(n):
        p0, p1 = points[i], points[(i + 1) % n]
        steps = max(1, int(math.ceil(np.linalg.norm(p1 - p0) / max_step)))
        for k in range(steps):
            out.append(p0 + (p1 - p0) * (k / steps))
    return np.array(out)


def _chaikin(points: np.ndarray, iterations: int) -> np.ndarray:
    for _ in range(iterations):
        new = []
        n = len(points)
        for i in range(n):
            p0, p1 = points[i], points[(i + 1) % n]
            new.append(0.75 * p0 + 0.25 * p1)
            new.append(0.25 * p0 + 0.75 * p1)
        points = np.array(new)
    return points


def build_path() -> np.ndarray:
    """Ruta cerrada suavizada, remuestreada cada PATH_SPACING metros."""
    pts = _densify(np.array(PATH_WAYPOINTS, dtype=np.float64), 0.3)
    pts = _chaikin(pts, 3)
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    n = int(cum[-1] / PATH_SPACING)
    s = np.arange(n) * PATH_SPACING
    return np.stack([np.interp(s, cum, closed[:, 0]), np.interp(s, cum, closed[:, 1])], axis=1)


class EnclosureMap:
    """Ruta y consultas de colision 2D vectorizadas en torch."""

    def __init__(self, device: str):
        self.device = device
        path = build_path()
        self.path = torch.tensor(path, dtype=torch.float32, device=device)
        self.num_points = self.path.shape[0]
        self.length = self.num_points * PATH_SPACING
        fwd = torch.roll(self.path, -5, dims=0)
        bwd = torch.roll(self.path, 5, dims=0)
        tangent = fwd - bwd
        tangent = tangent / torch.clamp(torch.linalg.vector_norm(tangent, dim=1, keepdim=True), min=1e-6)
        self.tangent = tangent
        self.normal = torch.stack((-tangent[:, 1], tangent[:, 0]), dim=1)

        boxes = [(cx, cy, sx / 2.0, sy / 2.0) for _, cx, cy, sx, sy, _ in STATIC_BOXES]
        self.static_boxes = torch.tensor(boxes, dtype=torch.float32, device=device)
        self.circles = torch.tensor([(cx, cy, r) for _, cx, cy, r, _ in COLUMNS], dtype=torch.float32, device=device)
        self.moving_x = torch.tensor([b[1] for b in MOVING_BOXES], dtype=torch.float32, device=device)
        self.moving_y0 = torch.tensor([b[2] for b in MOVING_BOXES], dtype=torch.float32, device=device)
        self.moving_amp = torch.tensor([b[3] for b in MOVING_BOXES], dtype=torch.float32, device=device)
        self.moving_period = torch.tensor([b[4] for b in MOVING_BOXES], dtype=torch.float32, device=device)
        self.moving_half = torch.tensor([b[5] for b in MOVING_BOXES], dtype=torch.float32, device=device)
        self.num_moving = len(MOVING_BOXES)

    def path_at(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Posicion, tangente y normal de la ruta para longitudes de arco ``s`` [E]."""
        u = torch.remainder(s, self.length) / PATH_SPACING
        i0 = torch.floor(u).long() % self.num_points
        i1 = (i0 + 1) % self.num_points
        frac = (u - torch.floor(u)).unsqueeze(1)
        pos = self.path[i0] * (1.0 - frac) + self.path[i1] * frac
        return pos, self.tangent[i0], self.normal[i0]

    def moving_centers(self, time: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """Centros [E,B,2] de las cajas moviles. ``phase`` es [E,B]."""
        if self.num_moving == 0:
            return torch.empty((time.shape[0], 0, 2), dtype=torch.float32, device=self.device)
        angle = 2.0 * math.pi * time.unsqueeze(1) / self.moving_period.unsqueeze(0) + phase
        y = self.moving_y0.unsqueeze(0) + self.moving_amp.unsqueeze(0) * torch.sin(angle)
        x = self.moving_x.unsqueeze(0).expand_as(y)
        return torch.stack((x, y), dim=-1)

    def signed_distance(self, p: torch.Tensor, moving: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Distancia con signo [E,Q] y normal exterior [E,Q,2] a todos los obstaculos."""
        num_env = p.shape[0]
        static_c = self.static_boxes[:, :2].unsqueeze(0).expand(num_env, -1, -1)
        if self.num_moving > 0 and moving.numel() > 0:
            centers = torch.cat((static_c, moving), dim=1)
            half = torch.cat((self.static_boxes[:, 2:], self.moving_half.unsqueeze(1).expand(-1, 2)), dim=0)
        else:
            centers = static_c
            half = self.static_boxes[:, 2:]
        d = p.unsqueeze(1) - centers
        q = d.abs() - half.unsqueeze(0)
        q_pos = torch.clamp(q, min=0.0)
        outside = torch.linalg.vector_norm(q_pos, dim=-1)
        inside = torch.clamp(torch.maximum(q[..., 0], q[..., 1]), max=0.0)
        sd_box = outside + inside
        sign = torch.where(d >= 0.0, 1.0, -1.0)
        n_out = sign * q_pos / torch.clamp(outside, min=1e-6).unsqueeze(-1)
        axis_x = (q[..., 0] > q[..., 1]).float()
        n_in = torch.stack((axis_x * sign[..., 0], (1.0 - axis_x) * sign[..., 1]), dim=-1)
        n_box = torch.where((outside > 1e-6).unsqueeze(-1), n_out, n_in)

        dc = p.unsqueeze(1) - self.circles[:, :2].unsqueeze(0)
        dist = torch.linalg.vector_norm(dc, dim=-1)
        sd_circ = dist - self.circles[:, 2].unsqueeze(0)
        n_circ = dc / torch.clamp(dist, min=1e-6).unsqueeze(-1)
        return torch.cat((sd_box, sd_circ), dim=1), torch.cat((n_box, n_circ), dim=1)

    def clearance(self, p: torch.Tensor, moving: torch.Tensor) -> torch.Tensor:
        """Distancia desde el punto a la superficie del obstaculo mas cercano [E]."""
        sd, _ = self.signed_distance(p, moving)
        return sd.min(dim=1).values

    def resolve(self, p: torch.Tensor, radius: float, moving: torch.Tensor, iterations: int = 3) -> torch.Tensor:
        """Expulsa un disco de ``radius`` fuera de los obstaculos (deslizamiento)."""
        for _ in range(iterations):
            sd, n = self.signed_distance(p, moving)
            best = sd.argmin(dim=1)
            idx = torch.arange(p.shape[0], device=p.device)
            depth = torch.clamp(radius - sd[idx, best], min=0.0)
            p = p + n[idx, best] * depth.unsqueeze(1)
        return p
