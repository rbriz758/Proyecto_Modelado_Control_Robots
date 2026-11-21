"""Sensor de vision artificial para TurtleBot3 Burger en Isaac Sim.

Este modulo gestiona la camara RGB montada en el chasis del TurtleBot3,
la captura de frames y la extraccion de caracteristicas visuales del objetivo.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from isaacsim.sensors.camera import Camera
except ImportError:
    Camera = None

try:
    from isaacsim.core.utils.stage import get_current_stage
except ImportError:
    get_current_stage = None


class VisualTargetFeatures:
    """Contenedor de caracteristicas visuales extraidas del frame RGB."""

    def __init__(
        self,
        detected: bool = False,
        center_x: float = 0.0,
        center_y: float = 0.0,
        pixel_count: int = 0,
        area_ratio: float = 0.0,
    ):
        self.detected = bool(detected)
        # Coordenadas normalizadas en [-1.0, 1.0], donde (0, 0) es el centro optico
        # center_x > 0 significa objetivo a la derecha
        # center_x < 0 significa objetivo a la izquierda
        self.center_x = float(center_x)
        self.center_y = float(center_y)
        self.pixel_count = int(pixel_count)
        self.area_ratio = float(area_ratio)

    def __repr__(self) -> str:
        return (
            f"VisualTargetFeatures(detected={self.detected}, "
            f"cx={self.center_x:+.2f}, cy={self.center_y:+.2f}, "
            f"pixels={self.pixel_count}, area_ratio={self.area_ratio:.3f})"
        )


class TurtleBot3Camera:
    """Camara RGB integrada en el TurtleBot3 Burger.

    Configuracion por defecto:
        - Ubicacion relativa al chasis (base_link): [0.05, 0.0, 0.12] m
        - Orientacion por defecto: Apuntando hacia adelante (+X del robot)
        - Resolucion estandar: 128x128 (balance ideal calidad / rendimiento para RL)
    """

    def __init__(
        self,
        robot_prim_path: str = "/World/TurtleBot3Burger",
        camera_subpath: str = "Geometry/base_footprint/base_link/rgb_camera",
        name: str = "turtlebot3_rgb_camera",
        resolution: Tuple[int, int] = (128, 128),
        translation: Tuple[float, float, float] = (0.05, 0.0, 0.12),
        orientation: Optional[np.ndarray] = None,
    ):
        self.prim_path = f"{robot_prim_path}/{camera_subpath}"
        self.name = name
        self.resolution = resolution
        self.translation = np.array(translation, dtype=np.float32)
        self.orientation = orientation
        self._camera: Optional[Camera] = None
        self._initialized = False

        if Camera is not None:
            self._camera = Camera(
                prim_path=self.prim_path,
                name=self.name,
                translation=self.translation,
                orientation=self.orientation,
                resolution=self.resolution,
            )

    def initialize(self) -> None:
        """Inicializa los buffers y render products de la camara."""
        if self._camera is None:
            raise RuntimeError("Isaac Sim no esta disponible para inicializar la camara")

        self._camera.initialize()

        if get_current_stage is None:
            raise RuntimeError("No se puede validar el prim de la camara en Isaac Sim")

        prim = get_current_stage().GetPrimAtPath(self.prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"El prim de la camara no existe: {self.prim_path}")

        self._initialized = True

    def get_rgb(self) -> Optional[np.ndarray]:
        """Obtiene el ultimo frame RGB renderizado como array uint8 (H, W, 3)."""
        if self._camera is None or not self._initialized:
            return None
        frame = self._camera.get_rgb()
        if frame is not None and len(frame.shape) == 3 and frame.shape[2] >= 3:
            return frame[:, :, :3]
        return frame

    def extract_red_target_features(
        self,
        rgb_image: Optional[np.ndarray] = None,
        min_pixels: int = 15,
    ) -> VisualTargetFeatures:
        """Segmenta el objetivo de color rojo y extrae su centroide y tamaño aparente.

        Args:
            rgb_image: Frame RGB (opcional, si es None se captura el actual).
            min_pixels: Numero minimo de pixeles para considerar deteccion valida.

        Returns:
            VisualTargetFeatures con centroide normalizado y ratio de ocupacion.
        """
        if rgb_image is None:
            rgb_image = self.get_rgb()

        if rgb_image is None:
            return VisualTargetFeatures(detected=False)

        h, w, _ = rgb_image.shape
        r = rgb_image[:, :, 0].astype(np.int16)
        g = rgb_image[:, :, 1].astype(np.int16)
        b = rgb_image[:, :, 2].astype(np.int16)

        # Criterio robusto de color rojo frente al entorno e iluminacion
        red_mask = (r > 110) & (r > (g + 30)) & (r > (b + 30))
        pixel_count = int(np.sum(red_mask))

        if pixel_count < min_pixels:
            return VisualTargetFeatures(detected=False, pixel_count=pixel_count)

        ys, xs = np.where(red_mask)
        raw_cx = float(np.mean(xs))
        raw_cy = float(np.mean(ys))

        # Normalizar a rango [-1.0, 1.0] con respecto al centro de la imagen
        norm_cx = (raw_cx - (w / 2.0)) / (w / 2.0)
        norm_cy = (raw_cy - (h / 2.0)) / (h / 2.0)
        area_ratio = float(pixel_count) / float(h * w)

        return VisualTargetFeatures(
            detected=True,
            center_x=norm_cx,
            center_y=norm_cy,
            pixel_count=pixel_count,
            area_ratio=area_ratio,
        )

    def save_snapshot(self, output_path: str, rgb_image: Optional[np.ndarray] = None) -> bool:
        """Guarda un fotograma actual en disco como imagen PNG/JPEG."""
        if Image is None:
            return False
        if rgb_image is None:
            rgb_image = self.get_rgb()
        if rgb_image is None:
            return False

        img = Image.fromarray(rgb_image.astype(np.uint8))
        img.save(output_path)
        return True

