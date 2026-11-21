# Guía de Comandos y Escenarios — TurtleBot3 Visual Following (RL)

*(Todos los comandos incluyen la ruta completa a `isaaclab.sh` para que puedas copiarlos y pegarlos desde cualquier carpeta de la terminal).*

---

## 1. Visualizar los 5 Escenarios de Comportamiento del Objetivo (Fase 1.5)

Carga el modelo entrenado de la Fase 1.5 (`checkpoints/fase1_5/model_fase1_5.pt`) y lanza en el simulador 3D cada uno de los **5 comportamientos diferenciados del objetivo móvil**:

### Escenario 1 — Órbita Circular a Velocidad Constante (`circular`)
La bola describe una circunferencia limpia de radio fijo a velocidad constante ($0.20\text{ m/s}$).
```bash
/home/rbriram/IsaacLab/isaaclab.sh -p "/home/rbriram/Modelado_control/proyecto/proyecto_ws/rl/play_ppo.py" --phase 1.5 --scenario circular
```

### Escenario 2 — Eslalon / Zig-Zag Sinusoidal Rápido en S (`slalom`)
La bola avanza enlazando contra-curvas rápidas izquierda $\leftrightarrow$ derecha en forma de S ($0.21\text{ m/s}$).
```bash
/home/rbriram/IsaacLab/isaaclab.sh -p "/home/rbriram/Modelado_control/proyecto/proyecto_ws/rl/play_ppo.py" --phase 1.5 --scenario slalom
```

### Escenario 3 — Marcha Intermitente con Paradas en Seco (`stop_and_go`)
La bola avanza durante $3.0\text{ s}$, se detiene por completo ($0.00\text{ m/s}$) durante $2.5\text{ s}$ y cambia de dirección al reanudar la marcha.
```bash
/home/rbriram/IsaacLab/isaaclab.sh -p "/home/rbriram/Modelado_control/proyecto/proyecto_ws/rl/play_ppo.py" --phase 1.5 --scenario stop_and_go
```

### Escenario 4 — Figura en 8 con Paso Lento y Esprints (`acelerones`)
La bola traza una figura en 8 amplia alternando tramos muy lentos ($0.07\text{ m/s}$) con acelerones rápidos ($0.31\text{ m/s}$).
```bash
/home/rbriram/IsaacLab/isaaclab.sh -p "/home/rbriram/Modelado_control/proyecto/proyecto_ws/rl/play_ppo.py" --phase 1.5 --scenario acelerones
```

### Escenario 5 — Movimiento Evasivo con Quiebros Bruscos 2D (`erratico`)
La bola realiza cambios bruscos e impredecibles de rumbo en 2D combinados con variaciones continuas de ritmo.
```bash
/home/rbriram/IsaacLab/isaaclab.sh -p "/home/rbriram/Modelado_control/proyecto/proyecto_ws/rl/play_ppo.py" --phase 1.5 --scenario erratico
```

### Modo Mixto Aleatorio (`mixto` — por defecto)
Sortea al azar entre los 5 escenarios en cada episodio:
```bash
/home/rbriram/IsaacLab/isaaclab.sh -p "/home/rbriram/Modelado_control/proyecto/proyecto_ws/rl/play_ppo.py" --phase 1.5 --scenario mixto
```

---

## 2. Otros Modos de Visualización

### Ver Fase 1 (Modelo base original a ~2.0 m):
```bash
/home/rbriram/IsaacLab/isaaclab.sh -p "/home/rbriram/Modelado_control/proyecto/proyecto_ws/rl/play_ppo.py" --phase 1
```

### Ver el Mapa del Recinto con los 11 Pilares (sin política / solo mapa):
```bash
/home/rbriram/IsaacLab/isaaclab.sh -p "/home/rbriram/Modelado_control/proyecto/proyecto_ws/rl/view_map.py"
```

### Ver el Robot de la Fase 1.5 dentro del Mapa de los 11 Pilares:
```bash
/home/rbriram/IsaacLab/isaaclab.sh -p "/home/rbriram/Modelado_control/proyecto/proyecto_ws/rl/play_ppo.py" --phase 2
```

---

## 3. Evaluar Métricas en CSV por Escenario (`evaluate.py`)

Para extraer las métricas cuantitativas (`mean_horizontal_error`, `mean_area`, `detection_rate`, `mean_distance`, `collision_rate`, `success_rate`) de cualquier escenario en un archivo `.csv`:

```bash
/home/rbriram/IsaacLab/isaaclab.sh -p "/home/rbriram/Modelado_control/proyecto/proyecto_ws/rl/evaluate.py" \
  --phase 1.5 \
  --scenario circular \
  --output "/home/rbriram/Modelado_control/proyecto/proyecto_ws/results/fase1_5/ppo_circular.csv"
```
*(Puedes cambiar `--scenario circular` por `slalom`, `stop_and_go`, `acelerones`, `erratico` o `mixto`).*
