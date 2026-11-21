# TurtleBot3 Visual Following con Reinforcement Learning

Comandos esenciales para entrenar y visualizar el robot en **Isaac Sim / Isaac Lab** con selector de fases a la carta.

---

## 1. Visualizar al robot en Isaac Sim (A la carta por fases)

Abre la interfaz gráfica 3D cargando automáticamente el modelo y la configuración de la fase elegida:

### Ver SOLO el mapa de la Fase 2 (recinto cerrado, sin política ni entrenamiento):
```bash
cd /home/rbriram/IsaacLab
./isaaclab.sh -p "/home/rbriram/Modelado_control/proyecto/proyecto_ws/rl/view_map.py"
```
*(Recinto 5.6×3.6 m: A slalom de columnas · B pasillo estrecho · C túnel · D cajas móviles. La bola recorre la ruta y colisiona con los elementos; el robot permanece quieto.)*

### Ver Fase 2 con el robot entrenado (recinto cerrado):
```bash
cd /home/rbriram/IsaacLab
./isaaclab.sh -p "/home/rbriram/Modelado_control/proyecto/proyecto_ws/rl/play_ppo.py" --phase 2
```

### Ver Fase 1.5 (Seguimiento cercano a ~0.8 m en espacio abierto):
```bash
cd /home/rbriram/IsaacLab
./isaaclab.sh -p "/home/rbriram/Modelado_control/proyecto/proyecto_ws/rl/play_ppo.py" --phase 1.5
```

### Ver Fase 1 (Modelo base original a ~2.0 m):
```bash
cd /home/rbriram/IsaacLab
./isaaclab.sh -p "/home/rbriram/Modelado_control/proyecto/proyecto_ws/rl/play_ppo.py" --phase 1
```

* **Finalizar:** Cierra la ventana de Isaac Sim o presiona `Ctrl+C` en la terminal.
* **Grabación de vídeo:** Pulsa `Ctrl+Alt+Shift+R` en Ubuntu para grabar demostraciones.

---

## 2. Entrenar / Acumular aprendizaje por fases

Entrena al robot en la GPU en modo headless. Carga automáticamente el checkpoint base de la fase anterior y consolida el modelo final en `checkpoints/faseX/`:

### Entrenar Fase 2 (recinto cerrado; observación de 9 valores: bola + 5 sectores de obstáculo):
```bash
cd /home/rbriram/IsaacLab
./isaaclab.sh -p "/home/rbriram/Modelado_control/proyecto/proyecto_ws/rl/train_ppo.py" \
  --phase 2 \
  --num-envs 64 \
  --max-iterations 1200
```
*(Carga `checkpoints/fase1_5/model_fase1_5.pt` como base ampliando las entradas de la red de 4 a 9, y guarda el modelo final en `checkpoints/fase2/model_fase2.pt`. La versión anterior de la Fase 2 está en `checkpoints/fase2_v1_obstaculos/`.)*.

---

### Reentrenar Fase 1.5 (Seguimiento cercano):
```bash
cd /home/rbriram/IsaacLab
./isaaclab.sh -p "/home/rbriram/Modelado_control/proyecto/proyecto_ws/rl/train_ppo.py" \
  --phase 1.5 \
  --num-envs 64 \
  --max-iterations 500
```
### Comando para continuar entrenando sobre la Fase 1. 5 ya entrenada
```bash
cd /home/rbriram/IsaacLab
./isaaclab.sh -p "/home/rbriram/Modelado_control/proyecto/proyecto_ws/rl/train_ppo.py" \
  --phase 1.5 \
  --num-envs 64 \
  --max-iterations 800 \
  --checkpoint "/home/rbriram/Modelado_control/proyecto/proyecto_ws/checkpoints/fase1_5/model_fase1_5.pt"