---
title: Clasificador de Enfermedades en Plantas
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.19.0
app_file: app.py
python_version: "3.10"
pinned: false
---

# Clasificador de enfermedades en plantas (ExecuTorch)

Espacio de inferencia para el proyecto final de PDI. Sube la foto de una hoja y el modelo
predice su estado de salud entre las 38 clases del dataset PlantVillage.

El modelo es un MobileNetV2 entrenado con *transfer learning* y exportado a ExecuTorch
(`model.pte`, backend XNNPACK). La interfaz está hecha con Gradio.

## Archivos de este Space

- `app.py` — interfaz Gradio y lógica de inferencia con el runtime de ExecuTorch.
- `model.pte` — modelo exportado.
- `labels.json` — nombres de las clases en el orden del entrenamiento.
- `requirements.txt` — dependencias (versiones fijadas para coincidir con la exportación).
- `examples/` — imágenes de ejemplo para probar la demo.

## Despliegue

1. Crear un Space de tipo **Gradio** en Hugging Face.
2. Subir estos archivos (`app.py`, `model.pte`, `labels.json`, `requirements.txt` y `examples/`).
3. El Space instala las dependencias y queda público para la prueba en vivo.
