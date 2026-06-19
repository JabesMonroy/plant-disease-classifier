# Exportación del modelo con ExecuTorch

Este documento describe cómo se convierte el modelo entrenado en PyTorch
(`artifacts/best.pt`) a un archivo `model.pte` de **ExecuTorch**, que es el formato
que carga el Hugging Face Space para hacer inferencia.

## Por qué ExecuTorch

ExecuTorch toma un modelo de PyTorch y lo compila a un formato (`.pte`) optimizado para
ejecución eficiente, sin necesidad del intérprete completo de PyTorch. Usamos el backend
**XNNPACK**, que acelera las operaciones en CPU (la que ofrece el Space gratuito).

## Regla crítica de versiones

El `.pte` debe generarse con las **mismas versiones** de `torch` y `executorch` que usará
el Space. Una diferencia de versión es la causa más común de que el modelo cargue en Colab
pero falle en el despliegue. Las versiones usadas en este proyecto son:

```
torch==2.5.0
torchvision==0.20.0
executorch==0.4.0
```

> `executorch==0.4.0` fija de forma exacta `torch==2.5.0` y `torchvision==0.20.0`; usar otra
> versión de torch (por ejemplo 2.5.1) provoca un conflicto de dependencias en pip.

Estas mismas versiones aparecen en `space/requirements.txt`.

## Flujo de exportación

El script `export_executorch.py` realiza estos pasos:

1. **Reconstruye el modelo** MobileNetV2 con el número de clases leído de `labels.json` y
   carga los pesos de `best.pt` en modo evaluación.
2. **Traza el grafo** con `torch.export.export` usando una entrada de ejemplo de
   tamaño `(1, 3, 224, 224)`.
3. **Convierte y baja a XNNPACK** con `to_edge_transform_and_lower(..., partitioner=[XnnpackPartitioner()])`.
4. **Serializa** el programa con `to_executorch()` y escribe el `.pte`.
5. **Verifica** que la predicción del `.pte` coincide con la del modelo original.

## Cómo ejecutarlo

```bash
pip install executorch==0.4.0

python export/export_executorch.py \
  --checkpoint artifacts/best.pt \
  --labels artifacts/labels.json \
  --output space/model.pte \
  --img-size 224
```

Al terminar quedan en `space/`: el modelo `model.pte` y una copia de `labels.json` con el
orden exacto de las clases (necesario para mapear el índice predicho a su nombre).

## Salida esperada de la verificación

```
Modelo exportado en space/model.pte
Verificación | clase eager=<n> | clase pte=<n> | diferencia máx=<~0.00001>
```

Si las clases coinciden y la diferencia es cercana a cero, la exportación es correcta.
