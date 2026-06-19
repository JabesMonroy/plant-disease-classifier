# Clasificación de Enfermedades en Plantas (PlantVillage)

Proyecto final de Procesamiento Digital de Imágenes. Clasificador de imágenes que
identifica el estado de salud de hojas de cultivo a partir de una foto, entrenado
con *transfer learning* sobre **MobileNetV2** y desplegado en Hugging Face Spaces
mediante un modelo exportado con **ExecuTorch**.

## Contexto del problema

Las enfermedades en cultivos causan pérdidas significativas en la producción agrícola
y amenazan la seguridad alimentaria. El diagnóstico temprano permite tratar a tiempo
y reducir el uso innecesario de agroquímicos. La tarea de visión por computador es la
**clasificación de imágenes**: dada la foto de una hoja, predecir su clase (cultivo y
enfermedad, o sano).

## Base de datos (obligatorio)

- **Nombre:** PlantVillage Dataset
- **Fuente pública (Kaggle):** https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
- **Variante utilizada:** carpeta `color` (imágenes a color).
- **Número de clases:** 38 (combinaciones de cultivo y enfermedad, incluyendo hojas sanas).
- **Tamaño aproximado:** ~54.000 imágenes.
- **Organización:** una subcarpeta por clase (compatible con `torchvision.datasets.ImageFolder`).

> Esta es la base de datos reportada para la prueba en caliente: cualquier imagen de la
> carpeta `color` del dataset puede subirse al Hugging Face Space para inferencia.

## Arquitectura y decisiones técnicas

| Componente | Elección | Justificación |
|---|---|---|
| Modelo | MobileNetV2 (preentrenado en ImageNet) | Ligero, rápido en CPU y compatible con ExecuTorch. |
| Función de pérdida | CrossEntropyLoss con *label smoothing* | Estándar para clasificación multiclase; regulariza la confianza. |
| Métricas | Accuracy (principal) y Macro-F1 | Accuracy mide el desempeño global; Macro-F1 controla el desbalance de clases. |
| Optimizador | Adam | Convergencia estable con poco ajuste de hiperparámetros. |
| Exportación | ExecuTorch + XNNPACK | Inferencia eficiente en la CPU del Space. |

## Estructura del repositorio

```
.
├── src/                    # Código de entrenamiento
│   ├── dataset.py          # Transformaciones y DataLoaders
│   ├── model.py            # MobileNetV2 con transfer learning
│   ├── metrics.py          # Accuracy, Macro-F1 y matriz de confusión
│   ├── plots.py            # Curvas de pérdida y accuracy
│   └── train.py            # Bucle de entrenamiento y validación
├── export/                 # Exportación a ExecuTorch
│   ├── export_executorch.py
│   └── EXPORT.md           # Documentación del proceso de exportación
├── space/                  # Despliegue en Hugging Face Spaces
│   ├── app.py              # Interfaz Gradio + runtime ExecuTorch
│   ├── requirements.txt
│   └── README.md
├── notebooks/              # Notebook de Colab (orquesta todo el flujo)
│   └── entrenamiento.ipynb
├── artifacts/              # Modelos, gráficas y labels.json generados
└── requirements-train.txt  # Dependencias de entrenamiento
```

## Cómo entrenar

```bash
pip install -r requirements-train.txt

python src/train.py \
  --data-dir "ruta/al/plantvillage dataset/color" \
  --out-dir artifacts \
  --epochs 10 \
  --batch-size 32
```

Al finalizar se generan en `artifacts/`: el mejor modelo (`best.pt`), las curvas de
pérdida y accuracy, la matriz de confusión y `labels.json` con el orden de las clases.

> En Google Colab puedes ejecutar todo el flujo (descarga del dataset, entrenamiento y
> exportación) con el notebook `notebooks/entrenamiento.ipynb`.

## Exportar a ExecuTorch

```bash
pip install executorch==0.4.0

python export/export_executorch.py \
  --checkpoint artifacts/best.pt \
  --labels artifacts/labels.json \
  --output space/model.pte
```

Genera `space/model.pte` y copia `labels.json` junto a él. El detalle del proceso está en
[export/EXPORT.md](export/EXPORT.md).

## Despliegue en Hugging Face Spaces

El contenido de la carpeta `space/` se sube a un Space de tipo Gradio: `app.py`,
`model.pte`, `labels.json`, `requirements.txt` y las imágenes de `examples/`. El Space
carga el `.pte` con el runtime de ExecuTorch y permite subir una imagen para inferencia.

## Resultados

_Pendiente: se completará con los valores reales de test (accuracy y Macro-F1) y las
gráficas tras el entrenamiento._

## Hugging Face Space

_Pendiente: enlace al Space desplegado._
