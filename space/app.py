"""Hugging Face Space: clasificación de enfermedades en plantas con ExecuTorch."""

import json
import glob

import torch
import gradio as gr
from PIL import Image
from torchvision import transforms
from executorch.extension.pybindings.portable_lib import _load_for_executorch

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

with open("labels.json", "r", encoding="utf-8") as f:
    CLASSES = json.load(f)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Carga única del modelo ExecuTorch al iniciar el Space.
_model = _load_for_executorch("model.pte")


def _probabilities(image):
    """Devuelve el vector de probabilidades para una imagen PIL."""
    tensor = transform(image.convert("RGB")).unsqueeze(0)
    logits = _model.forward([tensor])[0]
    return torch.softmax(logits, dim=1)[0]


def predict(image):
    """Clasifica una imagen y devuelve las probabilidades por clase."""
    probs = _probabilities(image)
    return {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}


def predict_batch(paths):
    """Clasifica varias imágenes y devuelve una galería con la clase predicha."""
    galeria = []
    for path in paths:
        probs = _probabilities(Image.open(path))
        idx = int(probs.argmax())
        galeria.append((path, f"{CLASSES[idx]} ({probs[idx] * 100:.1f}%)"))
    return galeria


examples = sorted(glob.glob("examples/*"))

una_imagen = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Imagen de la hoja"),
    outputs=gr.Label(num_top_classes=3, label="Predicción"),
    examples=examples or None,
    description="Sube la foto de una hoja para identificar su estado de salud.",
)

varias_imagenes = gr.Interface(
    fn=predict_batch,
    inputs=gr.File(file_count="multiple", type="filepath", label="Imágenes de hojas"),
    outputs=gr.Gallery(label="Predicciones", columns=4),
    description="Sube varias imágenes a la vez para clasificarlas todas.",
)

demo = gr.TabbedInterface(
    [una_imagen, varias_imagenes],
    ["Una imagen", "Varias imágenes"],
    title="Clasificador de enfermedades en plantas (PlantVillage, 38 clases)",
)

if __name__ == "__main__":
    demo.launch()
