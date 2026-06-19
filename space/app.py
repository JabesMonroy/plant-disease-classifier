"""Hugging Face Space: clasificación de enfermedades en plantas con ExecuTorch."""

import json
import glob

import torch
import gradio as gr
from torchvision import transforms
from executorch.runtime import Runtime

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
_runtime = Runtime.get()
_program = _runtime.load_program("model.pte")
_method = _program.load_method("forward")


def predict(image):
    """Predice la clase de la hoja y devuelve las probabilidades por clase."""
    tensor = transform(image.convert("RGB")).unsqueeze(0)
    logits = _method.execute([tensor])[0]
    probs = torch.softmax(logits, dim=1)[0]
    return {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}


examples = sorted(glob.glob("examples/*"))

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Imagen de la hoja"),
    outputs=gr.Label(num_top_classes=3, label="Predicción"),
    title="Clasificador de enfermedades en plantas",
    description="Sube la foto de una hoja para identificar su estado de salud "
                "(dataset PlantVillage, 38 clases). Modelo MobileNetV2 exportado con ExecuTorch.",
    examples=examples or None,
)

if __name__ == "__main__":
    demo.launch()
