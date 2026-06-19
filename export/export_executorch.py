"""Exporta el modelo entrenado de PyTorch a un archivo .pte de ExecuTorch (backend XNNPACK)."""

import os
import sys
import json
import argparse

import torch
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from model import build_model


def load_model(checkpoint_path, num_classes):
    """Reconstruye MobileNetV2 y carga los pesos entrenados en modo evaluación."""
    model = build_model(num_classes, pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def export_to_executorch(model, img_size, output_path):
    """Convierte el modelo a ExecuTorch con lowering a XNNPACK y guarda el .pte."""
    example_inputs = (torch.randn(1, 3, img_size, img_size),)
    exported_program = torch.export.export(model, example_inputs)
    lowered = to_edge_transform_and_lower(
        exported_program, partitioner=[XnnpackPartitioner()])
    executorch_program = lowered.to_executorch()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(executorch_program.buffer)
    return example_inputs


def verify(model, pte_path, example_inputs):
    """Compara la predicción del modelo original con la del .pte exportado."""
    from executorch.extension.pybindings.portable_lib import _load_for_executorch

    with torch.no_grad():
        eager_out = model(*example_inputs)

    et_module = _load_for_executorch(pte_path)
    et_out = et_module.forward(list(example_inputs))[0]

    eager_cls = int(eager_out.argmax(1))
    et_cls = int(et_out.argmax(1))
    max_diff = (eager_out - et_out).abs().max().item()
    print(f"Verificación | clase eager={eager_cls} | clase pte={et_cls} | "
          f"diferencia máx={max_diff:.6f}")
    return eager_cls == et_cls


def main():
    parser = argparse.ArgumentParser(description="Exportación a ExecuTorch")
    parser.add_argument("--checkpoint", default="artifacts/best.pt")
    parser.add_argument("--labels", default="artifacts/labels.json")
    parser.add_argument("--output", default="space/model.pte")
    parser.add_argument("--img-size", type=int, default=224)
    args = parser.parse_args()

    with open(args.labels, "r", encoding="utf-8") as f:
        classes = json.load(f)

    model = load_model(args.checkpoint, len(classes))
    example_inputs = export_to_executorch(model, args.img_size, args.output)
    print(f"Modelo exportado en {args.output}")

    # Copia las etiquetas junto al modelo para que el Space tenga todo lo necesario.
    labels_out = os.path.join(os.path.dirname(args.output), "labels.json")
    with open(labels_out, "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)

    verify(model, args.output, example_inputs)


if __name__ == "__main__":
    main()
