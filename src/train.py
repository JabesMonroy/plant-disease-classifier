"""Entrenamiento y validación del clasificador de enfermedades en plantas."""

import os
import sys
import json
import argparse

import torch
import torch.nn as nn
from torch import optim

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import create_dataloaders
from model import build_model
from metrics import accuracy, macro_f1, confusion
from plots import plot_curves, plot_confusion


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Entrena una época y devuelve la pérdida promedio y el accuracy."""
    model.train()
    running_loss, y_true, y_pred = 0.0, [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        y_pred.extend(outputs.argmax(1).cpu().tolist())
        y_true.extend(labels.cpu().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss, accuracy(y_true, y_pred)


def evaluate(model, loader, criterion, device):
    """Evalúa el modelo y devuelve pérdida, accuracy y las etiquetas reales/predichas."""
    model.eval()
    running_loss, y_true, y_pred = 0.0, [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            y_pred.extend(outputs.argmax(1).cpu().tolist())
            y_true.extend(labels.cpu().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss, accuracy(y_true, y_pred), y_true, y_pred


def fit(model, train_loader, valid_loader, criterion, optimizer, device, epochs, ckpt_path):
    """Ejecuta el bucle de entrenamiento, guarda el mejor modelo y devuelve el historial."""
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, valid_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Época {epoch:02d}/{epochs} | "
              f"loss {train_loss:.4f}/{val_loss:.4f} | "
              f"acc {train_acc:.4f}/{val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Mejor modelo guardado (val_acc={best_acc:.4f})")

    return history


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento PlantVillage (MobileNetV2)")
    parser.add_argument("--data-dir", required=True, help="Carpeta del dataset (formato ImageFolder)")
    parser.add_argument("--out-dir", default="artifacts", help="Carpeta de salida de artefactos")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--freeze", action="store_true", help="Congela el extractor de características")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    train_loader, valid_loader, test_loader, classes = create_dataloaders(
        args.data_dir, batch_size=args.batch_size, img_size=args.img_size)
    num_classes = len(classes)
    print(f"{num_classes} clases detectadas")

    model = build_model(num_classes, pretrained=True, freeze_features=args.freeze).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    ckpt_path = os.path.join(args.out_dir, "best.pt")
    history = fit(model, train_loader, valid_loader, criterion, optimizer,
                  device, args.epochs, ckpt_path)

    # Evaluación final sobre el conjunto de test con el mejor modelo.
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    test_f1 = macro_f1(y_true, y_pred)
    print(f"\nTest | loss {test_loss:.4f} | acc {test_acc:.4f} | macro-F1 {test_f1:.4f}")

    # Gráficas y metadatos.
    plot_curves(history, args.out_dir)
    cm = confusion(y_true, y_pred, num_classes)
    plot_confusion(cm, classes, os.path.join(args.out_dir, "matriz_confusion.png"))

    with open(os.path.join(args.out_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)
    print(f"Artefactos guardados en {args.out_dir}")


if __name__ == "__main__":
    main()
