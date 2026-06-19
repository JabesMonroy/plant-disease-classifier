"""Gráficas de entrenamiento: curvas de pérdida, accuracy y matriz de confusión."""

import os
import matplotlib.pyplot as plt


def plot_curves(history, out_dir):
    """Genera las curvas de pérdida y de accuracy (train vs validación).

    history es un dict con las listas: train_loss, val_loss, train_acc, val_acc.
    Guarda 'curva_perdida.png' y 'curva_accuracy.png' en out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Entrenamiento")
    plt.plot(epochs, history["val_loss"], label="Validación")
    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.title("Curva de pérdida")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "curva_perdida.png"), bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Entrenamiento")
    plt.plot(epochs, history["val_acc"], label="Validación")
    plt.xlabel("Época")
    plt.ylabel("Accuracy")
    plt.title("Curva de accuracy")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "curva_accuracy.png"), bbox_inches="tight")
    plt.close()


def plot_confusion(cm, classes, out_path):
    """Dibuja y guarda la matriz de confusión."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 0.4),
                                    max(8, len(classes) * 0.4)))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=6)
    ax.set_yticklabels(classes, fontsize=6)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusión")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
