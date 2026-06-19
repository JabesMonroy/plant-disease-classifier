"""Métricas de evaluación para la clasificación multiclase."""

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def accuracy(y_true, y_pred):
    """Exactitud global (proporción de aciertos)."""
    return accuracy_score(y_true, y_pred)


def macro_f1(y_true, y_pred):
    """F1 promedio por clase (macro), robusto frente al desbalance de clases."""
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def confusion(y_true, y_pred, num_classes):
    """Matriz de confusión de tamaño num_classes x num_classes."""
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
