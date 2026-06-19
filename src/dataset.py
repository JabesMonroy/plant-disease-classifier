"""Carga del dataset PlantVillage y creación de los DataLoaders."""

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

# Estadísticas de normalización de ImageNet (usadas por MobileNetV2 preentrenado).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(img_size=224):
    """Devuelve las transformaciones de entrenamiento (con aumento) y de evaluación."""
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def create_dataloaders(data_dir, batch_size=32, img_size=224,
                       val_split=0.15, test_split=0.15, num_workers=2, seed=42):
    """Construye los DataLoaders de train, validación y test con split estratificado.

    El dataset se espera como carpetas por clase (formato ImageFolder).
    Devuelve (train_loader, valid_loader, test_loader, classes).
    """
    train_tf, eval_tf = build_transforms(img_size)
    train_base = datasets.ImageFolder(data_dir, transform=train_tf)
    eval_base = datasets.ImageFolder(data_dir, transform=eval_tf)
    classes = train_base.classes
    targets = train_base.targets

    indices = list(range(len(targets)))
    train_idx, temp_idx = train_test_split(
        indices, test_size=val_split + test_split,
        stratify=targets, random_state=seed)
    temp_targets = [targets[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=test_split / (val_split + test_split),
        stratify=temp_targets, random_state=seed)

    train_loader = DataLoader(
        Subset(train_base, train_idx), batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(
        Subset(eval_base, val_idx), batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(
        Subset(eval_base, test_idx), batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, valid_loader, test_loader, classes
