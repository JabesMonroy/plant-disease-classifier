"""Definición del modelo: MobileNetV2 con transfer learning."""

import torch.nn as nn
from torchvision import models


def build_model(num_classes, pretrained=True, freeze_features=False):
    """Crea un MobileNetV2 y reemplaza el clasificador para num_classes.

    Si pretrained es True usa los pesos de ImageNet. Si freeze_features es True
    congela el extractor de características y solo entrena el clasificador.
    """
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
