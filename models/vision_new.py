from typing import Callable, Dict

import torch.nn as nn
from torchvision import models, transforms

from .vision import LeNet, weights_init as lenet_weights_init

CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR100_STD = [0.2675, 0.2565, 0.2761]
CIFAR_NORMALIZE = transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD)


def _resolve_weights(enum_name: str, pretrained: bool):
    """Handle torchvision's new/old API differences for pretrained weights."""
    if not pretrained:
        return None, False
    weights_enum = getattr(models, enum_name, None)
    if weights_enum is not None:
        return weights_enum.DEFAULT, True
    return None, False


def _build_resnet18(num_classes: int, pretrained: bool = False):
    weights, use_weights_kw = _resolve_weights("ResNet18_Weights", pretrained)
    try:
        model = models.resnet18(weights=weights if use_weights_kw else None)
    except TypeError:
        model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _build_resnet34(num_classes: int, pretrained: bool = False):
    weights, use_weights_kw = _resolve_weights("ResNet34_Weights", pretrained)
    try:
        model = models.resnet34(weights=weights if use_weights_kw else None)
    except TypeError:
        model = models.resnet34(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _build_resnet50(num_classes: int, pretrained: bool = False):
    weights, use_weights_kw = _resolve_weights("ResNet50_Weights", pretrained)
    try:
        model = models.resnet50(weights=weights if use_weights_kw else None)
    except TypeError:
        model = models.resnet50(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _build_mobilenet_v2(num_classes: int, pretrained: bool = False):
    weights, use_weights_kw = _resolve_weights("MobileNet_V2_Weights", pretrained)
    try:
        model = models.mobilenet_v2(weights=weights if use_weights_kw else None)
    except TypeError:
        model = models.mobilenet_v2(pretrained=pretrained)
    classifier_in = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(classifier_in, num_classes)
    return model


MODEL_BUILDERS: Dict[str, Callable[[int, bool], nn.Module]] = {
    "lenet": lambda num_classes, pretrained=False: LeNet().apply(lenet_weights_init),
    "resnet18": _build_resnet18,
    "resnet34": _build_resnet34,
    "resnet50": _build_resnet50,
    "mobilenet_v2": _build_mobilenet_v2,
}


def build_model(arch: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    arch_key = arch.lower()
    if arch_key not in MODEL_BUILDERS:
        raise ValueError(f"Unsupported architecture '{arch}'. Available: {sorted(MODEL_BUILDERS.keys())}")
    return MODEL_BUILDERS[arch_key](num_classes, pretrained)


def default_transform(arch: str):
    arch_key = arch.lower()
    if arch_key == "lenet":
        return transforms.ToTensor()
    return transforms.Compose([transforms.ToTensor(), CIFAR_NORMALIZE])
