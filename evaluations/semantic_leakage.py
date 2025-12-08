import json
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_mse_psnr(reconstructed: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    """Compute MSE and PSNR between reconstructed and target images (expects [0,1])."""
    mse = F.mse_loss(reconstructed, target, reduction="mean").item()
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 10 * torch.log10(torch.tensor(1.0 / mse)).item()
    return mse, psnr


class TinyEncoder(nn.Module):
    """Fallback encoder if no pretrained backbone is available."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        out = self.net(x)
        return out.view(out.size(0), -1)


def get_default_encoder(device: str) -> nn.Module:
    """Try to load a small pretrained encoder; fallback to TinyEncoder."""
    try:
        from torchvision.models import resnet18, ResNet18_Weights

        encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
        encoder.fc = nn.Identity()
    except Exception:
        encoder = TinyEncoder()
    return encoder.to(device).eval()


def compute_feature_similarity(encoder: nn.Module, reconstructed: torch.Tensor, target: torch.Tensor) -> float:
    """Compute average cosine similarity between features of reconstructed and target images."""
    with torch.no_grad():
        f_recon = encoder(reconstructed)
        f_target = encoder(target)
        sim = F.cosine_similarity(f_recon, f_target, dim=1)
    return sim.mean().item()


def attribute_classification_accuracy(model: nn.Module, reconstructed: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute how often reconstructed samples are classified as the correct label by a classifier."""
    model_eval = model.eval()
    with torch.no_grad():
        logits = model_eval(reconstructed)
        preds = torch.argmax(logits, dim=1)
    correct = (preds == labels.view(-1)).float().mean().item()
    return correct


def summarize_leakage(
    reconstructed: torch.Tensor,
    target: torch.Tensor,
    labels: torch.Tensor,
    classifier: nn.Module,
    encoder: nn.Module,
) -> Dict[str, Any]:
    """Compute a suite of semantic leakage metrics."""
    mse, psnr = compute_mse_psnr(reconstructed, target)
    feature_sim = compute_feature_similarity(encoder, reconstructed, target)
    cls_acc = attribute_classification_accuracy(classifier, reconstructed, labels)
    return {"mse": mse, "psnr": psnr, "feature_similarity": feature_sim, "class_accuracy": cls_acc}


def save_metrics(metrics: Dict[str, Any], path: str):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
