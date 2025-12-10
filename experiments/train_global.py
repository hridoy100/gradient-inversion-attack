# -*- coding: utf-8 -*-
"""
Quick helper to train a global LeNet on CIFAR100 and save a checkpoint for inversion demos.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.vision_new import MODEL_BUILDERS, build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train a global model on CIFAR100 and save a checkpoint.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum.")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay.")
    parser.add_argument("--step-size", type=int, default=20, help="StepLR step size.")
    parser.add_argument("--gamma", type=float, default=0.1, help="StepLR gamma.")
    parser.add_argument("--data-root", type=str, default="~/.torch", help="Dataset root for CIFAR100.")
    parser.add_argument("--output", type=str, default="checkpoints/global.pth", help="Path to save state_dict.")
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU if available; raises if requested but CUDA is not available.",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="lenet",
        choices=sorted(MODEL_BUILDERS.keys()),
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use torchvision pretrained weights when available for the selected architecture.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available but --gpu was requested.")
        device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    train_set = datasets.CIFAR100(args.data_root, train=True, download=True, transform=transform)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)

    model = build_model(args.arch, num_classes=100, pretrained=args.pretrained).to(device)

    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            logits = model(data)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.size(0)
        avg_loss = total_loss / len(train_set)
        scheduler.step()
        print(
            f"Epoch {epoch+1}/{args.epochs} - loss: {avg_loss:.4f} - lr: {scheduler.get_last_lr()[0]:.5f}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Saved global model checkpoint to {output_path}")


if __name__ == "__main__":
    main()
