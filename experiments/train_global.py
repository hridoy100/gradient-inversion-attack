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
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.vision import LeNet, weights_init


def parse_args():
    parser = argparse.ArgumentParser(description="Train a global LeNet on CIFAR100 and save a checkpoint.")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum.")
    parser.add_argument("--data-root", type=str, default="~/.torch", help="Dataset root for CIFAR100.")
    parser.add_argument("--output", type=str, default="checkpoints/global.pth", help="Path to save state_dict.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = transforms.ToTensor()
    train_set = datasets.CIFAR100(args.data_root, train=True, download=True, transform=transform)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)

    model = LeNet().to(device)
    model.apply(weights_init)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

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
        print(f"Epoch {epoch+1}/{args.epochs} - loss: {avg_loss:.4f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Saved global model checkpoint to {output_path}")


if __name__ == "__main__":
    main()
