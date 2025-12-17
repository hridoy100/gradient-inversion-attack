"""
Lightweight fine-tuning of the CIFAR-10 diffusion prior to CIFAR-100.

This trains an unconditional 32x32 DDPM (UNet2DModel) for a small number of
steps to adapt the prior to CIFAR-100. The resulting checkpoint can be used
with `--diffusion-model checkpoints/cifar100_ddpm` in
evaluations/diffusion_reconstruction/run.py.

Usage:
    python3 scripts/finetune_cifar100_diffusion.py --steps 2000 --batch-size 128
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    from diffusers import DDPMScheduler, UNet2DModel
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError("Please install diffusers: pip install diffusers") from exc


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune CIFAR-10 DDPM to CIFAR-100.")
    parser.add_argument(
        "--pretrained-repo",
        type=str,
        default="google/ddpm-cifar10-32",
        help="Base diffusion checkpoint to start from (UNet2DModel + scheduler).",
    )
    parser.add_argument("--data-root", type=str, default="~/.torch", help="Where to download/load CIFAR-100.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--steps", type=int, default=2000, help="Number of training steps.")
    parser.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/cifar100_ddpm",
        help="Directory to save the fine-tuned UNet and scheduler.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),  # map [0,1] -> [-1,1]
        ]
    )
    dataset = datasets.CIFAR100(root=args.data_root, train=True, download=True, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    unet = UNet2DModel.from_pretrained(args.pretrained_repo).to(device)
    scheduler = DDPMScheduler.from_pretrained(args.pretrained_repo)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr)

    total_steps = args.steps
    step = 0
    unet.train()
    while step < total_steps:
        for x, _ in loader:
            x = x.to(device)
            bsz = x.size(0)
            t = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            noise = torch.randn_like(x)
            noisy = scheduler.add_noise(x, noise, t)
            pred = unet(noisy, t).sample
            loss = F.mse_loss(pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step % 100 == 0 or step == 1:
                print(f"step {step}/{total_steps} loss {loss.item():.4f}")
            if step >= total_steps:
                break

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(out_dir)
    scheduler.save_pretrained(out_dir)
    print(f"Saved fine-tuned model to {out_dir}")


if __name__ == "__main__":
    main()
