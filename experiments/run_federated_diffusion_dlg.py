# -*- coding: utf-8 -*-
"""
Experiment: Diffusion-latent DLG attack in a federated setting.

We optimize a latent z under a diffusion-style prior to match a client's observed gradients,
reconstructing the client's private input data.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torchvision import datasets, transforms

from attacks.diffusion_dlg import DiffusionDLGAttack
from attacks.diffusion_prior import SimpleDiffusionWrapper
from evaluations.semantic_leakage import get_default_encoder, summarize_leakage, save_metrics
from federated import FederatedClient
from models.vision import LeNet, weights_init
from utils import label_to_onehot


def to_safe_pil(img_tensor: torch.Tensor):
    img_tensor = torch.nan_to_num(img_tensor.detach().cpu(), nan=0.0, posinf=1.0, neginf=0.0)
    img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
    return transforms.ToPILImage()(img_tensor)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Diffusion-DLG attack on federated gradients.")
    parser.add_argument("--round-id", type=int, default=0, help="Federated round identifier (for logging).")
    parser.add_argument("--client-id", type=int, default=0, help="Client id to attack.")
    parser.add_argument("--steps", type=int, default=100, help="Optimization steps for Diffusion-DLG.")
    parser.add_argument("--lambda-prior", type=float, default=1e-3, help="Weight for diffusion prior term.")
    parser.add_argument("--output-dir", type=str, default="outputs/diffusion_dlg", help="Directory to save outputs.")
    parser.add_argument("--num-clients", type=int, default=3, help="Number of simulated clients.")
    parser.add_argument("--samples-per-client", type=int, default=1, help="Samples per client.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for client sampling.")
    parser.add_argument("--data-root", type=str, default="~/.torch", help="Dataset root for CIFAR100.")
    parser.add_argument("--log-every", type=int, default=10, help="Log interval for attack history.")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate for the diffusion DLG optimizer.")
    parser.add_argument("--restarts", type=int, default=1, help="Random restarts for the diffusion DLG optimizer.")
    parser.add_argument(
        "--init-scale",
        type=float,
        default=1.0,
        help="Stddev multiplier for latent init (smaller for stability with stronger priors).",
    )
    parser.add_argument(
        "--fix-labels",
        action="store_true",
        help="Use the true labels for the client batch instead of optimizing labels.",
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default=None,
        help="Path to a saved global model checkpoint (state_dict) to load before inversion.",
    )
    return parser.parse_args()


def build_clients(dataset, indices, num_clients, samples_per_client, device, num_classes):
    clients = []
    idx = 0
    for client_id in range(num_clients):
        batch = []
        labels = []
        for _ in range(samples_per_client):
            x, y = dataset[indices[idx]]
            batch.append(x)
            labels.append(y)
            idx += 1
        batch_tensor = torch.stack(batch).to(device)
        label_tensor = torch.tensor(labels, device=device).long()
        clients.append(FederatedClient(client_id, batch_tensor, label_tensor, device, num_classes))
    return clients


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset and sample clients
    transform = transforms.ToTensor()
    dataset = datasets.CIFAR100(args.data_root, download=True, transform=transform)
    total_samples = args.num_clients * args.samples_per_client
    indices = torch.randperm(len(dataset))[:total_samples].tolist()

    model = LeNet().to(device)
    model.apply(weights_init)

    # If a checkpoint is provided, load it; otherwise, attempt a default path.
    ckpt_path = args.model_checkpoint
    default_ckpt = ROOT / "checkpoints" / "global.pth"
    if ckpt_path is None and default_ckpt.exists():
        ckpt_path = default_ckpt
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        model.load_state_dict(ckpt)
        print(f"Loaded model checkpoint from {ckpt_path}")
    num_classes = 100

    clients = build_clients(dataset, indices, args.num_clients, args.samples_per_client, device, num_classes)

    if args.client_id >= len(clients):
        raise ValueError(f"client-id {args.client_id} is out of range for {len(clients)} clients.")

    target_client = clients[args.client_id]
    # Compute client gradient on local batch
    model.zero_grad()
    preds = model(target_client.data)
    onehot = label_to_onehot(target_client.labels, num_classes=num_classes)
    loss = torch.mean(torch.sum(-onehot * torch.log_softmax(preds, dim=-1), 1))
    target_grads = torch.autograd.grad(loss, model.parameters())

    # Set up diffusion prior and attack
    diffusion_prior = SimpleDiffusionWrapper(target_client.data.shape).to(device)
    attack = DiffusionDLGAttack(
        model=model,
        target_gradients=list(target_grads),
        data_shape=target_client.data.shape,
        num_classes=num_classes,
        device=device,
        diffusion_model=diffusion_prior,
        lambda_prior=args.lambda_prior,
        lr=args.lr,
        steps=args.steps,
        log_every=args.log_every,
        restarts=args.restarts,
        optimize_labels=not args.fix_labels,
        fixed_labels=onehot if args.fix_labels else None,
        init_scale=args.init_scale,
    )

    result = attack.run()
    recon = result["reconstructed_data"]
    recon_labels = result["reconstructed_labels"]

    # Save reconstruction
    img_path = Path(args.output_dir) / f"client_{args.client_id}_round_{args.round_id}.png"
    to_safe_pil(recon[0]).save(img_path)
    orig_img_path = Path(args.output_dir) / f"original_client_{args.client_id}_round_{args.round_id}.png"
    to_safe_pil(target_client.data[0]).save(orig_img_path)

    # Save history
    hist_path = Path(args.output_dir) / f"loss_client_{args.client_id}_round_{args.round_id}.json"
    with open(hist_path, "w") as f:
        json.dump(result["history"], f, indent=2)

    # Semantic leakage evaluation
    encoder = get_default_encoder(device)
    metrics = summarize_leakage(
        reconstructed=recon,
        target=target_client.data,
        labels=target_client.labels,
        classifier=model,
        encoder=encoder,
    )
    metrics_path = Path(args.output_dir) / f"metrics_client_{args.client_id}_round_{args.round_id}.json"
    save_metrics(metrics, metrics_path)

    print(f"Saved original to {orig_img_path}")
    print(f"Saved reconstruction to {img_path}")
    print(f"Loss history to {hist_path}")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
