# -*- coding: utf-8 -*-
import argparse
import json
import time
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import datasets, transforms

from federated import FederatedClient, FederatedServer
from models.vision_new import CIFAR100_MEAN, CIFAR100_STD, MODEL_BUILDERS, build_model, default_transform


def to_safe_pil(img_tensor: torch.Tensor, denormalize=None) -> Image.Image:
    """Convert tensor to PIL image, optional unnormalize first, clamp to [0, 1]."""
    img_tensor = img_tensor.detach().cpu()
    if denormalize:
        img_tensor = denormalize(img_tensor)
    img_tensor = torch.nan_to_num(img_tensor, nan=0.0, posinf=1.0, neginf=0.0)
    img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
    return transforms.ToPILImage()(img_tensor)


def build_denormalize(arch: str):
    """Return a denormalization fn for visualization given the chosen arch."""
    if arch.lower() == "lenet":
        return None
    mean = torch.tensor(CIFAR100_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR100_STD).view(3, 1, 1)

    def denorm(t: torch.Tensor) -> torch.Tensor:
        return t * std + mean

    return denorm


def save_image(tensor: torch.Tensor, path: Path, denormalize=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    to_safe_pil(tensor, denormalize).save(path)


def parse_args():
    parser = argparse.ArgumentParser(description="Federated Deep Leakage from Gradients.")
    parser.add_argument(
        "--arch",
        type=str,
        default="lenet",
        choices=sorted(MODEL_BUILDERS.keys()),
        help="Model architecture to use for clients and server.",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use torchvision pretrained weights when available for the selected architecture.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a model checkpoint to load (state_dict). Overrides random init/fc when provided.",
    )
    parser.add_argument(
        "--bn-eval",
        dest="bn_eval",
        action="store_true",
        default=True,
        help="Force BatchNorm layers to eval mode (recommended for tiny batch sizes like 1).",
    )
    parser.add_argument(
        "--no-bn-eval",
        dest="bn_eval",
        action="store_false",
        help="Keep BatchNorm layers in training mode.",
    )
    parser.add_argument("--num-clients", type=int, default=3, help="Number of federated clients.")
    parser.add_argument("--samples-per-client", type=int, default=1, help="Samples held by each client.")
    parser.add_argument("--iterations", type=int, default=300, help="LBFGS steps for reconstruction.")
    parser.add_argument("--log-every", type=int, default=25, help="Store intermediate reconstructions every N steps.")
    parser.add_argument("--restarts", type=int, default=1, help="Number of random restarts for inversion (best loss kept).")
    parser.add_argument("--tv-weight", type=float, default=0.0, help="Total variation weight to encourage smooth reconstructions.")
    parser.add_argument(
        "--init-scale",
        type=float,
        default=1.0,
        help="Stddev multiplier for dummy data init (smaller can help aggregated inversion).",
    )
    parser.add_argument(
        "--dummy-seed",
        type=int,
        default=None,
        help="Base seed for initializing dummy data/labels during inversion (per-client offset applied).",
    )
    parser.add_argument(
        "--reconstruct-mode",
        choices=["aggregated", "per-client", "both"],
        default="per-client",
        help="Reconstruct from aggregated gradients, per-client gradients, or both.",
    )
    parser.add_argument(
        "--normalize-gradients",
        action="store_true",
        help="L2-normalize client gradients before aggregation to reduce dominance.",
    )
    parser.add_argument(
        "--apply-agg-step",
        action="store_true",
        help="Apply one gradient descent step on the server model using the aggregated gradients before inversion.",
    )
    parser.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        default=True,
        help="Disable tqdm progress bars during inversion iterations.",
    )
    parser.add_argument(
        "--agg-lr",
        type=float,
        default=0.1,
        help="Learning rate for the aggregated gradient step when --apply-agg-step is set.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="outputs/reconstructions",
        help="Directory to save reconstruction outputs (images/metadata).",
    )
    parser.add_argument(
        "--no-save",
        dest="save_outputs",
        action="store_false",
        default=True,
        help="Disable saving reconstruction outputs to disk.",
    )
    parser.add_argument(
        "--client-indices",
        type=str,
        default=None,
        help="Comma separated CIFAR100 indices (must equal num_clients * samples_per_client).",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Seed for sampling client data.")
    parser.add_argument(
        "--data-root",
        type=str,
        default="~/.torch",
        help="Location to download/load CIFAR100.",
    )
    return parser.parse_args()


def pick_indices(args, total_samples: int, dataset_size: int) -> List[int]:
    """Parse/produce dataset indices for each client sample."""
    if args.client_indices:
        parsed = [int(x.strip()) for x in args.client_indices.split(",") if x.strip()]
        if len(parsed) != total_samples:
            raise ValueError(
                f"--client-indices provided {len(parsed)} indices, but {total_samples} are required "
                "(num_clients * samples_per_client)."
            )
        return parsed

    if total_samples > dataset_size:
        raise ValueError(f"Requested {total_samples} samples but dataset only has {dataset_size}.")

    generator = torch.Generator().manual_seed(args.seed)
    return torch.randperm(dataset_size, generator=generator)[:total_samples].tolist()


def build_clients(dataset, indices: List[int], num_clients: int, samples_per_client: int, device: str, num_classes: int):
    """Create federated clients from the selected dataset indices."""
    clients = []
    idx = 0
    for client_id in range(num_clients):
        data_batch = []
        label_batch = []
        for _ in range(samples_per_client):
            data, label = dataset[indices[idx]]
            data_batch.append(data)
            label_batch.append(label)
            idx += 1
        batch_tensor = torch.stack(data_batch).to(device)
        label_tensor = torch.tensor(label_batch, device=device).long()
        clients.append(FederatedClient(client_id, batch_tensor, label_tensor, device, num_classes))
    return clients


def visualize_per_client_recovery(per_client_results, samples_per_client: int, denormalize=None):
    """Plot original vs reconstructed data for each client separately."""
    for result in per_client_results:
        client = result["client"]
        recovered_data = result["recovered_data"]
        history = result["history"]

        cols = samples_per_client
        plt.figure(figsize=(2.5 * cols, 5))
        for idx in range(cols):
            plt.subplot(2, cols, idx + 1)
            plt.imshow(to_safe_pil(client.data[idx], denormalize))
            plt.title(f"Client {client.client_id} sample {idx}")
            plt.axis("off")

            plt.subplot(2, cols, idx + 1 + cols)
            plt.imshow(to_safe_pil(recovered_data[idx]))
            plt.title("Reconstructed")
            plt.axis("off")

        if history:
            plt.figure(figsize=(3 * len(history), 3))
            for i, entry in enumerate(history):
                plt.subplot(1, len(history), i + 1)
                plt.imshow(to_safe_pil(entry["data"][0], denormalize))
                plt.title(f"C{client.client_id} Iter {entry['iteration']}\nLoss {entry['loss']:.2f}")
                plt.axis("off")

        plt.tight_layout()


def visualize_aggregated_recovery(all_real_data: torch.Tensor, recovered_data: torch.Tensor, history, denormalize=None):
    """Plot aggregated reconstruction against the stacked real data."""
    total_samples = min(recovered_data.size(0), all_real_data.size(0))
    cols = min(total_samples, 8)
    rows = 2

    plt.figure(figsize=(2.5 * cols, 5))
    for idx in range(cols):
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(to_safe_pil(all_real_data[idx], denormalize))
        plt.title(f"Real {idx}")
        plt.axis("off")

        plt.subplot(rows, cols, idx + 1 + cols)
        plt.imshow(to_safe_pil(recovered_data[idx]))
        plt.title("Reconstructed")
        plt.axis("off")

    if history:
        plt.figure(figsize=(3 * len(history), 3))
        for i, entry in enumerate(history):
            plt.subplot(1, len(history), i + 1)
            plt.imshow(to_safe_pil(entry["data"][0], denormalize))
            plt.title(f"Aggregated Iter {entry['iteration']}\nLoss {entry['loss']:.2f}")
            plt.axis("off")

        plt.tight_layout()


def save_reconstructions(
    run_dir: Path,
    per_client_results,
    aggregated_result,
    clients,
    denormalize,
    indices,
    args,
    all_real_data: torch.Tensor = None,
):
    """Persist reconstructions and metadata to disk."""
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "arch": args.arch,
        "pretrained": args.pretrained,
        "seed": args.seed,
        "indices": indices,
        "num_clients": args.num_clients,
        "samples_per_client": args.samples_per_client,
        "iterations": args.iterations,
        "tv_weight": args.tv_weight,
        "init_scale": args.init_scale,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    for result in per_client_results:
        client = result["client"]
        recovered_data = result["recovered_data"]
        history = result["history"]
        client_dir = run_dir / f"client_{client.client_id}"
        for idx in range(recovered_data.size(0)):
            save_image(client.data[idx], client_dir / f"original_{idx}.png", denormalize)
            save_image(recovered_data[idx], client_dir / f"reconstructed_{idx}.png")
        if history:
            hist_dir = client_dir / "history"
            for entry in history:
                save_image(entry["data"][0], hist_dir / f"iter_{entry['iteration']}.png", denormalize)

    if aggregated_result and all_real_data is not None:
        agg_dir = run_dir / "aggregated"
        recovered_data = aggregated_result["recovered_data"]
        history = aggregated_result["history"]
        total_samples = min(recovered_data.size(0), all_real_data.size(0))
        for idx in range(total_samples):
            save_image(all_real_data[idx], agg_dir / f"original_{idx}.png", denormalize)
            save_image(recovered_data[idx], agg_dir / f"reconstructed_{idx}.png")
        if history:
            hist_dir = agg_dir / "history"
            for entry in history:
                save_image(entry["data"][0], hist_dir / f"iter_{entry['iteration']}.png", denormalize)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    transform = default_transform(args.arch)
    dataset = datasets.CIFAR100(args.data_root, download=True, transform=transform)
    num_classes = 100
    total_samples = args.num_clients * args.samples_per_client
    denormalize = build_denormalize(args.arch)

    torch.manual_seed(args.seed)
    indices = pick_indices(args, total_samples, len(dataset))
    print(f"Using dataset indices: {indices}")

    model = build_model(args.arch, num_classes=num_classes, pretrained=args.pretrained).to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        # Handle both raw state_dict and dicts with a 'state_dict' key.
        state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from {args.checkpoint}. missing_keys={missing}, unexpected_keys={unexpected}")
    if args.bn_eval and args.arch.lower() != "lenet":
        # Keep BatchNorm layers in eval mode to avoid small-batch failures (batch=1 and 1x1 feature maps).
        model.eval()

    clients = build_clients(dataset, indices, args.num_clients, args.samples_per_client, device, num_classes)
    server = FederatedServer(model=model, device=device, num_classes=num_classes)

    client_gradients = []
    for client in clients:
        grads, loss = client.compute_gradients(model)
        client_gradients.append(grads)
        print(f"Client {client.client_id}: local loss={loss:.4f}")

    # Reconstruct per-client before any aggregation (helps avoid averaging artifacts).
    per_client_results = []
    if args.reconstruct_mode in ("per-client", "both"):
        print("Starting per-client reconstructions...")
        for client, grads in zip(clients, client_gradients):
            recovered_data, recovered_labels, history = server.reconstruct_data(
                grads,
                data_shape=client.data.shape,
                iterations=args.iterations,
                log_every=args.log_every,
                restarts=args.restarts,
                init_seed=args.dummy_seed + client.client_id if args.dummy_seed is not None else None,
                tv_weight=args.tv_weight,
                init_scale=args.init_scale,
                progress=args.progress,
            )
            per_client_results.append(
                {
                    "client": client,
                    "recovered_data": recovered_data,
                    "recovered_labels": recovered_labels,
                    "history": history,
                }
            )
            print(f"Client {client.client_id}: reconstruction complete.")

    aggregated_result = None
    if args.reconstruct_mode in ("aggregated", "both"):
        aggregated_gradients = server.aggregate_gradients(client_gradients, normalize=args.normalize_gradients)
        print(f"Aggregated gradients collected from clients. normalize_gradients={args.normalize_gradients}")

        if args.apply_agg_step:
            print(f"Applying aggregated gradient step to server model with lr={args.agg_lr}")
            server.apply_gradient_step(aggregated_gradients, lr=args.agg_lr)

        data_shape = torch.Size((total_samples, *clients[0].data.shape[1:]))
        recovered_data, recovered_labels, history = server.reconstruct_data(
            aggregated_gradients,
            data_shape=data_shape,
            iterations=args.iterations,
            log_every=args.log_every,
            restarts=args.restarts,
            init_seed=args.dummy_seed,
            tv_weight=args.tv_weight,
            init_scale=args.init_scale,
            progress=args.progress,
        )
        aggregated_result = {"recovered_data": recovered_data, "history": history, "recovered_labels": recovered_labels}
        print("Aggregated reconstruction complete.")

    all_real_data = torch.cat([client.data for client in clients], dim=0) if aggregated_result else None

    if args.save_outputs:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = Path(args.save_dir) / f"run_{timestamp}"
        save_reconstructions(
            run_dir,
            per_client_results,
            aggregated_result,
            clients,
            denormalize,
            indices,
            args,
            all_real_data,
        )
        print(f"Saved reconstructions to {run_dir}")

    if per_client_results:
        visualize_per_client_recovery(per_client_results, args.samples_per_client, denormalize)
    if aggregated_result:
        visualize_aggregated_recovery(all_real_data, aggregated_result["recovered_data"], aggregated_result["history"], denormalize)

    plt.show()


if __name__ == "__main__":
    main()
