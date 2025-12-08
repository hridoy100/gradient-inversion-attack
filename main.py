# -*- coding: utf-8 -*-
import argparse
from typing import List

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import datasets, transforms

from federated import FederatedClient, FederatedServer
from models.vision import LeNet, weights_init


def to_safe_pil(img_tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL image, sanitizing NaNs/Infs and clamping to [0, 1]."""
    img_tensor = torch.nan_to_num(img_tensor.detach().cpu(), nan=0.0, posinf=1.0, neginf=0.0)
    img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
    return transforms.ToPILImage()(img_tensor)


def parse_args():
    parser = argparse.ArgumentParser(description="Federated Deep Leakage from Gradients.")
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
        "--agg-lr",
        type=float,
        default=0.1,
        help="Learning rate for the aggregated gradient step when --apply-agg-step is set.",
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


def visualize_per_client_recovery(per_client_results, samples_per_client: int):
    """Plot original vs reconstructed data for each client separately."""
    for result in per_client_results:
        client = result["client"]
        recovered_data = result["recovered_data"]
        history = result["history"]

        cols = samples_per_client
        plt.figure(figsize=(2.5 * cols, 5))
        for idx in range(cols):
            plt.subplot(2, cols, idx + 1)
            plt.imshow(to_safe_pil(client.data[idx]))
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
                plt.imshow(to_safe_pil(entry["data"][0]))
                plt.title(f"C{client.client_id} Iter {entry['iteration']}\nLoss {entry['loss']:.2f}")
                plt.axis("off")

        plt.tight_layout()


def visualize_aggregated_recovery(all_real_data: torch.Tensor, recovered_data: torch.Tensor, history):
    """Plot aggregated reconstruction against the stacked real data."""
    total_samples = min(recovered_data.size(0), all_real_data.size(0))
    cols = min(total_samples, 8)
    rows = 2

    plt.figure(figsize=(2.5 * cols, 5))
    for idx in range(cols):
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(to_safe_pil(all_real_data[idx]))
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
            plt.imshow(to_safe_pil(entry["data"][0]))
            plt.title(f"Aggregated Iter {entry['iteration']}\nLoss {entry['loss']:.2f}")
            plt.axis("off")

    plt.tight_layout()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    transform = transforms.ToTensor()
    dataset = datasets.CIFAR100(args.data_root, download=True, transform=transform)
    num_classes = 100
    total_samples = args.num_clients * args.samples_per_client

    torch.manual_seed(args.seed)
    indices = pick_indices(args, total_samples, len(dataset))
    print(f"Using dataset indices: {indices}")

    model = LeNet().to(device)
    model.apply(weights_init)

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
        )
        aggregated_result = {"recovered_data": recovered_data, "history": history, "recovered_labels": recovered_labels}
        print("Aggregated reconstruction complete.")

    if per_client_results:
        visualize_per_client_recovery(per_client_results, args.samples_per_client)
    if aggregated_result:
        all_real_data = torch.cat([client.data for client in clients], dim=0)
        visualize_aggregated_recovery(all_real_data, aggregated_result["recovered_data"], aggregated_result["history"])

    plt.show()


if __name__ == "__main__":
    main()
