# -*- coding: utf-8 -*-
import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms
from torchvision.utils import make_grid

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


def to_safe_01(img_tensor: torch.Tensor, denormalize=None) -> torch.Tensor:
    """Convert tensor to [0, 1] space for metrics/matching."""
    img_tensor = img_tensor.detach().cpu()
    if denormalize:
        img_tensor = denormalize(img_tensor)
    img_tensor = torch.nan_to_num(img_tensor, nan=0.0, posinf=1.0, neginf=0.0)
    return torch.clamp(img_tensor, 0.0, 1.0)


def to_safe_grid_pil(
    img_batch: torch.Tensor,
    denormalize=None,
    max_cols: int = 8,
    padding: int = 2,
    pad_value: float = 1.0,
) -> Image.Image:
    """Convert a (N, C, H, W) batch into a tiled PIL image for visualization."""
    if img_batch.dim() == 3:
        img_batch = img_batch.unsqueeze(0)
    if img_batch.dim() != 4:
        raise ValueError(f"Expected image batch of shape (N,C,H,W) or (C,H,W), got {tuple(img_batch.shape)}")
    img_batch = img_batch.detach().cpu()
    if denormalize:
        img_batch = denormalize(img_batch)
    img_batch = torch.nan_to_num(img_batch, nan=0.0, posinf=1.0, neginf=0.0)
    img_batch = torch.clamp(img_batch, 0.0, 1.0)
    nrow = max(1, int(max_cols))
    grid = make_grid(img_batch, nrow=nrow, padding=padding, pad_value=pad_value)
    return transforms.ToPILImage()(grid)


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


def save_grid_image(batch_tensor: torch.Tensor, path: Path, denormalize=None, max_cols: int = 8):
    path.parent.mkdir(parents=True, exist_ok=True)
    to_safe_grid_pil(batch_tensor, denormalize=denormalize, max_cols=max_cols).save(path)


def _linear_sum_assignment(cost: List[List[float]]) -> List[int]:
    """Return column indices assigned to each row (Hungarian algorithm, O(n^3))."""
    n = len(cost)
    if n == 0:
        return []
    m = len(cost[0])
    if any(len(row) != m for row in cost):
        raise ValueError("Cost matrix must be rectangular.")
    if n != m:
        raise ValueError("Only square cost matrices are supported.")

    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(0, n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = [0] * n
    for j in range(1, n + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    return assignment


def align_reconstruction_to_original(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    recovered_labels: Optional[torch.Tensor],
    history: Optional[list],
    denormalize=None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[list]]:
    """Align reconstructed batch order to match original batch order (batch gradient is permutation-invariant)."""
    if reconstructed is None:
        return reconstructed, recovered_labels, history

    n = min(original.size(0), reconstructed.size(0))
    if n <= 1:
        return reconstructed, recovered_labels, history

    original_01 = to_safe_01(original[:n], denormalize).flatten(1)
    reconstructed_01 = to_safe_01(reconstructed[:n], denormalize).flatten(1)
    diff = original_01[:, None, :] - reconstructed_01[None, :, :]
    cost_tensor = (diff * diff).mean(dim=-1)
    perm = _linear_sum_assignment(cost_tensor.tolist())
    perm_t = torch.as_tensor(perm, device=reconstructed.device, dtype=torch.long)

    reconstructed = reconstructed.index_select(0, perm_t)
    if recovered_labels is not None:
        recovered_labels = recovered_labels.index_select(0, perm_t)
    if history:
        for entry in history:
            if "data" in entry and isinstance(entry["data"], torch.Tensor) and entry["data"].dim() == 4:
                entry["data"] = entry["data"].index_select(0, perm_t.to(entry["data"].device))
    return reconstructed, recovered_labels, history


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
        base_fig = plt.figure(figsize=(2.5 * cols, 5))
        for idx in range(cols):
            plt.subplot(2, cols, idx + 1)
            plt.imshow(to_safe_pil(client.data[idx], denormalize))
            plt.title(f"Client {client.client_id} sample {idx}")
            plt.axis("off")

            plt.subplot(2, cols, idx + 1 + cols)
            plt.imshow(to_safe_pil(recovered_data[idx], denormalize))
            plt.title("Reconstructed")
            plt.axis("off")

        if history:
            n = len(history)
            grid_cols = min(cols, 8)
            grid_rows = int(math.ceil(cols / grid_cols)) if cols else 1
            fig_h = max(3.6, 1.7 * grid_rows + 1.2)
            fig, axes = plt.subplots(
                1,
                n,
                figsize=(max(3 * n, 6), fig_h),
                constrained_layout=True,
                squeeze=False,
            )
            for i, entry in enumerate(history):
                ax = axes[0][i]
                ax.imshow(to_safe_grid_pil(entry["data"], denormalize, max_cols=grid_cols))
                ax.set_title(
                    f"C{client.client_id} Iter {entry['iteration']}\nGradLoss {entry['loss']:.2f}",
                    fontsize=8,
                    pad=4,
                )
                ax.title.set_wrap(True)
                ax.axis("off")

        base_fig.tight_layout()


def visualize_aggregated_recovery(
    all_real_data: torch.Tensor,
    recovered_data: torch.Tensor,
    history,
    num_clients: int,
    samples_per_client: int,
    denormalize=None,
):
    """Plot aggregated reconstruction against the stacked real data."""
    total_samples = min(recovered_data.size(0), all_real_data.size(0))
    cols = min(total_samples, 8)
    rows = 2

    base_fig = plt.figure(figsize=(2.5 * cols, 5))
    for idx in range(cols):
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(to_safe_pil(all_real_data[idx], denormalize))
        plt.title(f"Real {idx}")
        plt.axis("off")

        plt.subplot(rows, cols, idx + 1 + cols)
        plt.imshow(to_safe_pil(recovered_data[idx], denormalize))
        plt.title("Reconstructed")
        plt.axis("off")

    base_fig.tight_layout()

    if history:
        # Show aggregated iteration history in separate windows per client, mirroring per-client view.
        for client_id in range(num_clients):
            start = client_id * samples_per_client
            end = min(start + samples_per_client, total_samples)
            if start >= total_samples:
                break
            n = len(history)
            grid_cols = min(end - start, 8) if end > start else 1
            grid_rows = int(math.ceil((end - start) / grid_cols)) if end > start else 1
            fig_h = max(3.6, 1.7 * grid_rows + 1.2)
            fig, axes = plt.subplots(
                1,
                n,
                figsize=(max(3 * n, 6), fig_h),
                constrained_layout=True,
                squeeze=False,
            )
            for i, entry in enumerate(history):
                ax = axes[0][i]
                batch = entry["data"][start:end]
                ax.imshow(to_safe_grid_pil(batch, denormalize, max_cols=grid_cols))
                mse = float(
                    F.mse_loss(
                        to_safe_01(batch, denormalize),
                        to_safe_01(all_real_data[start:end], denormalize),
                    ).item()
                )
                ax.set_title(
                    f"Agg C{client_id} Iter {entry['iteration']}\nGradLoss {entry['loss']:.2f}  MSE {mse:.4f}",
                    fontsize=8,
                    pad=4,
                )
                ax.title.set_wrap(True)
                ax.axis("off")


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

    def _to_01(x: torch.Tensor) -> torch.Tensor:
        x = x.detach().cpu()
        if denormalize:
            x = denormalize(x)
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        return torch.clamp(x, 0.0, 1.0)

    def _mean_std(t: torch.Tensor) -> Dict[str, float]:
        return {"mean": float(t.mean().item()), "std": float(t.std(unbiased=False).item())}

    def _metrics_rows(
        *,
        mode: str,
        client_id: Optional[int],
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        labels: torch.Tensor,
        recovered_labels: torch.Tensor,
        dataset_indices: List[int],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        original_01 = _to_01(original)
        reconstructed_01 = _to_01(reconstructed)
        diff = reconstructed_01 - original_01
        mse_per = (diff * diff).flatten(1).mean(dim=1)
        psnr_per = 10.0 * torch.log10(1.0 / torch.clamp(mse_per, min=1e-8))
        cos_per = F.cosine_similarity(original_01.flatten(1), reconstructed_01.flatten(1), dim=1)
        pred = recovered_labels.detach().cpu().argmax(dim=1)
        labels_cpu = labels.detach().cpu()
        correct = (pred == labels_cpu).to(torch.float32)

        rows: List[Dict[str, Any]] = []
        for i in range(original_01.size(0)):
            rows.append(
                {
                    "mode": mode,
                    "client_id": "" if client_id is None else int(client_id),
                    "sample_idx": int(i),
                    "dataset_idx": int(dataset_indices[i]),
                    "mse": float(mse_per[i].item()),
                    "psnr": float(psnr_per[i].item()),
                    "feature_similarity": float(cos_per[i].item()),
                    "true_label": int(labels_cpu[i].item()),
                    "pred_label": int(pred[i].item()),
                    "correct": int(correct[i].item()),
                }
            )

        summary: Dict[str, Any] = {
            "count": int(original_01.size(0)),
            "mse": _mean_std(mse_per),
            "psnr": _mean_std(psnr_per),
            "feature_similarity": _mean_std(cos_per),
            "class_accuracy": _mean_std(correct),
        }
        return rows, summary

    metrics_rows: List[Dict[str, Any]] = []
    metrics_summary: Dict[str, Any] = {"per_client": {}, "aggregated": None}

    for result in per_client_results:
        client = result["client"]
        recovered_data = result["recovered_data"]
        recovered_labels = result.get("recovered_labels")
        history = result["history"]
        client_dir = run_dir / f"client_{client.client_id}"
        if recovered_data is None:
            client_dir.mkdir(parents=True, exist_ok=True)
            (client_dir / "error.txt").write_text("Reconstruction failed (no recovered_data). Try more restarts or smaller init-scale.")
            continue
        for idx in range(recovered_data.size(0)):
            save_image(client.data[idx], client_dir / f"original_{idx}.png", denormalize)
            save_image(recovered_data[idx], client_dir / f"reconstructed_{idx}.png", denormalize)
        if history:
            hist_dir = client_dir / "history"
            for entry in history:
                save_grid_image(
                    entry["data"],
                    hist_dir / f"iter_{entry['iteration']}.png",
                    denormalize=denormalize,
                    max_cols=min(recovered_data.size(0), 8),
                )

        if recovered_labels is not None:
            start = client.client_id * args.samples_per_client
            end = start + args.samples_per_client
            rows, summary = _metrics_rows(
                mode="per-client",
                client_id=client.client_id,
                original=client.data,
                reconstructed=recovered_data,
                labels=client.labels,
                recovered_labels=recovered_labels,
                dataset_indices=indices[start:end],
            )
            metrics_rows.extend(rows)
            metrics_summary["per_client"][str(client.client_id)] = summary

    if aggregated_result and all_real_data is not None:
        agg_dir = run_dir / "aggregated"
        recovered_data = aggregated_result["recovered_data"]
        history = aggregated_result["history"]
        recovered_labels = aggregated_result.get("recovered_labels")
        if recovered_data is None:
            agg_dir.mkdir(parents=True, exist_ok=True)
            (agg_dir / "error.txt").write_text("Reconstruction failed (no recovered_data). Try more restarts or smaller init-scale.")
            return
        total_samples = min(recovered_data.size(0), all_real_data.size(0))
        for idx in range(total_samples):
            save_image(all_real_data[idx], agg_dir / f"original_{idx}.png", denormalize)
            save_image(recovered_data[idx], agg_dir / f"reconstructed_{idx}.png", denormalize)
        if history:
            hist_dir = agg_dir / "history"
            for entry in history:
                for client_id in range(args.num_clients):
                    start = client_id * args.samples_per_client
                    end = min(start + args.samples_per_client, total_samples)
                    if start >= total_samples:
                        break
                    save_grid_image(
                        entry["data"][start:end],
                        hist_dir / f"client_{client_id}_iter_{entry['iteration']}.png",
                        denormalize=denormalize,
                        max_cols=min(end - start, 8),
                    )

        if recovered_labels is not None:
            labels = torch.cat([client.labels for client in clients], dim=0)[:total_samples]
            rows, summary = _metrics_rows(
                mode="aggregated",
                client_id=None,
                original=all_real_data[:total_samples],
                reconstructed=recovered_data[:total_samples],
                labels=labels,
                recovered_labels=recovered_labels[:total_samples],
                dataset_indices=indices[:total_samples],
            )
            metrics_rows.extend(rows)
            metrics_summary["aggregated"] = summary

    if metrics_rows:
        (run_dir / "metrics_summary.json").write_text(json.dumps(metrics_summary, indent=2))
        with (run_dir / "metrics.csv").open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "mode",
                    "client_id",
                    "sample_idx",
                    "dataset_idx",
                    "mse",
                    "psnr",
                    "feature_similarity",
                    "true_label",
                    "pred_label",
                    "correct",
                ],
            )
            writer.writeheader()
            writer.writerows(metrics_rows)


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
            recovered_data, recovered_labels, history = align_reconstruction_to_original(
                client.data, recovered_data, recovered_labels, history, denormalize=denormalize
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
        all_real_data = torch.cat([client.data for client in clients], dim=0)
        recovered_data, recovered_labels, history = align_reconstruction_to_original(
            all_real_data, recovered_data, recovered_labels, history, denormalize=denormalize
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
        visualize_aggregated_recovery(
            all_real_data,
            aggregated_result["recovered_data"],
            aggregated_result["history"],
            num_clients=args.num_clients,
            samples_per_client=args.samples_per_client,
            denormalize=denormalize,
        )

    plt.show()


if __name__ == "__main__":
    main()
