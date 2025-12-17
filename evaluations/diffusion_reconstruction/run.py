# -*- coding: utf-8 -*-
"""
Diffusion-guided reconstruction evaluation.

This script reconstructs training data from gradients using a pretrained DDPM
prior (from Hugging Face diffusers) to denoise intermediate iterates. It
saves reconstructed images, loss traces, and quality metrics per client.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

# Ensure repository root is importable when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Newer diffusers expects torch.xpu; older torch (<2.0) lacks it, so stub it for compatibility.
if not hasattr(torch, "xpu"):
    class _DummyXPU:
        @staticmethod
        def empty_cache():
            return None
    torch.xpu = _DummyXPU()  # type: ignore[attr-defined]

try:
    from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    UNet2DModel,
)
except ImportError as exc:  # pragma: no cover - dependency message only
    raise ImportError(
        "diffusers is required for diffusion-guided reconstruction. "
        "Install it with `pip install diffusers`."
    ) from exc

from federated import FederatedClient, FederatedServer
from models.vision_new import CIFAR100_MEAN, CIFAR100_STD, MODEL_BUILDERS, build_model, default_transform
from utils import cross_entropy_for_onehot


def to_safe_pil(img_tensor: torch.Tensor, denormalize=None):
    """Convert tensor to PIL image with optional denormalization and clamping."""
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


def pick_indices(total_samples: int, dataset_size: int, seed: int, client_indices: str = None) -> List[int]:
    """Parse/produce dataset indices for each client sample."""
    if client_indices:
        parsed = [int(x.strip()) for x in client_indices.split(",") if x.strip()]
        if len(parsed) != total_samples:
            raise ValueError(
                f"--client-indices provided {len(parsed)} indices, but {total_samples} are required "
                "(num_clients * samples_per_client)."
            )
        return parsed

    if total_samples > dataset_size:
        raise ValueError(f"Requested {total_samples} samples but dataset only has {dataset_size}.")

    generator = torch.Generator().manual_seed(seed)
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


def normalize_for_model(x: torch.Tensor, arch: str) -> torch.Tensor:
    """Map diffusion space [-1, 1] to model input space (optionally normalized)."""
    x01 = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
    if arch.lower() == "lenet":
        return x01
    mean = torch.tensor(CIFAR100_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR100_STD, device=x.device).view(1, 3, 1, 1)
    return (x01 - mean) / std


def total_variation(img: torch.Tensor) -> torch.Tensor:
    tv_h = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]).mean()
    tv_v = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]).mean()
    return tv_h + tv_v


def gradient_match_loss(
    dummy_data: torch.Tensor,
    label_logits: torch.Tensor,
    model: torch.nn.Module,
    target_gradients: List[torch.Tensor],
    arch: str,
    tv_weight: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Squared gradient matching loss with optional TV regularization."""
    model_input = normalize_for_model(dummy_data, arch)
    pred = model(model_input)
    dummy_onehot = F.softmax(label_logits, dim=-1)
    dummy_loss = cross_entropy_for_onehot(pred, dummy_onehot)
    dummy_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)
    grad_loss = sum(((gx - gy) ** 2).sum() for gx, gy in zip(dummy_dy_dx, target_gradients))

    tv_loss = torch.tensor(0.0, device=dummy_data.device)
    if tv_weight > 0:
        tv_loss = total_variation((dummy_data + 1.0) / 2.0)
    total_loss = grad_loss + tv_weight * tv_loss
    return total_loss, grad_loss, tv_loss


def diffusion_guided_reconstruction(
    model: torch.nn.Module,
    target_gradients: List[torch.Tensor],
    data_shape: torch.Size,
    num_classes: int,
    arch: str,
    diffusion_model: UNet2DModel,
    scheduler: DDPMScheduler,
    diffusion_steps: int = 50,
    match_lr: float = 0.1,
    match_lr_end: float = 0.01,
    match_steps: int = 5,
    label_lr: float = 0.05,
    label_lr_end: float = 0.01,
    tv_weight: float = 0.0,
    log_every: int = 20,
):
    """Reconstruct data by alternating gradient matching with diffusion denoising."""
    device = target_gradients[0].device
    scheduler.set_timesteps(diffusion_steps)
    current = torch.randn(data_shape, device=device) * scheduler.init_noise_sigma
    label_logits = torch.randn((data_shape[0], num_classes), device=device, requires_grad=True)
    label_opt = torch.optim.Adam([label_logits], lr=label_lr)

    history = []
    total_steps = len(scheduler.timesteps)
    for step, timestep in enumerate(scheduler.timesteps):
        # Cosine-ish decay to allow coarse steps early and fine steps late.
        progress = step / max(total_steps - 1, 1)
        match_lr_now = match_lr * (1 - progress) + match_lr_end * progress
        label_lr_now = label_lr * (1 - progress) + label_lr_end * progress
        for g in label_opt.param_groups:
            g["lr"] = label_lr_now

        for _ in range(match_steps):
            current = current.detach().requires_grad_(True)
            label_opt.zero_grad()

            total_loss, grad_loss, tv_loss = gradient_match_loss(
                current, label_logits, model, target_gradients, arch, tv_weight=tv_weight
            )
            total_loss.backward()

            with torch.no_grad():
                current = current - match_lr_now * current.grad
                current.clamp_(-1.0, 1.0)  # keep iterates in diffusion's input range
            label_opt.step()

        with torch.no_grad():
            noisy_latent = current.detach()
            noise_pred = diffusion_model(noisy_latent, timestep).sample
            diffusion_out = scheduler.step(noise_pred, timestep, noisy_latent)
            # Some schedulers (e.g., DDPM/UniPC) expose pred_original_sample; others (e.g., DPM++) do not.
            if hasattr(diffusion_out, "pred_original_sample") and diffusion_out.pred_original_sample is not None:
                prior_loss = F.mse_loss(noisy_latent, diffusion_out.pred_original_sample).item()
            elif hasattr(diffusion_out, "prev_sample") and diffusion_out.prev_sample is not None:
                prior_loss = F.mse_loss(noisy_latent, diffusion_out.prev_sample).item()
            else:
                prior_loss = 0.0
            current = diffusion_out.prev_sample

        if step % log_every == 0 or step == len(scheduler.timesteps) - 1:
            history.append(
                {
                    "step": int(step),
                    "grad_loss": float(grad_loss.item()),
                    "prior_loss": float(prior_loss),
                    "total_loss": float(total_loss.item()),
                    "tv_loss": float(tv_loss.item()) if tv_weight > 0 else 0.0,
                }
            )

    with torch.no_grad():
        recovered_data = torch.clamp((current + 1.0) / 2.0, 0.0, 1.0)
        recovered_labels = F.softmax(label_logits.detach(), dim=-1)
    return recovered_data, recovered_labels, history


def compute_metrics(
    original: torch.Tensor, reconstructed: torch.Tensor, labels: torch.Tensor, recovered_labels: torch.Tensor
) -> dict:
    """Compute simple reconstruction quality metrics."""
    mse_val = F.mse_loss(reconstructed, original).item()
    psnr = 10 * math.log10(1.0 / max(mse_val, 1e-8))
    feature_sim = F.cosine_similarity(
        original.view(original.size(0), -1), reconstructed.view(reconstructed.size(0), -1)
    ).mean().item()
    recovered_pred = recovered_labels.argmax(dim=1)
    class_acc = (recovered_pred == labels).float().mean().item()
    return {
        "mse": mse_val,
        "psnr": psnr,
        "feature_similarity": feature_sim,
        "class_accuracy": class_acc,
    }


def save_client_outputs(
    save_dir: Path,
    client_id: int,
    round_id: int,
    original_batch: torch.Tensor,
    reconstructed_batch: torch.Tensor,
    denormalize,
    metrics: dict,
    history: list,
):
    client_dir = save_dir / f"client_{client_id}_round_{round_id}"
    client_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(original_batch.size(0)):
        save_image(original_batch[idx], client_dir / f"original_{idx}.png", denormalize)
        save_image(reconstructed_batch[idx], client_dir / f"reconstructed_{idx}.png")

    (client_dir / f"metrics_client_{client_id}_round_{round_id}.json").write_text(
        json.dumps(metrics, indent=2)
    )
    (client_dir / f"loss_client_{client_id}_round_{round_id}.json").write_text(
        json.dumps(history, indent=2)
    )


def load_diffusion_prior(device: str, repo_id: str = "google/ddpm-cifar10-32", scheduler_type: str = "ddpm"):
    """Load a CIFAR-scale diffusion prior with a configurable scheduler."""
    schedulers = {
        "ddpm": DDPMScheduler,
        "ddim": DDIMScheduler,
        "dpmpp": DPMSolverMultistepScheduler,
        "euler": EulerDiscreteScheduler,
        "euler-ancestral": EulerAncestralDiscreteScheduler,
    }
    if scheduler_type not in schedulers:
        raise ValueError(f"Unsupported scheduler '{scheduler_type}'. Choose from {sorted(schedulers.keys())}.")

    scheduler_cls = schedulers[scheduler_type]
    scheduler = scheduler_cls.from_pretrained(repo_id)
    diffusion_model = UNet2DModel.from_pretrained(repo_id).to(device)
    diffusion_model.eval()
    return diffusion_model, scheduler


def parse_args():
    parser = argparse.ArgumentParser(description="Diffusion-based reconstruction from gradients.")
    parser.add_argument("--arch", type=str, default="lenet", choices=sorted(MODEL_BUILDERS.keys()))
    parser.add_argument("--pretrained", action="store_true", help="Use torchvision pretrained weights when available.")
    parser.add_argument(
        "--bn-eval",
        dest="bn_eval",
        action="store_true",
        default=True,
        help="Force BatchNorm layers to eval mode (recommended for tiny batches).",
    )
    parser.add_argument(
        "--no-bn-eval", dest="bn_eval", action="store_false", help="Keep BatchNorm layers in training mode."
    )
    parser.add_argument("--num-clients", type=int, default=1, help="Number of federated clients.")
    parser.add_argument("--samples-per-client", type=int, default=1, help="Samples held by each client.")
    parser.add_argument(
        "--reconstruct-mode",
        choices=["per-client", "aggregated", "both"],
        default="per-client",
        help="Reconstruct per-client gradients, aggregated gradients, or both.",
    )
    parser.add_argument(
        "--diffusion-steps", type=int, default=50, help="Number of denoising steps for the diffusion prior."
    )
    parser.add_argument("--match-steps", type=int, default=5, help="Gradient updates per diffusion denoise step.")
    parser.add_argument("--match-lr", type=float, default=0.1, help="Initial step size for gradient matching updates.")
    parser.add_argument("--match-lr-end", type=float, default=0.01, help="Final step size for gradient matching.")
    parser.add_argument("--label-lr", type=float, default=0.05, help="Initial step size for label logit updates.")
    parser.add_argument("--label-lr-end", type=float, default=0.01, help="Final step size for label logit updates.")
    parser.add_argument("--tv-weight", type=float, default=0.0, help="Total-variation weight on reconstructed images.")
    parser.add_argument("--log-every", type=int, default=20, help="Store loss entries every N diffusion steps.")
    parser.add_argument(
        "--diffusion-model",
        type=str,
        default="google/ddpm-cifar10-32",
        help="Diffusion prior repo id (Hugging Face diffusers UNet2DModel).",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="dpmpp",
        choices=["ddpm", "ddim", "dpmpp", "euler", "euler-ancestral"],
        help="Scheduler used for the diffusion prior (dpmpp is a strong modern default).",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="outputs/diffusion_dlg",
        help="Directory to save reconstructed images/metrics.",
    )
    parser.add_argument("--round-id", type=int, default=0, help="Round id used in output filenames.")
    parser.add_argument("--seed", type=int, default=1234, help="Seed for sampling client data.")
    parser.add_argument(
        "--client-indices",
        type=str,
        default=None,
        help="Comma separated CIFAR100 indices (must equal num_clients * samples_per_client).",
    )
    parser.add_argument(
        "--normalize-gradients",
        action="store_true",
        help="L2-normalize client gradients before aggregation.",
    )
    parser.add_argument(
        "--apply-agg-step",
        action="store_true",
        help="Apply one gradient step on the model using aggregated gradients before inversion.",
    )
    parser.add_argument(
        "--agg-lr",
        type=float,
        default=0.1,
        help="Learning rate for the aggregated gradient step (used with --apply-agg-step).",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="~/.torch",
        help="Location to download/load CIFAR100.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional model checkpoint to load before computing gradients.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running diffusion-guided reconstruction on {device}")

    transform = default_transform(args.arch)
    dataset = datasets.CIFAR100(args.data_root, download=True, transform=transform)
    num_classes = 100
    total_samples = args.num_clients * args.samples_per_client
    denormalize = build_denormalize(args.arch)

    torch.manual_seed(args.seed)
    indices = pick_indices(total_samples, len(dataset), seed=args.seed, client_indices=args.client_indices)
    print(f"Using dataset indices: {indices}")

    model = build_model(args.arch, num_classes=num_classes, pretrained=args.pretrained).to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint from {args.checkpoint}")
    if args.bn_eval and args.arch.lower() != "lenet":
        model.eval()

    clients = build_clients(dataset, indices, args.num_clients, args.samples_per_client, device, num_classes)
    diffusion_model, scheduler = load_diffusion_prior(
        device, repo_id=args.diffusion_model, scheduler_type=args.scheduler
    )
    server = FederatedServer(model, device=device, num_classes=num_classes)

    run_dir = Path(args.save_dir) / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "arch": args.arch,
        "pretrained": args.pretrained,
        "seed": args.seed,
        "indices": indices,
        "num_clients": args.num_clients,
        "samples_per_client": args.samples_per_client,
        "diffusion_steps": args.diffusion_steps,
        "match_steps": args.match_steps,
        "match_lr": args.match_lr,
        "match_lr_end": args.match_lr_end,
        "label_lr": args.label_lr,
        "label_lr_end": args.label_lr_end,
        "tv_weight": args.tv_weight,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": args.checkpoint,
        "reconstruct_mode": args.reconstruct_mode,
        "normalize_gradients": args.normalize_gradients,
        "apply_agg_step": args.apply_agg_step,
        "agg_lr": args.agg_lr,
        "diffusion_model": args.diffusion_model,
        "scheduler": args.scheduler,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    client_gradients = []
    for client in clients:
        grads, loss = client.compute_gradients(model)
        grads = [g.to(device) for g in grads]
        client_gradients.append(grads)
        print(f"Client {client.client_id}: local loss={loss:.4f}")

    if args.reconstruct_mode in ("per-client", "both"):
        for client, grads in zip(clients, client_gradients):
            recovered_data, recovered_labels, history = diffusion_guided_reconstruction(
                model=model,
                target_gradients=grads,
                data_shape=client.data.shape,
                num_classes=num_classes,
                arch=args.arch,
                diffusion_model=diffusion_model,
                scheduler=scheduler,
                diffusion_steps=args.diffusion_steps,
                match_lr=args.match_lr,
                match_steps=args.match_steps,
                label_lr=args.label_lr,
                tv_weight=args.tv_weight,
                log_every=args.log_every,
            )

            original_batch = client.data.detach().cpu()
            reconstructed_batch = recovered_data.detach().cpu()
            labels_cpu = client.labels.detach().cpu()
            recovered_labels_cpu = recovered_labels.detach().cpu()
            visual_original = original_batch
            if denormalize:
                visual_original = torch.clamp(denormalize(visual_original), 0.0, 1.0)
            visual_reconstructed = torch.clamp(reconstructed_batch, 0.0, 1.0)

            metrics = compute_metrics(
                visual_original, visual_reconstructed, labels_cpu, recovered_labels_cpu
            )
            save_client_outputs(
                run_dir,
                client_id=client.client_id,
                round_id=args.round_id,
                original_batch=visual_original,
                reconstructed_batch=visual_reconstructed,
                denormalize=None,
                metrics=metrics,
                history=history,
            )
            print(f"Client {client.client_id}: reconstruction saved with metrics {metrics}")

    if args.reconstruct_mode in ("aggregated", "both"):
        aggregated_gradients = server.aggregate_gradients(
            client_gradients, normalize=args.normalize_gradients
        )
        print(f"Aggregated gradients collected. normalize_gradients={args.normalize_gradients}")

        if args.apply_agg_step:
            print(f"Applying aggregated gradient step to model with lr={args.agg_lr}")
            server.apply_gradient_step(aggregated_gradients, lr=args.agg_lr)

        total_samples = args.num_clients * args.samples_per_client
        data_shape = torch.Size((total_samples, *clients[0].data.shape[1:]))
        recovered_data, recovered_labels, history = diffusion_guided_reconstruction(
            model=model,
            target_gradients=aggregated_gradients,
            data_shape=data_shape,
            num_classes=num_classes,
            arch=args.arch,
            diffusion_model=diffusion_model,
            scheduler=scheduler,
            diffusion_steps=args.diffusion_steps,
            match_lr=args.match_lr,
            match_steps=args.match_steps,
            label_lr=args.label_lr,
            tv_weight=args.tv_weight,
            log_every=args.log_every,
        )

        all_original = torch.cat([client.data for client in clients], dim=0).detach().cpu()
        all_labels = torch.cat([client.labels for client in clients], dim=0).detach().cpu()
        all_reconstructed = torch.clamp(recovered_data.detach().cpu(), 0.0, 1.0)
        visual_original = all_original
        if denormalize:
            visual_original = torch.clamp(denormalize(visual_original), 0.0, 1.0)

        metrics = compute_metrics(
            visual_original,
            all_reconstructed,
            all_labels,
            recovered_labels.detach().cpu(),
        )

        agg_dir = run_dir / "aggregated"
        agg_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(visual_original.size(0)):
            save_image(visual_original[idx], agg_dir / f"original_{idx}.png", None)
            save_image(all_reconstructed[idx], agg_dir / f"reconstructed_{idx}.png", None)

        (agg_dir / "metrics_aggregated.json").write_text(json.dumps(metrics, indent=2))
        (agg_dir / "loss_aggregated.json").write_text(json.dumps(history, indent=2))
        print(f"Aggregated reconstruction saved with metrics {metrics}")


if __name__ == "__main__":
    main()
