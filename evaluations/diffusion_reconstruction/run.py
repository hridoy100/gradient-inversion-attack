# -*- coding: utf-8 -*-
"""
Diffusion-guided reconstruction evaluation.

This script reconstructs training data from gradients using a pretrained DDPM
prior (from Hugging Face diffusers) to denoise intermediate iterates. It
saves reconstructed images, loss traces, and quality metrics per client.
"""
import argparse
import csv
import json
import math
import os
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

# Newer diffusers may reference torch.mps; some torch builds (non-macOS or older versions) lack it.
if not hasattr(torch, "mps"):
    class _DummyMPS:
        @staticmethod
        def empty_cache():
            return None

        @staticmethod
        def is_available():
            return False

    torch.mps = _DummyMPS()  # type: ignore[attr-defined]

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
    original_01: torch.Tensor,
    reconstructed_01: torch.Tensor,
    recovered_labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Align reconstructed batch order to match original batch order (batch gradient is permutation-invariant)."""
    n = min(original_01.size(0), reconstructed_01.size(0))
    if n <= 1:
        return reconstructed_01, recovered_labels

    original_flat = original_01[:n].detach().cpu().flatten(1)
    recon_flat = reconstructed_01[:n].detach().cpu().flatten(1)
    diff = original_flat[:, None, :] - recon_flat[None, :, :]
    cost = (diff * diff).mean(dim=-1)
    perm = _linear_sum_assignment(cost.tolist())
    perm_t = torch.as_tensor(perm, device=reconstructed_01.device, dtype=torch.long)
    return reconstructed_01.index_select(0, perm_t), recovered_labels.index_select(0, perm_t)


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
    grad_loss_mode: str = "l2mean",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Squared gradient matching loss with optional TV regularization."""
    model_input = normalize_for_model(dummy_data, arch)
    pred = model(model_input)
    dummy_onehot = F.softmax(label_logits, dim=-1)
    dummy_loss = cross_entropy_for_onehot(pred, dummy_onehot)
    dummy_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)
    if grad_loss_mode not in {"l2sum", "l2mean", "cosine"}:
        raise ValueError("grad_loss_mode must be one of: l2sum, l2mean, cosine")

    if grad_loss_mode == "cosine":
        grad_loss = torch.tensor(0.0, device=dummy_data.device)
        for gx, gy in zip(dummy_dy_dx, target_gradients):
            gx_f = gx.reshape(-1)
            gy_f = gy.reshape(-1)
            denom = (torch.norm(gx_f) * torch.norm(gy_f)).clamp_min(1e-12)
            grad_loss = grad_loss + (1.0 - torch.sum(gx_f * gy_f) / denom)
    else:
        reduce_fn = torch.sum if grad_loss_mode == "l2sum" else torch.mean
        grad_loss = sum(reduce_fn((gx - gy) ** 2) for gx, gy in zip(dummy_dy_dx, target_gradients))

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
    prior_mode: str = "denoise",
    prior_weight: float = 0.0,
    prior_t_min: int = 0,
    prior_t_max: int = None,
    grad_loss_mode: str = "l2mean",
    diffusion_steps: int = 50,
    match_lr: float = 0.1,
    match_lr_end: float = 0.01,
    match_steps: int = 5,
    label_lr: float = 0.05,
    label_lr_end: float = 0.01,
    tv_weight: float = 0.0,
    log_every: int = 20,
):
    """Reconstruct data from gradients using a diffusion prior.

    prior_mode:
      - "denoise": alternate gradient matching with scheduler denoising steps (legacy; can be unstable).
      - "score": optimize a clean image directly with a score-matching prior term (more stable).
    """
    device = target_gradients[0].device
    if prior_t_max is None:
        prior_t_max = int(getattr(scheduler.config, "num_train_timesteps", 1000) - 1)

    if prior_mode == "denoise":
        scheduler.set_timesteps(diffusion_steps)
        current = torch.randn(data_shape, device=device) * scheduler.init_noise_sigma
    elif prior_mode == "score":
        # Optimize a clean image estimate directly in diffusion space [-1, 1].
        current = torch.randn(data_shape, device=device).clamp_(-1.0, 1.0)
    else:
        raise ValueError("prior_mode must be 'denoise' or 'score'.")

    label_logits = torch.randn((data_shape[0], num_classes), device=device, requires_grad=True)
    label_opt = torch.optim.Adam([label_logits], lr=label_lr)

    history = []
    best = {"loss": float("inf"), "data": None, "labels": None}
    total_steps = diffusion_steps if prior_mode == "score" else len(scheduler.timesteps)
    iterator = range(total_steps)
    for step in iterator:
        timestep = scheduler.timesteps[step] if prior_mode == "denoise" else None
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
                current,
                label_logits,
                model,
                target_gradients,
                arch,
                tv_weight=tv_weight,
                grad_loss_mode=grad_loss_mode,
            )
            prior_loss = torch.tensor(0.0, device=device)
            if prior_mode == "score" and prior_weight > 0:
                # Score-matching prior regularizer: encourage the current image to be likely under the diffusion model.
                t = torch.randint(prior_t_min, prior_t_max + 1, (current.size(0),), device=device).long()
                noise = torch.randn_like(current)
                x_t = scheduler.add_noise(current, noise, t)
                x_t_in = scheduler.scale_model_input(x_t, t) if hasattr(scheduler, "scale_model_input") else x_t
                noise_pred = diffusion_model(x_t_in, t).sample
                prior_loss = F.mse_loss(noise_pred, noise)
                total_loss = total_loss + prior_weight * prior_loss

            total_loss.backward()

            with torch.no_grad():
                current = current - match_lr_now * current.grad
                current.clamp_(-1.0, 1.0)  # keep iterates in diffusion's input range
            label_opt.step()

        if prior_mode == "denoise":
            with torch.no_grad():
                noisy_latent = current.detach()
                # Some schedulers expect inputs to be scaled before step.
                model_latent = (
                    scheduler.scale_model_input(noisy_latent, timestep)
                    if hasattr(scheduler, "scale_model_input")
                    else noisy_latent
                )
                noise_pred = diffusion_model(model_latent, timestep).sample
                diffusion_out = scheduler.step(noise_pred, timestep, noisy_latent)
                current = diffusion_out.prev_sample
            prior_loss_val = 0.0
        else:
            prior_loss_val = float(prior_loss.item()) if prior_weight > 0 else 0.0

        if step % log_every == 0 or step == total_steps - 1:
            history.append(
                {
                    "step": int(step),
                    "grad_loss": float(grad_loss.item()),
                    "prior_loss": float(prior_loss_val),
                    "total_loss": float(total_loss.item()),
                    "tv_loss": float(tv_loss.item()) if tv_weight > 0 else 0.0,
                }
            )
        # Track best iterate by total_loss to avoid late divergence.
        if total_loss.item() < best["loss"]:
            best["loss"] = float(total_loss.item())
            best["data"] = current.detach().clone()
            best["labels"] = label_logits.detach().clone()

    with torch.no_grad():
        data_final = best["data"] if best["data"] is not None else current
        labels_final = best["labels"] if best["labels"] is not None else label_logits
        recovered_data = torch.clamp((data_final + 1.0) / 2.0, 0.0, 1.0)
        recovered_labels = F.softmax(labels_final, dim=-1)
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
    # We never update the diffusion prior; freezing avoids allocating grads for its parameters
    # (while still allowing gradients to flow to its inputs in prior_mode='score').
    for p in diffusion_model.parameters():
        p.requires_grad_(False)
    return diffusion_model, scheduler


def parse_args():
    parser = argparse.ArgumentParser(description="Diffusion-based reconstruction from gradients.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run on. 'auto' picks CUDA if available.",
    )
    parser.add_argument(
        "--no-tf32",
        action="store_true",
        help="Disable TF32 on CUDA matmul/cuDNN (often improves stability on some GPUs).",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic CUDA/cuDNN algorithms (slower, but can improve reproducibility/stability).",
    )
    parser.add_argument(
        "--cuda-stable",
        action="store_true",
        help="Convenience flag: equivalent to setting --no-tf32 and --deterministic.",
    )
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
        "--diffusion-steps",
        type=int,
        default=50,
        help="Number of outer steps (denoise steps in prior-mode=denoise; optimization steps in prior-mode=score).",
    )
    parser.add_argument("--match-steps", type=int, default=5, help="Gradient updates per diffusion denoise step.")
    parser.add_argument("--match-lr", type=float, default=0.1, help="Initial step size for gradient matching updates.")
    parser.add_argument("--match-lr-end", type=float, default=0.01, help="Final step size for gradient matching.")
    parser.add_argument("--label-lr", type=float, default=0.05, help="Initial step size for label logit updates.")
    parser.add_argument("--label-lr-end", type=float, default=0.01, help="Final step size for label logit updates.")
    parser.add_argument("--tv-weight", type=float, default=0.0, help="Total-variation weight on reconstructed images.")
    parser.add_argument("--log-every", type=int, default=20, help="Store loss entries every N diffusion steps.")
    parser.add_argument(
        "--prior-mode",
        choices=["denoise", "score"],
        default="score",
        help="How to use the diffusion model as a prior (score is generally more stable).",
    )
    parser.add_argument(
        "--prior-weight",
        type=float,
        default=1.0,
        help="Weight of diffusion score-matching prior when --prior-mode=score.",
    )
    parser.add_argument(
        "--prior-t-min",
        type=int,
        default=0,
        help="Minimum diffusion timestep sampled for the prior regularizer (score mode).",
    )
    parser.add_argument(
        "--prior-t-max",
        type=int,
        default=None,
        help="Maximum diffusion timestep sampled for the prior regularizer (score mode). Default: scheduler config.",
    )
    parser.add_argument(
        "--grad-loss-mode",
        choices=["l2mean", "l2sum", "cosine"],
        default="l2mean",
        help="Gradient matching loss reduction; l2mean usually scales better than l2sum.",
    )
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
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda but CUDA is not available.")
        device = "cuda"
    else:
        device = "cpu"

    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        if args.cuda_stable:
            args.no_tf32 = True
            args.deterministic = True
        if args.no_tf32:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        if args.deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                if os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
                    torch.use_deterministic_algorithms(True)
                else:
                    print(
                        "Warning: --deterministic requested, but CUBLAS_WORKSPACE_CONFIG is not set. "
                        "Some CUDA ops may be nondeterministic. To enforce determinism, set "
                        "CUBLAS_WORKSPACE_CONFIG=:4096:8 (or :16:8) before running."
                    )
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
    scheduler_type = args.scheduler
    if args.prior_mode == "score" and scheduler_type != "ddpm":
        # Score mode relies on a training-like noise schedule (add_noise); DDPM is the safest default.
        print("prior-mode=score: overriding --scheduler to 'ddpm' for add_noise compatibility.")
        scheduler_type = "ddpm"
    diffusion_model, scheduler = load_diffusion_prior(device, repo_id=args.diffusion_model, scheduler_type=scheduler_type)
    server = FederatedServer(model, device=device, num_classes=num_classes)

    run_dir = Path(args.save_dir) / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows = []

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
        "scheduler": scheduler_type,
        "prior_mode": args.prior_mode,
        "prior_weight": args.prior_weight,
        "prior_t_min": args.prior_t_min,
        "prior_t_max": args.prior_t_max,
        "grad_loss_mode": args.grad_loss_mode,
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
                prior_mode=args.prior_mode,
                prior_weight=args.prior_weight,
                prior_t_min=args.prior_t_min,
                prior_t_max=args.prior_t_max,
                grad_loss_mode=args.grad_loss_mode,
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

            visual_reconstructed, recovered_labels_cpu = align_reconstruction_to_original(
                visual_original, visual_reconstructed, recovered_labels_cpu
            )
            metrics = compute_metrics(
                visual_original, visual_reconstructed, labels_cpu, recovered_labels_cpu
            )
            start = client.client_id * args.samples_per_client
            end = start + args.samples_per_client
            metrics_rows.append(
                {
                    "round_id": int(args.round_id),
                    "mode": "per-client",
                    "client_id": int(client.client_id),
                    "dataset_indices": ",".join(str(i) for i in indices[start:end]),
                    **{k: float(v) for k, v in metrics.items()},
                }
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
            prior_mode=args.prior_mode,
            prior_weight=args.prior_weight,
            prior_t_min=args.prior_t_min,
            prior_t_max=args.prior_t_max,
            grad_loss_mode=args.grad_loss_mode,
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

        recovered_labels_cpu = recovered_labels.detach().cpu()
        all_reconstructed, recovered_labels_cpu = align_reconstruction_to_original(
            visual_original, all_reconstructed, recovered_labels_cpu
        )
        metrics = compute_metrics(
            visual_original,
            all_reconstructed,
            all_labels,
            recovered_labels_cpu,
        )
        metrics_rows.append(
            {
                "round_id": int(args.round_id),
                "mode": "aggregated",
                "client_id": "",
                "dataset_indices": ",".join(str(i) for i in indices),
                **{k: float(v) for k, v in metrics.items()},
            }
        )

        agg_dir = run_dir / "aggregated"
        agg_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(visual_original.size(0)):
            save_image(visual_original[idx], agg_dir / f"original_{idx}.png", None)
            save_image(all_reconstructed[idx], agg_dir / f"reconstructed_{idx}.png", None)

        (agg_dir / "metrics_aggregated.json").write_text(json.dumps(metrics, indent=2))
        (agg_dir / "loss_aggregated.json").write_text(json.dumps(history, indent=2))
        print(f"Aggregated reconstruction saved with metrics {metrics}")

    if metrics_rows:
        with (run_dir / "metrics.csv").open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "round_id",
                    "mode",
                    "client_id",
                    "dataset_indices",
                    "mse",
                    "psnr",
                    "feature_similarity",
                    "class_accuracy",
                ],
            )
            writer.writeheader()
            writer.writerows(metrics_rows)


if __name__ == "__main__":
    main()
