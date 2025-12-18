# Federated Gradient Inversion

Lightweight demo of reconstructing training data from shared gradients in a simulated federated setting. Multiple clients hold CIFAR100 samples, compute gradients on a shared LeNet, send those gradients to the server, and the server runs gradient matching to recover the data.

## Prerequisites
- Python >= 3.6
- PyTorch >= 1.0
- torchvision >= 0.4

## Usage
```
# Three clients, one sample each, 200 inversion steps, reconstruct per client with 3 restarts
python3 main.py --num-clients 3 --samples-per-client 1 --iterations 200 --log-every 20 --reconstruct-mode per-client --restarts 3

# Use specific dataset indices for reproducibility
python3 main.py --num-clients 2 --samples-per-client 2 --client-indices 1,5,42,100

# Run both per-client and aggregated reconstructions
python3 main.py --reconstruct-mode both

# Aggregated reconstruction with TV prior, normalized gradients, and smaller init noise
python3 main.py --reconstruct-mode aggregated --tv-weight 1e-3 --normalize-gradients --init-scale 0.05 --iterations 400 --restarts 5

# Apply aggregated gradient step to the server model before inversion
python3 main.py --reconstruct-mode aggregated --apply-agg-step --agg-lr 0.05

# Force CPU (useful if CUDA numerics are unstable on your environment)
python3 main.py --device cpu --reconstruct-mode aggregated --iterations 300 --restarts 2
```

Key options:
- `--device`: `auto` (default), `cpu`, or `cuda`.
- `--dataset`: `cifar100` (default) or `tiny-imagenet`.
- `--dataset-split`: `train` (default) or `val` (Tiny-ImageNet).
- `--image-size`: input size used for Tiny-ImageNet transforms (default `64`).
- `--arch`: model architecture for the federated model (`lenet`, `resnet18`, `resnet34`, `resnet50`, `mobilenet_v2`).
- `--pretrained`: initialize torchvision architectures with pretrained weights when available.
- `--bn-eval/--no-bn-eval`: keep BatchNorm layers in eval mode (default) to avoid tiny-batch failures, or override.
- `--num-clients`: number of simulated clients.
- `--samples-per-client`: number of images per client.
- `--client-indices`: comma-separated dataset indices to force the client samples.
- `--iterations`: LBFGS iterations for inversion.
- `--log-every`: interval for snapshotting reconstructed images during optimization.
- `--reconstruct-mode`: choose `per-client` (default), `aggregated`, or `both`.
- `--restarts`: number of random restarts for inversion (keeps the best loss).
- `--dummy-seed`: base seed for dummy init; per-client offset is added automatically.
- `--normalize-gradients`: L2-normalize client gradients before aggregation.
- `--tv-weight`: strength of total-variation prior to smooth reconstructions (helpful for aggregated).
- `--init-scale`: std multiplier for dummy data initialization (smaller can help aggregated runs).
- `--apply-agg-step`: apply one gradient step to the server model using aggregated gradients before inversion.
- `--agg-lr`: learning rate for the aggregated step (used with `--apply-agg-step`).
- `--checkpoint`: path to a model state_dict to load (overrides random init).
- `--no-progress`: disable tqdm progress bars during inversion (enabled by default).
- `--save-dir`: base directory to save reconstruction images/metadata (`outputs/reconstructions` by default).
- `--no-save`: disable saving reconstructions to disk.
- `--data-root`: dataset location. For CIFAR100 it is the torchvision download/cache root (defaults to `~/.torch`). For Tiny-ImageNet it should contain `tiny-imagenet-200/`.

Notes:
- Aggregated reconstruction matches batch gradients, which are permutation-invariant. The script aligns reconstructed samples to real samples for visualization/metrics, but perfect 1:1 ordering is not guaranteed in general.
- Tiny-ImageNet is not downloaded automatically. Download/extract it separately and pass `--data-root` accordingly.

### Tiny-ImageNet (64x64) example
Assuming you have `datasets/tiny-imagenet-200/` locally:
```
python3 main.py \
  --dataset tiny-imagenet --data-root datasets --dataset-split train --image-size 64 \
  --arch resnet18 --pretrained \
  --reconstruct-mode aggregated \
  --num-clients 1 --samples-per-client 1 \
  --iterations 1500 --restarts 4 --init-scale 0.01 --tv-weight 1e-2
```

## Diffusion-based reconstruction (evaluation)
`evaluations/diffusion_reconstruction/run.py` reconstructs client data while denoising each iterate with a pretrained DDPM prior (`google/ddpm-cifar10-32` from Hugging Face diffusers). Install the extra dependency with `pip install diffusers`.

Example:
```
python3 evaluations/diffusion_reconstruction/run.py --num-clients 1 --samples-per-client 1 --diffusion-steps 50 --log-every 20 --save-dir outputs/diffusion_dlg
```
Outputs (images, loss traces, metrics) are stored under a timestamped subfolder of `--save-dir`.

## LaTeX metrics table
Create the paper-ready table from a `main.py` run directory (and optionally a diffusion run directory):
```
python3 scripts/make_metrics_table.py \
  --dlg-run outputs/reconstructions/run_YYYYMMDD_HHMMSS \
  --diffusion-run outputs/diffusion_dlg/run_YYYYMMDD_HHMMSS
```

## License
MIT; see [LICENSE](LICENSE).
