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

# Use specific CIFAR100 indices for reproducibility
python3 main.py --num-clients 2 --samples-per-client 2 --client-indices 1,5,42,100

# Run both per-client and aggregated reconstructions
python3 main.py --reconstruct-mode both

# Aggregated reconstruction with TV prior, normalized gradients, and smaller init noise
python3 main.py --reconstruct-mode aggregated --tv-weight 1e-3 --normalize-gradients --init-scale 0.05 --iterations 400 --restarts 5

# Apply aggregated gradient step to the server model before inversion
python3 main.py --reconstruct-mode aggregated --apply-agg-step --agg-lr 0.05
```

Key options:
- `--num-clients`: number of simulated clients.
- `--samples-per-client`: number of images per client.
- `--client-indices`: comma-separated CIFAR100 indices to force the client datasets.
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
- `--data-root`: dataset location (defaults to `~/.torch`; downloads CIFAR100 if missing).

## Diffusion-Latent DLG (federated)
We provide a plug-and-play Diffusion-DLG attack that optimizes a latent `z` under a diffusion-style prior instead of raw pixels, plus semantic leakage metrics.

```
# Train a global model checkpoint (optional but recommended)
python3 experiments/train_global.py --epochs 1 --batch-size 128 --lr 0.01 --output checkpoints/global.pth

# Minimal smoke test (toy settings)
python3 experiments/run_federated_diffusion_dlg.py --steps 50 --num-clients 1 --samples-per-client 1 --log-every 5

# Heavier run with stronger prior weight and custom output dir
python3 experiments/run_federated_diffusion_dlg.py --steps 200 --lambda-prior 1e-3 --client-id 0 --round-id 0 --output-dir outputs/diffusion_dlg

# Use the trained checkpoint and fixed labels for better fidelity
python3 experiments/run_federated_diffusion_dlg.py --model-checkpoint checkpoints/global.pth --fix-labels --steps 500 --lr 1e-2 --restarts 3
```

Artifacts:
- Reconstructed image: `outputs/diffusion_dlg/client_{k}_round_{r}.png`
- Loss history: `outputs/diffusion_dlg/loss_client_{k}_round_{r}.json`
- Semantic leakage metrics (MSE/PSNR/feature similarity/class accuracy): `outputs/diffusion_dlg/metrics_client_{k}_round_{r}.json`

## License
MIT; see [LICENSE](LICENSE).
