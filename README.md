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
- `--data-root`: dataset location (defaults to `~/.torch`; downloads CIFAR100 if missing).

## License
MIT; see [LICENSE](LICENSE).
