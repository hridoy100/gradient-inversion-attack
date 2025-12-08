import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleDiffusionWrapper(nn.Module):
    """
    Minimal stub standing in for a pretrained diffusion decoder.

    It maps a latent tensor `z` to an image-like tensor using a small
    convolutional decoder and a sigmoid to keep outputs in [0, 1].
    Replace this with a real diffusion model for stronger priors.
    """

    def __init__(self, output_shape):
        super().__init__()
        self.output_shape = output_shape
        channels = output_shape[1] if len(output_shape) > 1 else 3
        # Lightweight decoder: a couple of conv layers.
        self.decoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, channels, kernel_size=3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # If z is not in NCHW, reshape if possible; otherwise assume correct.
        if z.dim() == 2 and len(self.output_shape) == 4:
            # reshape (N, latent) to (N, C, H, W) using a simple projection
            n, latent_dim = z.shape
            c, h, w = self.output_shape[1:]
            if latent_dim != c * h * w:
                # fallback: expand to match expected shape
                z = z.view(n, c, 1, 1).expand(n, c, h, w)
            else:
                z = z.view(n, c, h, w)
        x = self.decoder(z)
        return torch.sigmoid(x)
