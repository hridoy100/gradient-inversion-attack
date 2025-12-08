import torch
import torch.nn.functional as F
from typing import List, Dict, Any

from utils import cross_entropy_for_onehot
from attacks.diffusion_prior import SimpleDiffusionWrapper


class DiffusionDLGAttack:
    """
    Diffusion-Latent Deep Leakage from Gradients.

    We optimize a latent variable z passed through a diffusion-like decoder to
    produce reconstructed inputs. Optimization minimizes gradient-matching loss
    against observed gradients plus a prior term on z.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_gradients: List[torch.Tensor],
        data_shape: torch.Size,
        num_classes: int,
        device: str,
        diffusion_model: torch.nn.Module = None,
        lambda_prior: float = 1e-3,
        lr: float = 0.1,
        steps: int = 300,
        log_every: int = 10,
        restarts: int = 1,
        optimize_labels: bool = True,
        fixed_labels: torch.Tensor = None,
        init_scale: float = 1.0,
    ):
        self.model = model
        self.target_gradients = [g.detach().to(device) for g in target_gradients]
        self.data_shape = data_shape
        self.num_classes = num_classes
        self.device = device
        self.lambda_prior = lambda_prior
        self.lr = lr
        self.steps = steps
        self.log_every = log_every
        self.diffusion_model = diffusion_model or SimpleDiffusionWrapper(data_shape).to(device)
        self.restarts = restarts
        self.optimize_labels = optimize_labels
        self.fixed_labels = fixed_labels.detach() if fixed_labels is not None else None
        self.init_scale = init_scale

    def run(self) -> Dict[str, Any]:
        batch_size = self.data_shape[0]
        best = {"loss": float("inf"), "data": None, "labels": None, "history": None}

        for attempt in range(self.restarts):
            # Latent same shape as data for the stub decoder; replace with true latent as needed.
            z = (torch.randn(self.data_shape, device=self.device) * self.init_scale).requires_grad_(True)
            params = [z]
            if self.optimize_labels:
                dummy_label = torch.randn((batch_size, self.num_classes), device=self.device, requires_grad=True)
                params.append(dummy_label)
            else:
                # Fixed labels provided as one-hot or class ids
                if self.fixed_labels is None:
                    raise ValueError("fixed_labels must be provided when optimize_labels=False")
                if self.fixed_labels.dim() == 1:
                    dummy_onehot_fixed = F.one_hot(self.fixed_labels.long(), num_classes=self.num_classes).float().to(
                        self.device
                    )
                else:
                    dummy_onehot_fixed = self.fixed_labels.to(self.device)
                dummy_label = dummy_onehot_fixed  # not a parameter

            optimizer = torch.optim.Adam(params, lr=self.lr)
            history = []
            final_loss = None

            for step in range(self.steps):
                optimizer.zero_grad()
                x_recon = self.diffusion_model(z)
                preds = self.model(x_recon)
                if self.optimize_labels:
                    dummy_onehot = F.softmax(dummy_label, dim=-1)
                else:
                    dummy_onehot = dummy_label
                loss_pred = cross_entropy_for_onehot(preds, dummy_onehot)
                dummy_grads = torch.autograd.grad(loss_pred, self.model.parameters(), create_graph=True)
                grad_loss = sum(((gx - gy) ** 2).sum() for gx, gy in zip(dummy_grads, self.target_gradients))

                prior_loss = (z ** 2).mean()
                total_loss = grad_loss + self.lambda_prior * prior_loss
                total_loss.backward()
                optimizer.step()
                final_loss = total_loss

                if step % self.log_every == 0 or step == self.steps - 1:
                    history.append(
                        {
                            "step": step,
                            "grad_loss": grad_loss.item(),
                            "prior_loss": prior_loss.item(),
                            "total_loss": total_loss.item(),
                        }
                    )

            if final_loss is not None and final_loss.item() < best["loss"]:
                best["loss"] = final_loss.item()
                best["data"] = x_recon.detach().clone()
                best["labels"] = dummy_onehot.detach().clone()
                best["history"] = history

        return {
            "reconstructed_data": best["data"],
            "reconstructed_labels": best["labels"],
            "history": best["history"] if best["history"] is not None else [],
        }
