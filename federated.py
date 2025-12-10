import torch
import torch.nn.functional as F
from typing import List, Tuple

from tqdm.auto import tqdm

from utils import label_to_onehot, cross_entropy_for_onehot


class FederatedClient:
    """Minimal federated client that holds local data and computes gradients."""

    def __init__(self, client_id: int, data: torch.Tensor, labels: torch.Tensor, device: str, num_classes: int):
        self.client_id = client_id
        self.data = data.to(device)
        self.labels = labels.to(device)
        self.device = device
        self.num_classes = num_classes

    def compute_gradients(self, model: torch.nn.Module) -> Tuple[List[torch.Tensor], float]:
        """Run one forward/backward pass on local data and return detached gradients."""
        model.zero_grad()
        pred = model(self.data)
        onehot = label_to_onehot(self.labels, num_classes=self.num_classes)
        loss = cross_entropy_for_onehot(pred, onehot)
        grads = torch.autograd.grad(loss, model.parameters())
        return [g.detach().clone() for g in grads], loss.item()


class FederatedServer:
    """Server keeps the reference model, aggregates gradients, and runs inversion."""

    def __init__(self, model: torch.nn.Module, device: str, num_classes: int):
        self.model = model
        self.device = device
        self.num_classes = num_classes

    def aggregate_gradients(self, client_gradients: List[List[torch.Tensor]], normalize: bool = False) -> List[torch.Tensor]:
        """Average gradients across clients, keeping shape/ordering aligned with model params.

        If normalize is True, each client's gradient tensor is L2-normalized before averaging to
        reduce domination from any single client with a larger loss/scale.
        """
        aggregated = []
        for grads_for_param in zip(*client_gradients):
            grads_on_device = []
            for g in grads_for_param:
                g = g.to(self.device)
                if normalize:
                    norm = torch.norm(g)
                    if norm > 0:
                        g = g / norm
                grads_on_device.append(g)
            aggregated.append(torch.stack(grads_on_device, dim=0).mean(dim=0))
        return aggregated

    def reconstruct_data(
        self,
        target_gradients: List[torch.Tensor],
        data_shape: torch.Size,
        iterations: int = 300,
        log_every: int = 20,
        restarts: int = 1,
        init_seed: int = None,
        tv_weight: float = 0.0,
        init_scale: float = 1.0,
        progress: bool = False,
    ):
        """Reconstruct training data that matches the provided gradients."""
        return gradient_inversion(
            model=self.model,
            target_gradients=target_gradients,
            data_shape=data_shape,
            num_classes=self.num_classes,
            device=self.device,
            iterations=iterations,
            log_every=log_every,
            restarts=restarts,
            init_seed=init_seed,
            tv_weight=tv_weight,
            init_scale=init_scale,
            progress=progress,
        )

    def apply_gradient_step(self, gradients: List[torch.Tensor], lr: float = 0.1):
        """Apply a single gradient descent step on the server model using aggregated gradients."""
        with torch.no_grad():
            for p, g in zip(self.model.parameters(), gradients):
                p.add_( -lr * g.to(self.device))


def gradient_inversion(
    model: torch.nn.Module,
    target_gradients: List[torch.Tensor],
    data_shape: torch.Size,
    num_classes: int,
    device: str,
    iterations: int = 300,
    log_every: int = 20,
    restarts: int = 1,
    init_seed: int = None,
    tv_weight: float = 0.0,
    init_scale: float = 1.0,
    progress: bool = False,
):
    """Run gradient matching to recover training data from shared gradients."""
    best = {"loss": float("inf"), "data": None, "labels": None, "history": None}

    for attempt in range(restarts):
        if init_seed is not None:
            torch.manual_seed(init_seed + attempt)

        dummy_data = (torch.randn(data_shape, device=device) * init_scale).requires_grad_(True)
        dummy_label = torch.randn((data_shape[0], num_classes), device=device, requires_grad=True)
        optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

        history = []

        def gradient_match_loss():
            """Compute squared distance between dummy and target gradients."""
            pred = model(dummy_data)
            dummy_onehot = F.softmax(dummy_label, dim=-1)
            dummy_loss = cross_entropy_for_onehot(pred, dummy_onehot)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)
            grad_diff = sum(((gx - gy) ** 2).sum() for gx, gy in zip(dummy_dy_dx, target_gradients))

            if tv_weight > 0:
                tv_h = torch.abs(dummy_data[:, :, :, :-1] - dummy_data[:, :, :, 1:]).mean()
                tv_v = torch.abs(dummy_data[:, :, :-1, :] - dummy_data[:, :, 1:, :]).mean()
                tv = tv_h + tv_v
                grad_diff = grad_diff + tv_weight * tv
            return grad_diff

        last_loss_val = None
        iterator = range(iterations)
        if progress:
            iterator = tqdm(iterator, desc=f"Inversion {attempt+1}/{restarts}", leave=False)

        for iters in iterator:
            def closure():
                optimizer.zero_grad()
                loss = gradient_match_loss()
                loss.backward()
                return loss

            loss_val = optimizer.step(closure)
            last_loss_val = loss_val

            if iters % log_every == 0 or iters == iterations - 1:
                history.append(
                    {
                        "iteration": iters,
                        "loss": loss_val.item(),
                        "data": dummy_data.detach().clone(),
                    }
                )

        final_loss = last_loss_val.item() if last_loss_val is not None else float("inf")
        if final_loss < best["loss"]:
            with torch.no_grad():
                best["loss"] = final_loss
                best["data"] = dummy_data.detach().clone()
                best["labels"] = F.softmax(dummy_label, dim=-1).detach().clone()
                best["history"] = history

    return best["data"], best["labels"], best["history"]
