"""Temperature scaling for value models.

The subset optimizers compare value scores against acquisition costs, so the
scores need to mean what they claim. A single temperature fitted on held-out
data corrects systematic over- or under-confidence without touching the
model's ranking.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from ..progress import progress


class TemperatureScaled(nn.Module):
    """Wraps a value model and divides its logits by a learned temperature."""

    def __init__(self, value_model: nn.Module, init: float = 1.0):
        super().__init__()
        self.value_model = value_model
        for p in self.value_model.parameters():
            p.requires_grad = False
        self.temperature = nn.Parameter(torch.tensor([float(init)]))

    @property
    def value_fn_name(self) -> str:
        return self.value_model.value_fn_name

    @property
    def mode_embedding(self) -> nn.Embedding:
        return self.value_model.mode_embedding

    def forward(self, x: Tensor, acquired: Tensor, proposed: Tensor) -> Tensor:
        return self.value_model(x, acquired, proposed) / self.temperature


def fit_temperature(
    value_model: nn.Module,
    loader,
    value_fn,
    n_epochs: int = 30,
    lr: float = 1e-3,
    device: str | None = None,
) -> TemperatureScaled:
    """Fit a temperature on ``loader``, returning the wrapped model."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = (
        value_model
        if isinstance(value_model, TemperatureScaled)
        else TemperatureScaled(value_model)
    ).to(device)
    model.value_model.eval()
    optimizer = torch.optim.Adam([model.temperature], lr=lr)
    criterion = value_fn.loss()
    for _ in range(n_epochs):
        total, count = 0.0, 0
        pbar = progress(loader, leave=False)
        for x, target, acquired, proposed in pbar:
            x = x.to(device)
            target = target.to(device)
            acquired = acquired.to(device)
            proposed = proposed.to(device)
            optimizer.zero_grad()
            logits = model(x, acquired, proposed)
            loss = value_fn.compute_loss(criterion, logits, target)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(target)
            count += len(target)
            pbar.set_description(
                f"calibration loss {total / count:.4f} | "
                f"T {model.temperature.item():.3f}"
            )
    return model
