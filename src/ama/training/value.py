"""Training and evaluating the value model."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from ..progress import progress

from ..masking import Masker
from ..value_fns import ValueFunction


def submodular_hinge_loss(
    value_model: nn.Module,
    x: torch.Tensor,
    acquired: torch.Tensor,
    proposed: torch.Tensor,
    masker: Masker,
    context_idx: int,
    value_fn: ValueFunction,
    lam: float = 1e-3,
) -> torch.Tensor:
    """Penalize proposals worth more from a richer state than from the poorest one.

    Diminishing returns says the same proposal cannot be more valuable once
    more has already been observed. This is a soft one-sided version of that
    constraint against the context-only state, which nudges the value model
    toward the submodular behaviour the subset optimizers assume.
    """
    context_set = torch.zeros_like(acquired)
    context_set[:, context_idx] = 1.0
    context_x = masker.mask(x, masker.indicator([context_idx]))
    value = value_fn.score(value_model(x, acquired, proposed))
    value_base = value_fn.score(value_model(context_x, context_set, proposed))
    return lam * torch.relu(value - value_base).mean()


def train_value_model(
    value_model: nn.Module,
    train_loader,
    val_loader,
    value_fn: ValueFunction,
    masker: Masker,
    context_idx: int,
    n_epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    use_hinge_loss: bool = False,
    hinge_lam: float = 1e-3,
    checkpoint_path: str | Path | None = None,
    device: str | None = None,
) -> tuple[nn.Module, float, list[float]]:
    """Train the value model, keeping the best epoch by the value function's metric."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    value_model = value_model.to(device)
    criterion = value_fn.loss()
    optimizer = torch.optim.Adam(
        value_model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, n_epochs)

    best_score = np.inf if not value_fn.higher_is_better else -np.inf
    best_model = deepcopy(value_model)
    val_scores: list[float] = []

    for epoch in range(n_epochs):
        value_model.train()
        total_loss, total_target, count = 0.0, 0.0, 0
        pbar = progress(train_loader, leave=False)
        for x, target, acquired, proposed in pbar:
            x = x.to(device)
            target = target.to(device)
            acquired = acquired.to(device)
            proposed = proposed.to(device)
            optimizer.zero_grad()
            logits = value_model(x, acquired, proposed)
            loss = value_fn.compute_loss(criterion, logits, target)
            if use_hinge_loss:
                loss = loss + submodular_hinge_loss(
                    value_model, x, acquired, proposed, masker,
                    context_idx, value_fn, lam=hinge_lam,
                )
            loss.backward()
            optimizer.step()
            batch = len(target)
            total_loss += loss.item() * batch
            total_target += target.sum().item()
            count += batch
            pbar.set_description(
                f"train loss {total_loss / count:.4f} | "
                f"mean target {total_target / count:.4f}"
            )

        score, val_loss = _validate(value_model, val_loader, value_fn, criterion, device)
        val_scores.append(score)
        improved = (
            score > best_score if value_fn.higher_is_better else score < best_score
        )
        print(
            f"epoch {epoch + 1}/{n_epochs} | train loss {total_loss / count:.4f} | "
            f"val loss {val_loss:.4f} | val {value_fn.name} score {score:.4f}"
            + ("  *" if improved else "")
        )
        if improved:
            best_score = score
            best_model = deepcopy(value_model)
            if checkpoint_path is not None:
                path = Path(checkpoint_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_model.state_dict(), path)
        scheduler.step()
    return best_model, best_score, val_scores


@torch.no_grad()
def _validate(value_model, loader, value_fn, criterion, device):
    value_model.eval()
    total_loss, count = 0.0, 0
    all_logits, all_targets = [], []
    for x, target, acquired, proposed in progress(loader, leave=False):
        x = x.to(device)
        target = target.to(device)
        acquired = acquired.to(device)
        proposed = proposed.to(device)
        logits = value_model(x, acquired, proposed)
        batch = len(target)
        total_loss += value_fn.compute_loss(criterion, logits, target).item() * batch
        count += batch
        all_logits.append(logits)
        all_targets.append(target)
    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    return value_fn.validate(logits, targets), total_loss / count


@torch.no_grad()
def eval_value_model(
    value_model: nn.Module,
    loader,
    value_fn: ValueFunction,
    device: str | None = None,
) -> dict:
    """Score the value model on held-out data."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    value_model = value_model.to(device).eval()
    criterion = value_fn.loss()
    score, loss = _validate(value_model, loader, value_fn, criterion, device)
    return {"score": score, "loss": loss}


def mean_target(loader) -> float:
    """Mean training target, the constant-prediction baseline for info gain."""
    targets = torch.cat([target for _, target, _, _ in loader])
    return float(targets.float().mean().item())


def constant_predictor_mse(loader, constant: float) -> float:
    """MSE of always predicting ``constant``, the do-nothing regression baseline."""
    targets = torch.cat([target for _, target, _, _ in loader]).float()
    return float(torch.mean((targets - constant) ** 2).item())


def regression_skill(model_mse: float, baseline_mse: float) -> float:
    """Fraction of the constant predictor's squared error the model removes.

    ``1 - model_mse / baseline_mse``: a relative reduction, not a difference of
    MSEs. Zero means the model is no better than predicting the training mean,
    one means it predicts perfectly, and negative means it is worse than the
    constant. This is the number that says whether a regression-valued value
    function learned anything at all.

    The baseline predicts the *training* mean rather than the mean of the data
    being scored, so no information leaks from the evaluation set. That makes
    this a skill score rather than strictly R-squared, though the two coincide
    when the two means are close.
    """
    if baseline_mse == 0:
        return 0.0
    return 1.0 - model_mse / baseline_mse
