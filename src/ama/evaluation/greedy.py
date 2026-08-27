"""Static greedy modality ordering.

Ranks modalities by how much each one adds to validation performance when
appended to the set chosen so far. The result is a single order used by every
example, which is exactly what an adaptive policy is meant to improve on, so
it serves as the reference baseline and as the backbone of the
``adaptive_greedy`` policy.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..data.masked import MaskedDataset
from ..data.sources import ExampleSource
from ..masking import Masker
from ..metrics import ClassificationMetrics
from ..value_fns import ValueFunction


@torch.no_grad()
def score_subset(
    model: nn.Module,
    source: ExampleSource,
    masker: Masker,
    context_idx: int,
    include_modes,
    value_fn: ValueFunction | None = None,
    batch_size: int = 32,
    device: str | None = None,
) -> float:
    """Validation score with a fixed set of modalities observed.

    Accuracy normally; mean cross-entropy for ``info_gain``, whose ordering is
    defined by loss reduction rather than by accuracy.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    use_loss = value_fn is not None and value_fn.name == "info_gain"
    dataset = MaskedDataset(
        source, masker, context_idx, deterministic=True, include_modes=include_modes
    )
    loader = DataLoader(dataset, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    correct, total, total_loss = 0, 0, 0.0
    for x, y, observed in loader:
        x, y, observed = x.to(device), y.to(device), observed.to(device)
        logits = model(x, observed)
        correct += (logits.argmax(1) == y.long()).sum().item()
        total += y.size(0)
        total_loss += criterion(logits, y.long()).item()
    return total_loss / total if use_loss else correct / total


def greedy_modality_order(
    model: nn.Module,
    source: ExampleSource,
    masker: Masker,
    context_idx: int = 0,
    value_fn: ValueFunction | None = None,
    max_modes: int | None = None,
    device: str | None = None,
    verbose: bool = True,
) -> list[int]:
    """Order modalities by greedy validation improvement."""
    n_modes = masker.n_modes
    remaining = {i for i in range(n_modes) if i != context_idx}
    observed = [False] * n_modes
    observed[context_idx] = True
    lower_is_better = value_fn is not None and value_fn.name == "info_gain"

    limit = min(n_modes - 1, max_modes if max_modes is not None else n_modes - 1)
    order: list[int] = []
    for _ in range(limit):
        best_score = np.inf if lower_is_better else 0.0
        best_idx = None
        for candidate in sorted(remaining):
            observed[candidate] = True
            score = score_subset(
                model, source, masker, context_idx, observed, value_fn, device=device
            )
            observed[candidate] = False
            better = score < best_score if lower_is_better else score > best_score
            if better:
                best_score, best_idx = score, candidate
        if best_idx is None:
            break
        observed[best_idx] = True
        remaining.remove(best_idx)
        order.append(best_idx)
        if verbose:
            print(f"  +modality {best_idx} -> {best_score:.4f}")
    return order


def greedy_prefix_gains(
    model: nn.Module,
    source: ExampleSource,
    masker: Masker,
    order: list[int],
    context_idx: int = 0,
    device: str | None = None,
) -> tuple[float, list[float]]:
    """Validation accuracy gain of each prefix of ``order`` over context alone.

    These gains are what the ``adaptive_greedy`` policy weighs against the
    realized costs when it picks a stopping point per example.
    """
    n_modes = masker.n_modes
    base_modes = [False] * n_modes
    base_modes[context_idx] = True
    baseline = score_subset(model, source, masker, context_idx, base_modes, device=device)

    gains = []
    for prefix_len in range(1, len(order) + 1):
        modes = base_modes.copy()
        for mode in order[:prefix_len]:
            modes[mode] = True
        gains.append(
            score_subset(model, source, masker, context_idx, modes, device=device)
            - baseline
        )
    return baseline, gains


def evaluate_by_prefix(
    model: nn.Module,
    source: ExampleSource,
    masker: Masker,
    order: list[int],
    metrics: ClassificationMetrics,
    context_idx: int = 0,
    batch_size: int = 32,
    device: str | None = None,
) -> list[dict]:
    """Accuracy and AUC after acquiring each prefix of the greedy order."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    n_modes = masker.n_modes
    observed = [False] * n_modes
    observed[context_idx] = True

    rows = []
    for prefix_len, mode in enumerate(order, start=1):
        observed[mode] = True
        dataset = MaskedDataset(
            source, masker, context_idx, deterministic=True, include_modes=observed
        )
        loader = DataLoader(dataset, batch_size=batch_size)
        correct, total = 0, 0
        probs, labels = [], []
        with torch.no_grad():
            for x, y, obs in loader:
                x, y, obs = x.to(device), y.to(device), obs.to(device)
                logits = model(x, obs)
                correct += (logits.argmax(1) == y.long()).sum().item()
                total += y.size(0)
                probs.append(torch.softmax(logits, 1))
                labels.append(y)
        probs, labels = torch.cat(probs), torch.cat(labels)
        rows.append(
            {
                "n_modalities": prefix_len,
                "modality": mode,
                "accuracy": correct / total,
                "auc": metrics.auc(probs, labels),
            }
        )
    return rows
