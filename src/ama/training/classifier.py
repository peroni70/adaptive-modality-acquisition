"""Training the masked classifier.

One classifier serves every acquisition state, so it is trained on examples
masked to random modality subsets rather than on complete inputs.
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
from ..progress import progress

from ..metrics import ClassificationMetrics


def train_classifier(
    model: nn.Module,
    train_loader,
    val_loader,
    n_epochs: int = 15,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: str | None = None,
) -> tuple[nn.Module, float]:
    """Train on randomly masked examples, returning the best model by val accuracy."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_acc = 0.0
    best_model = deepcopy(model)
    for epoch in range(n_epochs):
        model.train()
        total_loss, count = 0.0, 0
        pbar = progress(train_loader, leave=False)
        for x, y, observed in pbar:
            x, y, observed = x.to(device), y.to(device), observed.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x, observed), y.long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            count += len(y)
            pbar.set_description(f"train loss {total_loss / count:.4f}")

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y, observed in val_loader:
                x, y, observed = x.to(device), y.to(device), observed.to(device)
                preds = model(x, observed).argmax(1)
                correct += (preds == y.long()).sum().item()
                total += y.size(0)
        acc = correct / total
        print(
            f"epoch {epoch + 1}/{n_epochs} | "
            f"train loss {total_loss / count:.4f} | val acc {acc:.4f}"
        )
        if acc > best_acc:
            best_acc = acc
            best_model = deepcopy(model)
    return best_model, best_acc


@torch.no_grad()
def eval_classifier(
    model: nn.Module, loader, metrics: ClassificationMetrics, device: str | None = None
) -> dict:
    """Accuracy and AUC on a loader of masked examples."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    correct, total = 0, 0
    probs, labels = [], []
    for x, y, observed in loader:
        x, y, observed = x.to(device), y.to(device), observed.to(device)
        logits = model(x, observed)
        correct += (logits.argmax(1) == y.long()).sum().item()
        total += y.size(0)
        probs.append(torch.softmax(logits, 1))
        labels.append(y)
    probs = torch.cat(probs)
    labels = torch.cat(labels)
    result = {"accuracy": correct / total, "auc": metrics.auc(probs, labels)}
    if metrics.top_k is not None:
        result[f"top_{metrics.top_k}_accuracy"] = (
            metrics.top_k_correct(probs, labels) / total
        )
    return result
