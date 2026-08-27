"""Scoring an acquisition policy.

A policy is judged on what it buys and what it spends. The headline number is
reward: accuracy gained over observing the context alone, minus the cost of
the modalities acquired to gain it.

    reward = accuracy - baseline_accuracy - cost

Per-example rows carry the same quantities so that two methods can be compared
on identical examples under identical sampled costs.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ..progress import progress

from ..costs import CostModel
from ..data.policy import PolicyDataset
from ..data.sources import ExampleSource
from ..masking import Masker
from ..metrics import ClassificationMetrics
from ..results import SampleResult


@torch.no_grad()
def evaluate_policy(
    model: nn.Module,
    loader,
    context_idx: int,
    n_modes: int,
    metrics: ClassificationMetrics,
    baseline_accuracy: float = 0.0,
    baseline_correct: list[float] | None = None,
    method_name: str = "policy",
    collect_samples: bool = True,
    device: str | None = None,
) -> tuple[dict, list[SampleResult]]:
    """Evaluate the observations a policy produced.

    Args:
        loader: yields ``(x, acquired, y, costs[, n_stages])`` from a PolicyDataset.
        baseline_accuracy: context-only accuracy, subtracted to form reward.
        baseline_correct: per-example context-only correctness, so per-example
            reward is paired rather than measured against a population mean.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    criterion = nn.CrossEntropyLoss(reduction="none")

    correct, total, total_loss, total_cost = 0.0, 0, 0.0, 0.0
    top_k_correct = 0
    acquired_counts = torch.zeros(n_modes)
    probs_all, labels_all, samples = [], [], []
    offset = 0
    start = time.time()

    pbar = progress(loader, leave=False)
    for batch in pbar:
        if len(batch) == 5:
            x, acquired, y, costs, n_stages = batch
        else:
            x, acquired, y, costs = batch
            n_stages = None
        x = x.to(device)
        labels = y.long().to(device).view(-1)
        acquired = acquired.to(device)
        batch_size = len(labels)

        logits = model(x, acquired)
        probs = torch.softmax(logits, 1)
        losses = criterion(logits, labels)
        preds = logits.argmax(dim=-1)
        correct_mask = (preds == labels).float()

        # The context modality is free, so it never enters the bill.
        billable = acquired.clone()
        billable[:, context_idx] = 0.0
        batch_costs = torch.sum(costs.to(device) * billable, 1)

        correct += correct_mask.sum().item()
        total += batch_size
        total_loss += losses.sum().item()
        total_cost += batch_costs.sum().item()
        acquired_counts += billable.sum(0).cpu()
        probs_all.append(probs)
        labels_all.append(labels)
        if metrics.top_k is not None:
            top_k_correct += metrics.top_k_correct(logits, labels)

        accuracy, avg_cost = correct / total, total_cost / total
        pbar.set_description(
            f"acc {accuracy:.4f} | cost {avg_cost:.4f} | "
            f"reward {accuracy - baseline_accuracy - avg_cost:.4f}"
        )

        if collect_samples:
            samples.extend(
                _sample_rows(
                    method_name, offset, batch_size, n_modes, labels, preds,
                    metrics.predicted_probability(probs, preds), losses,
                    correct_mask, batch_costs, billable, n_stages,
                    baseline_correct, baseline_accuracy,
                )
            )
        offset += batch_size

    accuracy = correct / total
    avg_cost = total_cost / total
    probs_all = torch.cat(probs_all)
    labels_all = torch.cat(labels_all)
    summary = {
        "loss": total_loss / total,
        "cost": avg_cost,
        "accuracy": accuracy,
        "skipped": 1.0 - (acquired_counts.sum().item() / ((n_modes - 1) * total)),
        "reward": accuracy - baseline_accuracy - avg_cost,
        "efficiency": float(np.divide(accuracy - baseline_accuracy, avg_cost))
        if avg_cost
        else 0.0,
        "auc": metrics.auc(probs_all, labels_all),
        "seconds": time.time() - start,
    }
    if metrics.top_k is not None:
        summary[f"top_{metrics.top_k}_accuracy"] = top_k_correct / total
    return summary, samples


def _sample_rows(
    method_name, offset, batch_size, n_modes, labels, preds, pred_probs, losses,
    correct_mask, batch_costs, billable, n_stages, baseline_correct, baseline_accuracy,
):
    """Build one SampleResult per example in the batch."""
    labels = labels.cpu()
    preds = preds.cpu()
    pred_probs = pred_probs.cpu()
    losses = losses.cpu()
    correct_mask = correct_mask.cpu()
    batch_costs = batch_costs.cpu()
    billable = billable.cpu()
    counts = billable.sum(1)

    if baseline_correct is None:
        base = torch.full((batch_size,), float(baseline_accuracy))
    else:
        base = torch.tensor(
            baseline_correct[offset : offset + batch_size], dtype=torch.float32
        )
    if n_stages is None:
        stages = [None] * batch_size
    else:
        stages = [None if s < 0 else int(s) for s in n_stages.view(-1).tolist()]

    rows = []
    for i in range(batch_size):
        modes = torch.nonzero(billable[i], as_tuple=False).view(-1).tolist()
        rows.append(
            SampleResult(
                name=method_name,
                sample_idx=offset + i,
                label=int(labels[i]),
                pred=int(preds[i]),
                pred_proba=float(pred_probs[i]),
                loss=float(losses[i]),
                correct=float(correct_mask[i]),
                baseline_correct=float(base[i]),
                cost=float(batch_costs[i]),
                reward=float(correct_mask[i] - base[i] - batch_costs[i]),
                n_acquired=int(counts[i]),
                skipped=float(1.0 - counts[i] / (n_modes - 1)),
                selected_modes=",".join(str(m) for m in modes),
                num_stages=stages[i],
            )
        )
    return rows


def evaluate_static_baselines(
    model: nn.Module,
    source: ExampleSource,
    masker: Masker,
    context_idx: int,
    order: list[int],
    metrics: ClassificationMetrics,
    cost_model: CostModel | None = None,
    prefix_lengths=None,
    batch_size: int = 64,
    device: str | None = None,
) -> tuple[list[tuple[str, dict]], float, list[SampleResult], list[float]]:
    """Evaluate fixed greedy prefixes, the non-adaptive comparison.

    The zero-length prefix is the context-only baseline; its accuracy and its
    per-example correctness anchor the reward of every other method, so it is
    always evaluated first.
    """
    n_modes = masker.n_modes
    if prefix_lengths is None:
        prefix_lengths = range(n_modes)
    prefix_lengths = sorted(prefix_lengths)
    if prefix_lengths and prefix_lengths[0] != 0:
        raise ValueError("prefix_lengths must include 0, the context-only baseline")

    results, samples = [], []
    baseline_accuracy = 0.0
    baseline_correct = None

    for prefix_len in prefix_lengths:
        include_modes = [False] * n_modes
        for mode in order[:prefix_len]:
            include_modes[mode] = True
        dataset = PolicyDataset(
            source, masker, context_idx, method="preset",
            include_modes=include_modes, deterministic=True,
            cost_model=cost_model, device=device,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        name = f"static_greedy_{prefix_len}"
        summary, rows = evaluate_policy(
            model, loader, context_idx, n_modes, metrics,
            baseline_accuracy=baseline_accuracy,
            baseline_correct=baseline_correct,
            method_name=name, device=device,
        )
        if prefix_len == 0:
            baseline_accuracy = summary["accuracy"]
            baseline_correct = [row.correct for row in rows]
            for row in rows:
                row.baseline_correct = row.correct
                row.reward = 0.0
            summary["reward"] = 0.0
            summary["efficiency"] = 0.0
        print(
            f"{name}: acc {summary['accuracy']:.4f} | cost {summary['cost']:.4f} | "
            f"reward {summary['reward']:.4f}"
        )
        results.append((name, summary))
        samples.extend(rows)
    return results, baseline_accuracy, samples, baseline_correct
