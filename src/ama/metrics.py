"""Classification metrics for the downstream task.

Binary and multiclass tasks need different AUC calls, and some tasks want a
top-k accuracy alongside top-1.  Both choices follow from the task itself, so
they live in one small object rather than in branches at each call site.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor


@dataclass
class ClassificationMetrics:
    """AUC and accuracy conventions for a classification task."""

    n_classes: int
    #: Also report top-k accuracy; ``None`` to report top-1 only.
    top_k: int | None = None

    @property
    def is_binary(self) -> bool:
        return self.n_classes == 2

    def auc(self, probs: Tensor, labels: Tensor) -> float:
        """Area under the ROC curve, one-vs-one averaged when multiclass."""
        probs = probs.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        if self.is_binary:
            return float(roc_auc_score(labels, probs[:, 1]))
        return float(roc_auc_score(labels, probs, multi_class="ovo"))

    def top_k_correct(self, logits: Tensor, labels: Tensor) -> int:
        """Number of examples whose true label is in the top ``k`` predictions."""
        if self.top_k is None:
            raise ValueError("top_k was not configured for this task")
        topk = logits.topk(self.top_k, dim=-1).indices
        return int((topk == labels.unsqueeze(-1)).any(dim=-1).sum().item())

    def predicted_probability(self, probs: Tensor, preds: Tensor) -> Tensor:
        """Probability the model assigned to the class it predicted."""
        return probs.gather(1, preds.unsqueeze(1)).squeeze(1)


def build_metrics(spec: dict) -> ClassificationMetrics:
    """Construct metrics from a config mapping."""
    return ClassificationMetrics(
        n_classes=int(spec["n_classes"]), top_k=spec.get("top_k")
    )
