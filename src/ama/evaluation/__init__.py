"""Evaluation: greedy ordering baselines and policy scoring."""

from .greedy import (
    evaluate_by_prefix,
    greedy_modality_order,
    greedy_prefix_gains,
    score_subset,
)
from .policy import evaluate_policy, evaluate_static_baselines

__all__ = [
    "evaluate_by_prefix",
    "evaluate_policy",
    "evaluate_static_baselines",
    "greedy_modality_order",
    "greedy_prefix_gains",
    "score_subset",
]
