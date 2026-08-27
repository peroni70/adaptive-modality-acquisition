"""Subset optimizers.

Each stage of acquisition asks the same question: given what has been observed
and what each modality costs, which subset maximizes value minus cost?

    max_{Q subseteq unacquired}  v(Q | P) - sum_{i in Q} c_i

The value model is not guaranteed submodular and the objective is not
monotone, so these are unconstrained-submodular-maximization heuristics rather
than exact solvers. They differ in how many value-model evaluations they spend:
``single_item_greedy`` uses one pass, ``enumerate`` is exhaustive and only
practical for small modality counts.

Every optimizer takes a ``score`` callable mapping a proposed subset indicator
to its value, which keeps them independent of the model and the data.
"""

from __future__ import annotations

from copy import deepcopy
from itertools import product
from typing import Callable

import numpy as np
import torch
from torch import Tensor

#: Maps an ``(n_modes,)`` float indicator of the proposed subset to its value.
ScoreFn = Callable[[Tensor], Tensor]


def _as_float(value) -> float:
    return value.item() if isinstance(value, Tensor) else float(value)


def greedy_usm(
    n_modes: int, acquired: np.ndarray, score: ScoreFn, costs: Tensor, rng
) -> tuple[np.ndarray, float]:
    """Add the best remaining modality until doing so stops helping."""
    device = costs.device
    a = torch.zeros(n_modes, device=device)
    best_value, current_value = 0.0, 0.0
    unacquired = [i for i, seen in enumerate(acquired) if not seen]
    best_idx = None
    while unacquired and current_value >= best_value:
        best_value = current_value
        a_next = a.clone()
        current_value = None
        for idx in unacquired:
            a_next[idx] = 1.0
            reward = score(a_next) - torch.sum(costs * a_next)
            if not current_value or reward > current_value:
                current_value = reward
                best_idx = idx
            a_next[idx] = 0.0
        if current_value >= best_value:
            a[best_idx] = 1.0
        unacquired.remove(best_idx)
    if current_value is not None and current_value >= best_value:
        best_value = current_value
    return a.to(torch.bool).cpu().numpy(), _as_float(best_value)


def single_item_greedy(
    n_modes: int, acquired: np.ndarray, score: ScoreFn, costs: Tensor, rng
) -> tuple[np.ndarray, float]:
    """Propose at most one modality: the single best. One pass, no recursion."""
    device = costs.device
    a = torch.zeros(n_modes, device=device)
    best_value = -np.inf
    best_idx = None
    unacquired = [i for i, seen in enumerate(acquired) if not seen]
    a_next = a.clone()
    for idx in unacquired:
        a_next[idx] = 1.0
        reward = score(a_next) - torch.sum(costs * a_next)
        if reward > best_value:
            best_value = reward
            best_idx = idx
        a_next[idx] = 0.0
    if best_idx is None:
        return a.to(torch.bool).cpu().numpy(), 0.0
    a[best_idx] = 1.0
    return a.to(torch.bool).cpu().numpy(), _as_float(best_value)


def randomized_usm(
    n_modes: int, acquired: np.ndarray, score: ScoreFn, costs: Tensor, rng
) -> tuple[np.ndarray, float]:
    """Buchbinder et al.'s double greedy for unconstrained maximization.

    Grows a set ``a`` from empty while shrinking a set ``b`` from the full
    unacquired set, deciding each modality by the relative marginal gain of
    including versus excluding it.
    """
    device = costs.device
    a = torch.zeros(n_modes, device=device)
    b = torch.tensor(~acquired, dtype=torch.float32, device=device)
    current_value = 0.0
    a_score, b_score = None, None
    for i in range(n_modes):
        if acquired[i]:
            continue
        a_next, b_next = a.clone(), b.clone()
        a_next[i], b_next[i] = 1.0, 0.0
        with torch.no_grad():
            if a.sum() == 0:
                a_score = 0
            elif a_score is None:
                a_score = score(a)
            if b.sum() == 0:
                b_score = 0
            elif b_score is None:
                b_score = score(b)
            a_i_score = 0.0 if a_next.sum() == 0 else score(a_next)
            b_i_score = 0.0 if b_next.sum() == 0 else score(b_next)
            v_a = a_score - torch.sum(costs * a)
            v_b = b_score - torch.sum(costs * b)
            v_a_i = a_i_score - torch.sum(costs * a_next)
            v_b_i = b_i_score - torch.sum(costs * b_next)
        v_a, v_b = _as_float(v_a), _as_float(v_b)
        v_a_i, v_b_i = _as_float(v_a_i), _as_float(v_b_i)
        q_a, q_b = max(v_a_i - v_a, 0), max(v_b_i - v_b, 0)
        prob = q_a / (q_a + q_b) if q_a + q_b > 0 else 0
        if rng.random() < prob:
            a = a_next
            current_value = v_a_i
            a_score = a_i_score
        else:
            b = b_next
            current_value = v_b_i
            b_score = b_i_score
    return a.to(torch.bool).cpu().numpy(), _as_float(current_value)


def hybrid_usm(
    n_modes: int, acquired: np.ndarray, score: ScoreFn, costs: Tensor, rng
) -> tuple[np.ndarray, float]:
    """Run both heuristics and keep whichever subset scored higher."""
    q1, r1 = randomized_usm(n_modes, acquired, score, costs, rng)
    q2, r2 = greedy_usm(n_modes, acquired, score, costs, rng)
    return (q2, r2) if r2 > r1 else (q1, r1)


def enumerate_subsets(
    n_modes: int, acquired: np.ndarray, score: ScoreFn, costs: Tensor, rng
) -> tuple[np.ndarray, float]:
    """Exhaustive search. Exact, but costs 2**n_modes value-model evaluations."""
    device = costs.device
    available = 1.0 - torch.tensor(acquired, dtype=torch.float32, device=device)
    best_reward = 0.0
    best_q = torch.zeros(n_modes, device=device)
    for combo in product([0, 1], repeat=n_modes):
        q = torch.tensor(combo, dtype=torch.float32, device=device) * available
        reward = score(q) - torch.sum(costs * q)
        if reward > best_reward:
            best_reward = reward
            best_q = deepcopy(q)
    return best_q.to(torch.bool).cpu().numpy(), _as_float(best_reward)


OPTIMIZERS = {
    "rand_usm": randomized_usm,
    "greedy_usm": greedy_usm,
    "hybrid_usm": hybrid_usm,
    "enum": enumerate_subsets,
    "single_item_greedy": single_item_greedy,
}


def optimize_subset(
    method: str,
    n_modes: int,
    acquired: np.ndarray,
    score: ScoreFn,
    costs: Tensor,
    rng,
) -> np.ndarray:
    """Run ``method`` and return its subset, or the empty set if it is not worth it.

    A non-positive optimum means no acquisition pays for itself at these costs,
    which is also how the multi-stage loop learns to stop.
    """
    try:
        optimizer = OPTIMIZERS[method]
    except KeyError:
        raise ValueError(
            f"unknown optimizer {method!r}; expected one of {sorted(OPTIMIZERS)}"
        ) from None
    subset, value = optimizer(n_modes, acquired, score, costs, rng)
    if value <= 0:
        return np.zeros(n_modes, dtype=bool)
    return subset
