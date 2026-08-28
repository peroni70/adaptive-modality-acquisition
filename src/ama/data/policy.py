"""Running an acquisition policy over a dataset.

This is where the value model is put to work. For each example the policy
starts from the context modality alone and repeatedly asks the subset
optimizer what to acquire next, stopping when nothing is worth its cost. The
result is the observation the downstream classifier finally sees, together
with the bill for getting there.

Three policies share the interface:

* ``eama``            - the adaptive multi-stage policy driven by the value model.
* ``adaptive_greedy`` - fixed greedy modality order, but the stopping point is
  chosen per example from the realized costs. A cost-aware static baseline.
* ``preset``          - a fixed subset for every example, used for the static
  greedy-prefix baselines.

Optimization runs per example rather than per batch: each example follows its
own trajectory through a different number of stages, and the reported wall
clock is meant to reflect the per-decision cost of the method.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from ..costs import CostModel
from ..masking import Masker
from ..optimizers import optimize_subset
from ..value_fns import ValueFunction
from .sources import ExampleSource, as_label_tensor

METHODS = ("eama", "adaptive_greedy", "preset")


class PolicyDataset(Dataset):
    """Yields ``(observed_x, acquired, y, costs[, n_stages])`` after running a policy.

    Args:
        source: indexable yielding ``(x, y)``.
        masker: how to zero out unobserved modalities.
        context_idx: modality observed for free, and never billed.
        value_models: one model or an ensemble whose scores are averaged.
        value_fn: interprets value-model logits as a scalar energy.
        method: one of ``METHODS``.
        opt_method: subset optimizer, for ``method="eama"``.
        cost_model: supplies each example's acquisition costs, already
            converted into value units. Defaults to free acquisition.
        single_stage: stop after the first acquisition round.
        reverse: negate the value score, for the reversed prob-change variant.
        include_modes: the fixed subset, for ``method="preset"``.
        greedy_order, greedy_prefix_gains: for ``method="adaptive_greedy"``,
            the modality order and its validation gains by prefix length.
        return_num_stages: also return how many acquisition rounds ran.
    """

    def __init__(
        self,
        source: ExampleSource,
        masker: Masker,
        context_idx: int,
        value_models=None,
        value_fn: ValueFunction | None = None,
        method: str = "eama",
        opt_method: str = "rand_usm",
        cost_model: CostModel | None = None,
        single_stage: bool = False,
        deterministic: bool = False,
        reverse: bool = False,
        include_modes=None,
        greedy_order=None,
        greedy_prefix_gains=None,
        device: str = "cpu",
        return_num_stages: bool = False,
        verbose: bool = False,
    ):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        self.source = source
        self.masker = masker
        self.n_modes = masker.n_modes
        self.context_idx = context_idx
        self.value_fn = value_fn
        self.method = method
        self.opt_method = opt_method
        self.cost_model = cost_model
        self.single_stage = single_stage
        self.deterministic = deterministic
        self.reverse = reverse
        self.device = device
        self.return_num_stages = return_num_stages
        self.verbose = verbose

        self.value_models = []
        if value_models is not None:
            if not isinstance(value_models, (list, tuple)):
                value_models = [value_models]
            for model in value_models:
                model.to(device)
                model.eval()
            self.value_models = list(value_models)
            if value_fn is None:
                raise ValueError("value_fn is required when value_models are given")

        self.costs = torch.zeros(self.n_modes, device=device)
        self.include_modes = (
            None if include_modes is None else np.asarray(include_modes, dtype=bool)
        )
        self.greedy_order = greedy_order
        self.greedy_prefix_gains = greedy_prefix_gains
        if method == "adaptive_greedy" and (
            greedy_order is None or greedy_prefix_gains is None
        ):
            raise ValueError(
                "method='adaptive_greedy' requires greedy_order and greedy_prefix_gains"
            )
        if method == "preset" and self.include_modes is None:
            raise ValueError("method='preset' requires include_modes")
        self.rng = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.source)

    def sample_costs(self, idx: int, rng) -> torch.Tensor:
        """Effective cost vector for one example, in value units."""
        if self.cost_model is None:
            return torch.zeros(self.n_modes, device=self.device)
        return self.cost_model.for_example(idx, rng)

    def score_subset(self, x_current, acquired, proposed) -> torch.Tensor:
        """Average value across the ensemble for one proposal."""
        total = 0.0
        with torch.no_grad():
            for model in self.value_models:
                logits = model(x_current, acquired, proposed.unsqueeze(0))
                total = total + self.value_fn.score(logits, reverse=self.reverse)
        return total / len(self.value_models)

    def _run_eama(self, x, acquired, rng):
        """Acquire in rounds until the optimizer declines to add anything."""
        n_stages = 0
        while True:
            x_current = self.masker.mask(x, acquired).to(self.device).unsqueeze(0)
            acquired_batch = (
                torch.from_numpy(acquired).float().unsqueeze(0).to(self.device)
            )
            score = lambda proposed: self.score_subset(  # noqa: E731
                x_current, acquired_batch, proposed
            )
            proposed = optimize_subset(
                self.opt_method, self.n_modes, acquired, score, self.costs, rng
            )
            if not proposed.any():
                break
            n_stages += 1
            acquired = acquired | proposed
            if self.verbose:
                print(acquired)
            if self.single_stage:
                break
        return acquired, n_stages

    def _run_adaptive_greedy(self, acquired):
        """Take the greedy prefix whose validation gain best beats its cost."""
        best_reward, best_len = 0.0, 0
        for prefix_len, gain in enumerate(self.greedy_prefix_gains, start=1):
            cost = torch.sum(self.costs[self.greedy_order[:prefix_len]]).item()
            if gain - cost > best_reward:
                best_reward = gain - cost
                best_len = prefix_len
        acquired = acquired.copy()
        acquired[list(self.greedy_order[:best_len])] = True
        return acquired, int(best_len > 0)

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(idx) if self.deterministic else self.rng
        self.costs = self.sample_costs(idx, rng)

        x, y = self.source[idx]
        acquired = np.zeros(self.n_modes, dtype=bool)
        acquired[self.context_idx] = True

        if self.method == "eama":
            acquired, n_stages = self._run_eama(x, acquired, rng)
        elif self.method == "adaptive_greedy":
            acquired, n_stages = self._run_adaptive_greedy(acquired)
        else:
            acquired = acquired | self.include_modes
            n_stages = None

        x_current = self.masker.mask(x, acquired)
        observed = torch.from_numpy(acquired).float().to(self.device)
        y = as_label_tensor(y)
        if self.return_num_stages:
            return x_current, observed, y, self.costs, (-1 if n_stages is None else n_stages)
        return x_current, observed, y, self.costs
