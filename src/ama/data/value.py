"""Training data for the value model.

Each example pairs an acquisition state with a proposal, labelled by what the
frozen classifier actually does when the proposal is granted. The classifier
is the oracle here; the value model learns to anticipate it without paying for
the data.

Subsets are drawn by first sampling a cardinality uniformly and then sampling
members. Drawing each modality independently would concentrate the training
distribution near half the modalities and rarely show the near-empty and
near-full states that a policy actually passes through.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from ..masking import Masker
from ..value_fns import ValueFunction
from .sources import ExampleSource, as_label_tensor


def sample_acquisition(n_modes: int, context_idx: int, rng):
    """Draw an acquired set and a non-empty proposal disjoint from it.

    The cardinality is drawn first and the members second. Drawing each
    modality independently would concentrate the training distribution near
    half the modalities and rarely show the near-empty and near-full states a
    policy actually passes through.
    """
    k = rng.integers(0, n_modes - 1)
    optional = [i for i in range(n_modes) if i != context_idx]
    chosen = set(rng.choice(optional, k, replace=False).tolist())
    chosen.add(context_idx)

    acquired = np.zeros(n_modes, dtype=bool)
    acquired[list(chosen)] = True

    unobserved = sorted(set(range(n_modes)) - chosen)
    size = rng.integers(1, n_modes - k)
    proposed = np.zeros(n_modes, dtype=bool)
    proposed[rng.choice(unobserved, size, replace=False)] = True
    return acquired, proposed


class ValueDataset(Dataset):
    """Yields ``(masked_x, value, acquired, proposed)`` for value-model training.

    Args:
        source: indexable yielding ``(x, y)``.
        masker: how to zero out unobserved modalities.
        classifier: frozen model supplying the value targets.
        context_idx: modality observed for free.
        value_fn: defines the target computed from the classifier.
        n_repeats: distinct (state, proposal) draws per example.
        deterministic: seed each draw, so validation targets are stable.
        include_state: when False, blank the observation to ablate the
            model's access to the current input.
    """

    def __init__(
        self,
        source: ExampleSource,
        masker: Masker,
        classifier: torch.nn.Module,
        context_idx: int,
        value_fn: ValueFunction,
        n_repeats: int = 1,
        deterministic: bool = False,
        include_state: bool = True,
        device: str = "cpu",
    ):
        self.source = source
        self.masker = masker
        self.n_modes = masker.n_modes
        self.context_idx = context_idx
        self.value_fn = value_fn
        self.n_repeats = n_repeats
        self.deterministic = deterministic
        self.include_state = include_state
        self.device = device
        self.classifier = classifier.to(device)
        self.classifier.eval()
        self._rng = None
        self._rng_pid = None

    def __len__(self) -> int:
        return self.n_repeats * len(self.source)

    @property
    def rng(self):
        """A generator private to this worker process.

        DataLoader workers are forked, so they inherit a copy of the parent's
        generator and would otherwise all draw the same sequence of subsets.
        Re-seed whenever the process changes.
        """
        import os

        if self._rng is None or self._rng_pid != os.getpid():
            from torch.utils.data import get_worker_info

            info = get_worker_info()
            seed = None if info is None else info.seed % (2**32)
            self._rng = np.random.default_rng(seed)
            self._rng_pid = os.getpid()
        return self._rng

    def sample_subsets(self, rng) -> tuple[np.ndarray, np.ndarray]:
        """Draw an acquired set and a non-empty proposal disjoint from it."""
        return sample_acquisition(self.n_modes, self.context_idx, rng)

    def compute_value(self, x_current, x_new, y, acquired, proposed) -> float:
        """Run the frozen classifier before and after, and score the change."""
        with torch.no_grad():
            x_current = x_current.to(self.device).unsqueeze(0)
            x_new = x_new.to(self.device).unsqueeze(0)
            y = y.to(self.device)
            after = (proposed.long() | acquired.long()).unsqueeze(0)
            orig_logits = self.classifier(x_current, acquired.unsqueeze(0))
            new_logits = self.classifier(x_new, after)
        return self.value_fn.target(orig_logits, new_logits, y)

    def __getitem__(self, idx: int):
        repeat, idx = divmod(idx, len(self.source))
        # Seed on the flat position so every (example, repeat) draws a distinct
        # subset. Seeding on ``repeat * (idx + 1)`` would give the whole first
        # repeat the same seed, and so the same subset for every example.
        rng = (
            np.random.default_rng(repeat * len(self.source) + idx + 1)
            if self.deterministic
            else self.rng
        )
        acquired, proposed = self.sample_subsets(rng)

        x, y = self.source[idx]
        x_current = self.masker.mask(x, acquired)
        x_new = self.masker.mask(x, acquired | proposed)

        y = as_label_tensor(y)
        acquired = torch.from_numpy(acquired).float()
        proposed = torch.from_numpy(proposed).float()
        value = self.compute_value(x_current, x_new, y, acquired, proposed)

        if not self.include_state:
            x_current = torch.zeros_like(x_current)
        return x_current, value, acquired, proposed
