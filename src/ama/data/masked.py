"""Training data for the masked classifier.

Each example is shown under a random subset of modalities, which is what
teaches one classifier to handle every acquisition state it will meet at
evaluation time. The context modality is always observed.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from ..masking import Masker
from .sources import ExampleSource, as_label_tensor


class MaskedDataset(Dataset):
    """Yields ``(masked_x, y, observed)`` under random or fixed modality subsets.

    Args:
        source: indexable yielding ``(x, y)``.
        masker: how to zero out unobserved modalities.
        context_idx: modality observed for free; ``None`` for no context.
        deterministic: seed the subset per index, so an epoch is reproducible.
        include_modes: fix the observed set instead of sampling it.
    """

    def __init__(
        self,
        source: ExampleSource,
        masker: Masker,
        context_idx: int | None,
        deterministic: bool = False,
        include_modes=None,
    ):
        self.source = source
        self.masker = masker
        self.n_modes = masker.n_modes
        self.context_idx = context_idx
        self.deterministic = deterministic
        self.include_modes = (
            None if include_modes is None else np.asarray(include_modes, dtype=bool)
        )
        self.rng = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.source)

    def observed_set(self, idx: int) -> np.ndarray:
        """The modality subset shown for example ``idx``."""
        if self.include_modes is None:
            rng = np.random.default_rng(idx) if self.deterministic else self.rng
            observed = rng.integers(2, size=self.n_modes).astype(bool)
        else:
            # Copy: the caller's array must not gain the context modality.
            observed = self.include_modes.copy()
        if self.context_idx is not None:
            observed[self.context_idx] = True
        return observed

    def __getitem__(self, idx: int):
        observed = self.observed_set(idx)
        x, y = self.source[idx]
        x = self.masker.mask(x, observed)
        return x, as_label_tensor(y).item(), torch.from_numpy(observed).float()
