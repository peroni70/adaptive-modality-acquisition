"""Adapters presenting a dataset as an indexable sequence of ``(x, y)`` pairs.

Every dataset in this package is consumed through this one interface, so a new
application only has to say how to produce examples, not how they will be
masked, sampled or batched.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class ExampleSource(Protocol):
    """Anything supporting ``len`` and ``source[i] -> (x, y)``."""

    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> tuple[Tensor, object]: ...


class TensorSource:
    """Wraps parallel feature and label tensors."""

    def __init__(self, x, y):
        self.x = x if isinstance(x, Tensor) else torch.as_tensor(x)
        self.y = y if isinstance(y, Tensor) else torch.as_tensor(y)
        if len(self.x) != len(self.y):
            raise ValueError(
                f"x and y must be the same length, got {len(self.x)} and {len(self.y)}"
            )

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.x[idx], self.y[idx]


def as_label_tensor(y) -> Tensor:
    """Normalize a label of any scalar-ish type to a one-element tensor."""
    if isinstance(y, Tensor):
        return y.reshape(-1)[:1]
    return torch.tensor([y])
