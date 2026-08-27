"""Feature encoders.

An encoder knows nothing about modality selection. It turns a masked example
into features, and the wrappers in :mod:`ama.modeling` add everything to do
with acquired and proposed subsets.

Two shapes are supported, declared by the ``per_modality`` attribute:

* ``False`` - ``(B, ...) -> (B, out_dim)``, a single pooled feature vector.
* ``True``  - ``(B, M, ...) -> (B, M, out_dim)``, one vector per modality,
  which the wrappers mask and mean-pool. Suits inputs whose modalities are
  separate channels or streams.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor


class Encoder(nn.Module):
    """Base class fixing the contract the wrappers rely on."""

    #: Width of the features produced for each example (or each modality).
    out_dim: int
    #: Whether ``forward`` returns one vector per modality.
    per_modality: bool = False


class FlatEncoder(Encoder):
    """Pass a flat feature vector straight through.

    Useful when modalities are already embeddings concatenated into one vector
    and the wrapper's trunk is all the encoding that is needed.
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.out_dim = in_dim

    def forward(self, x: Tensor) -> Tensor:
        return x.flatten(start_dim=1)


class ConvEncoder(Encoder):
    """Small two-layer CNN for single-channel images."""

    def __init__(
        self,
        image_size: int = 28,
        in_channels: int = 1,
        channels: Sequence[int] = (32, 64),
        kernel_size: int = 3,
        pool: int = 2,
    ):
        super().__init__()
        c1, c2 = channels
        self.conv1 = nn.Conv2d(in_channels, c1, kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=kernel_size)
        self.pool = nn.MaxPool2d(kernel_size=pool)
        self.relu = nn.ReLU()
        # Two valid convolutions then one pooling stage.
        size = (image_size - 2 * (kernel_size - 1)) // pool
        self.out_dim = c2 * size * size

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        return x.flatten(start_dim=1)


class MLPEncoder(Encoder):
    """Fully connected encoder for tabular or pre-embedded inputs."""

    def __init__(self, in_dim: int, hidden: Sequence[int], dropout: float = 0.0):
        super().__init__()
        from torchvision.ops import MLP

        self.mlp = MLP(
            in_dim, list(hidden), norm_layer=nn.BatchNorm1d, dropout=dropout
        )
        self.out_dim = hidden[-1]

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x.flatten(start_dim=1))
