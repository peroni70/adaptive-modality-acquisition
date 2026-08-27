"""Patch-MNIST.

MNIST digits cut into a regular grid of square patches, each patch treated as
a separately acquirable modality. The centre patch is the free context, which
by itself is rarely enough to identify a digit, so a policy has to decide
which surrounding patches are worth uncovering.

Costs nothing to obtain and runs on a laptop, which makes it the reference
example for the rest of the package.
"""

from __future__ import annotations

from torch.utils.data import Subset, random_split
import torch

from ..registry import register_app


class _Limited:
    """First ``n`` examples of a source, for quick runs."""

    def __init__(self, source, n: int):
        self.source = source
        self.n = min(n, len(source))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        return self.source[idx]


@register_app("patch_mnist")
def load_splits(cfg) -> dict:
    """Download MNIST and split it into train / val / test sources."""
    from torchvision import datasets, transforms

    data = cfg.data
    root = data.get("root", "~/.cache/ama/mnist")
    transform = transforms.Compose([transforms.ToTensor()])
    full_train = datasets.MNIST(
        root=root, train=True, download=True, transform=transform
    )
    test = datasets.MNIST(root=root, train=False, download=True, transform=transform)

    train_fraction = float(data.get("train_fraction", 0.9))
    n_train = int(train_fraction * len(full_train))
    train, val = random_split(
        full_train,
        [n_train, len(full_train) - n_train],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    splits = {"train": train, "val": val, "test": test}
    limits = data.get("max_examples") or {}
    for name, limit in limits.items():
        if limit:
            splits[name] = _Limited(splits[name], int(limit))
    return splits
