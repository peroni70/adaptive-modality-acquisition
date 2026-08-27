"""Representing a set of modalities as a vector."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ModalitySetEmbedding(nn.Module):
    """Mean of the learned embeddings of the modalities present in a set.

    Averaging rather than summing keeps the representation on a comparable
    scale regardless of how many modalities the set contains, so a value model
    can compare a proposal of one modality against a proposal of five.
    """

    def __init__(self, n_modes: int, dim: int, max_norm: float | None = 1.0):
        super().__init__()
        self.n_modes = n_modes
        self.dim = dim
        self.embedding = nn.Embedding(n_modes, dim, max_norm=max_norm)

    @property
    def weight(self) -> Tensor:
        return self.embedding.weight

    def forward(self, members: Tensor) -> Tensor:
        """``members``: ``(B, n_modes)`` boolean/float set indicator."""
        members = members.float().view(-1, self.n_modes)
        pooled = members @ self.embedding.weight
        # Divide by set cardinality. An empty set yields the zero vector rather
        # than a division by zero.
        count = members.sum(dim=1, keepdim=True).clamp(min=1.0)
        return pooled / count


def masked_mean(embeddings: Tensor, members: Tensor, eps: float = 1e-6) -> Tensor:
    """Mean of ``(B, M, d)`` per-modality embeddings over the members of a set."""
    members = members.float().view(embeddings.shape[0], embeddings.shape[1], 1)
    total = (embeddings * members).sum(dim=1)
    count = members.sum(dim=1).clamp(min=eps)
    return total / count
