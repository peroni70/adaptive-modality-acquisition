"""Classifier and value model wrappers.

Both models pair a modality-agnostic encoder with learned embeddings of the
modality sets involved. The classifier is told what has been observed; the
value model is additionally told what is being proposed.

The ``fusion`` argument controls where the set embeddings enter:

* ``"early"`` - concatenated to the encoder features before the trunk.
* ``"late"``  - concatenated after a first trunk, then passed through a second.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.ops import MLP

from ..value_fns import ValueFunction
from .encoders import Encoder
from .fusion import ModalitySetEmbedding, masked_mean

FUSIONS = ("early", "late")


def _mlp(in_dim: int, hidden: Sequence[int], dropout: float = 0.0) -> nn.Module:
    return MLP(in_dim, list(hidden), norm_layer=nn.BatchNorm1d, dropout=dropout)


class _SetConditionedModel(nn.Module):
    """Encoder plus trunk, conditioned on ``n_sets`` modality-set embeddings."""

    def __init__(
        self,
        encoder: Encoder,
        n_modes: int,
        n_sets: int,
        out_dim: int,
        hidden: Sequence[int],
        mode_emb_dim: int,
        fusion: str = "early",
        dropout: float = 0.0,
    ):
        super().__init__()
        if fusion not in FUSIONS:
            raise ValueError(f"fusion must be one of {FUSIONS}, got {fusion!r}")
        self.encoder = encoder
        self.n_modes = n_modes
        self.fusion = fusion
        self.set_embeddings = nn.ModuleList(
            ModalitySetEmbedding(n_modes, mode_emb_dim) for _ in range(n_sets)
        )
        cond_dim = n_sets * mode_emb_dim
        if fusion == "early":
            self.trunk = _mlp(encoder.out_dim + cond_dim, hidden, dropout)
            self.post = None
        else:
            self.trunk = _mlp(encoder.out_dim, hidden, dropout)
            self.post = _mlp(hidden[-1] + cond_dim, hidden, dropout)
        self.head = nn.Linear(hidden[-1], out_dim)

    @property
    def mode_embedding(self) -> nn.Embedding:
        """The embedding table over individual modalities."""
        return self.set_embeddings[0].embedding

    def encode(self, x: Tensor, acquired: Tensor) -> Tensor:
        features = self.encoder(x)
        if self.encoder.per_modality:
            # Per-modality encoders emit (B, M, d); drop the unobserved rows.
            features = masked_mean(features, acquired)
        return features

    def forward(self, x: Tensor, *sets: Tensor) -> Tensor:
        if len(sets) != len(self.set_embeddings):
            raise ValueError(
                f"expected {len(self.set_embeddings)} modality sets, got {len(sets)}"
            )
        features = self.encode(x, sets[0])
        cond = torch.cat(
            [emb(s) for emb, s in zip(self.set_embeddings, sets)], dim=1
        )
        if self.fusion == "early":
            hidden = self.trunk(torch.cat((features, cond), dim=1))
        else:
            hidden = self.post(torch.cat((self.trunk(features), cond), dim=1))
        return self.head(hidden)


class MaskedClassifier(_SetConditionedModel):
    """Predicts the label from a partially observed example.

    Conditioning on which modalities were observed lets one classifier serve
    every acquisition state, instead of training a separate model per subset.
    """

    def __init__(
        self,
        encoder: Encoder,
        n_modes: int,
        n_classes: int,
        hidden: Sequence[int] = (128, 64, 32),
        mode_emb_dim: int = 1028,
        fusion: str = "early",
        dropout: float = 0.0,
    ):
        super().__init__(
            encoder,
            n_modes,
            n_sets=1,
            out_dim=n_classes,
            hidden=hidden,
            mode_emb_dim=mode_emb_dim,
            fusion=fusion,
            dropout=dropout,
        )
        self.n_classes = n_classes

    def forward(self, x: Tensor, acquired: Tensor) -> Tensor:
        return super().forward(x, acquired)


class ValueModel(_SetConditionedModel):
    """Scores a proposed acquisition given the current observations.

    Takes the masked example, the set already acquired, and the set proposed;
    returns logits whose interpretation is set by the value function.
    """

    def __init__(
        self,
        encoder: Encoder,
        n_modes: int,
        value_fn: ValueFunction,
        hidden: Sequence[int] = (64, 32),
        mode_emb_dim: int = 1028,
        fusion: str = "early",
        dropout: float = 0.0,
    ):
        super().__init__(
            encoder,
            n_modes,
            n_sets=2,
            out_dim=value_fn.head_dim,
            hidden=hidden,
            mode_emb_dim=mode_emb_dim,
            fusion=fusion,
            dropout=dropout,
        )
        self.value_fn_name = value_fn.name

    def forward(self, x: Tensor, acquired: Tensor, proposed: Tensor) -> Tensor:
        return super().forward(x, acquired, proposed)
