"""Value functions.

A value function defines what the value model is trained to predict about a
proposed acquisition: how the classifier's answer changes when the proposed
modalities are added to what has already been observed.

Each one bundles together everything that used to be scattered across
``if value_fn == ...`` branches:

* ``head_dim``   - width of the value model's output layer
* ``target``     - the label computed from the frozen classifier
* ``loss``       - the training objective
* ``score``      - the scalar energy the subset optimizers maximize
* ``validate``   - the model-selection metric on held-out data

Adding a fifth value function means adding one subclass and registering it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch import Tensor


class ValueFunction(ABC):
    """Defines the target, loss and energy score for one notion of value."""

    name: str
    head_dim: int
    #: True when a larger validation score means a better model.
    higher_is_better: bool = True

    @abstractmethod
    def target(self, orig_logits: Tensor, new_logits: Tensor, y: Tensor):
        """Label for one example, from the classifier before and after acquiring."""

    @abstractmethod
    def loss(self) -> nn.Module:
        """Training criterion for the value model."""

    def compute_loss(self, criterion: nn.Module, logits: Tensor, targets: Tensor) -> Tensor:
        """Apply ``criterion`` with logits and targets in the shapes it expects.

        Owned here rather than at the call sites: a regression head emits
        ``(B, 1)`` against ``(B,)`` targets, and letting those broadcast turns
        the loss into a mean over every (prediction, target) pair, whose
        minimizer is the batch mean for every output.
        """
        return criterion(logits, self.prepare_targets(targets))

    @abstractmethod
    def score(self, logits: Tensor, reverse: bool = False) -> Tensor:
        """Map value-model logits to the scalar energy used for optimization."""

    @abstractmethod
    def validate(self, logits: Tensor, targets: Tensor) -> float:
        """Held-out score used to pick the best epoch."""

    def prepare_targets(self, targets: Tensor) -> Tensor:
        """Cast a batch of targets to what ``loss`` expects."""
        return targets.long()

    def positive_probability(self, logits: Tensor) -> Tensor:
        """Probability of the 'improved' class, for calibration diagnostics."""
        raise NotImplementedError(f"{self.name} is not a probabilistic value function")

    def binarize(self, targets: np.ndarray) -> np.ndarray:
        """Collapse targets to {0, 1} for a reliability diagram."""
        raise NotImplementedError(f"{self.name} is not a probabilistic value function")

    @property
    def is_probabilistic(self) -> bool:
        """Whether the value model emits class probabilities worth calibrating."""
        return True

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, head_dim={self.head_dim})"


class _Classification(ValueFunction):
    """Shared machinery for the value functions trained with cross entropy."""

    def loss(self) -> nn.Module:
        return nn.CrossEntropyLoss()


class BitFlip(_Classification):
    """1 when acquiring turns a wrong prediction into a right one."""

    name = "bit_flip"
    head_dim = 2

    def target(self, orig_logits, new_logits, y):
        orig = orig_logits.argmax(1)
        new = new_logits.argmax(1)
        return int(((orig != y) & (new == y)).long().item())

    def score(self, logits, reverse=False):
        return torch.softmax(logits, dim=1)[:, 1]

    def validate(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        return float(roc_auc_score(targets.cpu().numpy(), probs[:, 1].cpu().numpy()))

    def positive_probability(self, logits):
        return torch.softmax(logits, dim=1)[:, 1]

    def binarize(self, targets):
        return targets.astype(int)


class AccChange(_Classification):
    """Three-way outcome: correctness lost (0), unchanged (1), or gained (2).

    Its score, ``P(gained) - P(lost)``, is the expected change in accuracy from
    granting the proposal, which is what makes it directly comparable against a
    cost once the two are put in the same units.
    """

    name = "acc_change"
    head_dim = 3

    def target(self, orig_logits, new_logits, y):
        orig = orig_logits.argmax(1)
        new = new_logits.argmax(1)
        return int(((new == y).long() - (orig == y).long() + 1).item())

    def score(self, logits, reverse=False):
        probs = torch.softmax(logits, dim=1)
        value = probs[:, 2] - probs[:, 0]
        return -value if reverse else value

    def validate(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        return float(
            roc_auc_score(targets.cpu().numpy(), probs.cpu().numpy(), multi_class="ovo")
        )

    def positive_probability(self, logits):
        return torch.softmax(logits, dim=1)[:, 2]

    def binarize(self, targets):
        return np.where(targets == 2, 1, 0)


class InfoGain(ValueFunction):
    """Reduction in cross-entropy loss from acquiring the proposed modalities."""

    name = "info_gain"
    head_dim = 1
    higher_is_better = False

    def __init__(self):
        self._ce = nn.CrossEntropyLoss()

    def target(self, orig_logits, new_logits, y):
        h_orig = self._ce(orig_logits, y.long())
        h_new = self._ce(new_logits, y.long())
        return float((h_orig - h_new).item())

    def loss(self) -> nn.Module:
        return nn.MSELoss()

    def prepare_targets(self, targets):
        return targets.float()

    def compute_loss(self, criterion, logits, targets):
        return criterion(logits.squeeze(-1), self.prepare_targets(targets))

    def score(self, logits, reverse=False):
        return logits.squeeze(-1)

    def validate(self, logits, targets):
        return float(
            nn.functional.mse_loss(logits.squeeze(-1), targets.float()).item()
        )

    @property
    def is_probabilistic(self) -> bool:
        return False


VALUE_FUNCTIONS = {cls.name: cls for cls in (BitFlip, AccChange, InfoGain)}


def get_value_fn(name: str) -> ValueFunction:
    """Look up a value function by name."""
    try:
        return VALUE_FUNCTIONS[name]()
    except KeyError:
        raise ValueError(
            f"unknown value function {name!r}; expected one of {sorted(VALUE_FUNCTIONS)}"
        ) from None
