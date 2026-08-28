"""Diagnostics computed from a trained classifier alone.

Run these before committing to a value function: they describe the problem you
have, and that determines which value function can represent it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .data.sources import ExampleSource, as_label_tensor
from .data.value import sample_acquisition
from .masking import Masker
from .progress import progress


@dataclass
class ConfusionRates:
    """How a classifier responds to acquiring more, over random acquisitions.

    Every draw takes a random observed set and a random non-empty proposal,
    and asks what happens to the prediction when the proposal is granted. Each
    draw lands in exactly one of three outcomes.
    """

    n_draws: int
    #: Wrong before, right after. The gain acquisition is supposed to deliver.
    gained: float
    #: Right before, wrong after. This is the model confusion rate.
    lost: float
    #: Correctness unchanged, whether right or wrong throughout.
    unchanged: float
    accuracy_before: float
    accuracy_after: float
    #: Of the draws that were right before, the fraction acquisition broke.
    confusion_rate_given_correct: float

    @property
    def model_confusion_rate(self) -> float:
        """How often acquiring turns a correct prediction incorrect.

        The rate over all acquisitions. See
        :attr:`conditional_model_confusion_rate` for the rate among the
        acquisitions that had something to lose.
        """
        return self.lost

    @property
    def conditional_model_confusion_rate(self) -> float:
        """Model confusion rate among draws the classifier had right already."""
        return self.confusion_rate_given_correct

    @property
    def net_gain(self) -> float:
        """Expected change in accuracy from a random acquisition."""
        return self.gained - self.lost

    @property
    def recommended_value_fn(self) -> str:
        """Which value function can represent what this classifier does.

        ``bit_flip`` models gains only. When acquiring also destroys correct
        predictions at a non-trivial rate, that target is blind to a real part
        of the problem and ``acc_change``, which scores gains against losses,
        is the safer choice.
        """
        return "bit_flip" if self.lost < 0.01 else "acc_change"

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "model_confusion_rate": self.model_confusion_rate,
            "conditional_model_confusion_rate": self.conditional_model_confusion_rate,
            "net_gain": self.net_gain,
            "recommended_value_fn": self.recommended_value_fn,
        }

    def report(self) -> str:
        lines = [
            f"Draws                          : {self.n_draws}",
            f"Accuracy before acquiring      : {self.accuracy_before:.4f}",
            f"Accuracy after acquiring       : {self.accuracy_after:.4f}",
            "",
            "Outcome of a random acquisition",
            f"  gained    (wrong -> right)   : {self.gained:.4f}",
            f"  lost      (right -> wrong)   : {self.lost:.4f}",
            f"  unchanged                    : {self.unchanged:.4f}",
            "",
            f"MODEL CONFUSION RATE           : {self.model_confusion_rate:.4f}",
            "    share of all acquisitions that break a correct prediction",
            f"  given correct beforehand     : "
            f"{self.conditional_model_confusion_rate:.4f}",
            "    share of the correct predictions that acquiring destroys",
            "",
            f"Net accuracy change            : {self.net_gain:+.4f}",
            "",
            f"Suggested value function       : {self.recommended_value_fn}",
        ]
        if self.recommended_value_fn == "acc_change":
            lines.append(
                "  Acquiring destroys correct predictions often enough that a\n"
                "  gain-only target would ignore a real part of the problem."
            )
        else:
            lines.append(
                "  Acquiring almost never destroys a correct prediction, so the\n"
                "  simpler binary target loses little."
            )
        return "\n".join(lines)


class _AcquisitionPairs(Dataset):
    """Before/after views of an example under a random acquisition."""

    def __init__(self, source, masker, context_idx, n_repeats, seed):
        self.source = source
        self.masker = masker
        self.context_idx = context_idx
        self.n_repeats = n_repeats
        self.seed = seed

    def __len__(self) -> int:
        return self.n_repeats * len(self.source)

    def __getitem__(self, idx: int):
        repeat, i = divmod(idx, len(self.source))
        # Matches ValueDataset's deterministic seeding, so these draws are the
        # ones the value model is actually trained on.
        rng = np.random.default_rng(self.seed + repeat * len(self.source) + i + 1)
        acquired, proposed = sample_acquisition(
            self.masker.n_modes, self.context_idx, rng
        )
        x, y = self.source[i]
        return (
            self.masker.mask(x, acquired),
            self.masker.mask(x, acquired | proposed),
            torch.from_numpy(acquired).float(),
            torch.from_numpy(acquired | proposed).float(),
            as_label_tensor(y).item(),
        )


@torch.no_grad()
def model_confusion_rates(
    classifier: torch.nn.Module,
    source: ExampleSource,
    masker: Masker,
    context_idx: int,
    n_repeats: int = 1,
    batch_size: int = 128,
    seed: int = 0,
    device: str | None = None,
) -> ConfusionRates:
    """Measure how often acquiring helps, hurts, or changes nothing.

    Acquisitions are drawn exactly as value-model training draws them, so the
    rates describe the distribution the value model is fitted on.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    classifier = classifier.to(device).eval()
    loader = DataLoader(
        _AcquisitionPairs(source, masker, context_idx, n_repeats, seed),
        batch_size=batch_size,
        shuffle=False,
    )

    gained = lost = unchanged = 0
    correct_before = correct_after = total = 0
    for x_before, x_after, acquired, after, y in progress(loader):
        x_before, x_after = x_before.to(device), x_after.to(device)
        acquired, after = acquired.to(device), after.to(device)
        y = y.to(device).long()

        was_right = classifier(x_before, acquired).argmax(1) == y
        is_right = classifier(x_after, after).argmax(1) == y

        gained += int((~was_right & is_right).sum())
        lost += int((was_right & ~is_right).sum())
        unchanged += int((was_right == is_right).sum())
        correct_before += int(was_right.sum())
        correct_after += int(is_right.sum())
        total += len(y)

    return ConfusionRates(
        n_draws=total,
        gained=gained / total,
        lost=lost / total,
        unchanged=unchanged / total,
        accuracy_before=correct_before / total,
        accuracy_after=correct_after / total,
        confusion_rate_given_correct=(lost / correct_before) if correct_before else 0.0,
    )
