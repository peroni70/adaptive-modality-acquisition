import numpy as np
import torch

from ama.data.value import ValueDataset
from ama.diagnostics import ConfusionRates, model_confusion_rates
from ama.value_fns import get_value_fn

CONTEXT = 10


def rates(classifier, source, masker, **kw):
    return model_confusion_rates(classifier, source, masker, CONTEXT, **kw)


def test_outcomes_partition_the_draws(classifier, source, masker):
    r = rates(classifier, source, masker, n_repeats=2)
    assert r.gained + r.lost + r.unchanged == 1.0
    assert r.n_draws == 2 * len(source)


def test_net_gain_is_the_accuracy_difference(classifier, source, masker):
    r = rates(classifier, source, masker, n_repeats=2)
    assert r.net_gain == (r.accuracy_after - r.accuracy_before)


def test_a_classifier_that_ignores_its_input_is_never_confused(source, masker):
    """No prediction changes, so nothing is gained and nothing is lost."""

    class Constant(torch.nn.Module):
        def forward(self, x, acquired):
            return torch.zeros(len(x), 10).index_fill_(1, torch.tensor([3]), 1.0)

    r = rates(Constant(), source, masker, n_repeats=2)
    assert r.gained == 0.0 and r.lost == 0.0
    assert r.unchanged == 1.0
    assert r.recommended_value_fn == "bit_flip"


def test_the_rates_describe_the_value_model_training_distribution(
    classifier, source, masker
):
    """The diagnostic must measure the draws the value model is fitted on.

    acc_change labels each draw 0 lost / 1 unchanged / 2 gained, so its target
    histogram and these rates are two views of the same thing.
    """
    r = rates(classifier, source, masker, n_repeats=2)
    dataset = ValueDataset(
        source, masker, classifier, CONTEXT, get_value_fn("acc_change"),
        n_repeats=2, deterministic=True,
    )
    targets = np.array([dataset[i][1] for i in range(len(dataset))])
    assert np.isclose(r.lost, (targets == 0).mean())
    assert np.isclose(r.unchanged, (targets == 1).mean())
    assert np.isclose(r.gained, (targets == 2).mean())


def test_recommendation_follows_the_confusion_rate():
    common = dict(
        n_draws=100, unchanged=0.5, accuracy_before=0.5,
        accuracy_after=0.5, confusion_rate_given_correct=0.0,
    )
    assert ConfusionRates(gained=0.5, lost=0.0, **common).recommended_value_fn == "bit_flip"
    assert ConfusionRates(gained=0.4, lost=0.1, **common).recommended_value_fn == "acc_change"


def test_report_names_the_model_confusion_rate(classifier, source, masker):
    """The paper's term, spelled out - not left as an unexplained 'lost'."""
    text = rates(classifier, source, masker).report()
    assert "MODEL CONFUSION RATE" in text
    assert "Suggested value function" in text


def test_model_confusion_rate_is_exposed_by_name(classifier, source, masker):
    r = rates(classifier, source, masker)
    assert r.model_confusion_rate == r.lost
    assert r.conditional_model_confusion_rate == r.confusion_rate_given_correct
    assert r.to_dict()["model_confusion_rate"] == r.lost
