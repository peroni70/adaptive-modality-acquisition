import numpy as np
import pytest
import torch

from ama.optimizers import OPTIMIZERS, optimize_subset

N_MODES = 5
VALUABLE = [1, 3]


def planted_score():
    """Modalities 1 and 3 are worth 1.0 each; the rest are worthless."""
    weights = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0])
    return lambda proposal: (proposal * weights).sum()


def acquired_context():
    acquired = np.zeros(N_MODES, dtype=bool)
    acquired[0] = True
    return acquired


@pytest.mark.parametrize("method", sorted(OPTIMIZERS))
def test_cheap_value_is_worth_acquiring(method):
    subset = optimize_subset(
        method, N_MODES, acquired_context(), planted_score(),
        torch.full((N_MODES,), 0.1), np.random.default_rng(0),
    )
    chosen = np.nonzero(subset)[0].tolist()
    assert chosen, f"{method} acquired nothing when value clearly beat cost"
    assert set(chosen) <= set(VALUABLE)


@pytest.mark.parametrize("method", sorted(OPTIMIZERS))
def test_expensive_modalities_are_declined(method):
    subset = optimize_subset(
        method, N_MODES, acquired_context(), planted_score(),
        torch.full((N_MODES,), 5.0), np.random.default_rng(0),
    )
    assert not subset.any()


@pytest.mark.parametrize("method", sorted(OPTIMIZERS))
def test_already_acquired_modalities_are_never_reproposed(method):
    acquired = np.array([True, True, False, False, False])
    subset = optimize_subset(
        method, N_MODES, acquired, planted_score(),
        torch.full((N_MODES,), 0.1), np.random.default_rng(0),
    )
    assert not (subset & acquired).any()


def test_single_item_greedy_proposes_at_most_one():
    subset = optimize_subset(
        "single_item_greedy", N_MODES, acquired_context(), planted_score(),
        torch.full((N_MODES,), 0.1), np.random.default_rng(0),
    )
    assert subset.sum() == 1


def test_exhaustive_search_finds_the_planted_optimum():
    subset = optimize_subset(
        "enum", N_MODES, acquired_context(), planted_score(),
        torch.full((N_MODES,), 0.1), np.random.default_rng(0),
    )
    assert np.nonzero(subset)[0].tolist() == VALUABLE


def test_hybrid_is_never_worse_than_greedy_alone():
    score, costs = planted_score(), torch.full((N_MODES,), 0.1)
    _, greedy_value = OPTIMIZERS["greedy_usm"](
        N_MODES, acquired_context(), score, costs, np.random.default_rng(0)
    )
    _, hybrid_value = OPTIMIZERS["hybrid_usm"](
        N_MODES, acquired_context(), score, costs, np.random.default_rng(0)
    )
    assert hybrid_value >= greedy_value


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="unknown optimizer"):
        optimize_subset(
            "nope", N_MODES, acquired_context(), planted_score(),
            torch.zeros(N_MODES), np.random.default_rng(0),
        )
