import numpy as np
import torch

from ama.costs import CostModel
from ama.data import MaskedDataset, PolicyDataset, ValueDataset
from ama.value_fns import get_value_fn

CONTEXT = 10


def test_masked_dataset_always_shows_context(masker, source):
    dataset = MaskedDataset(source, masker, CONTEXT, deterministic=True)
    for i in range(len(dataset)):
        _, _, observed = dataset[i]
        assert observed[CONTEXT] == 1.0


def test_masked_dataset_does_not_mutate_include_modes(masker, source):
    include = [False] * masker.n_modes
    dataset = MaskedDataset(source, masker, CONTEXT, include_modes=include)
    dataset[0]
    assert include[CONTEXT] is False, "include_modes leaked the context modality"


def test_value_dataset_proposes_disjoint_non_empty_subsets(
    masker, source, classifier
):
    dataset = ValueDataset(
        source, masker, classifier, CONTEXT, get_value_fn("acc_change"),
        n_repeats=3, deterministic=True,
    )
    assert len(dataset) == 3 * len(source)
    for i in range(len(dataset)):
        _, _, acquired, proposed = dataset[i]
        assert acquired[CONTEXT] == 1.0
        assert proposed.sum() >= 1
        assert (acquired * proposed).sum() == 0


def test_value_dataset_varies_subsets_across_examples(masker, source, classifier):
    """Deterministic seeding must not give every example the same subset."""
    dataset = ValueDataset(
        source, masker, classifier, CONTEXT, get_value_fn("acc_change"),
        n_repeats=1, deterministic=True,
    )
    sizes = {int(dataset[i][2].sum()) for i in range(len(dataset))}
    assert len(sizes) > 1


def test_value_dataset_is_reproducible(masker, source, classifier):
    args = (source, masker, classifier, CONTEXT, get_value_fn("bit_flip"))
    first = ValueDataset(*args, deterministic=True)[2]
    second = ValueDataset(*args, deterministic=True)[2]
    assert torch.equal(first[2], second[2]) and torch.equal(first[3], second[3])


def test_include_state_false_blanks_the_observation(masker, source, classifier):
    dataset = ValueDataset(
        source, masker, classifier, CONTEXT, get_value_fn("bit_flip"),
        deterministic=True, include_state=False,
    )
    assert dataset[0][0].abs().sum() == 0


def test_policy_acquires_less_as_costs_rise(masker, source, value_model):
    value_fn = get_value_fn("acc_change")
    # Bias the head so every proposal looks worthwhile; costs alone decide.
    with torch.no_grad():
        value_model.head.weight.zero_()
        value_model.head.bias.copy_(torch.tensor([0.0, 0.0, 6.0]))

    counts = []
    for scale in (0.0, 0.9, 3.0):
        dataset = PolicyDataset(
            source, masker, CONTEXT, value_model, value_fn,
            opt_method="greedy_usm", deterministic=True,
            cost_model=CostModel(masker.n_modes, scale=scale),
        )
        counts.append(int(dataset[0][1].sum()))
    assert counts[0] > counts[-1], f"costs did not reduce acquisition: {counts}"


def test_lambda_sets_willingness_to_pay(masker, source, value_model):
    """Large lambda means accuracy is worth a lot, so the policy buys more."""
    value_fn = get_value_fn("acc_change")
    with torch.no_grad():
        value_model.head.weight.zero_()
        value_model.head.bias.copy_(torch.tensor([0.0, 0.0, 6.0]))

    costs = np.ones((len(source), masker.n_modes))
    counts = []
    for lam in (0.01, 100.0):
        dataset = PolicyDataset(
            source, masker, CONTEXT, value_model, value_fn,
            opt_method="greedy_usm", deterministic=True,
            cost_model=CostModel(masker.n_modes, costs=costs, lambda_=lam),
        )
        counts.append(int(dataset[0][1].sum()))
    assert counts[0] < counts[1], f"lambda did not raise willingness to pay: {counts}"


def test_per_example_costs_are_applied_per_example(masker, source, value_model):
    """Row i of the cost matrix governs example i."""
    value_fn = get_value_fn("acc_change")
    with torch.no_grad():
        value_model.head.weight.zero_()
        value_model.head.bias.copy_(torch.tensor([0.0, 0.0, 6.0]))

    # Example 0 faces free modalities; example 1 faces prohibitive ones.
    costs = np.zeros((len(source), masker.n_modes))
    costs[1] = 1000.0
    dataset = PolicyDataset(
        source, masker, CONTEXT, value_model, value_fn,
        opt_method="greedy_usm", deterministic=True,
        cost_model=CostModel(masker.n_modes, costs=costs, lambda_=1.0),
    )
    assert int(dataset[0][1].sum()) > 1, "free example should acquire"
    assert int(dataset[1][1].sum()) == 1, "priced-out example should hold at context"


def test_preset_policy_observes_exactly_what_it_was_given(masker, source):
    include = [i < 3 for i in range(masker.n_modes)]
    dataset = PolicyDataset(
        source, masker, CONTEXT, method="preset", include_modes=include,
        deterministic=True,
    )
    _, observed, _, _ = dataset[0]
    assert set(torch.nonzero(observed).view(-1).tolist()) == {0, 1, 2, CONTEXT}


def test_adaptive_greedy_stops_where_cost_overtakes_gain(masker, source):
    dataset = PolicyDataset(
        source, masker, CONTEXT, method="adaptive_greedy",
        greedy_order=[3, 7, 1], greedy_prefix_gains=[0.5, 0.7, 0.72],
        deterministic=True,
    )
    _, observed, _, _ = dataset[0]
    # Free acquisition, so it should take the whole order.
    assert set(torch.nonzero(observed).view(-1).tolist()) == {1, 3, 7, CONTEXT}


def test_parallel_workers_draw_independent_subsets(masker, source, classifier):
    """Forked workers inherit the parent's generator unless it is re-seeded."""
    from torch.utils.data import DataLoader

    dataset = ValueDataset(
        source, masker, classifier, CONTEXT, get_value_fn("bit_flip")
    )
    loader = DataLoader(dataset, batch_size=4, num_workers=4)
    draws = torch.cat([acquired for _, _, acquired, _ in loader])
    distinct = {tuple(row.tolist()) for row in draws}
    # Shared seeding would repeat one block of draws once per worker.
    assert len(distinct) > 0.7 * len(draws), (
        f"only {len(distinct)} distinct subsets from {len(draws)} draws"
    )
