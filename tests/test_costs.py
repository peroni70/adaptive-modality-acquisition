import numpy as np
import pytest
import torch

from ama.costs import CostModel, load_costs


def test_lambda_converts_costs_into_value_units():
    """effective = raw / lambda, so lambda is spend per unit of value."""
    model = CostModel(4, costs=[1.0, 2.0, 3.0, 4.0], lambda_=2.0)
    effective = model.for_example(0, np.random.default_rng(0))
    assert torch.allclose(effective, torch.tensor([0.5, 1.0, 1.5, 2.0]))


def test_larger_lambda_means_cheaper_in_value_units():
    rng = np.random.default_rng(0)
    generous = CostModel(2, costs=[1.0, 1.0], lambda_=100.0).for_example(0, rng)
    stingy = CostModel(2, costs=[1.0, 1.0], lambda_=0.01).for_example(0, rng)
    assert (generous < stingy).all()


def test_a_single_vector_applies_to_every_example():
    model = CostModel(3, costs=[1.0, 2.0, 3.0], lambda_=1.0)
    rng = np.random.default_rng(0)
    assert torch.equal(model.for_example(0, rng), model.for_example(7, rng))


def test_a_matrix_gives_each_example_its_own_costs():
    costs = np.array([[1.0, 1.0], [5.0, 5.0]])
    model = CostModel(2, costs=costs, lambda_=1.0)
    rng = np.random.default_rng(0)
    assert model.for_example(0, rng).tolist() == [1.0, 1.0]
    assert model.for_example(1, rng).tolist() == [5.0, 5.0]


def test_missing_cost_row_is_reported():
    model = CostModel(2, costs=np.ones((2, 2)), lambda_=1.0)
    with pytest.raises(IndexError, match="no cost row"):
        model.for_example(5, np.random.default_rng(0))


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"costs": np.ones((3, 7))}, "shape"),
        ({"costs": -np.ones((3, 4))}, "non-negative"),
        ({"costs": np.ones(4), "lambda_": 0.0}, "positive"),
        ({"costs": np.ones(4), "lambda_": -1.0}, "positive"),
    ],
)
def test_invalid_costs_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        CostModel(4, **kwargs)


def test_simulated_costs_are_non_negative_and_scaled():
    model = CostModel(6, scale=0.05)
    assert model.is_simulated
    drawn = torch.stack(
        [model.for_example(i, np.random.default_rng(i)) for i in range(20)]
    )
    assert (drawn >= 0).all()
    assert drawn.mean() < 0.5  # scale=0.05 around unit mean costs


@pytest.mark.parametrize("suffix", [".pt", ".npy", ".csv"])
def test_costs_round_trip_through_files(tmp_path, suffix):
    costs = np.array([[1.0, 2.0], [3.0, 4.0]])
    path = tmp_path / f"costs{suffix}"
    if suffix == ".pt":
        torch.save(torch.tensor(costs), path)
    elif suffix == ".npy":
        np.save(path, costs)
    else:
        np.savetxt(path, costs, delimiter=",")
    loaded = CostModel(2, costs=load_costs(path), lambda_=1.0)
    assert loaded.for_example(1, np.random.default_rng(0)).tolist() == [3.0, 4.0]


def test_unsupported_cost_file_is_rejected(tmp_path):
    path = tmp_path / "costs.json"
    path.write_text("[]")
    with pytest.raises(ValueError, match="unsupported cost file"):
        load_costs(path)
