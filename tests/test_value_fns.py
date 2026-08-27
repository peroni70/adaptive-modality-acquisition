import pytest
import torch

from ama.value_fns import VALUE_FUNCTIONS, get_value_fn

# Logits that predict class 1 confidently, and class 0 confidently.
RIGHT = torch.tensor([[0.0, 9.0, 0.0]])
WRONG = torch.tensor([[9.0, 0.0, 0.0]])
Y = torch.tensor([1])


@pytest.mark.parametrize("name", sorted(VALUE_FUNCTIONS))
def test_head_dim_matches_score_shape(name):
    value_fn = get_value_fn(name)
    logits = torch.randn(4, value_fn.head_dim)
    assert value_fn.score(logits).shape == (4,)


def test_bit_flip_fires_only_on_wrong_to_right():
    vf = get_value_fn("bit_flip")
    assert vf.target(WRONG, RIGHT, Y) == 1
    assert vf.target(RIGHT, WRONG, Y) == 0
    assert vf.target(RIGHT, RIGHT, Y) == 0


def test_acc_change_is_three_way():
    vf = get_value_fn("acc_change")
    assert vf.target(WRONG, RIGHT, Y) == 2  # gained
    assert vf.target(RIGHT, RIGHT, Y) == 1  # unchanged
    assert vf.target(RIGHT, WRONG, Y) == 0  # lost


def test_acc_change_score_is_signed_and_reversible():
    vf = get_value_fn("acc_change")
    gain = torch.tensor([[0.0, 0.0, 9.0]])
    assert vf.score(gain).item() > 0.9
    assert vf.score(gain, reverse=True).item() < -0.9


def test_info_gain_target_is_loss_reduction():
    vf = get_value_fn("info_gain")
    assert vf.target(WRONG, RIGHT, Y) > 0
    assert vf.target(RIGHT, WRONG, Y) < 0
    assert vf.higher_is_better is False


def test_info_gain_loss_does_not_broadcast():
    """The loss the trainer actually calls must not expand (B, 1) to (B, B)."""
    vf = get_value_fn("info_gain")
    logits = torch.zeros(8, 1)          # the value model's real output shape
    targets = torch.ones(8)
    loss = vf.compute_loss(vf.loss(), logits, targets)
    assert loss.shape == ()
    assert loss.item() == pytest.approx(1.0)


def test_info_gain_loss_rewards_a_perfect_prediction():
    """A perfect prediction must score zero; broadcasting would penalize it."""
    vf = get_value_fn("info_gain")
    logits = torch.tensor([[0.0], [10.0]])
    targets = torch.tensor([0.0, 10.0])
    assert vf.compute_loss(vf.loss(), logits, targets).item() == pytest.approx(0.0)


@pytest.mark.parametrize("name", sorted(VALUE_FUNCTIONS))
def test_every_value_function_takes_raw_model_output(name):
    """compute_loss accepts logits exactly as the value model emits them."""
    vf = get_value_fn(name)
    logits = torch.randn(8, vf.head_dim)
    targets = (
        torch.randn(8) if not vf.higher_is_better
        else torch.randint(0, vf.head_dim, (8,))
    )
    assert vf.compute_loss(vf.loss(), logits, targets).shape == ()


def test_unknown_name_is_rejected():
    with pytest.raises(ValueError, match="unknown value function"):
        get_value_fn("nope")

