"""The toy example carries the README's argument, so its structure is pinned."""

import torch

from ama.apps.toy_sensors import (
    INFORMATIVE_BY_TYPE,
    N_SENSORS,
    SENSOR_WIDTH,
    make_split,
)


def machine_types(x):
    """The free sensor encodes the type; positive means type 1."""
    return (x[:, 0] > 0).long()


def test_the_free_sensor_identifies_the_machine_type():
    x, _ = make_split(2000, torch.Generator().manual_seed(0))
    # Two well separated groups, both well represented.
    types = machine_types(x)
    assert 0.4 < types.float().mean() < 0.6
    assert x[types == 1, 0].min() > x[types == 0, 0].max()


def test_the_free_sensor_does_not_reveal_the_fault():
    """Context says where to look, never what the answer is."""
    x, y = make_split(4000, torch.Generator().manual_seed(0))
    for type_id in (0, 1):
        rows = machine_types(x) == type_id
        # Within a type, the label is a coin flip until you acquire something.
        assert 0.4 < y[rows].float().mean() < 0.6


def test_each_type_is_determined_by_its_own_sensors():
    """The label is a function of that type's informative sensors alone."""
    x, y = make_split(4000, torch.Generator().manual_seed(0))
    types = machine_types(x)
    for type_id, sensors in INFORMATIVE_BY_TYPE.items():
        rows = types == type_id
        signal = sum(
            x[rows][:, s * SENSOR_WIDTH : (s + 1) * SENSOR_WIDTH].sum(dim=1)
            for s in sensors
        )
        assert torch.equal((signal > 0).long(), y[rows])


def test_the_two_types_need_different_sensors():
    """Without this the example would not motivate adaptive acquisition."""
    assert set(INFORMATIVE_BY_TYPE[0]).isdisjoint(INFORMATIVE_BY_TYPE[1])


def test_shapes_line_up_with_the_configured_masker():
    x, y = make_split(16, torch.Generator().manual_seed(0))
    assert x.shape == (16, N_SENSORS * SENSOR_WIDTH)
    assert y.shape == (16,)
