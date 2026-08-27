"""A synthetic multi-sensor dataset, used as the worked example in the README.

Eight sensors report on a machine. Sensor 0 is free - the ambient reading -
and the other seven cost money to query.

The machines come in two types, and the free reading says which. On a type A
machine the fault shows up in sensors 2 and 5; on a type B machine it shows up
in sensors 3 and 4. Sensors 1, 6 and 7 are noise for everyone.

This is the point of the example. No single fixed set of sensors is efficient:
a population-level ranking has to buy all four informative sensors to cover
both machine types, paying twice over on every unit. A policy that reads the
free sensor first knows which type it is looking at, and buys only the two
that matter. Same accuracy, half the bill - and no ranking of sensors, however
good, can do this, because the right answer depends on the case.

Small enough to train end to end on a laptop CPU in a few minutes, and it
needs no downloads.
"""

from __future__ import annotations

import torch

from ..data.sources import TensorSource
from ..registry import register_app

#: Readings contributed by each sensor. Modality i owns columns
#: [i * SENSOR_WIDTH, (i + 1) * SENSOR_WIDTH).
SENSOR_WIDTH = 4
N_SENSORS = 8
#: Which sensors carry the fault signal, by machine type.
INFORMATIVE_BY_TYPE = {0: (2, 5), 1: (3, 4)}
#: How loudly the free sensor announces the machine type.
TYPE_SIGNAL = 3.0


def _block(x: torch.Tensor, sensor: int) -> torch.Tensor:
    return x[:, sensor * SENSOR_WIDTH : (sensor + 1) * SENSOR_WIDTH]


def make_split(n: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw ``n`` machines: their sensor readings and whether they are faulty."""
    x = torch.randn(n, N_SENSORS * SENSOR_WIDTH, generator=generator)
    machine_type = torch.randint(0, 2, (n,), generator=generator)

    # The free sensor identifies the machine type, and nothing else. Knowing
    # the type tells you where to look, never whether the machine is faulty.
    x[:, 0] = TYPE_SIGNAL * (2.0 * machine_type.float() - 1.0) + 0.3 * x[:, 0]

    y = torch.zeros(n, dtype=torch.long)
    for type_id, sensors in INFORMATIVE_BY_TYPE.items():
        rows = machine_type == type_id
        signal = sum(_block(x, sensor)[rows].sum(dim=1) for sensor in sensors)
        y[rows] = (signal > 0).long()
    return x, y


@register_app("toy_sensors")
def load_splits(cfg) -> dict:
    """Generate train / val / test splits of the synthetic sensor data."""
    sizes = cfg.data.get("sizes", {"train": 8000, "val": 2000, "test": 2000})
    generator = torch.Generator().manual_seed(cfg.seed)
    return {
        split: TensorSource(*make_split(int(sizes[split]), generator))
        for split in ("train", "val", "test")
    }
