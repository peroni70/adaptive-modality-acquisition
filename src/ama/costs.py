"""Acquisition costs.

Two situations, one interface.

*Studying* the method means sweeping cost levels to trace out how a policy
behaves as acquisition gets more expensive, so costs are simulated.

*Using* the method means the costs are real and already known - a dollar
figure per test, a latency budget, a radiation dose - and may differ per
subject. Those are supplied directly.

Either way the policy needs value and cost in the same units before it can
compare them, which is what ``lambda_`` does. See :class:`CostModel`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor


class CostModel:
    """Per-example acquisition costs, converted into value units.

    The subset optimizers maximize ``v(Q) - cost(Q)``, where ``v`` is measured
    in whatever the value function predicts. For ``acc_change`` that is a
    change in expected accuracy, so a raw cost in dollars cannot be subtracted
    from it directly. ``lambda_`` is the exchange rate that makes the two
    commensurable:

        effective_cost = raw_cost / lambda_

    with ``lambda_`` read as **how much you are willing to spend per unit of
    expected accuracy gained** - dollars per unit accuracy, in the same
    currency as ``costs``. A large ``lambda_`` means accuracy is worth a lot
    and the policy buys freely; a small one means it acquires only when the
    expected gain is large.

    Args:
        costs: ``None`` to simulate; a ``(n_modes,)`` vector shared by every
            example; or an ``(n_examples, n_modes)`` matrix of per-example
            costs.
        lambda_: exchange rate, in cost units per unit of value.
        mean_costs, cov_costs: distribution to simulate from when ``costs`` is
            ``None``.
        scale: multiplies simulated costs. This is the sweep knob; it is the
            reciprocal of ``lambda_`` and only one of the two should be set.
    """

    def __init__(
        self,
        n_modes: int,
        costs=None,
        lambda_: float | None = None,
        mean_costs=None,
        cov_costs=None,
        scale: float = 1.0,
        device: str = "cpu",
    ):
        if lambda_ is not None and lambda_ <= 0:
            raise ValueError(f"lambda_ must be positive, got {lambda_}")
        self.n_modes = n_modes
        self.device = device
        self.lambda_ = lambda_
        # lambda_ and scale express the same thing from opposite directions.
        self.scale = (1.0 / lambda_) if lambda_ is not None else float(scale)

        self.costs = None if costs is None else self._as_matrix(costs, n_modes)
        self.mean_costs = np.ones(n_modes) if mean_costs is None else np.asarray(mean_costs)
        self.cov_costs = np.eye(n_modes) if cov_costs is None else np.asarray(cov_costs)

    @staticmethod
    def _as_matrix(costs, n_modes: int) -> Tensor:
        costs = torch.as_tensor(costs, dtype=torch.float32)
        if costs.ndim == 1:
            costs = costs.unsqueeze(0)
        if costs.ndim != 2 or costs.shape[1] != n_modes:
            raise ValueError(
                f"costs must have shape (n_modes,) or (n_examples, n_modes) "
                f"with n_modes={n_modes}, got {tuple(costs.shape)}"
            )
        if (costs < 0).any():
            raise ValueError("costs must be non-negative")
        return costs

    @property
    def is_simulated(self) -> bool:
        return self.costs is None

    def for_example(self, idx: int, rng) -> Tensor:
        """Effective cost vector for one example, already in value units."""
        if self.costs is None:
            raw = rng.multivariate_normal(self.mean_costs, self.cov_costs)
            raw = torch.tensor(np.clip(raw, 0.0, None), dtype=torch.float32)
        elif self.costs.shape[0] == 1:
            raw = self.costs[0]
        else:
            if idx >= self.costs.shape[0]:
                raise IndexError(
                    f"no cost row for example {idx}: costs has "
                    f"{self.costs.shape[0]} rows"
                )
            raw = self.costs[idx]
        return (raw * self.scale).to(self.device)

    def __repr__(self) -> str:
        kind = "simulated" if self.is_simulated else f"given{tuple(self.costs.shape)}"
        return f"CostModel({kind}, scale={self.scale:g})"


def load_costs(path: str | Path):
    """Read a cost vector or matrix from ``.pt``, ``.npy`` or ``.csv``."""
    path = Path(path)
    if path.suffix == ".pt":
        return torch.load(path, map_location="cpu")
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix in {".csv", ".txt"}:
        return np.loadtxt(path, delimiter="," if path.suffix == ".csv" else None)
    raise ValueError(f"unsupported cost file {path.suffix!r}; use .pt, .npy or .csv")
