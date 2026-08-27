"""Experiment configuration.

One YAML file describes an application end to end. Anything a field does not
cover is a genuine difference in the data or the model, and belongs in the
application module rather than here.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from pathlib import Path
import yaml


@dataclass
class EncoderConfig:
    """Which feature encoder to build, and how."""

    type: str = "conv"
    #: Passed through to the encoder constructor.
    options: dict = field(default_factory=dict)


@dataclass
class ModelConfig:
    """A classifier or value model: encoder, trunk, and training schedule."""

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    hidden: list[int] = field(default_factory=lambda: [128, 64, 32])
    mode_emb_dim: int = 1028
    fusion: str = "early"
    dropout: float = 0.0
    epochs: int = 15
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 64


@dataclass
class ValueModelConfig(ModelConfig):
    """Value model, plus the sampling and regularization it needs."""

    hidden: list[int] = field(default_factory=lambda: [256, 128, 64])
    weight_decay: float = 1e-5
    batch_size: int = 32
    epochs: int = 100
    #: Distinct (state, proposal) draws per example, per split.
    n_repeats: dict = field(
        default_factory=lambda: {"train": 1, "val": 2, "test": 1}
    )
    use_hinge_loss: bool = False
    hinge_lam: float = 1e-3
    calibrate: bool = True
    calibration_epochs: int = 10
    #: Blank the observation, to ablate the value model's access to the input.
    include_state: bool = True
    #: Where the frozen classifier generates value targets. Targets are computed
    #: one example at a time, so "cpu" plus workers usually beats a GPU here.
    target_device: str = "cpu"
    #: DataLoader workers generating value targets in parallel. "auto" scales
    #: to the CPUs actually available to the process, which is what a batch
    #: scheduler allocates rather than what the machine physically has.
    num_workers: int | str = 0


@dataclass
class PolicyConfig:
    """The evaluation sweep: which value functions, costs and optimizers."""

    value_fns: list[str] = field(default_factory=lambda: ["acc_change", "bit_flip"])
    alphas: list[float] = field(
        default_factory=lambda: [0.0, 0.01, 0.03, 0.05, 0.07, 0.1]
    )
    optimizers: list[str] = field(
        default_factory=lambda: [
            "rand_usm",
            "greedy_usm",
            "hybrid_usm",
            "single_item_greedy",
        ]
    )
    #: Also run acc_change with the value score negated. A negative control:
    #: a policy that trusts the value model should then acquire nothing.
    include_reverse: bool = False
    #: Also run the cost-aware fixed-order baseline.
    include_adaptive_greedy: bool = True
    #: Evaluate calibrated as well as uncalibrated value models.
    include_calibrated: bool = True
    #: Path to real acquisition costs: a (n_modes,) vector or an
    #: (n_examples, n_modes) matrix, as .pt, .npy or .csv. When set, costs are
    #: taken from here instead of simulated, and ``alphas`` is ignored.
    costs_path: str | None = None
    #: Exchange rate for real costs: how much you will spend per unit of
    #: expected accuracy gained. Required when costs_path is set.
    cost_lambda: float | None = None
    batch_size: int = 64
    single_stage: bool = False


@dataclass
class Config:
    """A complete experiment."""

    app: str
    context_idx: int = 0
    seed: int = 0
    device: str = "auto"
    run_dir: str = "runs"
    data: dict = field(default_factory=dict)
    masker: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    classifier: ModelConfig = field(default_factory=ModelConfig)
    value_model: ValueModelConfig = field(default_factory=ValueModelConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)

    @property
    def run_path(self) -> Path:
        return Path(self.run_dir).expanduser()

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _coerce(text: str):
    """Parse a CLI override value using YAML rules, so types survive."""
    return yaml.safe_load(text)


def apply_overrides(data: dict, overrides) -> dict:
    """Apply ``a.b=value`` strings onto a nested mapping."""
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"override must look like key=value, got {override!r}")
        path, raw = override.split("=", 1)
        node = data
        keys = path.split(".")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
            if not isinstance(node, dict):
                raise ValueError(f"cannot descend into {path!r}: {key!r} is not a table")
        node[keys[-1]] = _coerce(raw)
    return data


def load_config(path: str | Path, overrides=None) -> Config:
    """Read a YAML config, apply overrides, and validate the result."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if "app" not in raw:
        raise ValueError(f"{path}: config must name an 'app'")
    raw = apply_overrides(raw, overrides)
    return _build_config(raw)


def _build_config(raw: dict) -> Config:
    typed = {
        "classifier": ModelConfig,
        "value_model": ValueModelConfig,
        "policy": PolicyConfig,
    }
    kwargs = dict(raw)
    for key, cls in typed.items():
        if key in kwargs:
            section = dict(kwargs[key])
            if "encoder" in section:
                section["encoder"] = EncoderConfig(**section["encoder"])
            _check_keys(cls, section)
            kwargs[key] = cls(**section)
    _check_keys(Config, kwargs)
    return Config(**kwargs)


def _check_keys(cls, data: dict) -> None:
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f"unknown config key(s) for {cls.__name__}: {sorted(unknown)}; "
            f"expected {sorted(known)}"
        )
