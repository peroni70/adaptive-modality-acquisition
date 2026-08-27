"""Constructing models and datasets from a config.

Keeps the mapping from config fields to objects in one place, so the pipeline
stages read as a sequence of steps rather than a sequence of constructors.
"""

from __future__ import annotations

import math
import os

import torch
from torch.utils.data import DataLoader

from .config import Config, EncoderConfig, ModelConfig, ValueModelConfig
from .data import MaskedDataset, ValueDataset
from .masking import Masker, build_masker
from .metrics import ClassificationMetrics, build_metrics
from .modeling import ConvEncoder, Encoder, FlatEncoder, MLPEncoder, MaskedClassifier, ValueModel
from .value_fns import ValueFunction

ENCODERS = {"conv": ConvEncoder, "mlp": MLPEncoder, "flat": FlatEncoder}

#: Beyond this, more target-generation workers stop paying for themselves.
MAX_AUTO_WORKERS = 8


def available_cpus() -> int:
    """CPUs this process may actually use, not what the machine has.

    Under a batch scheduler the two differ: the job is pinned to its
    allocation, and sizing a worker pool by the host's core count oversubscribes
    it badly.
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        return os.cpu_count() or 1


def resolve_workers(value) -> int:
    """Resolve a ``num_workers`` setting, which may be "auto"."""
    if isinstance(value, str):
        if value != "auto":
            raise ValueError(f"num_workers must be an int or 'auto', got {value!r}")
        # Leave a core for the training loop itself.
        return max(0, min(MAX_AUTO_WORKERS, available_cpus() - 1))
    return int(value)


def build_encoder(spec: EncoderConfig, example_shape) -> Encoder:
    """Instantiate an encoder, inferring input size from an example when needed."""
    try:
        cls = ENCODERS[spec.type]
    except KeyError:
        raise ValueError(
            f"unknown encoder {spec.type!r}; expected one of {sorted(ENCODERS)}"
        ) from None
    options = dict(spec.options)
    if cls is ConvEncoder:
        options.setdefault("image_size", int(example_shape[-1]))
        options.setdefault("in_channels", int(example_shape[0]))
    else:
        options.setdefault("in_dim", int(math.prod(example_shape)))
    return cls(**options)


def build_classifier(
    cfg: ModelConfig, masker: Masker, metrics: ClassificationMetrics, example_shape
) -> MaskedClassifier:
    return MaskedClassifier(
        encoder=build_encoder(cfg.encoder, example_shape),
        n_modes=masker.n_modes,
        n_classes=metrics.n_classes,
        hidden=cfg.hidden,
        mode_emb_dim=cfg.mode_emb_dim,
        fusion=cfg.fusion,
        dropout=cfg.dropout,
    )


def build_value_model(
    cfg: ValueModelConfig, masker: Masker, value_fn: ValueFunction, example_shape
) -> ValueModel:
    return ValueModel(
        encoder=build_encoder(cfg.encoder, example_shape),
        n_modes=masker.n_modes,
        value_fn=value_fn,
        hidden=cfg.hidden,
        mode_emb_dim=cfg.mode_emb_dim,
        fusion=cfg.fusion,
        dropout=cfg.dropout,
    )


def example_shape(source) -> tuple[int, ...]:
    """Shape of a single example, used to size encoders."""
    x, _ = source[0]
    return tuple(x.shape)


def masked_loaders(splits, masker, cfg: Config) -> dict[str, DataLoader]:
    """Loaders of randomly masked examples, for classifier training."""
    batch = cfg.classifier.batch_size
    loaders = {}
    for name, source in splits.items():
        train = name == "train"
        dataset = MaskedDataset(
            source, masker, cfg.context_idx, deterministic=not train
        )
        loaders[name] = DataLoader(dataset, batch_size=batch, shuffle=train)
    return loaders


def value_loaders(
    splits, masker, classifier, value_fn: ValueFunction, cfg: Config
) -> dict[str, DataLoader]:
    """Loaders of (state, proposal, value) triples, for value-model training."""
    vcfg = cfg.value_model
    workers = resolve_workers(vcfg.num_workers)
    loaders = {}
    for name, source in splits.items():
        train = name == "train"
        dataset = ValueDataset(
            source,
            masker,
            classifier,
            cfg.context_idx,
            value_fn,
            n_repeats=int(vcfg.n_repeats.get(name, 1)),
            deterministic=not train,
            include_state=vcfg.include_state,
            device=vcfg.target_device,
        )
        loaders[name] = DataLoader(
            dataset,
            batch_size=vcfg.batch_size,
            shuffle=train,
            num_workers=workers,
            persistent_workers=workers > 0,
        )
    return loaders


def build_masker_and_metrics(cfg: Config) -> tuple[Masker, ClassificationMetrics]:
    return build_masker(cfg.masker), build_metrics(cfg.metrics)
