"""Application registry.

An application supplies the one thing the framework cannot infer: how to load
its data. Masking, models, metrics and the whole pipeline come from the config.

    @register_app("my_dataset")
    def load_splits(cfg):
        return {"train": ..., "val": ..., "test": ...}
"""

from __future__ import annotations

import importlib
from typing import Callable

#: name -> callable(Config) -> {"train": source, "val": source, "test": source}
_APPS: dict[str, Callable] = {}

SPLITS = ("train", "val", "test")


def register_app(name: str):
    """Register a split loader under ``name``."""

    def decorator(fn: Callable) -> Callable:
        if name in _APPS:
            raise ValueError(f"application {name!r} is already registered")
        _APPS[name] = fn
        return fn

    return decorator


def get_app(name: str) -> Callable:
    """Look up a registered application, importing the built-ins on demand."""
    if name not in _APPS:
        try:
            importlib.import_module(f"ama.apps.{name}")
        except ModuleNotFoundError:
            pass
    try:
        return _APPS[name]
    except KeyError:
        raise ValueError(
            f"unknown application {name!r}; registered: {sorted(_APPS)}"
        ) from None


def load_splits(cfg) -> dict:
    """Load an application's splits and check it returned all of them."""
    splits = get_app(cfg.app)(cfg)
    missing = set(SPLITS) - set(splits)
    if missing:
        raise ValueError(
            f"application {cfg.app!r} did not return split(s) {sorted(missing)}"
        )
    return splits


def available_apps() -> list[str]:
    return sorted(_APPS)
