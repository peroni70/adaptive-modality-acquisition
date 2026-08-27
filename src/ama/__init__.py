"""Adaptive modality acquisition.

Learns a value model that scores a proposed subset of modalities against what
has already been observed, and uses it to acquire information in stages,
stopping when nothing left is worth its cost.

The pieces a new application supplies are its data loader and, if its layout
is unusual, a masker. Everything else - value functions, subset optimizers,
the acquisition loop, evaluation - is shared.
"""

from .config import Config, load_config
from .masking import ChannelMasker, Masker, PatchMasker, SliceMasker, build_masker
from .metrics import ClassificationMetrics
from .optimizers import OPTIMIZERS, optimize_subset
from .registry import available_apps, register_app
from .value_fns import VALUE_FUNCTIONS, ValueFunction, get_value_fn

__version__ = "0.1.0"

__all__ = [
    "OPTIMIZERS",
    "VALUE_FUNCTIONS",
    "ChannelMasker",
    "ClassificationMetrics",
    "Config",
    "Masker",
    "PatchMasker",
    "SliceMasker",
    "ValueFunction",
    "available_apps",
    "build_masker",
    "get_value_fn",
    "load_config",
    "optimize_subset",
    "register_app",
]
