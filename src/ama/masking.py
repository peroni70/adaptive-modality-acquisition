"""Modality masking.

The masker is the one component that is genuinely tied to how a dataset lays
out its modalities.  Everything else in this package is data agnostic.

A masker answers a single question: given one example and a boolean vector
saying which modalities are observed, what does the model actually see?
Unobserved modalities are zeroed out; regions of the input that no modality
claims are left untouched.

Subclassing only requires ``region``:

    class MyMasker(Masker):
        def region(self, mode):
            return (slice(None), slice(self.offsets[mode], self.offsets[mode + 1]))
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import torch
from torch import Tensor


class Masker(ABC):
    """Zero out the parts of an example belonging to unobserved modalities."""

    def __init__(self, n_modes: int):
        if n_modes < 1:
            raise ValueError(f"n_modes must be positive, got {n_modes}")
        self.n_modes = n_modes

    @abstractmethod
    def region(self, mode: int) -> tuple:
        """Index expression selecting modality ``mode`` of a single example."""

    def mask(self, x: Tensor, keep) -> Tensor:
        """Return a copy of ``x`` with every modality not in ``keep`` zeroed.

        Works on a single example or on a batch: ``region`` indexes trailing
        axes, so leading batch dimensions pass through untouched.
        """
        out = x.clone()
        for mode in range(self.n_modes):
            if not bool(keep[mode]):
                out[self.region(mode)] = 0
        return out

    def keep_only(self, x: Tensor, modes) -> Tensor:
        """Mask ``x`` down to the given iterable of modality indices."""
        return self.mask(x, self.indicator(modes))

    def indicator(self, modes) -> Tensor:
        """Boolean ``(n_modes,)`` vector for an iterable of modality indices."""
        keep = torch.zeros(self.n_modes, dtype=torch.bool)
        for mode in modes:
            keep[mode] = True
        return keep

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n_modes={self.n_modes})"


class SliceMasker(Masker):
    """Modalities are contiguous spans of a flat feature vector.

    ``spans[i] == (start, end)`` gives the half-open range owned by modality
    ``i``.  Suits datasets whose modalities have been concatenated into one
    vector, e.g. per-modality embeddings laid end to end.
    """

    def __init__(self, spans: Sequence[Sequence[int]]):
        super().__init__(len(spans))
        self.spans = [(int(s), int(e)) for s, e in spans]
        for start, end in self.spans:
            if end <= start:
                raise ValueError(f"span end must exceed start, got ({start}, {end})")

    def region(self, mode: int) -> tuple:
        start, end = self.spans[mode]
        return (..., slice(start, end))


class PatchMasker(Masker):
    """Modalities are rectangular patches of a ``(C, H, W)`` image.

    ``boxes[i] == (row_start, row_end, col_start, col_end)``.
    """

    def __init__(self, boxes: Sequence[Sequence[int]]):
        super().__init__(len(boxes))
        self.boxes = [tuple(int(v) for v in box) for box in boxes]
        for box in self.boxes:
            if len(box) != 4:
                raise ValueError(f"box must be (r0, r1, c0, c1), got {box}")

    def region(self, mode: int) -> tuple:
        r0, r1, c0, c1 = self.boxes[mode]
        return (..., slice(r0, r1), slice(c0, c1))

    @classmethod
    def grid(cls, image_size: int, patch_size: int) -> "PatchMasker":
        """Tile a square image into a regular grid of patches, row-major."""
        if image_size % patch_size:
            raise ValueError(
                f"image_size {image_size} is not divisible by patch_size {patch_size}"
            )
        n = image_size // patch_size
        boxes = [
            (i * patch_size, (i + 1) * patch_size, j * patch_size, (j + 1) * patch_size)
            for i in range(n)
            for j in range(n)
        ]
        return cls(boxes)


class ChannelMasker(Masker):
    """Modalities are entries along the modality axis, e.g. ECG leads ``(M, T)``."""

    def region(self, mode: int) -> tuple:
        return (..., mode, slice(None))


def build_masker(spec: dict) -> Masker:
    """Construct a masker from a config mapping."""
    spec = dict(spec)
    kind = spec.pop("type", None)
    if kind == "slice":
        return SliceMasker(spec["spans"])
    if kind == "patch":
        if "grid" in spec:
            return PatchMasker.grid(**spec["grid"])
        return PatchMasker(spec["boxes"])
    if kind == "channel":
        return ChannelMasker(spec["n_modes"])
    raise ValueError(f"unknown masker type {kind!r}; expected slice, patch or channel")
