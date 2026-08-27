import pytest
import torch

from ama.masking import (
    ChannelMasker,
    Masker,
    PatchMasker,
    SliceMasker,
    build_masker,
)


def test_grid_tiles_the_image():
    masker = PatchMasker.grid(image_size=28, patch_size=7)
    assert masker.n_modes == 16
    # Every pixel belongs to exactly one patch, so keeping all keeps everything.
    x = torch.ones(1, 28, 28)
    assert masker.mask(x, torch.ones(16, dtype=bool)).sum() == 28 * 28
    assert masker.mask(x, torch.zeros(16, dtype=bool)).sum() == 0


def test_grid_rejects_indivisible_sizes():
    with pytest.raises(ValueError, match="divisible"):
        PatchMasker.grid(image_size=28, patch_size=5)


@pytest.mark.parametrize(
    "masker,shape,per_mode",
    [
        (PatchMasker.grid(28, 7), (1, 28, 28), 49),
        (SliceMasker([[0, 30], [30, 60]]), (60,), 30),
        (ChannelMasker(12), (12, 100), 100),
    ],
)
def test_keeping_one_modality_keeps_its_region(masker, shape, per_mode):
    x = torch.ones(shape)
    assert masker.keep_only(x, [0]).sum() == per_mode


@pytest.mark.parametrize(
    "masker,shape",
    [
        (PatchMasker.grid(28, 7), (1, 28, 28)),
        (SliceMasker([[0, 30], [30, 60]]), (60,)),
        (ChannelMasker(12), (12, 100)),
    ],
)
def test_masking_is_batch_safe(masker, shape):
    x = torch.rand((5,) + shape)
    keep = masker.indicator([0])
    batched = masker.mask(x, keep)
    for i in range(5):
        assert torch.equal(batched[i], masker.mask(x[i], keep))


def test_mask_does_not_mutate_input(masker_x=None):
    masker = PatchMasker.grid(28, 7)
    x = torch.ones(1, 28, 28)
    masker.keep_only(x, [0])
    assert x.sum() == 28 * 28


def test_build_masker_from_config():
    masker = build_masker({"type": "patch", "grid": {"image_size": 28, "patch_size": 7}})
    assert isinstance(masker, PatchMasker) and masker.n_modes == 16
    with pytest.raises(ValueError, match="unknown masker type"):
        build_masker({"type": "nope"})


def test_custom_masker_needs_only_a_region(masker_cls=None):
    """The extension point documented in the README: subclass, define region."""

    class InterleavedMasker(Masker):
        def region(self, mode):
            return (..., slice(mode, None, self.n_modes))

    masker = InterleavedMasker(8)
    kept = masker.keep_only(torch.arange(32.0), [2])
    assert torch.nonzero(kept).view(-1).tolist() == [2, 10, 18, 26]
    # The eight modalities partition the input, so nothing is unhideable.
    claimed = sum(
        int(masker.keep_only(torch.ones(32), [i]).sum()) for i in range(8)
    )
    assert claimed == 32
