import numpy as np
import pytest
import torch

from ama.data.sources import TensorSource
from ama.masking import PatchMasker
from ama.modeling import ConvEncoder, MaskedClassifier, ValueModel
from ama.value_fns import get_value_fn


@pytest.fixture
def masker():
    return PatchMasker.grid(image_size=28, patch_size=7)


@pytest.fixture
def source():
    torch.manual_seed(0)
    return TensorSource(torch.randn(16, 1, 28, 28), torch.randint(0, 10, (16,)))


@pytest.fixture
def classifier(masker):
    torch.manual_seed(0)
    return MaskedClassifier(
        ConvEncoder(28), masker.n_modes, 10, hidden=[32, 16], mode_emb_dim=8
    ).eval()


@pytest.fixture
def value_model(masker):
    torch.manual_seed(0)
    return ValueModel(
        ConvEncoder(28),
        masker.n_modes,
        get_value_fn("acc_change"),
        hidden=[32, 16],
        mode_emb_dim=8,
    ).eval()


@pytest.fixture
def rng():
    return np.random.default_rng(0)
