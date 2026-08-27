import torch

from ama.modeling import ModalitySetEmbedding, ValueModel, ConvEncoder
from ama.value_fns import get_value_fn


def test_set_embedding_averages_over_members():
    embedding = ModalitySetEmbedding(4, 8)
    members = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    expected = (embedding.weight[0] + embedding.weight[1]) / 2
    assert torch.allclose(embedding(members)[0], expected, atol=1e-6)


def test_set_embedding_normalizes_by_cardinality_not_magnitude():
    """A set's embedding must scale with |S|, independent of the vectors' values."""
    embedding = ModalitySetEmbedding(4, 8)
    with torch.no_grad():
        embedding.embedding.weight.fill_(2.0)
    one = embedding(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    two = embedding(torch.tensor([[1.0, 1.0, 0.0, 0.0]]))
    assert torch.allclose(one, two)
    assert torch.allclose(one, torch.full_like(one, 2.0))


def test_empty_set_embeds_to_zero_not_nan():
    embedding = ModalitySetEmbedding(4, 8)
    out = embedding(torch.zeros(1, 4))
    assert torch.isfinite(out).all() and out.abs().sum() == 0


def test_value_model_output_width_follows_the_value_function():
    for name, width in [("bit_flip", 2), ("acc_change", 3), ("info_gain", 1)]:
        model = ValueModel(
            ConvEncoder(28), 16, get_value_fn(name), hidden=[16, 8], mode_emb_dim=4
        ).eval()
        x = torch.randn(2, 1, 28, 28)
        sets = torch.zeros(2, 16)
        assert model(x, sets, sets).shape == (2, width)


def test_late_fusion_runs():
    model = ValueModel(
        ConvEncoder(28), 16, get_value_fn("bit_flip"),
        hidden=[16, 8], mode_emb_dim=4, fusion="late",
    ).eval()
    x = torch.randn(2, 1, 28, 28)
    sets = torch.zeros(2, 16)
    assert model(x, sets, sets).shape == (2, 2)
