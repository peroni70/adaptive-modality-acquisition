import pytest

from ama.config import apply_overrides, load_config

CONFIG = "configs/patch_mnist.yaml"


def test_shipped_config_loads():
    cfg = load_config(CONFIG)
    assert cfg.app == "patch_mnist"
    assert cfg.context_idx == 10
    assert cfg.classifier.epochs == 15
    assert cfg.value_model.use_hinge_loss is True


def test_overrides_preserve_types():
    cfg = load_config(
        CONFIG, ["value_model.epochs=5", "policy.alphas=[0.0,0.5]", "device=cpu"]
    )
    assert cfg.value_model.epochs == 5
    assert cfg.policy.alphas == [0.0, 0.5]
    assert cfg.resolve_device() == "cpu"


def test_misspelled_key_is_rejected():
    with pytest.raises(ValueError, match="unknown config key"):
        load_config(CONFIG, ["value_model.epocs=5"])


def test_override_without_equals_is_rejected():
    with pytest.raises(ValueError, match="key=value"):
        apply_overrides({}, ["novalue"])
