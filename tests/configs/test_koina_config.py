"""Tests for the shared Koina configuration."""

from pathlib import Path

from omegaconf import OmegaConf

CONFIG_DIR = Path(__file__).parents[2] / "winnow" / "configs"


def test_koina_config_defaults_to_the_public_endpoint():
    """The default server is the public Koina endpoint, over TLS."""
    cfg = OmegaConf.load(CONFIG_DIR / "koina.yaml")

    assert cfg.koina.server_url == "koina.wilhelmlab.org:443"
    assert cfg.koina.ssl is True


def test_calibrator_interpolates_the_koina_server_into_koina_features():
    """Features calling Koina take their server from the shared block."""
    cfg = OmegaConf.load(CONFIG_DIR / "calibrator.yaml")
    features = cfg.calibrator.features

    koina_features = [
        name
        for name, feature in features.items()
        if feature is not None and "koina_server_url" in feature
    ]
    assert koina_features, "expected at least one Koina-backed feature"

    for name in koina_features:
        assert (
            features[name]._get_node("koina_server_url")._value()
            == "${koina.server_url}"
        ), name
        assert features[name]._get_node("koina_ssl")._value() == "${koina.ssl}", name


def test_every_feature_naming_a_koina_model_also_names_a_server():
    """A feature that calls a Koina model must say which server to call."""
    cfg = OmegaConf.load(CONFIG_DIR / "calibrator.yaml")

    for name, feature in cfg.calibrator.features.items():
        if feature is None:
            continue
        keys = set(feature.keys())
        if {"intensity_model_name", "irt_model_name"} & keys:
            assert "koina_server_url" in keys, name
            assert "koina_ssl" in keys, name
