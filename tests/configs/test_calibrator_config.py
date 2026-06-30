"""Tests for the default calibrator configuration."""

from pathlib import Path

from omegaconf import OmegaConf


def test_default_calibrator_keeps_token_score_features_opt_in():
    """Test token score features are not enabled in the default calibrator."""
    config_path = Path(__file__).parents[2] / "winnow" / "configs" / "calibrator.yaml"
    cfg = OmegaConf.load(config_path)

    assert "token_score_features" not in cfg.calibrator.features


def test_default_calibrator_keeps_peptide_language_model_features_opt_in():
    """Test peptide language model features are not enabled by default."""
    config_path = Path(__file__).parents[2] / "winnow" / "configs" / "calibrator.yaml"
    cfg = OmegaConf.load(config_path)

    assert "peptide_language_model_feature" not in cfg.calibrator.features
