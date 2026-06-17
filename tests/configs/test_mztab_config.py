"""Tests for MZTab configuration compatibility."""

from pathlib import Path

from omegaconf import OmegaConf


def test_default_mztab_config_loads_beams_for_default_beam_features() -> None:
    """Default MZTab loading must satisfy the stock beam-dependent features."""
    config_dir = Path(__file__).parents[2] / "winnow" / "configs"
    calibrator_cfg = OmegaConf.load(config_dir / "calibrator.yaml")
    mztab_cfg = OmegaConf.load(config_dir / "data_loader" / "mztab.yaml")

    default_features = calibrator_cfg.calibrator.features
    has_beam_dependent_features = (
        "beam_features" in default_features or "chimeric_features" in default_features
    )

    assert has_beam_dependent_features
    assert mztab_cfg.load_beams is True
