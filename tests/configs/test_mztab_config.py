"""Tests for MZTab configuration compatibility."""

from pathlib import Path

from omegaconf import OmegaConf


def test_default_mztab_config_loads_beams_when_calibrator_needs_them() -> None:
    """Beam-dependent calibrator features require MZTab beam loading to be enabled."""
    config_dir = Path(__file__).parents[2] / "winnow" / "configs"
    calibrator_cfg = OmegaConf.load(config_dir / "calibrator.yaml")
    mztab_cfg = OmegaConf.load(config_dir / "data_loader" / "mztab.yaml")

    default_features = calibrator_cfg.calibrator.features
    needs_beams = (
        "beam_features" in default_features or "chimeric_features" in default_features
    )

    assert not needs_beams or mztab_cfg.load_beams is True
