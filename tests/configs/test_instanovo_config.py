"""Tests for InstaNovo configuration compatibility."""

from pathlib import Path

from omegaconf import OmegaConf


def test_default_instanovo_config_loads_beams_when_calibrator_needs_them() -> None:
    """Beam-dependent calibrator features require InstaNovo beam loading to be enabled."""
    config_dir = Path(__file__).parents[2] / "winnow" / "configs"
    calibrator_cfg = OmegaConf.load(config_dir / "calibrator.yaml")
    instanovo_cfg = OmegaConf.load(config_dir / "data_loader" / "instanovo.yaml")

    default_features = calibrator_cfg.calibrator.features
    needs_beams = (
        "beam_features" in default_features or "chimeric_features" in default_features
    )

    loads_beams = instanovo_cfg.get("beam_columns") is not None

    assert not needs_beams or loads_beams
