"""Tests for Koina intensity model input configuration helpers."""

import pytest
import typer

from winnow.utils.koina_intensity_config import (
    resolve_feature_model_inputs,
    strip_runtime_keys_from_feature_config,
    validate_koina_intensity_config,
)


def test_resolve_feature_model_inputs_defaults_columns():
    """Null constants/columns resolve to default metadata column names."""
    constants, columns = resolve_feature_model_inputs(
        {"collision_energies": None, "fragmentation_types": None},
        {},
    )
    assert constants is None
    assert columns == {
        "collision_energies": "collision_energy",
        "fragmentation_types": "frag_type",
    }


def test_strip_runtime_keys_from_feature_config():
    """Runtime-only Koina keys are removed from saved feature configs."""
    cfg = {
        "_target_": "winnow.calibration.features.fragment_match.FragmentMatchFeatures",
        "mz_tolerance": 0.02,
        "model_input_constants": {"collision_energies": 27},
        "model_input_columns": {"fragmentation_types": "frag_type"},
    }
    stripped = strip_runtime_keys_from_feature_config(cfg)
    assert "model_input_constants" not in stripped
    assert "model_input_columns" not in stripped
    assert stripped["mz_tolerance"] == 0.02


def test_validate_koina_intensity_config_conflicting_sources():
    """Dual CE/frag specification exits with code 1."""
    koina_cfg = {
        "input_constants": {"collision_energies": 27},
        "input_columns": {"collision_energies": "collision_energy"},
    }
    with pytest.raises(typer.Exit) as exc:
        validate_koina_intensity_config(
            koina_cfg,
            hydra_overrides=["koina.input_columns.collision_energies=collision_energy"],
        )
    assert exc.value.exit_code == 1
