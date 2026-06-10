"""Tests for winnow.utils.koina_intensity_config."""

from omegaconf import OmegaConf
import pytest
import typer

from winnow.utils.koina_intensity_config import (
    DEFAULT_KOINA_INPUT_COLUMNS,
    format_resolved_koina_setting,
    parse_koina_intensity_config,
    resolve_feature_model_inputs,
    validate_koina_intensity_config,
)


def test_resolve_feature_model_inputs_defaults_columns():
    constants, columns = resolve_feature_model_inputs(None, None)
    assert constants is None
    assert columns == dict(DEFAULT_KOINA_INPUT_COLUMNS)


def test_parse_koina_intensity_config_extracts_blocks():
    koina_cfg = OmegaConf.create(
        {
            "input_constants": {"collision_energies": 27},
            "input_columns": {"fragmentation_types": "frag_type"},
        }
    )
    constants, columns = parse_koina_intensity_config(koina_cfg)
    assert constants == {"collision_energies": 27}
    assert columns == {"fragmentation_types": "frag_type"}


def test_validate_koina_intensity_config_rejects_dual_specification():
    koina_cfg = OmegaConf.create(
        {
            "input_constants": {"collision_energies": 27},
            "input_columns": {"collision_energies": "collision_energy"},
        }
    )
    with pytest.raises(typer.Exit):
        validate_koina_intensity_config(
            koina_cfg,
            hydra_overrides=["koina.input_columns.collision_energies=collision_energy"],
        )


def test_validate_koina_intensity_config_allows_default_columns_with_null_constants():
    koina_cfg = OmegaConf.create(
        {
            "input_constants": {"collision_energies": None},
            "input_columns": {"collision_energies": "collision_energy"},
        }
    )
    validate_koina_intensity_config(koina_cfg, hydra_overrides=None)


def test_format_resolved_koina_setting_matches_irt_log_style():
    assert (
        format_resolved_koina_setting(
            "collision_energies",
            {"collision_energies": 27},
            None,
        )
        == "collision_energies=27"
    )
    assert (
        format_resolved_koina_setting(
            "fragmentation_types",
            {"fragmentation_types": "HCD"},
            None,
        )
        == "fragmentation_types='HCD'"
    )
    assert (
        format_resolved_koina_setting("collision_energies", None, None)
        == "collision_energies='collision_energy'"
    )
