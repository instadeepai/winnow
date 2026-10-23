"""Tests for Koina intensity model input configuration helpers."""

import logging

import pytest
import typer

from winnow.utils.koina_intensity_config import (
    apply_koina_intensity_config,
    parse_unsupported_residues,
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


# unsupported_residues overrides


class _Feature:
    """Stand-in for a Koina-backed feature carrying an exclusion list."""

    def __init__(self, residues):
        self.unsupported_residues = list(residues)
        self.model_input_constants = None
        self.model_input_columns = None


class _PlainFeature:
    """A feature that does no Koina filtering."""


class _Calibrator:
    def __init__(self, feature_dict):
        self.feature_dict = feature_dict

    def apply_koina_model_input_overrides(self, **_kwargs):
        pass

    def apply_unsupported_residues_override(self, unsupported_residues):
        if unsupported_residues is None:
            return []
        updated = []
        for name, feature in self.feature_dict.items():
            if not hasattr(feature, "unsupported_residues"):
                continue
            feature.unsupported_residues = list(unsupported_residues)
            updated.append(name)
        return updated


@pytest.fixture()
def calibrator():
    return _Calibrator(
        {
            "Fragment Match Features": _Feature(["[UNIMOD:5]"]),
            "iRT Feature": _Feature(["[UNIMOD:5]"]),
            "Beam Features": _PlainFeature(),
        }
    )


def test_parse_unsupported_residues_missing_block():
    """A config without the constraints block yields no list."""
    assert parse_unsupported_residues(None) is None
    assert parse_unsupported_residues({"constraints": None}) is None
    assert parse_unsupported_residues({}) is None


def test_parse_unsupported_residues_reads_list():
    """The configured tokens come back as a plain list."""
    cfg = {"constraints": {"unsupported_residues": ["P[UNIMOD:425]", "U"]}}
    assert parse_unsupported_residues(cfg) == ["P[UNIMOD:425]", "U"]


def test_apply_replaces_saved_list_when_overridden(calibrator, caplog):
    """An explicit override replaces the list on every Koina-backed feature."""
    cfg = {"constraints": {"unsupported_residues": ["P[UNIMOD:425]", "K[UNIMOD:259]"]}}
    apply_koina_intensity_config(
        calibrator,
        cfg,
        logging.getLogger(__name__),
        hydra_overrides=[
            'koina.constraints.unsupported_residues=["P[UNIMOD:425]"]',
        ],
    )
    for name in ("Fragment Match Features", "iRT Feature"):
        assert calibrator.feature_dict[name].unsupported_residues == [
            "P[UNIMOD:425]",
            "K[UNIMOD:259]",
        ]


def test_apply_leaves_saved_list_alone_without_override(calibrator):
    """Without the override the calibrator keeps the list it was fitted with."""
    cfg = {"constraints": {"unsupported_residues": ["P[UNIMOD:425]"]}}
    apply_koina_intensity_config(
        calibrator,
        cfg,
        logging.getLogger(__name__),
        hydra_overrides=["fdr_control.fdr_threshold=0.05"],
    )
    for name in ("Fragment Match Features", "iRT Feature"):
        assert calibrator.feature_dict[name].unsupported_residues == ["[UNIMOD:5]"]


def test_apply_skips_features_without_the_attribute(calibrator):
    """Features that do no Koina filtering are left untouched."""
    cfg = {"constraints": {"unsupported_residues": ["U"]}}
    apply_koina_intensity_config(
        calibrator,
        cfg,
        logging.getLogger(__name__),
        hydra_overrides=["koina.constraints.unsupported_residues=[U]"],
    )
    assert not hasattr(calibrator.feature_dict["Beam Features"], "unsupported_residues")
