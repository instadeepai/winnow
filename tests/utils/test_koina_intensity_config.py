"""Tests for Koina intensity model input configuration helpers."""

import logging

import pytest
import typer

from winnow.utils.koina_intensity_config import (
    apply_koina_intensity_config,
    parse_koina_model_names,
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

    def __init__(self, residues, intensity_model=None, irt_model=None):
        self.unsupported_residues = list(residues)
        self.model_input_constants = None
        self.model_input_columns = None
        if intensity_model is not None:
            self.intensity_model_name = intensity_model
        if irt_model is not None:
            self.irt_model_name = irt_model


class _PlainFeature:
    """A feature that does no Koina filtering."""


class _Calibrator:
    def __init__(self, feature_dict):
        self.feature_dict = feature_dict

    def apply_koina_model_input_overrides(self, **_kwargs):
        pass

    def apply_koina_model_name_overrides(self, intensity_model=None, irt_model=None):
        wanted = {
            "intensity_model_name": intensity_model,
            "irt_model_name": irt_model,
        }
        updated = []
        for name, feature in self.feature_dict.items():
            changed = False
            for attribute, value in wanted.items():
                if value is None or not hasattr(feature, attribute):
                    continue
                setattr(feature, attribute, value)
                changed = True
            if changed:
                updated.append(name)
        return updated

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
            "Fragment Match Features": _Feature(
                ["[UNIMOD:5]"], intensity_model="Prosit_2025_intensity_22PTM"
            ),
            "iRT Feature": _Feature(["[UNIMOD:5]"], irt_model="Prosit_2025_irt_22PTM"),
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


def test_parse_koina_model_names_needs_the_override_key():
    """A name that is only the shipped default is not a request to change it."""
    cfg = {
        "intensity_model": "AlphaPeptDeep_ms2_generic",
        "irt_model": "Prosit_2025_irt_22PTM",
    }
    assert parse_koina_model_names(cfg, set()) == (None, None)
    assert parse_koina_model_names(cfg, {"koina.intensity_model"}) == (
        "AlphaPeptDeep_ms2_generic",
        None,
    )
    assert parse_koina_model_names(cfg, {"koina.irt_model"}) == (
        None,
        "Prosit_2025_irt_22PTM",
    )


def test_apply_points_features_at_an_overridden_intensity_model(calibrator, caplog):
    """An explicit override renames the intensity model and warns about calibration."""
    cfg = {"intensity_model": "AlphaPeptDeep_ms2_generic"}
    with caplog.at_level(logging.WARNING):
        apply_koina_intensity_config(
            calibrator,
            cfg,
            logging.getLogger(__name__),
            hydra_overrides=["koina.intensity_model=AlphaPeptDeep_ms2_generic"],
        )
    fragment = calibrator.feature_dict["Fragment Match Features"]
    assert fragment.intensity_model_name == "AlphaPeptDeep_ms2_generic"
    # The iRT model is a separate key and was not asked for.
    assert calibrator.feature_dict["iRT Feature"].irt_model_name == (
        "Prosit_2025_irt_22PTM"
    )
    assert "refit" in caplog.text


def test_apply_leaves_saved_model_alone_without_override(calibrator):
    """Without the override the calibrator keeps the model it was fitted against."""
    cfg = {"intensity_model": "AlphaPeptDeep_ms2_generic"}
    apply_koina_intensity_config(
        calibrator,
        cfg,
        logging.getLogger(__name__),
        hydra_overrides=["fdr_control.fdr_threshold=0.05"],
    )
    assert (
        calibrator.feature_dict["Fragment Match Features"].intensity_model_name
        == "Prosit_2025_intensity_22PTM"
    )
