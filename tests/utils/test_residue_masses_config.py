"""Tests for supplementing a loaded calibrator's residue mass table."""

import logging

import pytest

from winnow.utils.residue_masses_config import (
    apply_residue_masses_config,
    parse_residue_masses,
)


class _MassFeature:
    def __init__(self, masses):
        self.residue_masses = dict(masses)


class _OtherFeature:
    pass


class _Calibrator:
    def __init__(self, feature_dict):
        self.feature_dict = feature_dict
        self.calls = []

    def add_residue_masses(self, residue_masses):
        self.calls.append(residue_masses)
        added, updated = set(), []
        for name, feature in self.feature_dict.items():
            table = getattr(feature, "residue_masses", None)
            if table is None:
                continue
            missing = {k: v for k, v in residue_masses.items() if k not in table}
            if not missing:
                continue
            feature.residue_masses = {**table, **missing}
            added.update(missing)
            updated.append(name)
        return sorted(added), updated


@pytest.fixture()
def calibrator():
    return _Calibrator(
        {
            "Mass Error (Da)": _MassFeature({"G": 57.021464}),
            "Beam Features": _OtherFeature(),
        }
    )


def test_parse_residue_masses_absent():
    """A config with no table yields nothing."""
    assert parse_residue_masses(None) is None
    assert parse_residue_masses({}) is None
    assert parse_residue_masses({"residue_masses": None}) is None


def test_parse_residue_masses_coerces_types():
    """Tokens come back as strings mapped to floats."""
    assert parse_residue_masses({"residue_masses": {"G": 57}}) == {"G": 57.0}


def test_apply_adds_missing_residues(calibrator):
    """Residues with no mass are added to the feature's table."""
    cfg = {"residue_masses": {"G": 57.021464, "W[UNIMOD:35]": 202.074228}}
    apply_residue_masses_config(calibrator, cfg, logging.getLogger(__name__))
    table = calibrator.feature_dict["Mass Error (Da)"].residue_masses
    assert table["W[UNIMOD:35]"] == 202.074228


def test_apply_keeps_the_mass_the_calibrator_was_fitted_with(calibrator):
    """A residue already present is not moved to the configured value."""
    cfg = {"residue_masses": {"G": 99.9}}
    apply_residue_masses_config(calibrator, cfg, logging.getLogger(__name__))
    assert calibrator.feature_dict["Mass Error (Da)"].residue_masses["G"] == 57.021464


def test_apply_without_a_configured_table_is_a_noop(calibrator):
    """No table in the config means nothing is touched."""
    apply_residue_masses_config(calibrator, {}, logging.getLogger(__name__))
    assert calibrator.calls == []


def test_apply_skips_features_without_a_table(calibrator):
    """Features that compute no masses are not given a table."""
    cfg = {"residue_masses": {"U": 150.953636}}
    apply_residue_masses_config(calibrator, cfg, logging.getLogger(__name__))
    assert not hasattr(calibrator.feature_dict["Beam Features"], "residue_masses")
