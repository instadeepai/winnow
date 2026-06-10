"""Tests for winnow.utils.irt_calibration_config."""

import logging

import pytest
import typer

from winnow.calibration.features.retention_time import RetentionTimeFeature
from winnow.calibration.calibrator import ProbabilityCalibrator
from winnow.utils.irt_calibration_config import (
    apply_irt_calibration_config,
    explicit_irt_calibration_fields,
    validate_irt_calibration_config,
)


def _predict_cfg(
    *,
    train_fraction=None,
    min_train_points=None,
    irt_regressor_path=None,
):
    return {
        "calibrator": {
            "irt_regressor_path": irt_regressor_path,
            "irt_calibration": {
                "train_fraction": train_fraction,
                "min_train_points": min_train_points,
            },
        }
    }


def _calibrator_with_rt(*, train_fraction=0.1, min_train_points=5):
    calibrator = ProbabilityCalibrator()
    calibrator.add_feature(
        RetentionTimeFeature(
            train_fraction=train_fraction,
            min_train_points=min_train_points,
            learn_from_missing=False,
        )
    )
    return calibrator


def test_explicit_irt_calibration_fields_from_hydra_override():
    cfg = _predict_cfg()
    explicit = explicit_irt_calibration_fields(
        cfg["calibrator"]["irt_calibration"],
        hydra_overrides=["calibrator.irt_calibration.train_fraction=0.3"],
    )
    assert explicit == {"train_fraction"}


def test_explicit_irt_calibration_fields_from_non_null_config():
    cfg = _predict_cfg(train_fraction=0.3)
    explicit = explicit_irt_calibration_fields(cfg["calibrator"]["irt_calibration"])
    assert explicit == {"train_fraction"}


def test_validate_irt_calibration_config_errors_when_regressor_path_set():
    cfg = _predict_cfg(
        train_fraction=0.3,
        irt_regressor_path="irt_regressors.safetensors",
    )
    with pytest.raises(typer.Exit):
        validate_irt_calibration_config(
            cfg,
            hydra_overrides=["calibrator.irt_calibration.train_fraction=0.3"],
        )


def test_validate_irt_calibration_config_allows_regressor_path_without_overrides():
    cfg = _predict_cfg(irt_regressor_path="irt_regressors.safetensors")
    validate_irt_calibration_config(cfg)


def test_apply_irt_calibration_config_keeps_model_defaults_when_null(caplog):
    calibrator = _calibrator_with_rt(train_fraction=0.1, min_train_points=5)
    cfg = _predict_cfg()

    with caplog.at_level(logging.INFO):
        apply_irt_calibration_config(
            calibrator,
            cfg["calibrator"]["irt_calibration"],
            logger=logging.getLogger("test"),
        )

    rt = calibrator.feature_dict["iRT Feature"]
    assert rt.train_fraction == 0.1
    assert rt.min_train_points == 5
    assert not any("Overriding iRT" in r.message for r in caplog.records)


def test_apply_irt_calibration_config_overrides_and_warns(caplog):
    calibrator = _calibrator_with_rt(train_fraction=0.1, min_train_points=5)
    cfg = _predict_cfg(train_fraction=0.3)

    with caplog.at_level(logging.WARNING):
        apply_irt_calibration_config(
            calibrator,
            cfg["calibrator"]["irt_calibration"],
            hydra_overrides=["calibrator.irt_calibration.train_fraction=0.3"],
            logger=logging.getLogger("test"),
        )

    rt = calibrator.feature_dict["iRT Feature"]
    assert rt.train_fraction == 0.3
    assert any("Overriding iRT" in r.message for r in caplog.records)
