"""Tests for predict/diagnose iRT regressor calibration helpers."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from winnow.calibration.features.retention_time import RetentionTimeFeature
from winnow.utils.irt_calibration_config import maybe_load_irt_regressors


def test_maybe_load_irt_regressors_noop_when_path_null():
    """Unset irt_regressor_path does not call load_regressors."""
    rt_feature = MagicMock(spec=RetentionTimeFeature)
    calibrator = SimpleNamespace(feature_dict={"iRT Feature": rt_feature})
    cfg = {"calibrator": {"irt_regressor_path": None}}

    maybe_load_irt_regressors(cfg, calibrator, logger=None)

    rt_feature.load_regressors.assert_not_called()


def test_maybe_load_irt_regressors_loads_when_path_and_feature_present():
    """Configured path loads regressors on the RetentionTimeFeature."""
    rt_feature = MagicMock(spec=RetentionTimeFeature)
    calibrator = SimpleNamespace(feature_dict={"iRT Feature": rt_feature})
    cfg = {"calibrator": {"irt_regressor_path": "/tmp/irt_regressors.safetensors"}}
    logger = MagicMock()

    maybe_load_irt_regressors(cfg, calibrator, logger=logger)

    rt_feature.load_regressors.assert_called_once_with(
        "/tmp/irt_regressors.safetensors"
    )
    logger.info.assert_called_once()


def test_maybe_load_irt_regressors_skips_when_feature_absent():
    """Path set but no iRT Feature does not raise."""
    calibrator = SimpleNamespace(feature_dict={})
    cfg = {"calibrator": {"irt_regressor_path": "/tmp/irt_regressors.safetensors"}}

    maybe_load_irt_regressors(cfg, calibrator, logger=None)
