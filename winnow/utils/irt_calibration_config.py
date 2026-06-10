"""Predict-time RT→iRT linear regressor calibration settings (RetentionTimeFeature)."""

from __future__ import annotations

from typing import Any, List, Optional, Set

import typer

from winnow.utils.hydra_overrides import hydra_override_keys

IRT_CALIBRATION_PREFIX = "calibrator.irt_calibration."
IRT_CALIBRATION_FIELDS = ("train_fraction", "min_train_points")


def _irt_calibration_block(cfg: Any) -> Any:
    calibrator_cfg = cfg.get("calibrator")
    if calibrator_cfg is None:
        return None
    return calibrator_cfg.get("irt_calibration")


def explicit_irt_calibration_fields(
    irt_calibration_cfg: Any,
    hydra_overrides: Optional[List[str]] = None,
) -> Set[str]:
    """Field names the user explicitly set via config or Hydra overrides."""
    explicit: Set[str] = set()
    override_keys = hydra_override_keys(hydra_overrides)
    for field in IRT_CALIBRATION_FIELDS:
        hydra_path = f"{IRT_CALIBRATION_PREFIX}{field}"
        if hydra_path in override_keys:
            explicit.add(field)
            continue
        if (
            irt_calibration_cfg is not None
            and irt_calibration_cfg.get(field) is not None
        ):
            explicit.add(field)
    return explicit


def validate_irt_calibration_config(
    cfg: Any,
    hydra_overrides: Optional[List[str]] = None,
) -> None:
    """Exit if ``irt_calibration`` overrides conflict with ``irt_regressor_path``."""
    calibrator_cfg = cfg.get("calibrator") or {}
    irt_regressor_path = calibrator_cfg.get("irt_regressor_path")
    if not irt_regressor_path:
        return

    irt_calibration_cfg = _irt_calibration_block(cfg)
    explicit = explicit_irt_calibration_fields(irt_calibration_cfg, hydra_overrides)
    if not explicit:
        return

    from rich.console import Console

    fields = ", ".join(f"calibrator.irt_calibration.{f}" for f in sorted(explicit))
    lines = [
        "[bold red]Error:[/bold red] Cannot set iRT calibration training parameters "
        "when calibrator.irt_regressor_path is set.",
        "",
        f"Conflicting override(s): {fields}",
        "",
        "Pre-loaded regressors are used as-is. Unset "
        "calibrator.irt_regressor_path to re-fit with custom "
        "calibration settings, for example:",
        "  [dim]calibrator.irt_regressor_path=null "
        "calibrator.irt_calibration.train_fraction=0.3[/dim]",
    ]
    Console(stderr=True).print("\n".join(lines))
    raise typer.Exit(code=1)


def _get_retention_time_feature(calibrator: Any):
    from winnow.calibration.features.retention_time import RetentionTimeFeature

    rt_feature = calibrator.feature_dict.get("iRT Feature")
    if isinstance(rt_feature, RetentionTimeFeature):
        return rt_feature
    return None


def _resolved_irt_calibration(
    irt_calibration_cfg: Any,
    model_defaults: dict[str, Any],
) -> dict[str, Any]:
    resolved = dict(model_defaults)
    if irt_calibration_cfg is None:
        return resolved
    for field in IRT_CALIBRATION_FIELDS:
        value = irt_calibration_cfg.get(field)
        if value is not None:
            resolved[field] = value
    return resolved


def _log_irt_calibration_override(
    field: str,
    new_value: Any,
    old_value: Any,
    *,
    explicit: Set[str],
    logger: Any,
) -> None:
    if field not in explicit or logger is None:
        return
    if new_value != old_value:
        logger.warning(
            "Overriding iRT regressor training setting '%s' for predict: "
            "model default %s -> %s.",
            field,
            old_value,
            new_value,
        )
        return
    logger.warning(
        "Explicit iRT calibration override for '%s' matches the loaded model "
        "default (%s).",
        field,
        old_value,
    )


def apply_irt_calibration_config(
    calibrator: Any,
    irt_calibration_cfg: Any,
    hydra_overrides: Optional[List[str]] = None,
    logger: Any = None,
) -> None:
    """Apply optional predict-time overrides to RetentionTimeFeature calibration params."""
    rt_feature = _get_retention_time_feature(calibrator)
    if rt_feature is None:
        if logger is not None:
            logger.debug(
                "No RetentionTimeFeature on calibrator; skipping iRT calibration config."
            )
        return

    explicit = explicit_irt_calibration_fields(irt_calibration_cfg, hydra_overrides)
    model_defaults = {
        "train_fraction": rt_feature.train_fraction,
        "min_train_points": rt_feature.min_train_points,
    }
    resolved = _resolved_irt_calibration(irt_calibration_cfg, model_defaults)

    for field in IRT_CALIBRATION_FIELDS:
        _log_irt_calibration_override(
            field,
            resolved[field],
            model_defaults[field],
            explicit=explicit,
            logger=logger,
        )
        setattr(rt_feature, field, resolved[field])

    if logger is not None:
        logger.info(
            "iRT calibration settings: train_fraction=%s, min_train_points=%s",
            rt_feature.train_fraction,
            rt_feature.min_train_points,
        )
