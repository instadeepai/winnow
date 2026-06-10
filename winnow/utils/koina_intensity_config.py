"""Koina intensity-model runtime configuration for predict / train / compute-features.

Collision energy and fragmentation type resolution applies to intensity Koina models
(``FragmentMatchFeatures``, ``ChimericFeatures``). Server URL/SSL overrides also
apply to ``RetentionTimeFeature`` iRT Koina calls; CE/frag inputs do not.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import typer

from winnow.utils.hydra_overrides import hydra_override_keys

KOINA_RUNTIME_CONFIG_KEYS = frozenset(
    {
        "koina_server_url",
        "koina_ssl",
        "model_input_constants",
        "model_input_columns",
    }
)

KOINA_INPUT_KEYS = frozenset({"collision_energies", "fragmentation_types"})

DEFAULT_KOINA_INPUT_COLUMNS: Dict[str, str] = {
    "collision_energies": "collision_energy",
    "fragmentation_types": "frag_type",
}


def resolve_feature_model_inputs(
    model_input_constants: Optional[Dict[str, Any]],
    model_input_columns: Optional[Dict[str, str]],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, str]]]:
    """Resolve CE/frag sources for feature construction or checkpoint merge.

    Priority per key: non-null constant > non-null column > default column name.
    """
    active_constants = _active_non_null(model_input_constants)
    active_columns = _active_non_null(model_input_columns)
    for key in list(active_columns):
        if key in active_constants:
            del active_columns[key]
    for key, default_column in DEFAULT_KOINA_INPUT_COLUMNS.items():
        if key not in active_constants and key not in active_columns:
            active_columns[key] = default_column
    return (
        active_constants if active_constants else None,
        active_columns if active_columns else None,
    )


def strip_runtime_keys_from_feature_config(
    feature_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Return a copy of a saved feature config with runtime-only keys removed."""
    return {
        key: value
        for key, value in feature_config.items()
        if key not in KOINA_RUNTIME_CONFIG_KEYS
    }


def parse_koina_intensity_config(
    koina_cfg: Any,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, str]]]:
    """Extract input_constants / input_columns dicts from a Hydra koina config block."""
    if koina_cfg is None:
        return None, None
    from omegaconf import OmegaConf

    constants_cfg = koina_cfg.get("input_constants")
    columns_cfg = koina_cfg.get("input_columns")
    constants = (
        OmegaConf.to_container(constants_cfg, resolve=True)
        if constants_cfg is not None
        else None
    )
    columns = (
        OmegaConf.to_container(columns_cfg, resolve=True)
        if columns_cfg is not None
        else None
    )
    return constants, columns


def _active_non_null(d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not d:
        return {}
    return {k: v for k, v in d.items() if v is not None}


def user_specified_constant(
    input_constants: Optional[Dict[str, Any]], key: str
) -> bool:
    """True when ``input_constants[key]`` is explicitly non-null."""
    return _active_non_null(input_constants).get(key) is not None


def user_specified_column(
    input_columns: Optional[Dict[str, str]],
    key: str,
    override_keys: Set[str],
) -> bool:
    """True when the user intentionally set a column mapping for ``key``."""
    columns = _active_non_null(input_columns)
    value = columns.get(key)
    if value is None:
        return False
    if f"koina.input_columns.{key}" in override_keys:
        return True
    return value != DEFAULT_KOINA_INPUT_COLUMNS.get(key)


def validate_koina_intensity_config(
    koina_cfg: Any,
    hydra_overrides: Optional[List[str]] = None,
) -> None:
    """Exit with a pretty error if CE/frag are set as both constant and column.

    Shipped default columns with null constants are not treated as a dual specification.
    """
    constants, columns = parse_koina_intensity_config(koina_cfg)
    override_keys = hydra_override_keys(hydra_overrides)
    conflicts = [
        key
        for key in sorted(KOINA_INPUT_KEYS)
        if user_specified_constant(constants, key)
        and user_specified_column(columns, key, override_keys)
    ]
    if not conflicts:
        return

    from rich.console import Console

    lines = [
        "[bold red]Error:[/bold red] Koina model input(s) cannot be set in both "
        "koina.input_constants and koina.input_columns:",
        "",
    ]
    for key in conflicts:
        lines.append(f"  • {key}")
    lines.extend(
        [
            "",
            "Use exactly one source per key, for example:",
            "  [dim]koina.input_constants.collision_energies=27[/dim]",
            "  [dim]koina.input_columns.collision_energies=collision_energy[/dim]",
        ]
    )
    Console(stderr=True).print("\n".join(lines))
    raise typer.Exit(code=1)


def format_resolved_koina_input(
    key: str,
    constants: Optional[Dict[str, Any]],
    columns: Optional[Dict[str, str]],
) -> str:
    """Human-readable description of how a Koina input key will be resolved."""
    active_constants = _active_non_null(constants)
    active_columns = _active_non_null(columns)
    if key in active_constants:
        return f"constant {active_constants[key]!r}"
    if key in active_columns:
        return f"column {active_columns[key]!r}"
    default_col = DEFAULT_KOINA_INPUT_COLUMNS.get(key)
    if default_col is not None:
        return f"column {default_col!r} (default)"
    return "unset"


def log_resolved_koina_intensity_config(calibrator: Any, logger: Any) -> None:
    """Log how CE/frag inputs are resolved on intensity-based features."""
    logged: Set[str] = set()
    for feature in calibrator.feature_dict.values():
        if not (
            hasattr(feature, "model_input_constants")
            and hasattr(feature, "model_input_columns")
        ):
            continue
        parts = [
            f"{key} ← {format_resolved_koina_input(key, feature.model_input_constants, feature.model_input_columns)}"
            for key in sorted(KOINA_INPUT_KEYS)
        ]
        if not parts:
            continue
        feature_label = getattr(feature, "name", feature.__class__.__name__)
        if feature_label in logged:
            continue
        logged.add(feature_label)
        logger.info("Koina inputs (%s): %s", feature_label, "; ".join(parts))


def apply_koina_intensity_config(
    calibrator: Any,
    koina_cfg: Any,
    logger: Any,
) -> None:
    """Apply server URL/SSL and CE/frag overrides from predict config to a calibrator."""
    if koina_cfg is None:
        return

    constants, columns = parse_koina_intensity_config(koina_cfg)
    calibrator.apply_koina_model_input_overrides(
        model_input_constants=constants,
        model_input_columns=columns,
    )
    server_url = koina_cfg.get("server_url")
    ssl = koina_cfg.get("ssl")
    if server_url is not None or ssl is not None:
        calibrator.apply_koina_server_overrides(
            server_url=server_url,
            ssl=ssl,
        )
    log_resolved_koina_intensity_config(calibrator, logger)
