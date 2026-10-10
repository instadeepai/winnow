"""Residue mass table runtime configuration for predict.

The masses a run needs are those of the residue set its predictions were
decoded with, which a calibrator fitted on a narrower set cannot know. This
mirrors the Koina runtime overrides in
:mod:`winnow.utils.koina_intensity_config`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def parse_residue_masses(cfg: Any) -> Optional[Dict[str, float]]:
    """Extract ``residue_masses`` from a config as a plain dict."""
    if cfg is None:
        return None
    masses = cfg.get("residue_masses")
    if masses is None:
        return None
    from omegaconf import DictConfig, OmegaConf

    if isinstance(masses, DictConfig):
        masses = OmegaConf.to_container(masses, resolve=True)
    return {str(residue): float(mass) for residue, mass in dict(masses).items()}


def apply_residue_masses_config(
    calibrator: Any,
    cfg: Any,
    logger: Any,
) -> None:
    """Add configured residue masses to a loaded calibrator's table.

    The configured table supplements the saved one rather than replacing it.
    A calibrator's own table is the one its features were fitted against and is
    usually the wider of the two, so replacing it would drop residues the
    calibrator already handles; and where the two give different masses for the
    same residue, the saved value is the one its calibration was learned from.
    Supplementing only adds residues that had no mass at all, which is the case
    that cannot otherwise be computed.
    """
    configured = parse_residue_masses(cfg)
    if not configured:
        return
    added, updated = calibrator.add_residue_masses(configured)
    if updated:
        logger.info(
            "Added %d residue mass(es) from the configured residue set to %s.",
            len(added),
            ", ".join(sorted(updated)),
        )
        logger.debug("Residues added: %s", ", ".join(sorted(added)))
