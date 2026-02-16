"""Configuration path resolution utilities.

This module provides robust path resolution for config directories that works
in both development (cloned repo) and package (installed) modes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List
import logging
import shutil
import tempfile
import atexit

logger = logging.getLogger(__name__)

# Track temporary directories for cleanup
_temp_dirs: List[Path] = []


def _cleanup_temp_dirs() -> None:
    """Clean up temporary directories on exit."""
    for temp_dir in _temp_dirs:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


atexit.register(_cleanup_temp_dirs)


def get_config_dir() -> Path:
    """Get the primary config directory (package or dev mode).

    Returns:
        Path to the config directory. In package mode, returns the package
        config directory. In dev mode, returns the repo root config directory.

    Raises:
        FileNotFoundError: If config directory cannot be found in either mode.
    """
    # Try package mode first (when installed)
    try:
        from importlib.resources import files

        config_path = files("winnow").joinpath("configs")
        if config_path.is_dir():
            return Path(str(config_path))
    except (ModuleNotFoundError, TypeError, AttributeError):
        pass

    # Fallback to dev mode (cloned repo)
    # This file is in winnow/utils/, so go up to repo root
    utils_dir = Path(__file__).parent
    repo_root = utils_dir.parent.parent
    dev_configs = repo_root / "winnow" / "configs"

    if dev_configs.exists() and dev_configs.is_dir():
        return dev_configs

    # If neither works, try alternative dev location (configs at repo root)
    alt_dev_configs = repo_root / "configs"
    if alt_dev_configs.exists() and alt_dev_configs.is_dir():
        return alt_dev_configs

    raise FileNotFoundError(
        f"Could not locate configs directory. Tried:\n"
        f"  - Package configs: winnow.configs\n"
        f"  - Dev configs: {dev_configs}\n"
        f"  - Alt dev configs: {alt_dev_configs}"
    )


def get_config_search_path(custom_config_dir: Optional[str] = None) -> List[Path]:
    """Get ordered list of config directories for Hydra search path.

    The search path is ordered by priority (first directory has highest priority):
    1. Custom config directory (if provided)
    2. Package configs (when installed)
    3. Development configs (when running from cloned repo)

    Args:
        custom_config_dir: Optional path to custom config directory.
            If provided, this takes highest priority.

    Returns:
        List of config directory paths in priority order (highest first).
        All paths are absolute.

    Raises:
        FileNotFoundError: If custom_config_dir is provided but doesn't exist.
        ValueError: If custom_config_dir is provided but is not a directory.
    """
    search_path: List[Path] = []

    # 1. Custom config directory (highest priority)
    if custom_config_dir:
        custom_path = Path(custom_config_dir).resolve()
        if not custom_path.exists():
            raise FileNotFoundError(
                f"Custom config directory does not exist: {custom_config_dir}"
            )
        if not custom_path.is_dir():
            raise ValueError(
                f"Custom config path is not a directory: {custom_config_dir}"
            )
        search_path.append(custom_path)
        logger.info(f"Using custom config directory: {custom_path}")

    # 2. Package configs (fallback for files not in custom dir)
    try:
        package_config_dir = get_config_dir()
        # Only add if it's different from custom dir (avoid duplicates)
        if not search_path or package_config_dir.resolve() != search_path[0].resolve():
            search_path.append(package_config_dir.resolve())
            logger.debug(f"Added package config directory: {package_config_dir}")
    except FileNotFoundError:
        logger.warning("Package config directory not found, skipping")

    return search_path


def _merge_config_dirs(custom_dir: Path, package_dir: Path) -> Path:
    """Create a merged config directory with custom configs overriding package configs.

    Creates a temporary directory containing:
    - All files from custom_dir (highest priority)
    - Files from package_dir that don't exist in custom_dir (fallback)

    This allows partial configs to work with Hydra's single-directory search.

    Args:
        custom_dir: Custom config directory (highest priority).
        package_dir: Package config directory (fallback).

    Returns:
        Path to temporary merged config directory.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="winnow_configs_"))
    _temp_dirs.append(temp_dir)

    # First, copy all package configs (this provides fallback for missing files)
    if package_dir.exists():
        for item in package_dir.rglob("*"):
            if item.is_file():
                rel_path = item.relative_to(package_dir)
                dest_path = temp_dir / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dest_path)

    # Then, copy/override with custom configs (this takes precedence)
    if custom_dir.exists():
        for item in custom_dir.rglob("*"):
            if item.is_file():
                rel_path = item.relative_to(custom_dir)
                dest_path = temp_dir / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dest_path)
                logger.debug(f"Merged custom config: {rel_path}")

    return temp_dir


def get_primary_config_dir(custom_config_dir: Optional[str] = None) -> Path:
    """Get the primary config directory to use with Hydra.

    If custom_config_dir is provided, creates a merged directory containing
    both custom and package configs (custom takes precedence). This allows
    partial configs to work - users only need to include files they want to override.

    Otherwise returns package/dev config directory.

    Args:
        custom_config_dir: Optional path to custom config directory.

    Returns:
        Path to primary config directory (absolute). May be a temporary directory
        if custom_config_dir is provided.
    """
    if custom_config_dir:
        custom_path = Path(custom_config_dir).resolve()
        package_path = get_config_dir().resolve()
        # Merge custom and package configs so partial configs work
        merged_dir = _merge_config_dirs(custom_path, package_path)
        logger.info(
            f"Using merged config directory (custom: {custom_path}, "
            f"package: {package_path}) -> {merged_dir}"
        )
        return merged_dir
    return get_config_dir().resolve()
