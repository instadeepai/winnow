"""Utilities for resolving local paths or HuggingFace Hub repository IDs."""

from __future__ import annotations

import logging
from pathlib import Path

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)


def resolve_data_path(
    path_or_repo_id: str,
    repo_type: str = "model",
    cache_dir: Path | None = None,
) -> Path:
    """Resolve a local path or HuggingFace Hub repo ID to a local path.

    Resolution order (logged at INFO level):
      1. Check if ``path_or_repo_id`` exists as a local path (file or dir).
      2. If not, attempt HuggingFace Hub download via ``snapshot_download()``.
      3. If both fail, raise with a clear error message.

    Warning:
        If a local directory happens to match an HF repo ID pattern
        (e.g. ``myorg/mydata`` exists locally), the local path takes
        priority.  This is logged as a warning.

    Args:
        path_or_repo_id: A local file/directory path or a HuggingFace
            repository identifier (e.g. ``"InstaDeepAI/winnow-features"``).
        repo_type: HuggingFace repo type (``"model"``, ``"dataset"``, etc.).
        cache_dir: Optional directory for caching HuggingFace downloads.

    Returns:
        Resolved local ``Path``.

    Raises:
        FileNotFoundError: If the path does not exist locally and cannot
            be downloaded from HuggingFace Hub.
    """
    local = Path(path_or_repo_id)
    if local.exists():
        resolved = local.resolve()
        if "/" in path_or_repo_id and not path_or_repo_id.startswith("/"):
            logger.warning(
                "Local path %s matches an HF repo ID pattern but exists "
                "locally. Using local path. To force HF download, rename or "
                "remove the local directory.",
                resolved,
            )
        logger.info("Resolved path locally: %s", resolved)
        return resolved

    try:
        logger.info(
            "Path %s not found locally, downloading from HuggingFace Hub...",
            path_or_repo_id,
        )
        downloaded = Path(
            snapshot_download(
                repo_id=path_or_repo_id,
                repo_type=repo_type,
                cache_dir=str(cache_dir) if cache_dir else None,
            )
        )
        logger.info("Downloaded to: %s", downloaded)
        return downloaded
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not resolve '{path_or_repo_id}' as a local path or "
            f"HuggingFace Hub repository (repo_type={repo_type!r}). "
            f"HuggingFace error: {exc}"
        ) from exc
