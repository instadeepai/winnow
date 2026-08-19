#!/usr/bin/env python3
"""Download Figshare article files reconstructing folder paths from the API.

Fetches public article metadata (``files`` plus ``folder_structure``) and writes
each file under ``output_dir`` at ``{folder}/{name}``, matching the Figshare
folder layout used for paper reproduction artefacts.

By default targets article ``30147601`` version ``7``. Use ``--check-only`` to
verify expected relative paths already exist locally without downloading.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import Annotated, Any

import typer

logger = logging.getLogger(__name__)

FIGSHARE_API = "https://api.figshare.com/v2"
_DEFAULT_OUTPUT_DIR = Path("paper_data")

app = typer.Typer(
    add_completion=False,
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


def _configure_logging() -> None:
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def _article_url(article_id: int, version: int | None) -> str:
    if version is None:
        return f"{FIGSHARE_API}/articles/{article_id}"
    return f"{FIGSHARE_API}/articles/{article_id}/versions/{version}"


def _http_get_json(url: str) -> dict[str, Any]:
    """GET a JSON object from ``url`` via urllib."""
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "winnow-paper-scripts"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"Figshare API HTTP {exc.code} for {url}: {exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Figshare API request failed for {url}: {exc.reason}"
        ) from exc
    data = json.loads(payload.decode("utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(
            f"Expected JSON object from {url}, got {type(data).__name__}"
        )
    return data


def _fetch_article(article_id: int, version: int | None) -> dict[str, Any]:
    """Load article metadata, preferring the versioned endpoint when set."""
    if version is not None:
        url = _article_url(article_id, version)
        try:
            return _http_get_json(url)
        except RuntimeError as exc:
            logger.warning(
                "Versioned endpoint failed (%s); falling back to current article metadata.",
                exc,
            )
    return _http_get_json(_article_url(article_id, None))


def _relative_path(file_info: dict[str, Any], folder_structure: dict[str, Any]) -> str:
    """Build the reconstructed relative path for a Figshare file entry."""
    file_id = str(file_info.get("id", ""))
    folder = str(folder_structure.get(file_id, "") or "").strip("/")
    name = str(file_info.get("name", ""))
    if not name:
        raise ValueError(f"Figshare file {file_id!r} has no name")
    if folder:
        return f"{folder}/{name}"
    return name


def _matches_include(rel_path: str, include: list[str] | None) -> bool:
    if not include:
        return True
    return any(fnmatch.fnmatch(rel_path, pattern) for pattern in include)


def _list_article_entries(
    article: dict[str, Any],
    include: list[str] | None,
) -> list[tuple[str, dict[str, Any]]]:
    """Return ``(relative_path, file_info)`` pairs filtered by ``include`` globs."""
    folder_structure = article.get("folder_structure") or {}
    if not isinstance(folder_structure, dict):
        raise RuntimeError("Article metadata folder_structure must be an object")
    files = article.get("files") or []
    if not isinstance(files, list):
        raise RuntimeError("Article metadata files must be a list")

    entries: list[tuple[str, dict[str, Any]]] = []
    for file_info in files:
        if not isinstance(file_info, dict):
            continue
        rel_path = _relative_path(file_info, folder_structure)
        if not _matches_include(rel_path, include):
            continue
        entries.append((rel_path, file_info))
    entries.sort(key=lambda item: item[0])
    return entries


def _local_size_matches(path: Path, expected_size: int | None) -> bool:
    if expected_size is None or not path.is_file():
        return False
    return path.stat().st_size == int(expected_size)


def _download_file(download_url: str, dest: Path) -> None:
    """Download ``download_url`` to ``dest`` using urllib, writing via a temp file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_name(f".{dest.name}.partial")
    request = urllib.request.Request(
        download_url,
        headers={"User-Agent": "winnow-paper-scripts"},
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            with tmp_path.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
        tmp_path.replace(dest)
    except urllib.error.URLError as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"Download failed for {download_url}: {exc.reason}") from exc
    except OSError:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _check_paths(output_dir: Path, relative_paths: list[str]) -> int:
    """Return the number of missing paths under ``output_dir``."""
    missing = 0
    for rel in relative_paths:
        path = output_dir / rel
        if path.is_file():
            logger.info("OK %s", rel)
        else:
            logger.error("Missing %s", rel)
            missing += 1
    return missing


def _paths_for_check_only(
    article_id: int,
    version: int | None,
    include: list[str] | None,
    require: list[str] | None,
) -> list[str]:
    """Resolve relative paths to verify under ``--check-only``."""
    if require:
        to_check = list(require)
        if include:
            to_check = [path for path in to_check if _matches_include(path, include)]
        return to_check

    article = _fetch_article(article_id, version)
    to_check = [rel for rel, _ in _list_article_entries(article, include)]
    if not to_check:
        logger.error("No article files matched the selection for --check-only.")
        raise typer.Exit(code=1)
    return to_check


def _download_or_skip_entry(
    rel_path: str,
    file_info: dict[str, Any],
    output_dir: Path,
    *,
    dry_run: bool,
    force: bool,
) -> str:
    """Download one article file or skip it. Returns ``downloaded``, ``skipped``, or ``listed``."""
    size = file_info.get("size")
    expected_size = int(size) if size is not None else None
    download_url = file_info.get("download_url")
    dest = output_dir / rel_path
    size_label = str(expected_size) if expected_size is not None else "?"

    if dry_run:
        logger.info("DRY-RUN %s (%s bytes)", rel_path, size_label)
        return "listed"

    if not force and _local_size_matches(dest, expected_size):
        logger.info("Skip %s (size matches)", rel_path)
        return "skipped"

    if not download_url:
        raise RuntimeError(f"No download_url for {rel_path!r}")

    logger.info("Download %s (%s bytes)", rel_path, size_label)
    _download_file(str(download_url), dest)
    if expected_size is not None and not _local_size_matches(dest, expected_size):
        raise RuntimeError(
            f"Downloaded size mismatch for {rel_path!r}: "
            f"expected {expected_size}, got {dest.stat().st_size}"
        )
    return "downloaded"


@app.command()
def main(
    article_id: Annotated[
        int,
        typer.Option("--article-id", help="Figshare article id."),
    ] = 30147601,
    version: Annotated[
        int | None,
        typer.Option(
            "--version",
            help="Article version to fetch (uses /versions/{n} when set).",
        ),
    ] = 7,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Root directory for reconstructed article paths.",
        ),
    ] = _DEFAULT_OUTPUT_DIR,
    include: Annotated[
        list[str] | None,
        typer.Option(
            "--include",
            help=(
                "Glob pattern(s) matched against reconstructed relative paths. "
                "If unset, all files are selected."
            ),
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="List selected files without downloading."),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Re-download even when a local file already matches remote size.",
        ),
    ] = False,
    check_only: Annotated[
        bool,
        typer.Option(
            "--check-only",
            help=(
                "Verify expected relative paths exist under output-dir and exit "
                "(1 if any are missing). Does not download."
            ),
        ),
    ] = False,
    require: Annotated[
        list[str] | None,
        typer.Option(
            "--require",
            help=(
                "Relative path(s) that must exist when --check-only is set. "
                "If unset with --check-only, all selected article files are checked."
            ),
        ),
    ] = None,
) -> None:
    """Download or verify Figshare article files with reconstructed folder paths."""
    _configure_logging()

    if check_only:
        to_check = _paths_for_check_only(article_id, version, include, require)
        missing = _check_paths(output_dir, to_check)
        if missing:
            logger.error(
                "%d of %d required path(s) missing under %s",
                missing,
                len(to_check),
                output_dir,
            )
            raise typer.Exit(code=1)
        logger.info(
            "All %d required path(s) present under %s", len(to_check), output_dir
        )
        return

    if require:
        logger.warning("--require is only used with --check-only; ignoring.")

    article = _fetch_article(article_id, version)
    title = article.get("title", "?")
    article_version = article.get("version", version)
    entries = _list_article_entries(article, include)
    logger.info(
        "Article %s (version %s): %r — %d file(s) selected",
        article_id,
        article_version,
        title,
        len(entries),
    )
    if not entries:
        logger.warning("No files matched; nothing to do.")
        return

    downloaded = 0
    skipped = 0
    for rel_path, file_info in entries:
        outcome = _download_or_skip_entry(
            rel_path,
            file_info,
            output_dir,
            dry_run=dry_run,
            force=force,
        )
        if outcome == "downloaded":
            downloaded += 1
        elif outcome == "skipped":
            skipped += 1

    if dry_run:
        logger.info("Dry run complete (%d file(s) listed).", len(entries))
        return

    logger.info(
        "Done: downloaded %d, skipped %d, total selected %d under %s",
        downloaded,
        skipped,
        len(entries),
        output_dir,
    )


if __name__ == "__main__":
    app()
