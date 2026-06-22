#!/usr/bin/env python3
"""Upload staged winnow analysis outputs to a Figshare project article."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from tqdm import tqdm

from figshare_deposition_manifest import (
    collect_staging_paths,
    load_manifest,
    manifest_includes_general_results_metadata,
    rel_matches_glob,
    summarize_excluded_general_results_metadata,
)

FIGSHARE_API = "https://api.figshare.com/v2"
FIGSHARE_FILE_LIMIT = 500
FIGSHARE_DEFAULT_STORAGE_LIMIT_GIB = 20
MIN_REQUEST_INTERVAL_SECS = 1.0
GiB = 1024**3


class ByteProgress(Protocol):
    """Progress bar for byte-based uploads."""

    def update(self, n: int) -> None:
        """Report progress for n additional bytes."""
        ...

    def close(self) -> None:
        """Close the progress bar."""
        ...


class _NoOpByteProgress:
    """No-op implementation of ByteProgress."""

    def update(self, n: int) -> None:
        """No-op implementation of update."""
        return

    def close(self) -> None:
        """No-op implementation of close."""
        return


def _make_byte_progress(
    *,
    enabled: bool,
    total_bytes: int,
    initial_bytes: int,
) -> ByteProgress:
    if not enabled:
        return _NoOpByteProgress()
    return tqdm(
        total=total_bytes,
        initial=initial_bytes,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc="Uploading",
        smoothing=0.05,
    )


def _normalize_figshare_url(url: str) -> str:
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return f"{FIGSHARE_API}/{url.lstrip('/')}"


@dataclass
class UploadTarget:
    """One staged local file to upload with Figshare basename + folder_path."""

    staging_rel: str
    file_name: str
    folder_path: str
    local_path: Path
    size: int

    @property
    def display_path(self) -> str:
        """Staging-relative path as shown in Figshare folder + basename."""
        if self.folder_path:
            return f"{self.folder_path}/{self.file_name}"
        return self.file_name


class FigshareClient:
    """Minimal Figshare API v2 client for article file replacement."""

    def __init__(self, token: str) -> None:
        self.token = token
        self._last_request_at = 0.0

    def request(
        self,
        method: str,
        url: str,
        *,
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
        expected_status: tuple[int, ...] = (200,),
    ) -> Any:
        """Issue a rate-limited authenticated Figshare API request."""
        self._throttle()
        req_headers = {"Authorization": f"token {self.token}"}
        if headers:
            req_headers.update(headers)
        request = urllib.request.Request(
            url, data=data, headers=req_headers, method=method
        )
        try:
            with urllib.request.urlopen(request) as response:
                self._last_request_at = time.monotonic()
                if response.status not in expected_status:
                    raise RuntimeError(
                        f"{method} {url} unexpected status {response.status}"
                    )
                body = response.read()
                if not body:
                    return None
                text = body.decode()
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return text.strip() or None
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise RuntimeError(f"{method} {url} failed ({exc.code}): {detail}") from exc

    def list_project_articles(self, project_id: int) -> list[dict[str, Any]]:
        """List articles in a Figshare project."""
        return self.request(
            "GET",
            f"{FIGSHARE_API}/account/projects/{project_id}/articles?page_size=100",
        )

    def get_article(self, article_id: int) -> dict[str, Any]:
        """Fetch private article metadata including folder_structure."""
        result = self.request("GET", f"{FIGSHARE_API}/account/articles/{article_id}")
        if not isinstance(result, dict):
            raise RuntimeError(f"Unexpected article response: {result!r}")
        return result

    def list_article_files(self, article_id: int) -> list[dict[str, Any]]:
        """List files attached to an article."""
        return self.request(
            "GET", f"{FIGSHARE_API}/account/articles/{article_id}/files"
        )

    def delete_article_file(self, article_id: int, file_id: int) -> None:
        """Delete one file from an article."""
        self.request(
            "DELETE",
            f"{FIGSHARE_API}/account/articles/{article_id}/files/{file_id}",
            expected_status=(200, 204),
        )

    def initiate_file_upload(
        self,
        article_id: int,
        *,
        name: str,
        size: int,
        md5: str,
        folder_path: str = "",
    ) -> dict[str, Any]:
        """Start a multipart upload and return file metadata including upload_url."""
        body: dict[str, Any] = {"name": name, "size": size, "md5": md5}
        if folder_path:
            body["folder_path"] = folder_path
        payload = json.dumps(body).encode()
        init = self.request(
            "POST",
            f"{FIGSHARE_API}/account/articles/{article_id}/files",
            data=payload,
            headers={"Content-Type": "application/json"},
            expected_status=(201,),
        )
        if not isinstance(init, dict) or "location" not in init:
            raise RuntimeError(f"Unexpected Figshare upload init response: {init!r}")

        location = _normalize_figshare_url(str(init["location"]))
        file_info = self.request("GET", location)
        if not isinstance(file_info, dict):
            raise RuntimeError(f"Unexpected Figshare file info response: {file_info!r}")
        if "id" not in file_info or "upload_url" not in file_info:
            raise RuntimeError(
                f"Figshare file info missing id/upload_url: {file_info!r}"
            )
        file_info["location"] = location
        return file_info

    def get_upload_parts(self, upload_url: str) -> dict[str, Any]:
        """Fetch multipart upload part boundaries from the upload service."""
        return self.request("GET", upload_url)

    def upload_part(self, upload_url: str, part_number: int, chunk: bytes) -> None:
        """Upload one multipart chunk."""
        self._throttle()
        url = f"{upload_url}/{part_number}"
        request = urllib.request.Request(
            url,
            data=chunk,
            headers={
                "Authorization": f"token {self.token}",
                "Content-Type": "application/octet-stream",
            },
            method="PUT",
        )
        try:
            with urllib.request.urlopen(request) as response:
                if response.status not in (200, 201, 204):
                    raise RuntimeError(
                        f"Unexpected upload-part status {response.status}"
                    )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise RuntimeError(f"PUT {url} failed ({exc.code}): {detail}") from exc
        self._last_request_at = time.monotonic()

    def complete_file_upload(
        self,
        article_id: int,
        file_id: int,
        *,
        location: str | None = None,
    ) -> None:
        """Finalize a multipart upload."""
        url = _normalize_figshare_url(
            location or f"{FIGSHARE_API}/account/articles/{article_id}/files/{file_id}"
        )
        self.request("POST", url, data=b"", expected_status=(200, 202))

    def update_article(self, article_id: int, payload: dict[str, Any]) -> None:
        """Update article metadata."""
        body = json.dumps(payload).encode()
        self.request(
            "PUT",
            f"{FIGSHARE_API}/account/articles/{article_id}",
            data=body,
            headers={"Content-Type": "application/json"},
            expected_status=(200, 205),
        )

    def publish_article(self, article_id: int) -> None:
        """Publish a new version of the article."""
        self.request(
            "POST",
            f"{FIGSHARE_API}/account/articles/{article_id}/publish",
            expected_status=(201,),
        )

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < MIN_REQUEST_INTERVAL_SECS:
            time.sleep(MIN_REQUEST_INTERVAL_SECS - elapsed)


def _resolve_article_id(manifest: dict[str, Any], client: FigshareClient) -> int:
    article_cfg = manifest.get("article", {})
    article_id = article_cfg.get("id") or os.environ.get("FIGSHARE_ARTICLE_ID")
    if article_id:
        return int(article_id)

    project_id = manifest.get("project_id") or os.environ.get("FIGSHARE_PROJECT_ID")
    title = article_cfg.get("title", "Analysis outputs")
    if not project_id:
        raise ValueError(
            "Set article.id or FIGSHARE_ARTICLE_ID (or project_id for title lookup)"
        )

    articles = client.list_project_articles(int(project_id))
    for article in articles:
        if article.get("title") == title:
            return int(article["id"])
    available = ", ".join(sorted({article.get("title", "?") for article in articles}))
    raise ValueError(
        f"Article titled '{title}' not found in project {project_id}. Found: {available}"
    )


def _file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_file_part(handle, part: dict[str, Any]) -> bytes:
    start = int(part["startOffset"])
    end = int(part["endOffset"])
    handle.seek(start)
    return handle.read(end - start + 1)


def _figshare_paths_from_rel(rel: str) -> tuple[str, str]:
    """Split a staging relative path into Figshare basename and folder_path."""
    rel_posix = PurePosixPath(rel)
    parent = rel_posix.parent.as_posix()
    folder_path = "" if parent == "." else parent
    return rel_posix.name, folder_path


def _build_targets(
    manifest: dict[str, Any],
    staging_dir: Path,
    only_patterns: list[str] | None,
    *,
    include_general_results_metadata: bool | None = None,
) -> list[UploadTarget]:
    include_globs = manifest.get("include_globs", ["**/*"])
    exclude_globs = manifest.get("exclude_globs", [])
    include_metadata = manifest_includes_general_results_metadata(
        manifest,
        include_general_results_metadata=include_general_results_metadata,
    )
    rel_paths = collect_staging_paths(
        staging_dir,
        include_globs,
        exclude_globs,
        include_general_results_metadata=include_metadata,
    )
    targets: list[UploadTarget] = []

    for rel in rel_paths:
        if only_patterns and not any(
            rel_matches_glob(rel, pattern) for pattern in only_patterns
        ):
            continue
        local_path = staging_dir / rel
        file_name, folder_path = _figshare_paths_from_rel(rel)
        targets.append(
            UploadTarget(
                staging_rel=rel,
                file_name=file_name,
                folder_path=folder_path,
                local_path=local_path,
                size=local_path.stat().st_size,
            )
        )
    return targets


def _load_state(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"files": {}}
    try:
        with path.open() as handle:
            loaded = json.load(handle)
    except json.JSONDecodeError as exc:
        print(
            f"WARNING: ignoring corrupt resume state {path}: {exc}. Starting fresh.",
            file=sys.stderr,
        )
        return {"files": {}}
    if not isinstance(loaded, dict):
        print(
            f"WARNING: ignoring invalid resume state {path}. Starting fresh.",
            file=sys.stderr,
        )
        return {"files": {}}
    loaded.setdefault("files", {})
    return loaded


def _save_state(path: Path | None, state: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in state.items() if key != "_path"}
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    tmp_path.replace(path)


def _delete_existing_files(
    client: FigshareClient, article_id: int, dry_run: bool
) -> None:
    existing = client.list_article_files(article_id)
    print(f"Found {len(existing)} existing files in article {article_id}")
    for file_info in existing:
        file_id = int(file_info["id"])
        name = file_info.get("name", "?")
        if dry_run:
            print(f"DRY-RUN delete file {file_id}: {name}")
            continue
        print(f"Deleting file {file_id}: {name}")
        client.delete_article_file(article_id, file_id)


def _remote_file_display_path(
    file_info: dict[str, Any],
    folder_structure: dict[str, Any],
) -> str:
    file_id = str(file_info.get("id", ""))
    folder = str(folder_structure.get(file_id, "") or "").strip("/")
    name = str(file_info.get("name", ""))
    if folder:
        return f"{folder}/{name}"
    return name


def _list_article_files_with_folders(
    client: FigshareClient, article_id: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    article = client.get_article(article_id)
    folder_structure = article.get("folder_structure") or {}
    files = article.get("files") or client.list_article_files(article_id)
    return files, folder_structure


def _remote_file_matches_target(
    file_info: dict[str, Any],
    folder_structure: dict[str, Any],
    target: UploadTarget,
) -> bool:
    file_id = str(file_info.get("id", ""))
    folder = str(folder_structure.get(file_id, "") or "").strip("/")
    return file_info.get("name") == target.file_name and folder == target.folder_path


def _delete_remote_files_at_path(
    client: FigshareClient,
    article_id: int,
    target: UploadTarget,
    *,
    keep_file_ids: set[int] | None = None,
) -> int:
    """Remove remote copies at target's folder_path/name before a fresh upload."""
    keep = keep_file_ids or set()
    files, folder_structure = _list_article_files_with_folders(client, article_id)
    deleted = 0
    for file_info in files:
        if not _remote_file_matches_target(file_info, folder_structure, target):
            continue
        file_id = int(file_info["id"])
        if file_id in keep:
            continue
        remote_path = _remote_file_display_path(file_info, folder_structure)
        print(f"Deleting duplicate remote file {file_id}: {remote_path!r}")
        client.delete_article_file(article_id, file_id)
        deleted += 1
    return deleted


def _print_upload_verification(
    client: FigshareClient,
    article_id: int,
    target: UploadTarget,
    file_id: int,
) -> None:
    files, folder_structure = _list_article_files_with_folders(client, article_id)
    uploaded = next(
        (item for item in files if int(item.get("id", -1)) == file_id),
        None,
    )
    duplicates = [
        item
        for item in files
        if int(item.get("id", -1)) != file_id
        and _remote_file_matches_target(item, folder_structure, target)
    ]
    if uploaded is not None:
        remote_path = _remote_file_display_path(uploaded, folder_structure)
        print(
            f"Verified on article {article_id}: "
            f"id={uploaded.get('id')} path={remote_path!r} "
            f"size={uploaded.get('size')}"
        )
    else:
        print(
            f"WARNING: {target.display_path!r} (file_id={file_id}) not listed via "
            f"account API yet ({len(files)} files on article).",
            file=sys.stderr,
        )
    if duplicates:
        print(
            f"WARNING: {len(duplicates)} older duplicate(s) at {target.display_path!r} "
            f"still on article (ids: "
            f"{', '.join(str(item.get('id')) for item in duplicates)}). "
            "Re-run with --replace or delete them in the Figshare draft editor.",
            file=sys.stderr,
        )


def _try_complete_resumed_upload(
    client: FigshareClient,
    article_id: int,
    target: UploadTarget,
    file_state: dict[str, Any],
) -> bool:
    file_id = file_state.get("file_id")
    location = file_state.get("location")
    if file_state.get("status") != "UPLOADING" or not file_id or not location:
        return False
    client.complete_file_upload(article_id, int(file_id), location=str(location))
    file_state["status"] = "COMPLETED"
    print(f"Completed resumed upload for {target.display_path} (file_id={file_id})")
    _print_upload_verification(client, article_id, target, int(file_id))
    return True


def _upload_target(
    client: FigshareClient,
    article_id: int,
    target: UploadTarget,
    state: dict[str, Any],
    *,
    dry_run: bool,
    progress: ByteProgress | None = None,
) -> None:
    byte_progress = progress or _NoOpByteProgress()
    file_state = state["files"].get(target.staging_rel, {})
    if file_state.get("status") == "COMPLETED":
        print(f"SKIP completed {target.display_path}")
        return

    if dry_run:
        folder_note = (
            f" folder_path={target.folder_path!r}" if target.folder_path else ""
        )
        print(
            f"DRY-RUN upload {target.display_path} "
            f"(name={target.file_name!r}{folder_note}, {target.size} bytes) "
            f"<- {target.local_path}"
        )
        return

    if not target.local_path.exists():
        raise FileNotFoundError(
            f"Missing staged file: {target.local_path}. "
            "Run make stage-figshare-deposition or download-figshare-staging first."
        )

    if _try_complete_resumed_upload(client, article_id, target, file_state):
        state["files"][target.staging_rel] = file_state
        _save_state(state.get("_path"), state)
        byte_progress.update(target.size)
        return

    keep_ids: set[int] = set()
    if file_state.get("file_id") is not None:
        keep_ids.add(int(file_state["file_id"]))
    _delete_remote_files_at_path(client, article_id, target, keep_file_ids=keep_ids)

    md5 = _file_md5(target.local_path)
    file_info = client.initiate_file_upload(
        article_id,
        name=target.file_name,
        size=target.size,
        md5=md5,
        folder_path=target.folder_path,
    )
    file_id = int(file_info["id"])
    upload_url = str(file_info["upload_url"])
    parts_spec = client.get_upload_parts(upload_url)
    parts = parts_spec.get("parts")
    if not parts:
        raise RuntimeError(f"Figshare upload service returned no parts: {parts_spec!r}")

    file_state = {
        "file_id": file_id,
        "status": "UPLOADING",
        "parts_uploaded": file_state.get("parts_uploaded", []),
        "upload_url": upload_url,
        "location": file_info.get("location"),
    }
    state["files"][target.staging_rel] = file_state
    _save_state(state.get("_path"), state)

    with target.local_path.open("rb") as handle:
        for part in parts:
            part_number = int(part["partNo"])
            if part_number in file_state["parts_uploaded"]:
                continue
            chunk = _read_file_part(handle, part)
            client.upload_part(upload_url, part_number, chunk)
            byte_progress.update(len(chunk))
            file_state["parts_uploaded"].append(part_number)
            _save_state(state.get("_path"), state)

    client.complete_file_upload(
        article_id, file_id, location=str(file_info.get("location", ""))
    )
    file_state["status"] = "COMPLETED"
    _save_state(state.get("_path"), state)
    print(f"Uploaded {target.display_path} (file_id={file_id})")
    _print_upload_verification(client, article_id, target, file_id)


def _delete_remote_files_by_name(
    client: FigshareClient,
    article_id: int,
    names: list[str],
) -> int:
    files = client.list_article_files(article_id)
    deleted = 0
    for file_info in files:
        name = str(file_info.get("name", ""))
        if name not in names:
            continue
        file_id = int(file_info["id"])
        print(f"Deleting remote file {file_id}: {name!r}")
        client.delete_article_file(article_id, file_id)
        deleted += 1
    return deleted


# Pinned Hugging Face revisions for the Figshare article description (update when
# re-depositing after model/dataset releases).
# TODO: Set WINNOW_VERSION (or manifest winnow_version) to the release tag at publication.
FIGSHARE_HF_REVISIONS: dict[str, dict[str, str]] = {
    "winnow-ms-datasets": {
        "repo_id": "InstaDeepAI/winnow-ms-datasets",
        "kind": "dataset",
        "sha": "659802319d618a359de5ab90ec6b0195681e94a6",
        "url": "https://huggingface.co/datasets/InstaDeepAI/winnow-ms-datasets",
    },
    "winnow-general-model": {
        "repo_id": "InstaDeepAI/winnow-general-model",
        "kind": "model",
        "sha": "38a3541e5c097a5f12814211dd700d5440ea2c35",
        "url": "https://huggingface.co/InstaDeepAI/winnow-general-model",
    },
    "winnow-helaqc-model": {
        "repo_id": "InstaDeepAI/winnow-helaqc-model",
        "kind": "model",
        "sha": "9ad54d83cf4f5fce693665ea84d65bb9f56dc26d",
        "url": "https://huggingface.co/InstaDeepAI/winnow-helaqc-model",
    },
}

# Related Figshare calibrator articles in the same project/collection (not Hugging Face).
FIGSHARE_CALIBRATOR_ARTICLES: dict[str, dict[str, str]] = {
    "hela_casanovo_primenovo": {
        "article_id": "32744946",
        "doi": "10.6084/m9.figshare.32744946.v1",
        "title": "Additional HeLa Single Shot Models",
        "url": "https://doi.org/10.6084/m9.figshare.32744946",
        "folders": "casanovo_helaqc/, primenovo_helaqc/",
    },
    "hold_one_out_generalisation": {
        "article_id": "30147364",
        "doi": "10.6084/m9.figshare.30147364.v2",
        "title": "Hold-one-out generalisation models",
        "url": "https://doi.org/10.6084/m9.figshare.30147364",
        "folders": (
            "trained_on_gluc/, trained_on_helaqc/, trained_on_herceptin/, "
            "trained_on_immuno/, trained_on_sbrodae/, trained_on_snakevenoms/, "
            "trained_on_tplantibodies/, trained_on_woundfluids/"
        ),
    },
}

FIGSHARE_FOLDER_DESCRIPTIONS: dict[str, str] = {
    "helaqc_results": (
        "HeLa QC benchmark (PXD044934): InstaNovo, Casanovo, and PrimeNovo prediction "
        "outputs. instanovo/ uses InstaDeepAI/winnow-helaqc-model; casanovo/ and "
        "primenovo/ use the HeLa calibrators in Figshare article 32744946 "
        "(casanovo_helaqc/, primenovo_helaqc/). Layout: {tool}/{split}/metadata.csv "
        "and preds_and_fdr_metrics.csv, where split is test (held-out labelled "
        "spectra), unlabelled only, or full search space less the training set."
    ),
    "general_results": (
        "General-model evaluation on external benchmark datasets from "
        "InstaDeepAI/winnow-ms-datasets (general_model_evaluation/). labelled/ holds "
        "database-search reference runs; full/ holds full-search predictions. Each "
        "project folder contains preds_and_fdr_metrics.csv."
    ),
    "general_results_with_metadata": (
        "General-model evaluation on external benchmark datasets from "
        "InstaDeepAI/winnow-ms-datasets (general_model_evaluation/). labelled/ holds "
        "database-search reference runs; full/ holds full-search predictions. Each "
        "project folder contains metadata.csv and preds_and_fdr_metrics.csv."
    ),
    "feature_importance": (
        "Feature-importance analysis for the general model on PXD014877 (C. elegans): permutation "
        "importance (perm_importance.pkl) and SHAP values (shap_values.pkl)."
    ),
    "generalisation": (
        "Leave-one-source-out calibrator generalisation metrics "
        "(calibrator_generalisation_results.csv). Corresponding calibrator "
        "checkpoints are in Figshare article 30147364 (trained_on_*/ folders, one "
        "model per held-out training source)."
    ),
    "novelty": (
        "Novel-peptide and non-tryptic digest analyses: calibration behaviour on "
        "peptides outside the standard tryptic training distribution (summary and "
        "per-dataset CSV tables)."
    ),
    "upscored_fps": (
        "Up-scored false positives: false positives pushed into high-confidence "
        "regions by calibration, compared with true positives "
        "(upscored_summary.csv, upscored_fp_detail.csv)."
    ),
    "fdr_overlap": (
        "Winnow vs database-search overlap at 1 %, 5 %, and 10 % nominal FDR: "
        "retained PSM and unique-peptide counts, discordance categories, and "
        "per-project overlap summaries."
    ),
    "ablations": (
        "Feature-ablation evaluation: aggregated metrics across withheld feature "
        "groups (ablation_summary.csv)."
    ),
}


def _html_paragraph(text: str) -> str:
    return f"<p>{html.escape(text)}</p>"


def _html_hf_revision_item(key: str, *, note: str = "") -> str:
    entry = FIGSHARE_HF_REVISIONS[key]
    tree_url = f"{entry['url']}/tree/{entry['sha']}"
    note_suffix = f" ({html.escape(note)})" if note else ""
    return (
        "<li>"
        f'<a href="{html.escape(entry["url"])}">{html.escape(entry["repo_id"])}</a> '
        f"@ <code>{html.escape(entry['sha'])}</code>{note_suffix}<br/>"
        f'<a href="{html.escape(tree_url)}">{html.escape(tree_url)}</a>'
        "</li>"
    )


def _html_figshare_calibrator_item(key: str, *, note: str = "") -> str:
    entry = FIGSHARE_CALIBRATOR_ARTICLES[key]
    note_suffix = f" ({html.escape(note)})" if note else ""
    return (
        "<li>"
        f"{html.escape(entry['title'])} "
        f'(<a href="{html.escape(entry["url"])}">{html.escape(entry["doi"])}</a>)'
        f"{note_suffix}<br/>"
        f"Folders: {html.escape(entry['folders'])}"
        "</li>"
    )


def _article_description(
    manifest: dict[str, Any],
    targets: list[UploadTarget],
    *,
    include_general_results_metadata: bool,
) -> str:
    """Return Figshare article description as HTML (plain newlines are not rendered)."""
    model_name = manifest.get("model_name", "train_extra_small_mass_error_da")
    folders = sorted({target.staging_rel.split("/", 1)[0] for target in targets})
    blocks: list[str] = [
        _html_paragraph("Updated winnow analysis outputs (tabular results)."),
    ]
    blocks.extend(
        [
            "<p><strong>Models and training data (Hugging Face, pinned revisions):</strong></p>",
            "<ul>",
            _html_hf_revision_item(
                "winnow-ms-datasets",
                note="training/evaluation spectra and InstaNovo predictions",
            ),
            _html_hf_revision_item(
                "winnow-general-model",
                note=(
                    "calibrator for general_results/, feature_importance/, and related "
                    f"analyses; config key {model_name}"
                ),
            ),
            _html_hf_revision_item(
                "winnow-helaqc-model",
                note="InstaNovo HeLa calibrator (helaqc_results/instanovo/)",
            ),
            "</ul>",
            "<p><strong>Additional calibrators (Figshare, same project/collection):</strong></p>",
            "<ul>",
            _html_figshare_calibrator_item(
                "hela_casanovo_primenovo",
                note="helaqc_results/casanovo/ and helaqc_results/primenovo/",
            ),
            _html_figshare_calibrator_item(
                "hold_one_out_generalisation",
                note="companion to generalisation/calibrator_generalisation_results.csv",
            ),
            "</ul>",
            "<p><strong>Outputs are organised into folders:</strong></p>",
            "<ul>",
        ]
    )
    for folder in folders:
        if folder == "general_results" and not include_general_results_metadata:
            detail = (
                FIGSHARE_FOLDER_DESCRIPTIONS["general_results"]
                + " metadata.csv is omitted from this deposition to stay within "
                "Figshare storage limits."
            )
        elif folder == "general_results":
            detail = FIGSHARE_FOLDER_DESCRIPTIONS["general_results_with_metadata"]
        else:
            detail = FIGSHARE_FOLDER_DESCRIPTIONS.get(
                folder, "See staging tree for contents."
            )
        blocks.append(
            f"<li><strong>{html.escape(folder)}/</strong> — {html.escape(detail)}</li>"
        )
    blocks.append("</ul>")
    predict_note = (
        "Predict outputs use paired metadata.csv (spectrum metadata) and "
        "preds_and_fdr_metrics.csv (per-candidate scores, calibration, and "
        "FDR/q-value columns)."
        if include_general_results_metadata
        else (
            "Predict outputs use preds_and_fdr_metrics.csv (per-candidate scores, "
            "calibration, and FDR/q-value columns). helaqc_results/ also includes "
            "metadata.csv; general_results/ metadata.csv is omitted from this "
            "deposition to stay within Figshare storage limits."
        )
    )
    blocks.extend(
        [
            _html_paragraph(
                predict_note
                + " Column definitions: winnow docs/cli.md (predict output section)."
            )
        ]
    )
    return "\n".join(blocks)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("configs/figshare_deposition.yaml")
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=Path("figshare_staging"),
    )
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--resume-state", type=Path, default=None)
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Glob relative to staging root; repeat for preflight uploads.",
    )
    parser.add_argument("--skip-description-update", action="store_true")
    parser.add_argument(
        "--update-description-only",
        action="store_true",
        help="Update the article description from the manifest and exit.",
    )
    parser.add_argument(
        "--list-remote-files",
        action="store_true",
        help="List files on the target article via account API and exit.",
    )
    parser.add_argument(
        "--delete-remote-name",
        action="append",
        default=[],
        help="Delete remote files whose Figshare name matches exactly; repeat as needed.",
    )
    parser.add_argument(
        "--include-general-results-metadata",
        action="store_true",
        help=(
            "Upload general_results/**/metadata.csv (default: excluded; see manifest "
            "include_general_results_metadata)."
        ),
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show a tqdm progress bar with ETA for bytes uploaded.",
    )
    parser.add_argument(
        "--storage-limit-gib",
        type=float,
        default=None,
        help=(
            "Fail if total upload size exceeds this limit "
            f"(default: manifest figshare_storage_limit_gib or "
            f"{FIGSHARE_DEFAULT_STORAGE_LIMIT_GIB})."
        ),
    )
    return parser.parse_args(argv)


def _storage_limit_bytes(manifest: dict[str, Any], args: argparse.Namespace) -> int:
    if args.storage_limit_gib is not None:
        return int(args.storage_limit_gib * GiB)
    limit_gib = manifest.get("figshare_storage_limit_gib")
    if limit_gib is None:
        limit_gib = FIGSHARE_DEFAULT_STORAGE_LIMIT_GIB
    return int(float(limit_gib) * GiB)


def _format_gib(num_bytes: int) -> str:
    return f"{num_bytes / GiB:.2f} GiB"


def _include_general_results_metadata(
    manifest: dict[str, Any],
    args: argparse.Namespace,
) -> bool:
    return manifest_includes_general_results_metadata(
        manifest,
        include_general_results_metadata=(
            True if args.include_general_results_metadata else None
        ),
    )


def _run_dry_run(
    client: FigshareClient,
    manifest: dict[str, Any],
    targets: list[UploadTarget],
) -> None:
    try:
        article_id = _resolve_article_id(manifest, client)
        print(f"Resolved article_id={article_id}")
    except (ValueError, RuntimeError) as exc:
        print(f"Article resolution skipped in dry-run: {exc}")
    for target in targets:
        _upload_target(
            client,
            article_id=0,
            target=target,
            state={"files": {}},
            dry_run=True,
            progress=_NoOpByteProgress(),
        )


def _run_upload(
    client: FigshareClient,
    manifest: dict[str, Any],
    targets: list[UploadTarget],
    args: argparse.Namespace,
    *,
    include_general_results_metadata: bool,
) -> int:
    article_id = _resolve_article_id(manifest, client)
    print(f"Uploading to article_id={article_id}")
    print(
        "Note: for a published article, new files appear in the draft editor first:\n"
        f"  https://figshare.com/account/articles/{article_id}\n"
        "They are not visible on the public DOI page until you publish a new version."
    )

    state = _load_state(args.resume_state)
    state["_path"] = args.resume_state

    if args.replace:
        _delete_existing_files(client, article_id, dry_run=False)
        state["files"] = {}
        _save_state(args.resume_state, state)
        print("Cleared resume state after --replace (all files will be re-uploaded).")

    total_bytes = sum(
        target.size
        for target in targets
        if state["files"].get(target.staging_rel, {}).get("status") != "COMPLETED"
    )
    progress = _make_byte_progress(
        enabled=args.progress,
        total_bytes=total_bytes,
        initial_bytes=0,
    )
    try:
        for target in targets:
            _upload_target(
                client,
                article_id,
                target,
                state,
                dry_run=False,
                progress=progress,
            )
    finally:
        progress.close()

    if not args.skip_description_update:
        client.update_article(
            article_id,
            {
                "description": _article_description(
                    manifest,
                    targets,
                    include_general_results_metadata=include_general_results_metadata,
                )
            },
        )

    if args.publish:
        client.publish_article(article_id)
        print(f"Published article {article_id}")
    return 0


def _handle_remote_file_ops(
    client: FigshareClient,
    manifest: dict[str, Any],
    targets: list[UploadTarget],
    args: argparse.Namespace,
) -> int | None:
    """Run remote list/delete helpers; return exit code when upload should stop."""
    if not args.list_remote_files and not args.delete_remote_name:
        return None

    article_id = _resolve_article_id(manifest, client)
    if args.delete_remote_name:
        deleted = _delete_remote_files_by_name(
            client, article_id, args.delete_remote_name
        )
        print(f"Deleted {deleted} remote file(s) from article {article_id}")

    if args.list_remote_files:
        article = client.get_article(article_id)
        folder_structure = article.get("folder_structure") or {}
        files = article.get("files") or client.list_article_files(article_id)
        print(f"Article {article_id}: {len(files)} files")
        for item in sorted(
            files,
            key=lambda row: _remote_file_display_path(row, folder_structure),
        ):
            remote_path = _remote_file_display_path(item, folder_structure)
            print(f"  {item.get('id')}\t{remote_path}\t{item.get('size')}")

    if not targets:
        return 0
    return None


def _check_staging_dir(
    staging_dir: Path,
    *,
    allow_missing_staging: bool,
) -> tuple[list[UploadTarget], int | None] | None:
    """Return early when staging dir is missing; None when upload prep may continue."""
    if staging_dir.is_dir():
        return None
    if allow_missing_staging:
        return [], None
    print(f"ERROR: staging dir not found: {staging_dir}", file=sys.stderr)
    return [], 1


def _empty_only_patterns_error(
    targets: list[UploadTarget],
    staging_dir: Path,
    only_patterns: list[str] | None,
) -> int | None:
    """Return exit code when --only matched no files."""
    if targets or not only_patterns:
        return None
    print(
        f"ERROR: --only matched 0 files under {staging_dir}. "
        f"Patterns: {only_patterns!r}",
        file=sys.stderr,
    )
    return 1


def _print_excluded_metadata_summary(
    staging_dir: Path,
    include_globs: list[str],
    exclude_globs: list[str],
) -> None:
    excluded_count, excluded_bytes = summarize_excluded_general_results_metadata(
        staging_dir, include_globs, exclude_globs
    )
    if excluded_count:
        print(
            f"Excluded {excluded_count} general_results metadata file(s) "
            f"({_format_gib(excluded_bytes)})"
        )


def _upload_size_limit_error(
    targets: list[UploadTarget],
    *,
    limit_bytes: int | None,
    include_metadata: bool,
) -> int | None:
    """Return exit code when file count or total size exceeds Figshare limits."""
    if len(targets) > FIGSHARE_FILE_LIMIT:
        print(
            f"ERROR: {len(targets)} files exceeds Figshare limit of "
            f"{FIGSHARE_FILE_LIMIT}",
            file=sys.stderr,
        )
        return 1

    total_size = sum(target.size for target in targets)
    if limit_bytes is None:
        print(f"Total upload size: {_format_gib(total_size)}")
        return None

    print(
        f"Total upload size: {_format_gib(total_size)} "
        f"(Figshare limit: {_format_gib(limit_bytes)})"
    )
    if total_size <= limit_bytes:
        return None

    print(
        f"ERROR: total upload size {_format_gib(total_size)} exceeds "
        f"Figshare storage limit {_format_gib(limit_bytes)}.",
        file=sys.stderr,
    )
    if not include_metadata:
        print(
            "general_results/**/metadata.csv is already excluded. "
            "Reduce the staged file set or request a higher quota.",
            file=sys.stderr,
        )
    else:
        print(
            "Try omitting general_results/**/metadata.csv "
            "(default) or reduce the staged file set.",
            file=sys.stderr,
        )
    return 1


def _prepare_targets(
    manifest: dict[str, Any],
    staging_dir: Path,
    only_patterns: list[str] | None,
    *,
    allow_missing_staging: bool,
    include_general_results_metadata: bool | None = None,
    storage_limit_bytes: int | None = None,
) -> tuple[list[UploadTarget], int | None]:
    """Build upload targets; return (targets, error_code) when staging is required."""
    early = _check_staging_dir(staging_dir, allow_missing_staging=allow_missing_staging)
    if early is not None:
        return early

    include_globs = manifest.get("include_globs", ["**/*"])
    exclude_globs = manifest.get("exclude_globs", [])
    include_metadata = manifest_includes_general_results_metadata(
        manifest,
        include_general_results_metadata=include_general_results_metadata,
    )

    targets = _build_targets(
        manifest,
        staging_dir,
        only_patterns,
        include_general_results_metadata=include_general_results_metadata,
    )
    if not targets:
        return targets, _empty_only_patterns_error(targets, staging_dir, only_patterns)

    print(f"Prepared {len(targets)} files for upload")
    if not include_metadata:
        _print_excluded_metadata_summary(staging_dir, include_globs, exclude_globs)

    limit_error = _upload_size_limit_error(
        targets,
        limit_bytes=storage_limit_bytes,
        include_metadata=include_metadata,
    )
    if limit_error is not None:
        return targets, limit_error
    return targets, None


def main(argv: list[str] | None = None) -> int:
    """Upload deposition files from a local staging tree to Figshare."""
    args = _parse_args(argv)
    token = os.environ.get("FIGSHARE_PAT")
    if not token and not args.dry_run:
        print("FIGSHARE_PAT is required unless --dry-run is set", file=sys.stderr)
        return 1

    manifest = load_manifest(args.manifest)
    only_patterns = args.only or None
    include_metadata_override = True if args.include_general_results_metadata else None
    include_general_results_metadata = _include_general_results_metadata(manifest, args)
    remote_only = (
        args.list_remote_files
        or args.delete_remote_name
        or args.update_description_only
    )
    storage_limit_bytes = _storage_limit_bytes(manifest, args)
    targets, target_error = _prepare_targets(
        manifest,
        args.staging_dir,
        only_patterns,
        allow_missing_staging=(
            remote_only and not args.dry_run and not args.update_description_only
        ),
        include_general_results_metadata=include_metadata_override,
        storage_limit_bytes=storage_limit_bytes,
    )
    if target_error is not None:
        return target_error

    if not include_general_results_metadata:
        print(
            "Note: excluding general_results/**/metadata.csv "
            "(pass --include-general-results-metadata or set "
            "include_general_results_metadata: true in the manifest to upload them)."
        )

    if args.dry_run:
        _run_dry_run(FigshareClient(token or "dry-run"), manifest, targets)
        return 0

    assert token is not None
    client = FigshareClient(token)

    remote_exit = _handle_remote_file_ops(client, manifest, targets, args)
    if remote_exit is not None:
        return remote_exit

    if args.update_description_only:
        article_id = _resolve_article_id(manifest, client)
        description = _article_description(
            manifest,
            targets,
            include_general_results_metadata=include_general_results_metadata,
        )
        client.update_article(article_id, {"description": description})
        print(f"Updated description on article {article_id}")
        return 0

    return _run_upload(
        client,
        manifest,
        targets,
        args,
        include_general_results_metadata=include_general_results_metadata,
    )


if __name__ == "__main__":
    raise SystemExit(main())
