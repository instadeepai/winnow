#!/usr/bin/env python3
"""Upload staged winnow analysis outputs to a Figshare project article."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from figshare_deposition_manifest import collect_file_specs, load_manifest

FIGSHARE_API = "https://api.figshare.com/v2"
FIGSHARE_FILE_LIMIT = 500
DEFAULT_PART_SIZE = 10 * 1024 * 1024
MIN_REQUEST_INTERVAL_SECS = 1.0


@dataclass
class UploadTarget:
    """One staged local file to upload with its Figshare folder path."""

    figshare_name: str
    local_path: Path
    size: int


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
                return json.loads(body.decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise RuntimeError(f"{method} {url} failed ({exc.code}): {detail}") from exc

    def list_project_articles(self, project_id: int) -> list[dict[str, Any]]:
        """List articles in a Figshare project."""
        return self.request(
            "GET",
            f"{FIGSHARE_API}/account/projects/{project_id}/articles?page_size=100",
        )

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
    ) -> dict[str, Any]:
        """Start a multipart upload for one article file."""
        payload = json.dumps({"name": name, "size": size, "md5": md5}).encode()
        return self.request(
            "POST",
            f"{FIGSHARE_API}/account/articles/{article_id}/files",
            data=payload,
            headers={"Content-Type": "application/json"},
            expected_status=(201,),
        )

    def get_upload_parts(self, location: str) -> dict[str, Any]:
        """Fetch multipart upload part boundaries."""
        return self.request("GET", location)

    def upload_part(self, upload_url: str, part_number: int, chunk: bytes) -> None:
        """Upload one multipart chunk."""
        self._throttle()
        url = f"{upload_url}/{part_number}"
        request = urllib.request.Request(
            url,
            data=chunk,
            headers={"Content-Type": "application/octet-stream"},
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

    def complete_file_upload(self, article_id: int, file_id: int) -> None:
        """Finalize a multipart upload."""
        self.request(
            "POST",
            f"{FIGSHARE_API}/account/articles/{article_id}/files/{file_id}",
            expected_status=(202,),
        )

    def update_article(self, article_id: int, payload: dict[str, Any]) -> None:
        """Update article metadata."""
        body = json.dumps(payload).encode()
        self.request(
            "PUT",
            f"{FIGSHARE_API}/account/articles/{article_id}",
            data=body,
            headers={"Content-Type": "application/json"},
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


def _read_local_chunks(path: Path, part_size: int):
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(part_size)
            if not chunk:
                break
            yield chunk


def _s3_object_size(s3_uri: str, profile: str) -> int:
    cmd = ["aws", "s3", "ls", s3_uri]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise FileNotFoundError(f"S3 object not found: {s3_uri}")
    parts = lines[-1].split()
    return int(parts[2])


def _target_size(local_path: Path, s3_uri: str, profile: str) -> int:
    if local_path.exists():
        return local_path.stat().st_size
    return _s3_object_size(s3_uri, profile)


def _build_targets(
    manifest: dict[str, Any],
    staging_dir: Path,
    only_patterns: list[str] | None,
) -> list[UploadTarget]:
    specs = collect_file_specs(manifest)
    profile = manifest.get("aws_profile", "winnow")
    targets: list[UploadTarget] = []

    for spec in specs:
        if only_patterns and not any(
            PurePosixPath(spec.dest).match(pattern) for pattern in only_patterns
        ):
            continue
        local_path = staging_dir / spec.dest
        targets.append(
            UploadTarget(
                figshare_name=spec.dest,
                local_path=local_path,
                size=_target_size(local_path, spec.s3_uri, profile),
            )
        )
    return targets


def _load_state(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"files": {}}
    with path.open() as handle:
        return json.load(handle)


def _save_state(path: Path | None, state: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)


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


def _upload_target(
    client: FigshareClient,
    article_id: int,
    target: UploadTarget,
    state: dict[str, Any],
    *,
    dry_run: bool,
) -> None:
    file_state = state["files"].get(target.figshare_name, {})
    if file_state.get("status") == "COMPLETED":
        print(f"SKIP completed {target.figshare_name}")
        return

    if dry_run:
        print(
            f"DRY-RUN upload {target.figshare_name} ({target.size} bytes) "
            f"<- {target.local_path}"
        )
        return

    if not target.local_path.exists():
        raise FileNotFoundError(
            f"Missing staged file: {target.local_path}. "
            "Run make stage-figshare-deposition or download-figshare-staging first."
        )

    md5 = _file_md5(target.local_path)
    init = client.initiate_file_upload(
        article_id,
        name=target.figshare_name,
        size=target.size,
        md5=md5,
    )
    file_id = int(init["id"])
    location = init["location"]
    upload_info = client.get_upload_parts(location)
    part_size = int(upload_info.get("part_size", DEFAULT_PART_SIZE))

    file_state = {
        "file_id": file_id,
        "status": "UPLOADING",
        "parts_uploaded": file_state.get("parts_uploaded", []),
    }
    state["files"][target.figshare_name] = file_state
    _save_state(state.get("_path"), state)

    part_number = 1
    for chunk in _read_local_chunks(target.local_path, part_size):
        if part_number in file_state["parts_uploaded"]:
            part_number += 1
            continue
        client.upload_part(location, part_number, chunk)
        file_state["parts_uploaded"].append(part_number)
        _save_state(state.get("_path"), state)
        part_number += 1

    client.complete_file_upload(article_id, file_id)
    file_state["status"] = "COMPLETED"
    _save_state(state.get("_path"), state)
    print(f"Uploaded {target.figshare_name}")


def _article_description(manifest: dict[str, Any], targets: list[UploadTarget]) -> str:
    version = os.environ.get("WINNOW_VERSION", "unknown")
    model_name = manifest.get("model_name", "train_extra_small_mass_error_da")
    folders = sorted({target.figshare_name.split("/", 1)[0] for target in targets})
    lines = [
        "Updated winnow analysis outputs (tabular results; plots excluded).",
        f"Winnow version: {version}",
        f"Model for general_results/ and feature-importance analyses: {model_name}",
        "",
        "Top-level folders:",
    ]
    for folder in folders:
        lines.append(f"- {folder}/")
    lines.extend(
        [
            "",
            "Column glossary: see winnow docs/cli.md (predict output section).",
            "Hold-one-out calibrator models are deposited separately in this Figshare project.",
            "Plots live under new_eval_sets_plots/ on the analysis S3 bucket and are not included here.",
        ]
    )
    return "\n".join(lines)


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
    return parser.parse_args(argv)


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
        )


def _run_upload(
    client: FigshareClient,
    manifest: dict[str, Any],
    targets: list[UploadTarget],
    args: argparse.Namespace,
) -> int:
    article_id = _resolve_article_id(manifest, client)
    print(f"Uploading to article_id={article_id}")

    state = _load_state(args.resume_state)
    state["_path"] = args.resume_state

    if args.replace:
        _delete_existing_files(client, article_id, dry_run=False)

    for target in targets:
        _upload_target(
            client,
            article_id,
            target,
            state,
            dry_run=False,
        )

    if not args.skip_description_update:
        client.update_article(
            article_id,
            {"description": _article_description(manifest, targets)},
        )

    if args.publish:
        client.publish_article(article_id)
        print(f"Published article {article_id}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Upload deposition files from a local staging tree to Figshare."""
    args = _parse_args(argv)
    token = os.environ.get("FIGSHARE_TOKEN")
    if not token and not args.dry_run:
        print("FIGSHARE_TOKEN is required unless --dry-run is set", file=sys.stderr)
        return 1

    manifest = load_manifest(args.manifest)
    only_patterns = args.only or None
    targets = _build_targets(manifest, args.staging_dir, only_patterns)

    print(f"Prepared {len(targets)} files for upload")
    if len(targets) > FIGSHARE_FILE_LIMIT:
        print(
            f"ERROR: {len(targets)} files exceeds Figshare limit of {FIGSHARE_FILE_LIMIT}",
            file=sys.stderr,
        )
        return 1

    total_size = sum(target.size for target in targets)
    print(f"Total size: {total_size / (1024**3):.2f} GiB")

    if args.dry_run:
        _run_dry_run(FigshareClient(token or "dry-run"), manifest, targets)
        return 0

    assert token is not None
    return _run_upload(FigshareClient(token), manifest, targets, args)


if __name__ == "__main__":
    raise SystemExit(main())
