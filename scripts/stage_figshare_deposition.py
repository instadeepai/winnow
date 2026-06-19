#!/usr/bin/env python3
"""Stage Figshare deposition files from S3 into a local directory tree."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from figshare_deposition_manifest import FileSpec, collect_file_specs, load_manifest

FIGSHARE_FILE_LIMIT = 500


def _download_file(
    spec: FileSpec, staging_dir: Path, profile: str, dry_run: bool
) -> None:
    local_path = staging_dir / spec.dest
    if dry_run:
        print(f"DRY-RUN download {spec.s3_uri} -> {local_path}")
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "aws",
        "s3",
        "cp",
        spec.s3_uri,
        str(local_path),
    ]
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    """Download matched deposition files from S3 into a local staging tree."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/figshare_deposition.yaml"),
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=Path("figshare_staging"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    profile = manifest.get("aws_profile", "winnow")
    specs = collect_file_specs(manifest)

    print(f"Matched {len(specs)} deposition files")
    if len(specs) > FIGSHARE_FILE_LIMIT:
        print(
            "WARNING: file count exceeds Figshare 500-file article limit",
            file=sys.stderr,
        )

    for spec in specs:
        _download_file(spec, args.staging_dir, profile, args.dry_run)

    print(f"Staging complete under {args.staging_dir} ({len(specs)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
