#!/usr/bin/env python3
"""Drop mz_array and intensity_array from general_results metadata.csv files in place."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from figshare_deposition_manifest import is_general_results_metadata

ARRAY_COLUMNS = ("mz_array", "intensity_array")
GiB = 1024**3


def _format_bytes(num_bytes: int) -> str:
    if num_bytes >= GiB:
        return f"{num_bytes / GiB:.2f} GiB ({num_bytes:,} B)"
    if num_bytes >= 1024**2:
        return f"{num_bytes / (1024**2):.2f} MiB ({num_bytes:,} B)"
    return f"{num_bytes:,} B"


def _iter_general_results_metadata(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    paths: list[Path] = []
    for path in sorted(root.rglob("metadata.csv")):
        rel = path.relative_to(root).as_posix()
        if is_general_results_metadata(rel):
            paths.append(path)
    return paths


def _strip_arrays(path: Path, *, dry_run: bool) -> tuple[int, int, bool]:
    """Return (size_before, size_after, changed)."""
    size_before = path.stat().st_size
    header = pd.read_csv(path, nrows=0).columns.tolist()
    drop = [column for column in ARRAY_COLUMNS if column in header]
    if not drop:
        return size_before, size_before, False

    keep = [column for column in header if column not in drop]
    if dry_run:
        print(
            f"DRY-RUN {path}: would drop {drop} "
            f"({len(keep)} columns kept, {_format_bytes(size_before)} now)"
        )
        return size_before, size_before, True

    df = pd.read_csv(path, usecols=keep)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)
    size_after = path.stat().st_size
    return size_before, size_after, True


def main(argv: list[str] | None = None) -> int:
    """Drop mz_array and intensity_array from general_results metadata.csv files in place."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("figshare_staging"),
        help="Root directory containing general_results/ (default: figshare_staging).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report files and columns that would be changed without writing.",
    )
    args = parser.parse_args(argv)

    metadata_paths = _iter_general_results_metadata(args.root)
    if not metadata_paths:
        print(
            f"No general_results/**/metadata.csv files under {args.root}",
            file=sys.stderr,
        )
        return 1

    print(
        f"Found {len(metadata_paths)} general_results metadata file(s) under {args.root}"
    )

    rows: list[tuple[str, int, int, bool]] = []
    for path in metadata_paths:
        rel = path.relative_to(args.root).as_posix()
        size_before, size_after, changed = _strip_arrays(path, dry_run=args.dry_run)
        rows.append((rel, size_before, size_after, changed))
        if changed and not args.dry_run:
            saved = size_before - size_after
            print(
                f"Updated {rel}: {_format_bytes(size_before)} -> "
                f"{_format_bytes(size_after)} "
                f"(saved {_format_bytes(saved)})"
            )
        elif not changed:
            print(f"Skipped {rel}: no {list(ARRAY_COLUMNS)} columns")

    if args.dry_run:
        would_change = sum(1 for _, _, _, changed in rows if changed)
        print(f"DRY-RUN: would update {would_change} file(s)")
        return 0

    total_after = sum(size_after for _, _, size_after, _ in rows)
    total_before = sum(size_before for _, size_before, _, changed in rows if changed)
    total_after_changed = sum(
        size_after for _, _, size_after, changed in rows if changed
    )

    print()
    print("Metadata sizes after stripping:")
    for rel, _, size_after, _ in rows:
        print(f"  {rel}\t{_format_bytes(size_after)}")
    print()
    print(f"Total metadata size (all {len(rows)} files): {_format_bytes(total_after)}")
    if total_before:
        print(
            f"Total saved in updated files: "
            f"{_format_bytes(total_before - total_after_changed)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
