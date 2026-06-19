"""Resolve Figshare deposition file specs from configs/figshare_deposition.yaml."""

from __future__ import annotations

import fnmatch
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import yaml


@dataclass(frozen=True)
class FileSpec:
    """One S3 object mapped to a relative path in the Figshare article."""

    s3_uri: str
    dest: str


def load_manifest(path: Path) -> dict:
    """Load and parse the deposition YAML manifest."""
    with path.open() as handle:
        return yaml.safe_load(handle)


def _matches_any(path: str, patterns: list[str]) -> bool:
    normalized = path.replace("\\", "/")
    return any(PurePosixPath(normalized).match(pattern) for pattern in patterns)


def _should_include(
    dest: str,
    include_globs: list[str],
    exclude_globs: list[str],
) -> bool:
    if exclude_globs and _matches_any(dest, exclude_globs):
        return False
    if include_globs:
        return _matches_any(dest, include_globs)
    return True


def _basename_matches(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def _relative_dest(prefix_dest: str, key_suffix: str) -> str:
    suffix = key_suffix.lstrip("/")
    if prefix_dest:
        return f"{prefix_dest}/{suffix}".replace("//", "/")
    return suffix


# Flat keys under new_eval_sets_plots/fdr_overlap/ were uploaded without PXD folders.
FDR_OVERLAP_RUN_PROJECTS: dict[str, str] = {
    "20150708_QE3_UPLC8_DBJ_QC_HELA_39frac_Chymotrypsin": "PXD004452",
    "20151020_QE3_UPLC8_DBJ_SA_A549_Rep2_46": "PXD004452",
    "20151020_QE3_UPLC8_DBJ_SA_HCT116_Rep2_46": "PXD004452",
    "20170303_QEh1_LC2_FaMa_ChCh_SA_HLApI_JY_R1_exp2": "PXD006939",
    "20170609_QEh1_LC1_ChCh_FAMA_SA_HLAIIp_JY_all_R1": "PXD006939",
    "01747_C01_P018218_S00_I00_N03_R1": "PXD013868",
}

FDR_OVERLAP_FLAT_PROJECTS = frozenset({"PXD004732", "PXD014877", "PXD023064", "astral"})


def remap_fdr_overlap_rel_key(rel_key: str) -> str:
    """Place root-level fdr_overlap summaries under their PXD/run folders."""
    if "/" in rel_key:
        return rel_key

    name = PurePosixPath(rel_key).name
    if name == "all_projects_overlap_summary.csv":
        return rel_key

    if not name.endswith("_overlap_summary.csv"):
        return rel_key

    stem = name.removesuffix("_overlap_summary.csv")
    if stem in FDR_OVERLAP_RUN_PROJECTS:
        return f"{FDR_OVERLAP_RUN_PROJECTS[stem]}/{name}"
    if stem in FDR_OVERLAP_FLAT_PROJECTS:
        return f"{stem}/{name}"
    return rel_key


def _prefer_fdr_overlap_spec(existing: FileSpec, new: FileSpec) -> FileSpec:
    """When root and nested S3 keys map to the same dest, keep the nested key."""
    existing_depth = existing.s3_uri.count("/")
    new_depth = new.s3_uri.count("/")
    return new if new_depth > existing_depth else existing


def _list_s3_keys(s3_prefix: str, profile: str) -> list[str]:
    prefix = s3_prefix.rstrip("/") + "/"
    cmd = [
        "aws",
        "s3",
        "ls",
        prefix,
        "--recursive",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    uri = prefix.removeprefix("s3://")
    _, _, key_prefix = uri.partition("/")
    key_prefix = key_prefix.rstrip("/")

    relative_keys: list[str] = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        full_key = parts[-1]
        if key_prefix and full_key.startswith(key_prefix + "/"):
            relative_keys.append(full_key[len(key_prefix) + 1 :])
    return relative_keys


def _s3_join(prefix: str, suffix: str) -> str:
    return f"{prefix.rstrip('/')}/{suffix.lstrip('/')}"


def _collect_from_mappings(
    base: str,
    mappings: list[dict],
    filenames: list[str],
) -> list[FileSpec]:
    specs: list[FileSpec] = []
    for mapping in mappings:
        s3_dir = mapping["s3_dir"]
        dest_dir = mapping["dest"]
        for filename in filenames:
            specs.append(
                FileSpec(
                    s3_uri=_s3_join(_s3_join(base, s3_dir), filename),
                    dest=_relative_dest(dest_dir, filename),
                )
            )
    return specs


def _collect_from_explicit_files(entries: list[dict]) -> list[FileSpec]:
    return [FileSpec(s3_uri=item["s3"], dest=item["dest"]) for item in entries]


def _sync_key_matches(
    rel_key: str, file_patterns: list[str], *, use_basename_only: bool
) -> bool:
    basename = PurePosixPath(rel_key).name
    if use_basename_only:
        return _basename_matches(basename, file_patterns)
    if _matches_any(rel_key, file_patterns):
        return True
    return _basename_matches(basename, file_patterns)


def _collect_from_sync(
    sync_entries: list[dict],
    file_patterns: list[str],
    profile: str,
    *,
    use_basename_only: bool,
) -> list[FileSpec]:
    specs: list[FileSpec] = []
    for entry in sync_entries:
        s3_prefix = entry["s3_prefix"]
        dest_prefix = entry["dest"]
        for rel_key in _list_s3_keys(s3_prefix, profile):
            if not rel_key or rel_key.endswith("/"):
                continue
            if not _sync_key_matches(
                rel_key, file_patterns, use_basename_only=use_basename_only
            ):
                continue
            rel_dest = (
                remap_fdr_overlap_rel_key(rel_key)
                if dest_prefix == "fdr_overlap"
                else rel_key
            )
            specs.append(
                FileSpec(
                    s3_uri=_s3_join(s3_prefix, rel_key),
                    dest=_relative_dest(dest_prefix, rel_dest),
                )
            )
    return specs


def _sync_patterns(section_cfg: dict) -> list[str]:
    return section_cfg.get("files_in_sync") or section_cfg.get("files", [])


def _basename_only_patterns(patterns: list[str]) -> bool:
    return bool(
        patterns
        and all("/" not in pattern and "**" not in pattern for pattern in patterns)
    )


def _collect_explicit_file_specs(section_cfg: dict) -> list[FileSpec]:
    files = section_cfg.get("files")
    if not isinstance(files, list) or not files:
        return []
    if not isinstance(files[0], dict):
        return []
    return _collect_from_explicit_files(files)


def _collect_section_specs(section_cfg: dict, profile: str) -> list[FileSpec]:
    specs: list[FileSpec] = []
    if "mappings" in section_cfg:
        specs.extend(
            _collect_from_mappings(
                section_cfg["base"],
                section_cfg["mappings"],
                section_cfg.get("files", []),
            )
        )
    specs.extend(_collect_explicit_file_specs(section_cfg))
    if "sync" in section_cfg:
        patterns = _sync_patterns(section_cfg)
        specs.extend(
            _collect_from_sync(
                section_cfg["sync"],
                patterns,
                profile,
                use_basename_only=_basename_only_patterns(patterns),
            )
        )
    return specs


def collect_file_specs(manifest: dict) -> list[FileSpec]:
    """Return deduplicated deposition files from the manifest S3 sources."""
    profile = manifest.get("aws_profile", "winnow")
    include_globs = manifest.get("include_globs", [])
    exclude_globs = manifest.get("exclude_globs", [])
    specs: list[FileSpec] = []

    for section_cfg in manifest.get("s3_sources", {}).values():
        specs.extend(_collect_section_specs(section_cfg, profile))

    deduped: dict[str, FileSpec] = {}
    for spec in specs:
        if not _should_include(spec.dest, include_globs, exclude_globs):
            continue
        if spec.dest in deduped and spec.dest.startswith("fdr_overlap/"):
            deduped[spec.dest] = _prefer_fdr_overlap_spec(deduped[spec.dest], spec)
        else:
            deduped[spec.dest] = spec
    return sorted(deduped.values(), key=lambda item: item.dest)
