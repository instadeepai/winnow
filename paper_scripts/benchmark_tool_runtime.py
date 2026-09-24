#!/usr/bin/env python3
r"""One-shot HeLa QC CLI runtime benchmark: Winnow, Glissade, and NovoBoard.

Times each tool's shipped CLI on authentic prepared HeLa QC inputs (no row
filtering, no nested scaling). NovoBoard times FDR estimation only
(precomputed target+decoy CSVs); decoy spectrum generation and DNS on decoys
are omitted.

Usage:
    uv run python scripts/benchmark_tool_runtime.py \\
        --winnow-repo /path/to/winnow-checkout \\
        --novoboard-dir /path/to/novoboard/helaqc \\
        --novoboard-repo /path/to/NovoBoard \\
        --glissade-dir /path/to/glissade/build/helaqc \\
        --glissade-repo /path/to/glissade
    make benchmark-tools-helaqc
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import resource
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import polars as pl

_REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_WINNOW_SPECTRA = (
    _REPO_ROOT / "paper_data/winnow-ms-datasets/helaqc/unlabelled.parquet"
)
DEFAULT_WINNOW_PREDS = (
    _REPO_ROOT / "paper_data/winnow-ms-datasets/helaqc/instanovo/unlabelled_preds.csv"
)
DEFAULT_FASTA = _REPO_ROOT / "paper_data/winnow-ms-datasets/fasta/human.fasta"

DEFAULT_NOVOBOARD_DECOY_RATE = "0.50"

# Library API default is 10; CLI ships 100.
DEFAULT_GLISSADE_N_BOOTSTRAPS = 10

DEFAULT_WINNOW_MODEL = "InstaDeepAI/winnow-general-model"

DEFAULT_OUTPUT_DIR = _REPO_ROOT / "paper_results/runtime"
DEFAULT_SCRATCH_DIR = _REPO_ROOT / "paper_results/runtime/scratch_helaqc"


def _resolve_on_path(name: str) -> Path | None:
    """Return ``Path`` to *name* on ``PATH``, or ``None`` if missing."""
    found = shutil.which(name)
    return Path(found) if found else None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RepoIdentity:
    """Git and package provenance for one tool checkout."""

    name: str
    repo_root: str
    git_branch: str
    git_commit: str
    git_commit_short: str
    git_dirty: bool
    remote_name: str
    remote_url: str
    remote_ref: str
    remote_commit_url: str
    package_version: str
    cli_path: str


@dataclass
class InputPaths:
    """Authentic full-input paths for the one-shot HeLa QC timing."""

    n_winnow: int
    winnow_spectra: Path
    winnow_preds: Path
    novoboard_target: Path
    novoboard_decoy: Path
    glissade_denovo: Path
    glissade_labelled: Path
    counts: dict[str, int] = field(default_factory=dict)


@dataclass
class RunResult:
    """Timed CLI result for one method."""

    method: str
    n_rows: int
    wall_time_s: float
    returncode: int
    command: list[str]
    log_path: str
    repo: dict[str, Any] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Hardware / provenance
# ---------------------------------------------------------------------------


def get_hardware_info() -> dict[str, str]:
    """Collect coarse host hardware metadata."""
    info: dict[str, str] = {}
    cpu_name = None
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_name = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    info["cpu"] = cpu_name or platform.processor() or "unknown"
    info["cpu_cores"] = str(os.cpu_count() or "unknown")
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    info["ram_gb"] = f"{kb / (1024**2):.0f}"
                    break
    except OSError:
        info["ram_gb"] = "unknown"
    info["gpu"] = "none"
    return info


def _run_git(repo: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(repo),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _github_https(remote_url: str) -> str:
    url = remote_url.strip()
    if url.startswith("git@"):
        url = re.sub(r"^git@([^:]+):", r"https://\1/", url)
    if url.endswith(".git"):
        url = url[:-4]
    return url


def _read_pyproject_version(repo: Path) -> str:
    pyproject = repo / "pyproject.toml"
    if not pyproject.is_file():
        return "unknown"
    text = pyproject.read_text()
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return match.group(1) if match else "unknown"


def _resolve_remote_fields(root: Path, branch: str) -> tuple[str, str, str, str]:
    remote_name = _run_git(root, "config", f"branch.{branch}.remote") or "origin"
    remote_url = _run_git(root, "remote", "get-url", remote_name) or ""
    remote_ref = (
        _run_git(root, "rev-parse", "--abbrev-ref", f"{remote_name}/{branch}")
        or f"{remote_name}/{branch}"
    )
    commit = _run_git(root, "rev-parse", "HEAD") or ""
    https = _github_https(remote_url) if remote_url else ""
    commit_url = f"{https}/commit/{commit}" if https and commit else ""
    return remote_name, remote_url, remote_ref, commit_url


def resolve_repo_identity(
    name: str,
    repo_root: Path,
    *,
    cli_path: str,
) -> RepoIdentity:
    """Resolve git + package provenance for a tool checkout."""
    root = repo_root.resolve()
    branch = _run_git(root, "rev-parse", "--abbrev-ref", "HEAD") or "unknown"
    commit = _run_git(root, "rev-parse", "HEAD") or "unknown"
    short = commit[:12] if commit != "unknown" else "unknown"
    dirty_out = _run_git(root, "status", "--porcelain")
    dirty = bool(dirty_out)
    remote_name, remote_url, remote_ref, commit_url = _resolve_remote_fields(
        root, branch
    )
    return RepoIdentity(
        name=name,
        repo_root=str(root),
        git_branch=branch,
        git_commit=commit,
        git_commit_short=short,
        git_dirty=dirty,
        remote_name=remote_name,
        remote_url=remote_url,
        remote_ref=remote_ref,
        remote_commit_url=commit_url,
        package_version=_read_pyproject_version(root),
        cli_path=cli_path,
    )


def infer_repo_from_bin(bin_path: Path) -> Path:
    """Walk up from a venv binary to find a repo root with ``.git``."""
    cur = bin_path.resolve().parent
    for _ in range(8):
        if (cur / ".git").exists() or (cur / "pyproject.toml").is_file():
            if (cur / ".git").exists() or (cur / ".git").is_file():
                return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return bin_path.resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def build_input_paths(
    *,
    scratch_dir: Path,
    winnow_spectra: Path,
    winnow_preds: Path,
    novoboard_target: Path,
    novoboard_decoy: Path,
    glissade_denovo: Path,
    glissade_labelled: Path,
) -> InputPaths:
    """Use authentic full inputs as-is (no row filtering)."""
    n_winnow = int(pl.scan_parquet(winnow_spectra).select(pl.len()).collect().item())
    n_preds = int(pl.scan_csv(winnow_preds).select(pl.len()).collect().item())
    if n_winnow != n_preds:
        raise ValueError(
            f"Winnow spectra/preds length mismatch: {n_winnow} vs {n_preds}"
        )
    n_nb = int(pl.scan_csv(novoboard_target).select(pl.len()).collect().item())
    n_nb_decoy = int(pl.scan_csv(novoboard_decoy).select(pl.len()).collect().item())
    n_g_denovo = int(pl.scan_parquet(glissade_denovo).select(pl.len()).collect().item())
    n_g_lab = int(pl.scan_parquet(glissade_labelled).select(pl.len()).collect().item())

    counts = {
        "winnow_spectra": n_winnow,
        "novoboard_targets": n_nb,
        "novoboard_decoys": n_nb_decoy,
        "glissade_denovo": n_g_denovo,
        "glissade_labelled": n_g_lab,
    }
    print(
        "  Authentic HeLa QC inputs "
        f"(Winnow={n_winnow:,}; NovoBoard target/decoy="
        f"{n_nb:,}/{n_nb_decoy:,}; Glissade denovo/labelled="
        f"{n_g_denovo:,}/{n_g_lab:,})"
    )
    scratch_dir.mkdir(parents=True, exist_ok=True)
    return InputPaths(
        n_winnow=n_winnow,
        winnow_spectra=winnow_spectra,
        winnow_preds=winnow_preds,
        novoboard_target=novoboard_target,
        novoboard_decoy=novoboard_decoy,
        glissade_denovo=glissade_denovo,
        glissade_labelled=glissade_labelled,
        counts=counts,
    )


# ---------------------------------------------------------------------------
# CLI runners
# ---------------------------------------------------------------------------


def run_timed_cli(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    env: dict[str, str] | None = None,
) -> tuple[float, int]:
    """Run a CLI, capture logs, return (wall_time_s, returncode)."""
    cwd.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    t0 = time.perf_counter()
    with open(log_path, "w") as log_f:
        log_f.write("COMMAND: " + " ".join(command) + "\n")
        log_f.write(f"CWD: {cwd}\n\n")
        log_f.flush()
        completed = subprocess.run(
            list(command),
            cwd=str(cwd),
            env=merged_env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            check=False,
        )
    wall = time.perf_counter() - t0
    return wall, completed.returncode


def run_winnow(
    inputs: InputPaths,
    *,
    winnow_repo: Path,
    model: str,
    scratch_dir: Path,
    identity: RepoIdentity,
) -> RunResult:
    """Time ``uv run winnow predict`` from the main checkout."""
    out_dir = scratch_dir / "winnow_out"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = scratch_dir / "winnow.log"

    command = [
        "uv",
        "run",
        "winnow",
        "predict",
        f"dataset.spectrum_path_or_directory={inputs.winnow_spectra.resolve()}",
        f"dataset.predictions_path={inputs.winnow_preds.resolve()}",
        f"calibrator.pretrained_model_name_or_path={model}",
        f"output_folder={out_dir.resolve()}/",
    ]
    wall, rc = run_timed_cli(command, cwd=winnow_repo, log_path=log_path)
    if rc != 0:
        raise RuntimeError(f"Winnow predict failed (exit {rc}); see {log_path}")
    return RunResult(
        method="Winnow",
        n_rows=inputs.n_winnow,
        wall_time_s=wall,
        returncode=rc,
        command=command,
        log_path=str(log_path),
        repo=asdict(identity),
        detail={"input": "spectra+predictions", "n_spectra": inputs.n_winnow},
    )


def run_glissade(
    inputs: InputPaths,
    *,
    glissade_bin: Path,
    fasta: Path,
    scratch_dir: Path,
    identity: RepoIdentity,
    n_bootstraps: int = DEFAULT_GLISSADE_N_BOOTSTRAPS,
) -> RunResult:
    """Time Glissade parquet-mode CLI (``-n`` bootstraps)."""
    out_cwd = scratch_dir / "glissade_out"
    if out_cwd.exists():
        shutil.rmtree(out_cwd)
    out_cwd.mkdir(parents=True, exist_ok=True)
    log_path = scratch_dir / "glissade.log"

    command = [
        str(glissade_bin.resolve()),
        "--parquet",
        "-n",
        str(n_bootstraps),
        str(inputs.glissade_denovo.resolve()),
        str(inputs.glissade_labelled.resolve()),
        str(fasta.resolve()),
    ]
    wall, rc = run_timed_cli(command, cwd=out_cwd, log_path=log_path)
    if rc != 0:
        raise RuntimeError(f"Glissade failed (exit {rc}); see {log_path}")
    n_denovo = inputs.counts["glissade_denovo"]
    n_labelled = inputs.counts["glissade_labelled"]
    return RunResult(
        method="Glissade",
        n_rows=n_denovo,
        wall_time_s=wall,
        returncode=rc,
        command=command,
        log_path=str(log_path),
        repo=asdict(identity),
        detail={
            "n_bootstraps": n_bootstraps,
            "n_denovo": n_denovo,
            "n_labelled": n_labelled,
        },
    )


def run_novoboard(
    inputs: InputPaths,
    *,
    novoboard_bin: Path,
    scratch_dir: Path,
    identity: RepoIdentity,
) -> RunResult:
    """Time NovoBoard FDR estimation-only on real target+decoy CSVs."""
    out_cwd = scratch_dir / "novoboard_out"
    if out_cwd.exists():
        shutil.rmtree(out_cwd)
    out_cwd.mkdir(parents=True, exist_ok=True)
    log_path = scratch_dir / "novoboard.log"

    command = [
        str(novoboard_bin.resolve()),
        "fdr",
        "--target-file",
        str(inputs.novoboard_target.resolve()),
        "--decoy-files",
        str(inputs.novoboard_decoy.resolve()),
    ]
    wall, rc = run_timed_cli(command, cwd=out_cwd, log_path=log_path)
    if rc != 0:
        raise RuntimeError(f"NovoBoard fdr failed (exit {rc}); see {log_path}")
    n_targets = inputs.counts["novoboard_targets"]
    n_decoys = inputs.counts["novoboard_decoys"]
    return RunResult(
        method="NovoBoard",
        n_rows=n_targets,
        wall_time_s=wall,
        returncode=rc,
        command=command,
        log_path=str(log_path),
        repo=asdict(identity),
        detail={
            "n_targets": n_targets,
            "n_decoys": n_decoys,
            "scope": "fdr_estimation_only",
        },
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_provenance(identities: Sequence[RepoIdentity]) -> str:
    """Format tool provenance block for the printed report."""
    lines = ["Tool provenance:"]
    for ident in identities:
        lines.append(
            f"  {ident.name}: v{ident.package_version}  "
            f"branch={ident.git_branch}  commit={ident.git_commit_short}  "
            f"dirty={str(ident.git_dirty).lower()}"
        )
        lines.append(f"             remote={ident.remote_url}  ref={ident.remote_ref}")
        if ident.remote_commit_url:
            lines.append(f"             {ident.remote_commit_url}")
        lines.append(f"             cli={ident.cli_path}")
    return "\n".join(lines)


def format_report(
    hw: dict[str, str],
    identities: Sequence[RepoIdentity],
    results: Sequence[RunResult],
) -> str:
    """Format a human-readable wall-clock report."""
    lines = [
        "=" * 88,
        "  HeLa QC one-shot cross-tool runtime benchmark",
        "=" * 88,
        "",
        f"Hardware: {hw['cpu']} ({hw['cpu_cores']} cores), "
        f"{hw['ram_gb']} GB RAM, GPU: {hw['gpu']}",
        "",
        format_provenance(identities),
        "",
    ]
    col_w = [28, 12, 14, 12]
    hdr = (
        f"{'Method':<{col_w[0]}}| {'n_rows':>{col_w[1]}}| "
        f"{'Wall time (s)':>{col_w[2]}}| {'Exit':>{col_w[3]}}"
    )
    sep = (
        "-" * col_w[0]
        + "|"
        + "-" * (col_w[1] + 1)
        + "|"
        + "-" * (col_w[2] + 1)
        + "|"
        + "-" * (col_w[3] + 1)
    )
    lines.append(hdr)
    lines.append(sep)
    for r in results:
        lines.append(
            f"{r.method:<{col_w[0]}}| {r.n_rows:>{col_w[1]},}| "
            f"{r.wall_time_s:>{col_w[2]}.2f}| {r.returncode:>{col_w[3]}}"
        )
    lines.append(sep)
    lines.append("")
    n_boot = next(
        (
            int(r.detail["n_bootstraps"])
            for r in results
            if r.method == "Glissade" and "n_bootstraps" in r.detail
        ),
        DEFAULT_GLISSADE_N_BOOTSTRAPS,
    )
    lines.append("Notes:")
    lines.append("- Timed via each tool's shipped CLI on authentic HeLa QC inputs.")
    lines.append(
        "- NovoBoard: FDR estimation only; decoy MGF generation and DNS on "
        "decoys omitted."
    )
    lines.append(
        f"- Glissade: n_bootstraps={n_boot} (library API default is 10; CLI ships 100)."
    )
    lines.append("- Winnow: main checkout + InstaDeepAI/winnow-general-model.")
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    lines.append(f"- Parent process peak RSS: {peak_rss_mb:.1f} MB")
    lines.append("")
    return "\n".join(lines)


def build_json_report(
    hw: dict[str, str],
    identities: Sequence[RepoIdentity],
    results: Sequence[RunResult],
    *,
    glissade_n_bootstraps: int,
    novoboard_decoy_rate: str,
    input_counts: dict[str, int],
) -> dict[str, Any]:
    """Build structured JSON payload."""
    return {
        "mode": "helaqc_oneshot",
        "hardware": hw,
        "glissade_n_bootstraps": glissade_n_bootstraps,
        "input_counts": input_counts,
        "tools": {ident.name: asdict(ident) for ident in identities},
        "runs": [
            {
                "method": r.method,
                "n_rows": r.n_rows,
                "wall_time_s": r.wall_time_s,
                "returncode": r.returncode,
                "command": r.command,
                "log_path": r.log_path,
                "repo": r.repo,
                "detail": r.detail,
            }
            for r in results
        ],
        "omissions": {
            "novoboard_decoy_generation": "novoboard decoy MGF generation not timed",
            "novoboard_decoy_dns": "DNS re-sequencing of decoy spectra not timed",
            "novoboard_decoy_rate_tuning": (
                f"decoy-rate grid search not timed; fixed rate {novoboard_decoy_rate}"
            ),
            "winnow_training": "calibrator training not timed",
        },
        "process_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the HeLa QC one-shot runtime benchmark."""
    parser = argparse.ArgumentParser(
        description="One-shot HeLa QC CLI runtime: Winnow, Glissade, NovoBoard.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--spectrum-path",
        type=Path,
        default=DEFAULT_WINNOW_SPECTRA,
        help="Winnow HeLa QC spectra parquet.",
    )
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=DEFAULT_WINNOW_PREDS,
        help="Winnow HeLa QC predictions CSV.",
    )
    parser.add_argument(
        "--winnow-repo",
        type=Path,
        required=True,
        help="Winnow checkout used for CLI (expected: main).",
    )
    parser.add_argument(
        "--winnow-model",
        default=DEFAULT_WINNOW_MODEL,
        help="Calibrator path or HF id (default: InstaDeepAI/winnow-general-model).",
    )
    parser.add_argument(
        "--novoboard-dir",
        type=Path,
        required=True,
        help="NovoBoard HeLa QC dataset directory (parent of CSVs).",
    )
    parser.add_argument(
        "--novoboard-bin",
        type=Path,
        default=_resolve_on_path("novoboard"),
        help="Path to novoboard executable (default: first match on PATH).",
    )
    parser.add_argument(
        "--novoboard-repo",
        type=Path,
        required=True,
        help="NovoBoard repository root for provenance.",
    )
    parser.add_argument(
        "--novoboard-decoy-rate",
        default=DEFAULT_NOVOBOARD_DECOY_RATE,
        help="Decoy rate suffix for decoy CSV (default: 0.50).",
    )
    parser.add_argument(
        "--glissade-dir",
        type=Path,
        required=True,
        help="Glissade build/helaqc directory.",
    )
    parser.add_argument(
        "--glissade-bin",
        type=Path,
        default=_resolve_on_path("glissade"),
        help="Path to glissade executable (default: first match on PATH).",
    )
    parser.add_argument(
        "--glissade-repo",
        type=Path,
        required=True,
        help="Glissade repository root for provenance.",
    )
    parser.add_argument(
        "--glissade-n-bootstraps",
        type=int,
        default=DEFAULT_GLISSADE_N_BOOTSTRAPS,
        help=(
            "Glissade -n / --n_bootstraps (default: "
            f"{DEFAULT_GLISSADE_N_BOOTSTRAPS}, matching the library API)."
        ),
    )
    parser.add_argument(
        "--fasta",
        type=Path,
        default=DEFAULT_FASTA,
        help="Reference FASTA for Glissade (default: human).",
    )
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=DEFAULT_SCRATCH_DIR,
        help="Directory for CLI outputs and logs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for JSON report.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="JSON report path (default: <output-dir>/tool_runtime_helaqc_oneshot.json).",
    )
    parser.add_argument(
        "--skip-winnow",
        action="store_true",
        help="Skip Winnow CLI run.",
    )
    parser.add_argument(
        "--skip-glissade",
        action="store_true",
        help="Skip Glissade CLI run.",
    )
    parser.add_argument(
        "--skip-novoboard",
        action="store_true",
        help="Skip NovoBoard CLI run.",
    )
    return parser.parse_args()


def _require_bin(path: Path | None, *, name: str, flag: str) -> None:
    if path is None or not Path(path).is_file():
        raise FileNotFoundError(
            f"{name} binary not found: {path!r}. Pass {flag} or put '{name.lower()}' on PATH."
        )


def _preflight(args: argparse.Namespace) -> tuple[Path, Path]:
    """Validate inputs and return NovoBoard target/decoy paths."""
    nb_target = args.novoboard_dir / "raw_unlabelled.csv"
    nb_decoy = (
        args.novoboard_dir / f"raw_unlabelled_decoy_{args.novoboard_decoy_rate}.csv"
    )
    required_files = [
        args.spectrum_path,
        args.predictions_path,
        args.fasta,
        args.glissade_dir / "glissade_denovo.parquet",
        args.glissade_dir / "glissade_labelled.parquet",
        nb_target,
        nb_decoy,
    ]
    for path in required_files:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Required input not found: {path}")

    if not args.skip_winnow:
        if not args.winnow_repo.is_dir():
            raise FileNotFoundError(
                f"Winnow repo not found: {args.winnow_repo}. "
                "Create a main worktree (see Makefile benchmark-tools-main-worktree)."
            )
        if not (args.winnow_repo / "pyproject.toml").is_file():
            raise FileNotFoundError(
                f"Winnow repo looks invalid (no pyproject.toml): {args.winnow_repo}"
            )
    if not args.skip_glissade:
        _require_bin(args.glissade_bin, name="Glissade", flag="--glissade-bin")
    if not args.skip_novoboard:
        _require_bin(args.novoboard_bin, name="NovoBoard", flag="--novoboard-bin")
    return nb_target, nb_decoy


def _collect_identities(args: argparse.Namespace) -> list[RepoIdentity]:
    """Build provenance identities for the tools that will be timed."""
    identities: list[RepoIdentity] = []
    if not args.skip_winnow:
        identities.append(
            resolve_repo_identity(
                "Winnow",
                args.winnow_repo,
                cli_path=f"uv run winnow ({args.winnow_repo})",
            )
        )
    if not args.skip_glissade:
        g_repo = args.glissade_repo
        if not g_repo.is_dir():
            g_repo = infer_repo_from_bin(args.glissade_bin)
        identities.append(
            resolve_repo_identity(
                "Glissade",
                g_repo,
                cli_path=str(args.glissade_bin.resolve()),
            )
        )
    if not args.skip_novoboard:
        n_repo = args.novoboard_repo
        if not n_repo.is_dir():
            n_repo = infer_repo_from_bin(args.novoboard_bin)
        identities.append(
            resolve_repo_identity(
                "NovoBoard",
                n_repo,
                cli_path=str(args.novoboard_bin.resolve()),
            )
        )
    return identities


def _identity_by_name(identities: Sequence[RepoIdentity], name: str) -> RepoIdentity:
    """Return the provenance record for ``name``."""
    for ident in identities:
        if ident.name == name:
            return ident
    raise KeyError(f"No provenance identity named {name!r}")


def run_benchmarks(
    args: argparse.Namespace,
    inputs: InputPaths,
    identities: Sequence[RepoIdentity],
) -> list[RunResult]:
    """Time selected tool CLIs once on authentic HeLa QC inputs."""
    results: list[RunResult] = []
    if not args.skip_winnow:
        print("  Running Winnow...")
        results.append(
            run_winnow(
                inputs,
                winnow_repo=args.winnow_repo,
                model=args.winnow_model,
                scratch_dir=args.scratch_dir,
                identity=_identity_by_name(identities, "Winnow"),
            )
        )
        print(f"    {results[-1].wall_time_s:.2f}s")
    if not args.skip_glissade:
        print(f"  Running Glissade (n_bootstraps={args.glissade_n_bootstraps})...")
        results.append(
            run_glissade(
                inputs,
                glissade_bin=args.glissade_bin,
                fasta=args.fasta,
                scratch_dir=args.scratch_dir,
                identity=_identity_by_name(identities, "Glissade"),
                n_bootstraps=args.glissade_n_bootstraps,
            )
        )
        print(f"    {results[-1].wall_time_s:.2f}s")
    if not args.skip_novoboard:
        print("  Running NovoBoard...")
        results.append(
            run_novoboard(
                inputs,
                novoboard_bin=args.novoboard_bin,
                scratch_dir=args.scratch_dir,
                identity=_identity_by_name(identities, "NovoBoard"),
            )
        )
        print(f"    {results[-1].wall_time_s:.2f}s")
    return results


def main() -> None:
    """Run the HeLa QC one-shot cross-tool CLI runtime benchmark."""
    args = parse_args()
    nb_target, nb_decoy = _preflight(args)
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_json or (
        args.output_dir / "tool_runtime_helaqc_oneshot.json"
    )

    hw = get_hardware_info()
    identities = _collect_identities(args)
    print(format_provenance(identities))

    print("\nResolving authentic HeLa QC inputs...")
    inputs = build_input_paths(
        scratch_dir=args.scratch_dir,
        winnow_spectra=args.spectrum_path,
        winnow_preds=args.predictions_path,
        novoboard_target=nb_target,
        novoboard_decoy=nb_decoy,
        glissade_denovo=args.glissade_dir / "glissade_denovo.parquet",
        glissade_labelled=args.glissade_dir / "glissade_labelled.parquet",
    )

    print("\nTiming tools...")
    results = run_benchmarks(args, inputs, identities)

    report = format_report(hw, identities, results)
    print(report)

    json_report = build_json_report(
        hw,
        identities,
        results,
        glissade_n_bootstraps=args.glissade_n_bootstraps,
        novoboard_decoy_rate=args.novoboard_decoy_rate,
        input_counts=inputs.counts,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(json_report, f, indent=2)
    print(f"JSON results saved to {output_json}")


if __name__ == "__main__":
    main()
