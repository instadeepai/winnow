#!/usr/bin/env python3
r"""One-shot HeLa QC CLI runtime benchmark: Winnow, Glissade, and NovoBoard.

Times each tool's shipped CLI on authentic prepared HeLa QC inputs (no row
filtering, no nested scaling). NovoBoard times FDR estimation only
(precomputed target+decoy CSVs); decoy spectrum generation and DNS on decoys
are omitted.

Glissade parquet inputs are prepared from Hugging Face HeLa QC files when
``--prepare-glissade`` is set (default). NovoBoard CSVs come from the Figshare
``fdr_benchmark_inputs`` tree.

Usage:
    make -f Makefile.paper paper-recompute-tool-runtime

    uv run python paper_scripts/benchmark_tool_runtime.py
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import resource
import shutil
import subprocess
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

_REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_WINNOW_SPECTRA = (
    _REPO_ROOT / "paper_data/winnow-ms-datasets/helaqc/unlabelled.parquet"
)
DEFAULT_WINNOW_PREDS = (
    _REPO_ROOT / "paper_data/winnow-ms-datasets/helaqc/instanovo/unlabelled_preds.csv"
)
DEFAULT_FASTA = _REPO_ROOT / "paper_data/winnow-ms-datasets/fasta/human.fasta"

DEFAULT_TRAIN_SPECTRA = (
    _REPO_ROOT / "paper_data/winnow-ms-datasets/helaqc/train.parquet"
)
DEFAULT_TRAIN_PREDS = (
    _REPO_ROOT / "paper_data/winnow-ms-datasets/helaqc/instanovo/train_preds.csv"
)

DEFAULT_NOVOBOARD_DIR = (
    _REPO_ROOT / "paper_data/fdr_benchmark_inputs/novoboard/helaqc/novoboard"
)
DEFAULT_NOVOBOARD_DECOY_RATE = "0.50"

# Library API default is 10; CLI ships 100.
DEFAULT_GLISSADE_N_BOOTSTRAPS = 10

DEFAULT_GLISSADE_DIR = _REPO_ROOT / "paper_results/runtime/glissade_helaqc_inputs"

_LOCAL_GENERAL_MODEL = _REPO_ROOT / "paper_data/models/winnow-general-model"
DEFAULT_WINNOW_MODEL = (
    str(_LOCAL_GENERAL_MODEL)
    if _LOCAL_GENERAL_MODEL.is_dir()
    else "InstaDeepAI/winnow-general-model"
)

DEFAULT_OUTPUT_DIR = _REPO_ROOT / "paper_results/runtime"
DEFAULT_SCRATCH_DIR = _REPO_ROOT / "paper_results/runtime/scratch_helaqc"

# Pins from pyproject.toml [tool.uv.sources] / Makefile.paper.
GLISSADE_GIT_SHA = "7c723a2af4a88fda84a6bd4f223b351179bd36da"
NOVOBOARD_GIT_SHA = "a9faab3ef1af06987599c2f01e6ba96072c80172"
GLISSADE_REMOTE_URL = "https://github.com/JemmaLDaniel/glissade.git"
NOVOBOARD_REMOTE_URL = "https://github.com/JemmaLDaniel/NovoBoard.git"

EXPECTED_GLISSADE_DENOVO_ROWS = 42873
EXPECTED_GLISSADE_LABELLED_ROWS = 14147

GLISSADE_DENOVO_COLUMNS = ["predictions", "log_probs", "spectrum_id"]


def _resolve_on_path(name: str) -> Path | None:
    """Resolve an executable from ``PATH`` for default ``--*-bin`` flags.

    Reviewer runs should pick up ``glissade`` / ``novoboard`` from the paper
    extra venv without requiring absolute paths.

    Args:
        name: Executable basename (e.g. ``glissade``).

    Returns:
        Absolute path if found, otherwise ``None`` (caller must error clearly).
    """
    found = shutil.which(name)
    return Path(found) if found else None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RepoIdentity:
    """Provenance for one timed tool so JSON/report can cite exact pins.

    Wall times alone are not reproducible across machines; recording commit,
    package version, and CLI path lets a reviewer confirm they ran the same
    artefact even when seconds differ.
    """

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
    hardware: dict[str, str] = {}
    cpu_name = None
    try:
        with open("/proc/cpuinfo") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    cpu_name = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    hardware["cpu"] = cpu_name or platform.processor() or "unknown"
    hardware["cpu_cores"] = str(os.cpu_count() or "unknown")
    try:
        with open("/proc/meminfo") as meminfo:
            for line in meminfo:
                if line.startswith("MemTotal"):
                    mem_total_kb = int(line.split()[1])
                    hardware["ram_gb"] = f"{mem_total_kb / (1024**2):.0f}"
                    break
    except OSError:
        hardware["ram_gb"] = "unknown"
    hardware["gpu"] = "none"
    return hardware


def _run_git(repo: Path, *args: str) -> str | None:
    """Run a git subcommand in ``repo``, returning stdout or ``None`` on failure."""
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
    """Normalise a git remote URL to HTTPS without a ``.git`` suffix."""
    url = remote_url.strip()
    if url.startswith("git@"):
        url = re.sub(r"^git@([^:]+):", r"https://\1/", url)
    url = url.removesuffix(".git")
    return url


def _read_pyproject_version(repo: Path) -> str:
    """Read ``version`` from a checkout's ``pyproject.toml`` for provenance."""
    pyproject = repo / "pyproject.toml"
    if not pyproject.is_file():
        return "unknown"
    text = pyproject.read_text()
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return match.group(1) if match else "unknown"


def _package_version(dist_name: str) -> str:
    """Read the installed distribution version from the active environment."""
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _resolve_remote_fields(repo_root: Path, branch: str) -> tuple[str, str, str, str]:
    """Resolve remote name, URL, tracking ref, and commit URL for a checkout."""
    remote_name = _run_git(repo_root, "config", f"branch.{branch}.remote") or "origin"
    remote_url = _run_git(repo_root, "remote", "get-url", remote_name) or ""
    remote_ref = (
        _run_git(repo_root, "rev-parse", "--abbrev-ref", f"{remote_name}/{branch}")
        or f"{remote_name}/{branch}"
    )
    commit = _run_git(repo_root, "rev-parse", "HEAD") or ""
    remote_https_url = _github_https(remote_url) if remote_url else ""
    commit_url = (
        f"{remote_https_url}/commit/{commit}" if remote_https_url and commit else ""
    )
    return remote_name, remote_url, remote_ref, commit_url


def resolve_repo_identity(
    name: str,
    repo_root: Path,
    *,
    cli_path: str,
) -> RepoIdentity:
    """Build provenance from an explicit git checkout.

    Used for Winnow (this repository). Glissade and NovoBoard use
    ``resolve_pinned_identity`` from the paper-extra git pins.

    Args:
        name: Tool label in the report (``Winnow``, ``Glissade``, ``NovoBoard``).
        repo_root: Checkout root with a ``.git`` directory.
        cli_path: Path or command string recorded as the timed CLI.

    Returns:
        Filled ``RepoIdentity``.
    """
    resolved_root = repo_root.resolve()
    branch = _run_git(resolved_root, "rev-parse", "--abbrev-ref", "HEAD") or "unknown"
    commit = _run_git(resolved_root, "rev-parse", "HEAD") or "unknown"
    commit_short = commit[:12] if commit != "unknown" else "unknown"
    git_status_porcelain = _run_git(resolved_root, "status", "--porcelain")
    git_dirty = bool(git_status_porcelain)
    remote_name, remote_url, remote_ref, commit_url = _resolve_remote_fields(
        resolved_root, branch
    )
    return RepoIdentity(
        name=name,
        repo_root=str(resolved_root),
        git_branch=branch,
        git_commit=commit,
        git_commit_short=commit_short,
        git_dirty=git_dirty,
        remote_name=remote_name,
        remote_url=remote_url,
        remote_ref=remote_ref,
        remote_commit_url=commit_url,
        package_version=_read_pyproject_version(resolved_root),
        cli_path=cli_path,
    )


def resolve_pinned_identity(
    name: str,
    *,
    dist_name: str,
    git_commit: str,
    remote_url: str,
    cli_path: str,
) -> RepoIdentity:
    """Build provenance from the paper-extra git pin.

    Args:
        name: Tool label in the report.
        dist_name: Installed distribution name for ``importlib.metadata``.
        git_commit: Full SHA from ``pyproject.toml`` / ``Makefile.paper``.
        remote_url: Canonical GitHub URL for the pin.
        cli_path: Path of the timed executable.

    Returns:
        ``RepoIdentity`` with ``repo_root="uv.sources"`` and ``git_dirty=False``.
    """
    remote_https_url = _github_https(remote_url)
    return RepoIdentity(
        name=name,
        repo_root="uv.sources",
        git_branch="pinned",
        git_commit=git_commit,
        git_commit_short=git_commit[:12],
        git_dirty=False,
        remote_name="origin",
        remote_url=remote_url,
        remote_ref=f"rev={git_commit[:12]}",
        remote_commit_url=(
            f"{remote_https_url}/commit/{git_commit}" if remote_https_url else ""
        ),
        package_version=_package_version(dist_name),
        cli_path=cli_path,
    )


# ---------------------------------------------------------------------------
# Glissade input preparation (from Hugging Face HeLa QC)
# ---------------------------------------------------------------------------


def prepare_glissade_helaqc_inputs(
    *,
    train_preds: Path,
    unlabelled_preds: Path,
    train_spectra: Path,
    out_dir: Path,
    expected_denovo_rows: int | None = EXPECTED_GLISSADE_DENOVO_ROWS,
    expected_labelled_rows: int | None = EXPECTED_GLISSADE_LABELLED_ROWS,
) -> tuple[Path, Path]:
    """Build Glissade parquet inputs from Hugging Face HeLa QC files.

    Args:
        train_preds: InstaNovo train predictions CSV.
        unlabelled_preds: InstaNovo unlabelled predictions CSV.
        train_spectra: Train spectra parquet (must include ``sequence``).
        out_dir: Directory for ``glissade_denovo.parquet`` /
            ``glissade_labelled.parquet``.
        expected_denovo_rows: Optional hard check on denovo row count.
        expected_labelled_rows: Optional hard check on labelled row count.

    Returns:
        Paths to ``(glissade_denovo.parquet, glissade_labelled.parquet)``.

    Raises:
        FileNotFoundError: If a required input is missing.
        ValueError: On length mismatches or unexpected row counts.
    """
    for path in (train_preds, unlabelled_preds, train_spectra):
        if not path.is_file():
            raise FileNotFoundError(f"Required Glissade source not found: {path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    train_predictions = pl.read_csv(train_preds).select(GLISSADE_DENOVO_COLUMNS)
    unlabelled_predictions = pl.read_csv(unlabelled_preds).select(
        GLISSADE_DENOVO_COLUMNS
    )
    denovo_frame = pl.concat(
        [train_predictions, unlabelled_predictions], how="vertical"
    )

    train_sequences = pl.read_parquet(train_spectra, columns=["sequence"])
    if train_sequences.height != train_predictions.height:
        raise ValueError(
            f"Row count mismatch: {train_spectra} has {train_sequences.height} "
            f"rows, {train_preds} has {train_predictions.height} rows"
        )
    labelled_frame = pl.DataFrame(
        {
            "spectrum_id": train_predictions["spectrum_id"],
            "sequence": train_sequences["sequence"],
        }
    )

    if expected_denovo_rows is not None and denovo_frame.height != expected_denovo_rows:
        raise ValueError(
            f"Unexpected glissade_denovo rows: {denovo_frame.height} "
            f"(expected {expected_denovo_rows})"
        )
    if (
        expected_labelled_rows is not None
        and labelled_frame.height != expected_labelled_rows
    ):
        raise ValueError(
            f"Unexpected glissade_labelled rows: {labelled_frame.height} "
            f"(expected {expected_labelled_rows})"
        )

    denovo_path = out_dir / "glissade_denovo.parquet"
    labelled_path = out_dir / "glissade_labelled.parquet"
    denovo_frame.write_parquet(denovo_path)
    labelled_frame.write_parquet(labelled_path)
    print(
        f"  Prepared Glissade inputs under {out_dir} "
        f"(denovo={denovo_frame.height:,}, labelled={labelled_frame.height:,})"
    )
    return denovo_path, labelled_path


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
    """Validate and package authentic full HeLa QC inputs.

    Args:
        scratch_dir: Created if needed for later CLI outputs.
        winnow_spectra: Unlabelled spectra parquet for Winnow.
        winnow_preds: Matching InstaNovo predictions CSV.
        novoboard_target: NovoBoard ``raw_unlabelled.csv``.
        novoboard_decoy: Matching decoy CSV at the chosen rate.
        glissade_denovo: Prepared denovo parquet.
        glissade_labelled: Prepared labelled parquet.

    Returns:
        ``InputPaths`` including per-tool row counts.

    Raises:
        ValueError: If Winnow spectra and prediction row counts differ.
    """
    n_winnow = int(pl.scan_parquet(winnow_spectra).select(pl.len()).collect().item())
    n_winnow_predictions = int(
        pl.scan_csv(winnow_preds).select(pl.len()).collect().item()
    )
    if n_winnow != n_winnow_predictions:
        raise ValueError(
            "Winnow spectra/preds length mismatch: "
            f"{n_winnow} vs {n_winnow_predictions}"
        )
    n_novoboard_targets = int(
        pl.scan_csv(novoboard_target).select(pl.len()).collect().item()
    )
    n_novoboard_decoys = int(
        pl.scan_csv(novoboard_decoy).select(pl.len()).collect().item()
    )
    n_glissade_denovo = int(
        pl.scan_parquet(glissade_denovo).select(pl.len()).collect().item()
    )
    n_glissade_labelled = int(
        pl.scan_parquet(glissade_labelled).select(pl.len()).collect().item()
    )

    counts = {
        "winnow_spectra": n_winnow,
        "novoboard_targets": n_novoboard_targets,
        "novoboard_decoys": n_novoboard_decoys,
        "glissade_denovo": n_glissade_denovo,
        "glissade_labelled": n_glissade_labelled,
    }
    print(
        "  Authentic HeLa QC inputs "
        f"(Winnow={n_winnow:,}; NovoBoard target/decoy="
        f"{n_novoboard_targets:,}/{n_novoboard_decoys:,}; Glissade denovo/labelled="
        f"{n_glissade_denovo:,}/{n_glissade_labelled:,})"
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
    """Wall-clock a subprocess with merged stdout/stderr for post-mortem.

    Timing wraps the whole process (not in-process Python APIs) so the paper
    table reflects shipped CLI cost, including startup.

    Args:
        command: Argv list for ``subprocess.run``.
        cwd: Working directory for the child.
        log_path: File that receives the command header plus child output.
        env: Optional env overrides merged onto ``os.environ``.

    Returns:
        ``(wall_time_s, returncode)``.
    """
    cwd.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    start_time = time.perf_counter()
    with open(log_path, "w") as log_file:
        log_file.write("COMMAND: " + " ".join(command) + "\n")
        log_file.write(f"CWD: {cwd}\n\n")
        log_file.flush()
        completed = subprocess.run(
            list(command),
            cwd=str(cwd),
            env=merged_env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    wall_time_s = time.perf_counter() - start_time
    return wall_time_s, completed.returncode


def run_winnow(
    inputs: InputPaths,
    *,
    winnow_repo: Path,
    model: str,
    scratch_dir: Path,
    identity: RepoIdentity,
) -> RunResult:
    """Time bare ``uv run winnow predict``.

    Args:
        inputs: Validated spectra + predictions paths.
        winnow_repo: Checkout whose ``uv run winnow`` is invoked.
        model: Calibrator directory or Hugging Face id.
        scratch_dir: Parent for ``winnow_out`` and ``winnow.log``.
        identity: Provenance stamped into the result.

    Returns:
        Timed ``RunResult`` for method ``Winnow``.

    Raises:
        RuntimeError: If predict exits non-zero (see log path in the message).
    """
    winnow_output_dir = scratch_dir / "winnow_out"
    if winnow_output_dir.exists():
        shutil.rmtree(winnow_output_dir)
    winnow_output_dir.mkdir(parents=True, exist_ok=True)
    log_path = scratch_dir / "winnow.log"

    command = [
        "uv",
        "run",
        "winnow",
        "predict",
        f"dataset.spectrum_path_or_directory={inputs.winnow_spectra.resolve()}",
        f"dataset.predictions_path={inputs.winnow_preds.resolve()}",
        f"calibrator.pretrained_model_name_or_path={model}",
        f"output_folder={winnow_output_dir.resolve()}/",
    ]
    wall_time_s, return_code = run_timed_cli(
        command, cwd=winnow_repo, log_path=log_path
    )
    if return_code != 0:
        raise RuntimeError(
            f"Winnow predict failed (exit {return_code}); see {log_path}"
        )
    return RunResult(
        method="Winnow",
        n_rows=inputs.n_winnow,
        wall_time_s=wall_time_s,
        returncode=return_code,
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
    """Time Glissade's parquet-mode CLI with a fixed bootstrap count.

    Uses ``n_bootstraps=10`` (library API default) rather than the CLI default
    of 100 so the oneshot stays tractable.

    Args:
        inputs: Paths including denovo and labelled parquets.
        glissade_bin: ``glissade`` executable to invoke.
        fasta: Reference proteome for external-peptide definition.
        scratch_dir: Parent for output cwd and log.
        identity: Provenance stamped into the result.
        n_bootstraps: Value for ``-n`` / ``--n_bootstraps``.

    Returns:
        Timed ``RunResult`` for method ``Glissade``.

    Raises:
        RuntimeError: If the CLI exits non-zero.
    """
    glissade_output_dir = scratch_dir / "glissade_out"
    if glissade_output_dir.exists():
        shutil.rmtree(glissade_output_dir)
    glissade_output_dir.mkdir(parents=True, exist_ok=True)
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
    wall_time_s, return_code = run_timed_cli(
        command, cwd=glissade_output_dir, log_path=log_path
    )
    if return_code != 0:
        raise RuntimeError(f"Glissade failed (exit {return_code}); see {log_path}")
    n_glissade_denovo = inputs.counts["glissade_denovo"]
    n_glissade_labelled = inputs.counts["glissade_labelled"]
    return RunResult(
        method="Glissade",
        n_rows=n_glissade_denovo,
        wall_time_s=wall_time_s,
        returncode=return_code,
        command=command,
        log_path=str(log_path),
        repo=asdict(identity),
        detail={
            "n_bootstraps": n_bootstraps,
            "n_denovo": n_glissade_denovo,
            "n_labelled": n_glissade_labelled,
        },
    )


def run_novoboard(
    inputs: InputPaths,
    *,
    novoboard_bin: Path,
    scratch_dir: Path,
    identity: RepoIdentity,
) -> RunResult:
    """Time NovoBoard ``fdr`` on precomputed target and decoy CSVs only.

    Decoy MGF generation and DNS on decoys are deliberately omitted: regenerating
    them is out of scope for this oneshot and would dominate wall time.

    Args:
        inputs: Paths to target and decoy CSVs.
        novoboard_bin: ``novoboard`` executable to invoke.
        scratch_dir: Parent for output cwd and log.
        identity: Provenance stamped into the result.

    Returns:
        Timed ``RunResult`` for method ``NovoBoard``.

    Raises:
        RuntimeError: If ``novoboard fdr`` exits non-zero.
    """
    novoboard_output_dir = scratch_dir / "novoboard_out"
    if novoboard_output_dir.exists():
        shutil.rmtree(novoboard_output_dir)
    novoboard_output_dir.mkdir(parents=True, exist_ok=True)
    log_path = scratch_dir / "novoboard.log"

    command = [
        str(novoboard_bin.resolve()),
        "fdr",
        "--target-file",
        str(inputs.novoboard_target.resolve()),
        "--decoy-files",
        str(inputs.novoboard_decoy.resolve()),
    ]
    wall_time_s, return_code = run_timed_cli(
        command, cwd=novoboard_output_dir, log_path=log_path
    )
    if return_code != 0:
        raise RuntimeError(f"NovoBoard fdr failed (exit {return_code}); see {log_path}")
    n_novoboard_targets = inputs.counts["novoboard_targets"]
    n_novoboard_decoys = inputs.counts["novoboard_decoys"]
    return RunResult(
        method="NovoBoard",
        n_rows=n_novoboard_targets,
        wall_time_s=wall_time_s,
        returncode=return_code,
        command=command,
        log_path=str(log_path),
        repo=asdict(identity),
        detail={
            "n_targets": n_novoboard_targets,
            "n_decoys": n_novoboard_decoys,
            "scope": "fdr_estimation_only",
        },
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_provenance(identities: Sequence[RepoIdentity]) -> str:
    """Render a multi-line provenance block for stdout.

    Args:
        identities: Tools that will be or were timed.

    Returns:
        Human-readable string listing version, commit, remote, and CLI path.
    """
    lines = ["Tool provenance:"]
    for tool_identity in identities:
        lines.append(
            f"  {tool_identity.name}: v{tool_identity.package_version}  "
            f"branch={tool_identity.git_branch}  "
            f"commit={tool_identity.git_commit_short}  "
            f"dirty={str(tool_identity.git_dirty).lower()}"
        )
        lines.append(
            f"             remote={tool_identity.remote_url}  "
            f"ref={tool_identity.remote_ref}"
        )
        if tool_identity.remote_commit_url:
            lines.append(f"             {tool_identity.remote_commit_url}")
        lines.append(f"             cli={tool_identity.cli_path}")
    return "\n".join(lines)


def format_report(
    hardware_info: dict[str, str],
    identities: Sequence[RepoIdentity],
    results: Sequence[RunResult],
) -> str:
    """Format the human-readable wall-clock summary table and notes.

    Notes call out intentional omissions (NovoBoard decoy generation, Glissade
    bootstrap count) so a reader does not treat the table as end-to-end pipeline
    cost.

    Args:
        hardware_info: Output of ``get_hardware_info``.
        identities: Provenance rows to embed above the table.
        results: Timed runs in display order.

    Returns:
        Multi-line report string ending with a blank line.
    """
    lines = [
        "=" * 88,
        "  HeLa QC one-shot cross-tool runtime benchmark",
        "=" * 88,
        "",
        (
            f"Hardware: {hardware_info['cpu']} ({hardware_info['cpu_cores']} cores), "
            f"{hardware_info['ram_gb']} GB RAM, GPU: {hardware_info['gpu']}"
        ),
        "",
        format_provenance(identities),
        "",
    ]
    column_widths = [28, 12, 14, 12]
    header_row = (
        f"{'Method':<{column_widths[0]}}| {'n_rows':>{column_widths[1]}}| "
        f"{'Wall time (s)':>{column_widths[2]}}| {'Exit':>{column_widths[3]}}"
    )
    separator_row = (
        "-" * column_widths[0]
        + "|"
        + "-" * (column_widths[1] + 1)
        + "|"
        + "-" * (column_widths[2] + 1)
        + "|"
        + "-" * (column_widths[3] + 1)
    )
    lines.append(header_row)
    lines.append(separator_row)
    for run_result in results:
        lines.append(
            f"{run_result.method:<{column_widths[0]}}| "
            f"{run_result.n_rows:>{column_widths[1]},}| "
            f"{run_result.wall_time_s:>{column_widths[2]}.2f}| "
            f"{run_result.returncode:>{column_widths[3]}}"
        )
    lines.append(separator_row)
    lines.append("")
    glissade_n_bootstraps = next(
        (
            int(run_result.detail["n_bootstraps"])
            for run_result in results
            if run_result.method == "Glissade" and "n_bootstraps" in run_result.detail
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
        f"- Glissade: n_bootstraps={glissade_n_bootstraps} "
        "(library API default is 10; CLI ships 100)."
    )
    lines.append("- Winnow: shipped predict CLI + InstaDeepAI/winnow-general-model.")
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    lines.append(f"- Parent process peak RSS: {peak_rss_mb:.1f} MB")
    lines.append("")
    return "\n".join(lines)


def build_json_report(
    hardware_info: dict[str, str],
    identities: Sequence[RepoIdentity],
    results: Sequence[RunResult],
    *,
    glissade_n_bootstraps: int,
    novoboard_decoy_rate: str,
    input_counts: dict[str, int],
) -> dict[str, Any]:
    """Build the structured JSON artefact for archival and later comparison.

    Args:
        hardware_info: Hardware metadata.
        identities: Tool provenance records.
        results: Timed runs.
        glissade_n_bootstraps: Bootstrap count used for Glissade.
        novoboard_decoy_rate: Decoy rate suffix that selected the decoy CSV.
        input_counts: Row counts from ``build_input_paths``.

    Returns:
        JSON-serialisable dict with mode ``helaqc_oneshot``.
    """
    return {
        "mode": "helaqc_oneshot",
        "hardware": hardware_info,
        "glissade_n_bootstraps": glissade_n_bootstraps,
        "input_counts": input_counts,
        "tools": {
            tool_identity.name: asdict(tool_identity) for tool_identity in identities
        },
        "runs": [
            {
                "method": run_result.method,
                "n_rows": run_result.n_rows,
                "wall_time_s": run_result.wall_time_s,
                "returncode": run_result.returncode,
                "command": run_result.command,
                "log_path": run_result.log_path,
                "repo": run_result.repo,
                "detail": run_result.detail,
            }
            for run_result in results
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
    """Parse CLI flags for the oneshot benchmark.

    Defaults point at ``paper_data`` / ``paper_results`` layouts produced by
    ``paper-setup`` so ``make -f Makefile.paper paper-recompute-tool-runtime``
    needs no extra path arguments.

    Returns:
        Parsed ``argparse.Namespace``.
    """
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
        default=_REPO_ROOT,
        help="Winnow checkout used for CLI (default: this repository).",
    )
    parser.add_argument(
        "--winnow-model",
        default=DEFAULT_WINNOW_MODEL,
        help=(
            "Calibrator path or HF id (default: local paper_data/models/"
            "winnow-general-model when present)."
        ),
    )
    parser.add_argument(
        "--novoboard-dir",
        type=Path,
        default=DEFAULT_NOVOBOARD_DIR,
        help=(
            "NovoBoard HeLa QC CSV directory (parent of raw_unlabelled.csv; "
            "default: Figshare fdr_benchmark_inputs/.../helaqc/novoboard)."
        ),
    )
    parser.add_argument(
        "--novoboard-bin",
        type=Path,
        default=_resolve_on_path("novoboard"),
        help="Path to novoboard executable (default: first match on PATH).",
    )
    parser.add_argument(
        "--novoboard-decoy-rate",
        default=DEFAULT_NOVOBOARD_DECOY_RATE,
        help="Decoy rate suffix for decoy CSV (default: 0.50).",
    )
    parser.add_argument(
        "--glissade-dir",
        type=Path,
        default=DEFAULT_GLISSADE_DIR,
        help="Directory for glissade_denovo/labelled parquets.",
    )
    parser.add_argument(
        "--prepare-glissade",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Build Glissade parquets from HF HeLa QC train+unlabelled (default: true)."
        ),
    )
    parser.add_argument(
        "--train-preds-path",
        type=Path,
        default=DEFAULT_TRAIN_PREDS,
        help="Train predictions CSV used when preparing Glissade inputs.",
    )
    parser.add_argument(
        "--train-spectrum-path",
        type=Path,
        default=DEFAULT_TRAIN_SPECTRA,
        help="Train spectra parquet used when preparing Glissade inputs.",
    )
    parser.add_argument(
        "--glissade-bin",
        type=Path,
        default=_resolve_on_path("glissade"),
        help="Path to glissade executable (default: first match on PATH).",
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
    """Fail fast if a timed CLI binary is missing from PATH or flags.

    Args:
        path: Candidate executable path (may be ``None``).
        name: Human tool name for the error message.
        flag: CLI flag the user can pass (e.g. ``--glissade-bin``).

    Raises:
        FileNotFoundError: If ``path`` is missing or not a file.
    """
    if path is None or not Path(path).is_file():
        raise FileNotFoundError(
            f"{name} binary not found: {path!r}. Pass {flag} or put "
            f"'{name.lower()}' on PATH (uv sync --group paper; "
            "NovoBoard needs Python >=3.12)."
        )


def _preflight(args: argparse.Namespace) -> tuple[Path, Path]:
    """Validate required inputs and binaries before any timing starts.

    Catching missing Figshare/HF paths early avoids a long Glissade run that
    then fails on NovoBoard, or vice versa.

    Args:
        args: Parsed CLI namespace.

    Returns:
        ``(novoboard_target_csv, novoboard_decoy_csv)``.

    Raises:
        FileNotFoundError: If a required input or binary is missing.
    """
    novoboard_target_csv = args.novoboard_dir / "raw_unlabelled.csv"
    novoboard_decoy_csv = (
        args.novoboard_dir / f"raw_unlabelled_decoy_{args.novoboard_decoy_rate}.csv"
    )
    required_files = [
        args.spectrum_path,
        args.predictions_path,
        args.fasta,
        novoboard_target_csv,
        novoboard_decoy_csv,
    ]
    if not args.prepare_glissade:
        required_files.extend(
            [
                args.glissade_dir / "glissade_denovo.parquet",
                args.glissade_dir / "glissade_labelled.parquet",
            ]
        )
    for path in required_files:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Required input not found: {path}")

    if not args.skip_winnow:
        if not args.winnow_repo.is_dir():
            raise FileNotFoundError(f"Winnow repo not found: {args.winnow_repo}")
        if not (args.winnow_repo / "pyproject.toml").is_file():
            raise FileNotFoundError(
                f"Winnow repo looks invalid (no pyproject.toml): {args.winnow_repo}"
            )
    if not args.skip_glissade:
        _require_bin(args.glissade_bin, name="Glissade", flag="--glissade-bin")
    if not args.skip_novoboard:
        _require_bin(args.novoboard_bin, name="NovoBoard", flag="--novoboard-bin")
    return novoboard_target_csv, novoboard_decoy_csv


def _collect_identities(args: argparse.Namespace) -> list[RepoIdentity]:
    """Assemble provenance for each tool that is not skipped.

    Winnow is stamped from ``--winnow-repo``; Glissade and NovoBoard from the
    paper-extra git pins and installed package versions.

    Args:
        args: Parsed CLI namespace.

    Returns:
        List of ``RepoIdentity`` in timing order.
    """
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
        glissade_cli_path = str(Path(args.glissade_bin).resolve())
        identities.append(
            resolve_pinned_identity(
                "Glissade",
                dist_name="glissade",
                git_commit=GLISSADE_GIT_SHA,
                remote_url=GLISSADE_REMOTE_URL,
                cli_path=glissade_cli_path,
            )
        )
    if not args.skip_novoboard:
        novoboard_cli_path = str(Path(args.novoboard_bin).resolve())
        identities.append(
            resolve_pinned_identity(
                "NovoBoard",
                dist_name="novoboard",
                git_commit=NOVOBOARD_GIT_SHA,
                remote_url=NOVOBOARD_REMOTE_URL,
                cli_path=novoboard_cli_path,
            )
        )
    return identities


def _identity_by_name(identities: Sequence[RepoIdentity], name: str) -> RepoIdentity:
    """Look up a provenance record by tool name.

    Args:
        identities: List from ``_collect_identities``.
        name: Exact ``RepoIdentity.name`` (e.g. ``Winnow``).

    Returns:
        Matching ``RepoIdentity``.

    Raises:
        KeyError: If ``name`` was not collected (e.g. skipped on the CLI).
    """
    for tool_identity in identities:
        if tool_identity.name == name:
            return tool_identity
    raise KeyError(f"No provenance identity named {name!r}")


def run_benchmarks(
    args: argparse.Namespace,
    inputs: InputPaths,
    identities: Sequence[RepoIdentity],
) -> list[RunResult]:
    """Run each selected tool CLI once and collect wall-clock results.

    A single pass keeps the oneshot cheap; ``--skip-*`` flags support debugging
    one tool without re-running the others.

    Args:
        args: Parsed CLI namespace (skip flags, bins, model, bootstraps).
        inputs: Validated input paths and counts.
        identities: Provenance list aligned with tools that will run.

    Returns:
        ``RunResult`` list in execution order.
    """
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
    """Entrypoint: prepare inputs if needed, time CLIs, write JSON and report.

    Order is prepare → provenance → validate counts → time → print → dump JSON
    so a failure before timing still shows which pin/binary would have been used.
    """
    args = parse_args()
    novoboard_target_csv, novoboard_decoy_csv = _preflight(args)
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_json or (
        args.output_dir / "tool_runtime_helaqc_oneshot.json"
    )

    if args.prepare_glissade and not args.skip_glissade:
        print("Preparing Glissade HeLa QC parquets from HF inputs...")
        prepare_glissade_helaqc_inputs(
            train_preds=args.train_preds_path,
            unlabelled_preds=args.predictions_path,
            train_spectra=args.train_spectrum_path,
            out_dir=args.glissade_dir,
        )

    hardware_info = get_hardware_info()
    identities = _collect_identities(args)
    print(format_provenance(identities))

    print("\nResolving authentic HeLa QC inputs...")
    inputs = build_input_paths(
        scratch_dir=args.scratch_dir,
        winnow_spectra=args.spectrum_path,
        winnow_preds=args.predictions_path,
        novoboard_target=novoboard_target_csv,
        novoboard_decoy=novoboard_decoy_csv,
        glissade_denovo=args.glissade_dir / "glissade_denovo.parquet",
        glissade_labelled=args.glissade_dir / "glissade_labelled.parquet",
    )

    print("\nTiming tools...")
    results = run_benchmarks(args, inputs, identities)

    report = format_report(hardware_info, identities, results)
    print(report)

    json_report = build_json_report(
        hardware_info,
        identities,
        results,
        glissade_n_bootstraps=args.glissade_n_bootstraps,
        novoboard_decoy_rate=args.novoboard_decoy_rate,
        input_counts=inputs.counts,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as json_file:
        json.dump(json_report, json_file, indent=2)
    print(f"JSON results saved to {output_json}")


if __name__ == "__main__":
    main()
