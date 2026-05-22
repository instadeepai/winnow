"""Refit FDR on acfm predictions restricted to spectra not present in lcfm.

For each external project, spectra are matched on ``spectrum_id``. The acfm
(unlabelled) set is filtered to ``spectrum_id`` values absent from the paired
lcfm (labelled) predictions, FDR is re-estimated on ``calibrated_confidence``
for that subset only, and evaluation plots are written.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.logging import RichHandler

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.plot_eval_results import (  # noqa: E402
    _compute_diagnostics,
    _display_name,
    _fit_database_grounded_fdr,
    generate_all_plots,
)
from winnow.fdr.nonparametric import NonParametricFDRControl  # noqa: E402

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    logger.addHandler(RichHandler())

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


def _safe_basename(project: str) -> str:
    """Flat basename for outputs; project keys may contain path separators."""
    return project.replace("/", "_")


def _preds_csv_candidates(root: Path, project: str, *, role: str) -> list[Path]:
    """Paths to try for ``preds_and_fdr_metrics.csv`` under *root*."""
    if role == "labelled":
        return [
            root / project / "preds_and_fdr_metrics.csv",
            root / f"{project}_labelled" / "preds_and_fdr_metrics.csv",
        ]
    if role == "unlabelled":
        return [
            root / project / "preds_and_fdr_metrics.csv",
            root / f"{project}_unlabelled" / "preds_and_fdr_metrics.csv",
        ]
    raise ValueError(f"Unknown role {role!r}")


def _resolve_preds_csv(
    root: Path,
    project: str,
    *,
    role: str,
    alt_roots: list[Path] | None = None,
) -> Path:
    """Resolve ``preds_and_fdr_metrics.csv`` for lcfm (labelled) or acfm (unlabelled)."""
    roots_to_try: list[Path] = [root]
    if alt_roots:
        for alt in alt_roots:
            if alt.resolve() != root.resolve() and alt not in roots_to_try:
                roots_to_try.append(alt)

    tried: list[Path] = []
    for base in roots_to_try:
        for path in _preds_csv_candidates(base, project, role=role):
            tried.append(path)
            if path.is_file():
                if base.resolve() != root.resolve():
                    logger.info(
                        "Using %s predictions at %s (not under %s)",
                        role,
                        path,
                        root,
                    )
                return path

    hint = ""
    if root.is_dir():
        children = sorted(p.name for p in root.iterdir())[:12]
        hint = f" Children of {root}: {children}"
    raise FileNotFoundError(
        f"Missing {role} predictions for {project!r} under {root} "
        f"(tried: {', '.join(str(p) for p in tried)}){hint}"
    )


def _resolve_tree_root(
    predictions_root: Path,
    explicit: Path | None,
    *,
    role: str,
) -> tuple[Path, list[Path]]:
    """Pick labelled (lcfm) or unlabelled (acfm) root; return alternates to try."""
    sub = "lcfm" if role == "labelled" else "acfm"
    nested = predictions_root / sub
    alt_roots: list[Path] = []

    if explicit is None:
        root = nested if nested.is_dir() else predictions_root
        return root, alt_roots

    under_predictions = (
        predictions_root / explicit if not explicit.is_absolute() else None
    )
    if under_predictions is not None and under_predictions.is_dir():
        root = under_predictions
        if explicit.is_dir() and explicit.resolve() != root.resolve():
            alt_roots.append(explicit)
    else:
        root = explicit
        if nested.is_dir() and nested.resolve() != root.resolve():
            alt_roots.append(nested)

    return root, alt_roots


def _lcfm_spectrum_ids(
    labelled_root: Path,
    project: str,
    *,
    labelled_alt_roots: list[Path] | None = None,
) -> set[str]:
    labelled_path = _resolve_preds_csv(
        labelled_root,
        project,
        role="labelled",
        alt_roots=labelled_alt_roots,
    )
    ids = pd.read_csv(labelled_path, usecols=["spectrum_id"])["spectrum_id"]
    return set(ids.astype(str))


def _load_acfm_unlabelled(
    unlabelled_root: Path,
    project: str,
    *,
    unlabelled_alt_roots: list[Path] | None = None,
) -> pd.DataFrame:
    """Load acfm predict outputs with metadata merged (same as plot_eval_results)."""
    preds_path = _resolve_preds_csv(
        unlabelled_root,
        project,
        role="unlabelled",
        alt_roots=unlabelled_alt_roots,
    )
    folder = preds_path.parent
    preds_df = pd.read_csv(preds_path)
    meta_path = folder / "metadata.csv"
    if meta_path.is_file():
        meta_df = pd.read_csv(meta_path)
        overlap = [
            c for c in meta_df.columns if c in preds_df.columns and c != "spectrum_id"
        ]
        if overlap:
            meta_df = meta_df.drop(columns=overlap)
        df = preds_df.merge(meta_df, on="spectrum_id", how="left")
    else:
        df = preds_df

    if "proteome_hit" not in df.columns:
        raise ValueError(
            f"Expected 'proteome_hit' column for unlabelled acfm in {preds_path}"
        )
    df["correct"] = df["proteome_hit"].astype(float)
    required = ["confidence", "calibrated_confidence", "correct"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {preds_path}")
    return df


def filter_acfm_minus_lcfm(acfm_df: pd.DataFrame, lcfm_ids: set[str]) -> pd.DataFrame:
    """Keep acfm rows whose ``spectrum_id`` is not in the lcfm set."""
    mask = ~acfm_df["spectrum_id"].astype(str).isin(lcfm_ids)
    return acfm_df.loc[mask].copy()


def refit_fdr_on_confidence(
    df: pd.DataFrame,
    confidence_col: str = "calibrated_confidence",
) -> pd.DataFrame:
    """Fit non-parametric FDR on *df* and attach PSM FDR / q-value / PEP columns."""
    out = df.copy()
    for col in ("psm_fdr", "psm_q_value", "psm_pep"):
        if col in out.columns:
            out = out.drop(columns=[col])
    fdr_ctrl = NonParametricFDRControl()
    fdr_ctrl.fit(dataset=out[confidence_col])
    out = fdr_ctrl.add_psm_fdr(out, confidence_col=confidence_col)
    out = fdr_ctrl.add_psm_q_value(out, confidence_col=confidence_col)
    out = fdr_ctrl.add_psm_pep(out, confidence_col=confidence_col)
    return out


def process_project(
    labelled_root: Path,
    unlabelled_root: Path,
    project: str,
    output_dir: Path,
    *,
    labelled_alt_roots: list[Path] | None = None,
    unlabelled_alt_roots: list[Path] | None = None,
) -> dict[str, int]:
    """Filter acfm less lcfm, refit FDR, plot, and write tables for one project."""
    lcfm_ids = _lcfm_spectrum_ids(
        labelled_root, project, labelled_alt_roots=labelled_alt_roots
    )
    acfm_df = _load_acfm_unlabelled(
        unlabelled_root, project, unlabelled_alt_roots=unlabelled_alt_roots
    )
    subset_df = filter_acfm_minus_lcfm(acfm_df, lcfm_ids)

    counts = {
        "n_lcfm_spectrum_ids": len(lcfm_ids),
        "n_acfm": len(acfm_df),
        "n_acfm_minus_lcfm": len(subset_df),
    }
    if counts["n_acfm_minus_lcfm"] == 0:
        raise ValueError(
            f"{project}: no acfm spectra remain after excluding lcfm spectrum_id values"
        )

    logger.info(
        "%s: acfm=%s, lcfm ids=%s, acfm\\lcfm=%s",
        project,
        f"{counts['n_acfm']:,}",
        f"{counts['n_lcfm_spectrum_ids']:,}",
        f"{counts['n_acfm_minus_lcfm']:,}",
    )

    subset_df = refit_fdr_on_confidence(subset_df)

    project_dir = output_dir / project
    project_dir.mkdir(parents=True, exist_ok=True)
    safe = _safe_basename(project)
    subset_df.to_csv(project_dir / "preds_and_fdr_metrics.csv", index=False)

    true_fdr_ctrl = _fit_database_grounded_fdr(subset_df)
    db_fdr = true_fdr_ctrl.add_psm_fdr(
        subset_df[["calibrated_confidence"]].copy(),
        confidence_col="calibrated_confidence",
    )
    subset_df["db_grounded_psm_fdr"] = db_fdr["psm_fdr"]
    db_qval = true_fdr_ctrl.add_psm_q_value(
        subset_df[["calibrated_confidence"]].copy(),
        confidence_col="calibrated_confidence",
    )
    subset_df["db_grounded_psm_q_value"] = db_qval["psm_q_value"]

    summary_cols = [
        c
        for c in [
            "spectrum_id",
            "prediction",
            "confidence",
            "calibrated_confidence",
            "correct",
            "psm_fdr",
            "psm_q_value",
            "db_grounded_psm_fdr",
            "db_grounded_psm_q_value",
            "proteome_hit",
        ]
        if c in subset_df.columns
    ]
    subset_df[summary_cols].to_csv(project_dir / f"{safe}_summary.csv", index=False)

    diag = _compute_diagnostics(subset_df, "unlabelled")
    diag.to_csv(project_dir / f"{safe}_diagnostics.csv", index=False)

    pd.DataFrame([counts]).to_csv(project_dir / f"{safe}_counts.csv", index=False)

    generate_all_plots(subset_df, safe, "unlabelled", project_dir)
    return counts


@app.command()
def main(
    predictions_root: Annotated[
        Path,
        typer.Option(
            "--predictions-root",
            help=(
                "Default root for both trees when --labelled-dir / --unlabelled-dir "
                "are omitted."
            ),
        ),
    ],
    projects: Annotated[
        str,
        typer.Option(
            "--projects",
            help="Space- or comma-separated project keys (e.g. 'PXD009935 PXD014877').",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            help="Directory for per-project plots and refitted prediction tables.",
        ),
    ],
    labelled_dir: Annotated[
        Path | None,
        typer.Option(
            "--labelled-dir",
            help=(
                "Root with per-project lcfm folders ({project}/ or {project}_labelled/). "
                "Use e.g. new_eval_sets_results/lcfm when mirroring S3. "
                "If omitted, uses --predictions-root/lcfm when present."
            ),
        ),
    ] = None,
    unlabelled_dir: Annotated[
        Path | None,
        typer.Option(
            "--unlabelled-dir",
            help=(
                "Root with per-project acfm folders ({project}/ or {project}_unlabelled/). "
                "Use e.g. new_eval_sets_results/acfm when mirroring S3. "
                "If omitted, uses --predictions-root/acfm when present."
            ),
        ),
    ] = None,
) -> None:
    """Refit FDR on acfm less lcfm spectra and generate evaluation plots."""
    project_list = [p.strip() for p in projects.replace(",", " ").split() if p.strip()]
    if not project_list:
        raise typer.BadParameter("No projects specified.")

    labelled_root, labelled_alt = _resolve_tree_root(
        predictions_root, labelled_dir, role="labelled"
    )
    unlabelled_root, unlabelled_alt = _resolve_tree_root(
        predictions_root, unlabelled_dir, role="unlabelled"
    )
    logger.info("Labelled (lcfm) root: %s", labelled_root.resolve())
    logger.info("Unlabelled (acfm) root: %s", unlabelled_root.resolve())

    output_dir.mkdir(parents=True, exist_ok=True)
    all_counts: list[dict[str, int | str]] = []

    for project in project_list:
        display = _display_name(project)
        logger.info("Processing %s (%s)...", project, display)
        try:
            counts = process_project(
                labelled_root,
                unlabelled_root,
                project,
                output_dir,
                labelled_alt_roots=labelled_alt,
                unlabelled_alt_roots=unlabelled_alt,
            )
        except FileNotFoundError as exc:
            logger.warning("Skipping %s: %s", project, exc)
            continue
        all_counts.append({"project": project, **counts})

    if all_counts:
        pd.DataFrame(all_counts).to_csv(output_dir / "counts_summary.csv", index=False)
        logger.info("Wrote counts summary to %s", output_dir / "counts_summary.csv")
    else:
        raise typer.Exit(code=1)

    logger.info("Done. Outputs in %s", output_dir)


if __name__ == "__main__":
    app()
