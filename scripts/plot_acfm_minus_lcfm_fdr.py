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


def _lcfm_spectrum_ids(predictions_root: Path, project: str) -> set[str]:
    labelled_path = (
        predictions_root / f"{project}_labelled" / "preds_and_fdr_metrics.csv"
    )
    if not labelled_path.is_file():
        raise FileNotFoundError(f"Missing labelled predictions: {labelled_path}")
    ids = pd.read_csv(labelled_path, usecols=["spectrum_id"])["spectrum_id"]
    return set(ids.astype(str))


def _load_acfm_unlabelled(predictions_root: Path, project: str) -> pd.DataFrame:
    """Load acfm predict outputs with metadata merged (same as plot_eval_results)."""
    from scripts.plot_eval_results import _load_project_data  # noqa: E402

    return _load_project_data(predictions_root, project, "unlabelled", "unlabelled")


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
    predictions_root: Path,
    project: str,
    output_dir: Path,
) -> dict[str, int]:
    """Filter acfm less lcfm, refit FDR, plot, and write tables for one project."""
    lcfm_ids = _lcfm_spectrum_ids(predictions_root, project)
    acfm_df = _load_acfm_unlabelled(predictions_root, project)
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
    subset_df[summary_cols].to_csv(project_dir / f"{project}_summary.csv", index=False)

    diag = _compute_diagnostics(subset_df, "unlabelled")
    diag.to_csv(project_dir / f"{project}_diagnostics.csv", index=False)

    pd.DataFrame([counts]).to_csv(project_dir / f"{project}_counts.csv", index=False)

    generate_all_plots(subset_df, project, "unlabelled", project_dir)
    return counts


@app.command()
def main(
    predictions_root: Annotated[
        Path,
        typer.Option(
            "--predictions-root",
            help="Root with {project}_labelled/ and {project}_unlabelled/ predict folders.",
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
) -> None:
    """Refit FDR on acfm less lcfm spectra and generate evaluation plots."""
    project_list = [p.strip() for p in projects.replace(",", " ").split() if p.strip()]
    if not project_list:
        raise typer.BadParameter("No projects specified.")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_counts: list[dict[str, int | str]] = []

    for project in project_list:
        display = _display_name(project)
        logger.info("Processing %s (%s)...", project, display)
        try:
            counts = process_project(predictions_root, project, output_dir)
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
