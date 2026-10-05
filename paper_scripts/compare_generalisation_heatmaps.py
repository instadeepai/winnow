#!/usr/bin/env python3
"""Compare PR-AUC generalisation matrices between two results CSVs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import polars as pl
import typer

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "paper_scripts"))

from plot_calibrator_generalisation_heatmap import compute_pr_auc  # noqa: E402

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

# Align legacy deposit labels with HF ``source`` names.
_LABEL_ALIASES = {"PXD019483": "hepg2"}


def _normalise_dataset_label(value: str) -> str:
    return _LABEL_ALIASES.get(value, value)


def _pr_auc_matrix(results_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    lf = pl.scan_csv(results_path)
    trained = sorted(
        _normalise_dataset_label(v)
        for v in lf.select("trained_on_dataset")
        .unique()
        .collect()
        .to_series()
        .to_list()
    )
    tested = sorted(
        _normalise_dataset_label(v)
        for v in lf.select("test_dataset").unique().collect().to_series().to_list()
    )
    trained = sorted(set(trained))
    tested = sorted(set(tested))

    def cell(tr: str, te: str, conf: str) -> float:
        rev = {v: k for k, v in _LABEL_ALIASES.items()}
        train_keys = {tr, rev.get(tr, tr)}
        test_keys = {te, rev.get(te, te)}
        subset = (
            lf.filter(
                pl.col("trained_on_dataset").is_in(list(train_keys))
                & pl.col("test_dataset").is_in(list(test_keys))
            )
            .collect()
            .to_pandas()
        )
        if subset.empty:
            return float("nan")
        return compute_pr_auc(subset, conf, "correct")

    raw = pd.DataFrame(
        [[cell(tr, te, "confidence") for te in tested] for tr in trained],
        index=trained,
        columns=tested,
    )
    cal = pd.DataFrame(
        [[cell(tr, te, "calibrated_confidence") for te in tested] for tr in trained],
        index=trained,
        columns=tested,
    )
    return raw, cal


@app.command()
def main(
    original_path: Annotated[
        Path,
        typer.Argument(help="Reference CSV (e.g. Figshare deposit .bak)."),
    ],
    recomputed_path: Annotated[
        Path,
        typer.Argument(help="New reproduction CSV."),
    ],
) -> None:
    """Print PR-AUC deltas (recomputed − original) for shared train/test cells."""
    orig_raw, orig_cal = _pr_auc_matrix(original_path)
    new_raw, new_cal = _pr_auc_matrix(recomputed_path)

    shared_train = sorted(set(orig_cal.index) & set(new_cal.index))
    shared_test = sorted(set(orig_cal.columns) & set(new_cal.columns))

    diff_cal = (
        new_cal.loc[shared_train, shared_test] - orig_cal.loc[shared_train, shared_test]
    )
    diff_raw = (
        new_raw.loc[shared_train, shared_test] - orig_raw.loc[shared_train, shared_test]
    )

    print("=== Shared matrix shape (train x test) ===")
    print(f"{len(shared_train)} x {len(shared_test)}")
    print("\n=== Calibrated PR-AUC delta (recomputed - original) ===")
    print(diff_cal.to_string(float_format=lambda x: f"{x:+.4f}"))

    abs_cal = diff_cal.abs()
    print("\n=== Summary (calibrated) ===")
    print(f"mean abs delta: {abs_cal.mean().mean():.4f}")
    print(f"max abs delta:  {abs_cal.max().max():.4f}")
    print(f"RMSE:           {np.sqrt((diff_cal**2).mean().mean()):.4f}")

    only_new_train = sorted(set(new_cal.index) - set(orig_cal.index))
    only_new_test = sorted(set(new_cal.columns) - set(orig_cal.columns))
    if only_new_train or only_new_test:
        print("\n=== Only in recomputed (not in original matrix) ===")
        print("trained:", only_new_train)
        print("test:", only_new_test)

    print("\n=== Raw confidence PR-AUC delta summary ===")
    abs_raw = diff_raw.abs()
    print(f"mean abs delta: {abs_raw.mean().mean():.4f}")
    print(f"max abs delta:  {abs_raw.max().max():.4f}")


if __name__ == "__main__":
    app()
