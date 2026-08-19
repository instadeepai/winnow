#!/usr/bin/env python3
"""Subset general-model eval parquet + preds CSV by experiment name.

Used to reproduce the revisions-era PXD023064 / immuno2 cohort, which only
kept a fixed list of RAW runs (``PXD023064_FILES`` in ``Makefile.paper``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

logger = logging.getLogger(__name__)

app = typer.Typer(
    add_completion=False,
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _experiment_mask(spectrum_ids: pd.Series, experiments: set[str]) -> pd.Series:
    prefixes = spectrum_ids.astype("string").str.split(":", n=1).str[0]
    return prefixes.isin(experiments)


@app.command()
def main(
    spectra: Annotated[
        Path,
        typer.Option("--spectra", help="Input spectra parquet."),
    ],
    preds: Annotated[
        Path,
        typer.Option("--preds", help="Input InstantNovo predictions CSV."),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Directory for subset parquet + preds CSV."),
    ],
    stem: Annotated[
        str,
        typer.Option("--stem", help="Output basename stem (e.g. immuno2)."),
    ],
    experiment: Annotated[
        list[str],
        typer.Option(
            "--experiment",
            help="Keep rows whose spectrum_id prefix matches this experiment. Repeatable.",
        ),
    ],
) -> None:
    """Write ``{stem}.parquet`` and ``{stem}_preds.csv`` restricted to experiments."""
    _configure_logging()
    if not experiment:
        raise typer.BadParameter("Pass at least one --experiment")
    if not spectra.is_file():
        raise FileNotFoundError(f"Missing spectra parquet: {spectra}")
    if not preds.is_file():
        raise FileNotFoundError(f"Missing preds CSV: {preds}")

    wanted = set(experiment)
    spectra_df = pd.read_parquet(spectra)
    preds_df = pd.read_csv(preds)

    if "spectrum_id" not in spectra_df.columns:
        raise ValueError(f"{spectra} missing spectrum_id")
    if "spectrum_id" not in preds_df.columns:
        raise ValueError(f"{preds} missing spectrum_id")

    spectra_mask = _experiment_mask(spectra_df["spectrum_id"], wanted)
    preds_mask = _experiment_mask(preds_df["spectrum_id"], wanted)
    spectra_out = spectra_df.loc[spectra_mask].reset_index(drop=True)
    preds_out = preds_df.loc[preds_mask].reset_index(drop=True)

    present = set(
        spectra_out["spectrum_id"].astype("string").str.split(":", n=1).str[0].unique()
    )
    missing = sorted(wanted - present)
    if missing:
        logger.warning("Requested experiments absent from spectra: %s", missing)

    output_dir.mkdir(parents=True, exist_ok=True)
    spectra_path = output_dir / f"{stem}.parquet"
    preds_path = output_dir / f"{stem}_preds.csv"
    spectra_out.to_parquet(spectra_path, index=False)
    preds_out.to_csv(preds_path, index=False)
    logger.info(
        "Wrote %s (%d rows) and %s (%d rows) for experiments %s",
        spectra_path,
        len(spectra_out),
        preds_path,
        len(preds_out),
        sorted(wanted),
    )


if __name__ == "__main__":
    app()
