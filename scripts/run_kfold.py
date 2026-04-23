#!/usr/bin/env python3
r"""K-fold CV + full-label training + optional unlabelled prediction for Winnow.

1. Optionally restrict each predictions CSV to ``spectrum_id`` values present in
   the paired spectrum parquet (required when the CSV is a superset of
   experiments — the InstaNovo merge expects one prediction row per kept row).
2. Run ``winnow compute-features`` on the labelled parquet + (possibly filtered) CSV.
3. Export sequences and run ``split_peptides kfold`` so folds match unique peptides.
4. For each fold: train on the other folds' feature matrix, evaluate with
   ``winnow predict`` on held-out spectra.
5. Train on all labelled rows, save iRT regressors, then optionally predict on
   unlabelled data.

Example (single-parquet experiment, CSV may list extra runs):

  uv run python scripts/run_kfold.py \\
    --output-dir ./my_run \\
    --labelled-spectrum-parquet ./data/labelled.parquet \\
    --labelled-predictions-csv ./data/all_instanovo.csv \\
    --unlabelled-spectrum-parquet ./data/unlabelled.parquet \\
    --unlabelled-predictions-csv ./data/all_instanovo.csv
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Iterable, Optional

import polars as pl
import typer
from rich.logging import RichHandler

from winnow.datasets.data_loaders import _add_index_cols
from winnow.scripts.main import (
    compute_features_entry_point,
    predict_entry_point,
    train_entry_point,
)
from winnow.utils.config_path import get_config_dir


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _import_split_peptides():
    """Load ``scripts/split_peptides.py`` as a module (not on PYTHONPATH by default)."""
    import importlib.util

    path = _repo_root() / "scripts" / "split_peptides.py"
    spec = importlib.util.spec_from_file_location("split_peptides", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_sp = _import_split_peptides()
_sanitise_sequence = _sp.sanitise_sequence
_cmd_kfold = _sp.cmd_kfold


def _build_fold_series(
    meta: pl.DataFrame,
    fold_map: dict[str, int],
    sequence_col: str,
) -> list[int]:
    seqs = meta[sequence_col].to_list()
    out: list[int] = []
    missing: list[str] = []
    for s in seqs:
        key = _sanitise_sequence(str(s))
        if key not in fold_map:
            missing.append(key[:80])
        else:
            out.append(fold_map[key])
    if missing:
        raise RuntimeError(
            f"{len(missing)} metadata sequences not in fold map (first keys): {missing[:5]}"
        )
    return out


def _filter_predictions_csv_to_spectrum_ids(
    spectrum_parquet: Path,
    predictions_csv: Path,
    out_csv: Path,
) -> None:
    """Subset a predictions CSV to ``spectrum_id`` values in the spectrum parquet.

    The InstaNovo data loader requires the merge with spectra to return exactly
    as many rows as the (filtered) predictions table; a CSV with extra
    experiments must be reduced first.
    """
    spec = pl.read_parquet(spectrum_parquet)
    spec = _add_index_cols(spec, spectrum_parquet)
    if "spectrum_id" not in spec.columns:
        raise ValueError(
            f"Spectrum file {spectrum_parquet} has no spectrum_id (after index cols)."
        )
    ids = spec["spectrum_id"]
    preds = pl.read_csv(predictions_csv)
    if "spectrum_id" not in preds.columns:
        raise ValueError(
            f"Predictions CSV {predictions_csv} must contain a spectrum_id column."
        )
    n_before = len(preds)
    filt = preds.filter(pl.col("spectrum_id").is_in(ids))
    n_after = len(filt)
    if n_after < n_before:
        logging.info(
            "Filtered predictions %s: %d -> %d rows (spectra: %s)",
            predictions_csv.name,
            n_before,
            n_after,
            spectrum_parquet,
        )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    filt.write_csv(out_csv)


def _filter_parquet_csv_by_peptides(
    spectrum_parquet: Path,
    predictions_csv: Path,
    peptides: Iterable[str],
    out_parquet: Path,
    out_csv: Path,
    sequence_col: str,
) -> None:
    """Subset spectrum parquet and predictions CSV to rows for the given peptides.

    Peptides are matched on the **spectrum** table using ``sequence_col`` (ground
    truth). The predictions file is subset by ``spectrum_id`` only — InstaNovo
    CSVs typically have ``predictions`` / beam columns, not ``sequence``.
    """
    pep_set = set(peptides)
    spec = pl.read_parquet(spectrum_parquet)
    spec = _add_index_cols(spec, spectrum_parquet)
    if sequence_col not in spec.columns:
        raise ValueError(
            f"Spectrum parquet {spectrum_parquet} has no column {sequence_col!r} "
            "(ground truth sequence column)."
        )
    spec = spec.with_columns(
        pl.col(sequence_col)
        .map_elements(_sanitise_sequence, return_dtype=pl.Utf8)
        .alias("__sp")
    )
    filt = spec.filter(pl.col("__sp").is_in(list(pep_set))).drop("__sp")
    filt.write_parquet(out_parquet)

    ids = filt["spectrum_id"].unique().to_list()
    preds = pl.read_csv(predictions_csv)
    if "spectrum_id" not in preds.columns:
        raise ValueError(
            f"Predictions CSV {predictions_csv} must contain a spectrum_id column."
        )
    preds.filter(pl.col("spectrum_id").is_in(ids)).write_csv(out_csv)


def _cfg_dir_str(config_dir: Path) -> str:
    return str(config_dir.resolve())


def _resolve_unlabelled_inputs(
    spectrum: Optional[Path],
    predictions: Optional[Path],
) -> tuple[Optional[Path], Optional[Path]]:
    """Return resolved unlabelled paths, or (None, None). Exit if only one is set."""
    requested = spectrum is not None or predictions is not None
    if not requested:
        return None, None
    if spectrum is None or predictions is None:
        logging.error(
            "For unlabelled prediction, pass both --unlabelled-spectrum-parquet "
            "and --unlabelled-predictions-csv, or pass neither to skip that step."
        )
        raise typer.Exit(1)
    return spectrum.resolve(), predictions.resolve()


def _labelled_predictions_for_run(
    out: Path,
    labelled_spec: Path,
    labelled_csv: Path,
    no_filter: bool,
) -> Path:
    if no_filter:
        return labelled_csv
    path = out / "labelled_predictions_matched_spectra.csv"
    _filter_predictions_csv_to_spectrum_ids(labelled_spec, labelled_csv, path)
    return path


def _compute_features_if_needed(
    skip: bool,
    meta_csv: Path,
    train_matrix: Path,
    labelled_spec: Path,
    labelled_pred_path: Path,
    cfg_s: str,
) -> None:
    if not skip:
        compute_features_entry_point(
            overrides=[
                f"dataset.spectrum_path_or_directory={labelled_spec}",
                f"dataset.predictions_path={labelled_pred_path}",
                f"training_matrix_output_path={train_matrix}",
                f"metadata_output_path={meta_csv}",
                "labelled=true",
            ],
            config_dir=cfg_s,
        )
        return
    if not meta_csv.is_file() or not train_matrix.is_file():
        logging.error(
            "Missing %s or %s; run without --skip-compute-features.",
            meta_csv,
            train_matrix,
        )
        raise typer.Exit(1)


def _kfold_seq_column(meta: pl.DataFrame, sequence_col: str) -> str:
    col = sequence_col if sequence_col in meta.columns else "sequence"
    if col not in meta.columns:
        raise ValueError(
            f"Metadata has no peptide column {sequence_col!r} or 'sequence'. "
            f"Columns: {list(meta.columns)}"
        )
    return col


def _run_kfold_split(
    meta_csv: Path,
    seq_for_kfold: Path,
    assignments: Path,
    k: int,
    seed: int,
    sequence_col: str,
) -> tuple[pl.DataFrame, str]:
    meta = pl.read_csv(meta_csv)
    kfold_seq_col = _kfold_seq_column(meta, sequence_col)
    meta.select(kfold_seq_col).write_parquet(seq_for_kfold)
    _cmd_kfold(
        data=[str(seq_for_kfold)],
        assignments_out=assignments,
        k=k,
        sequence_col=kfold_seq_col,
        seed=seed,
    )
    return pl.read_csv(assignments), kfold_seq_col


def _training_matrix_with_folds(
    meta_csv: Path,
    train_matrix: Path,
    assign_df: pl.DataFrame,
    kfold_seq_col: str,
) -> pl.DataFrame:
    meta = pl.read_csv(meta_csv)
    fold_map = dict(
        zip(
            assign_df["standardised_sequence"].to_list(),
            assign_df["fold"].to_list(),
        )
    )
    tm = pl.read_parquet(train_matrix)
    if len(meta) != len(tm):
        raise RuntimeError(
            f"Metadata rows ({len(meta)}) != training matrix rows ({len(tm)})."
        )
    fold_series = pl.Series(
        "fold",
        _build_fold_series(meta, fold_map, kfold_seq_col),
        dtype=pl.Int64,
    )
    return tm.with_columns(fold=fold_series)


def _run_cv_folds(
    out: Path,
    k: int,
    tm: pl.DataFrame,
    assign_df: pl.DataFrame,
    labelled_spec: Path,
    labelled_pred_path: Path,
    sequence_col: str,
    cfg_s: str,
) -> None:
    for fold_id in range(k):
        fd = out / f"fold_{fold_id}"
        fd.mkdir(parents=True, exist_ok=True)
        train_path = fd / "train_matrix.parquet"
        tm.filter(pl.col("fold") != fold_id).drop("fold").write_parquet(train_path)

        hold_peptides = assign_df.filter(pl.col("fold") == fold_id)[
            "standardised_sequence"
        ].to_list()
        hold_spec = fd / "holdout_spectra.parquet"
        hold_csv = fd / "holdout_predictions.csv"
        _filter_parquet_csv_by_peptides(
            labelled_spec,
            Path(labelled_pred_path),
            hold_peptides,
            hold_spec,
            hold_csv,
            sequence_col,
        )

        model_dir = fd / "model"
        train_entry_point(
            overrides=[
                f"features_path={train_path}",
                "val_features_path=null",
                "validation_fraction=0.1",
                f"model_output_dir={model_dir}",
                f"dataset_output_path={fd / 'calibrated_dataset.csv'}",
                "irt_regressor_output_path=null",
                f"training_history_path={fd / 'training_history.json'}",
            ],
            config_dir=cfg_s,
        )

        pred_out = fd / "predictions"
        predict_entry_point(
            overrides=[
                f"dataset.spectrum_path_or_directory={hold_spec}",
                f"dataset.predictions_path={hold_csv}",
                f"calibrator.pretrained_model_name_or_path={model_dir}",
                "calibrator.irt_regressor_path=null",
                f"output_folder={pred_out}",
            ],
            config_dir=cfg_s,
        )


def _train_full_model(out: Path, train_matrix: Path, cfg_s: str) -> Path:
    full_model = out / "full_labelled_model"
    train_entry_point(
        overrides=[
            f"features_path={train_matrix}",
            "val_features_path=null",
            "validation_fraction=0.1",
            f"model_output_dir={full_model}",
            f"dataset_output_path={full_model / 'calibrated_dataset.csv'}",
            f"irt_regressor_output_path={full_model / 'irt_regressors.safetensors'}",
            f"training_history_path={full_model / 'training_history.json'}",
        ],
        config_dir=cfg_s,
    )
    return full_model


def _predict_unlabelled_if_configured(
    out: Path,
    u_spec: Optional[Path],
    u_csv: Optional[Path],
    full_model: Path,
    no_filter: bool,
    cfg_s: str,
) -> None:
    if u_spec is None or u_csv is None:
        return
    if no_filter:
        unlabelled_pred_path = u_csv
    else:
        unlabelled_pred_path = out / "unlabelled_predictions_matched_spectra.csv"
        _filter_predictions_csv_to_spectrum_ids(u_spec, u_csv, unlabelled_pred_path)
    unlab_out = out / "unlabelled_predictions"
    predict_entry_point(
        overrides=[
            f"dataset.spectrum_path_or_directory={u_spec}",
            f"dataset.predictions_path={unlabelled_pred_path}",
            f"calibrator.pretrained_model_name_or_path={full_model}",
            f"calibrator.irt_regressor_path={full_model / 'irt_regressors.safetensors'}",
            f"output_folder={unlab_out}",
        ],
        config_dir=cfg_s,
    )


def _cleanup_seq_parquet(seq_for_kfold: Path) -> None:
    if seq_for_kfold.is_file():
        try:
            seq_for_kfold.unlink()
        except OSError:
            pass


def _configure_script_logging() -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(logging.INFO)
    h = RichHandler(rich_tracebacks=True, show_path=False)
    h.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    root.addHandler(h)


app = typer.Typer(
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
    invoke_without_command=True,
    help=(
        "K-fold cross-validation and full Winnow run: one spectrum Parquet and "
        "InstaNovo-style predictions CSV."
    ),
)


@app.callback()
def main(
    output_dir: Annotated[
        Path,
        typer.Option(
            ...,
            "--output-dir",
            help="Working directory for matrices, models, and predictions.",
        ),
    ],
    labelled_spectrum_parquet: Annotated[
        Path,
        typer.Option(
            ...,
            help="One labelled spectrum Parquet (ground-truth sequence column).",
        ),
    ],
    labelled_predictions_csv: Annotated[
        Path,
        typer.Option(
            ...,
            help="Predictions CSV (may list more spectrum_id rows than this parquet).",
        ),
    ],
    k: Annotated[
        int,
        typer.Option(
            "--k",
            help="Number of cross-validation folds.",
        ),
    ] = 5,
    seed: Annotated[
        int,
        typer.Option(help="Seed for split_peptides kfold."),
    ] = 42,
    sequence_col: Annotated[
        str,
        typer.Option(
            help="Peptide sequence column in spectrum / metadata (InstaNovo: sequence).",
        ),
    ] = "sequence",
    skip_compute_features: Annotated[
        bool,
        typer.Option(
            "--skip-compute-features",
            help="Reuse existing full_metadata.csv and full_training_matrix.parquet in output-dir.",
        ),
    ] = False,
    config_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--config-dir",
            "-cp",
            help="Hydra config directory (default: packaged winnow configs).",
        ),
    ] = None,
    no_filter_predictions_to_spectra: Annotated[
        bool,
        typer.Option(
            "--no-filter-predictions-to-spectra",
            help="Do not pre-filter prediction CSVs to spectrum_id in each parquet. "
            "Use only if your CSV already has one row per spectrum in that parquet.",
        ),
    ] = False,
    unlabelled_spectrum_parquet: Annotated[
        Optional[Path],
        typer.Option(
            help="Optional unlabelled spectrum Parquet (requires unlabelled predictions).",
        ),
    ] = None,
    unlabelled_predictions_csv: Annotated[
        Optional[Path],
        typer.Option(
            help="Optional predictions CSV (may be a superset; matched to the unlabelled parquet).",
        ),
    ] = None,
) -> None:
    """K-fold cross-validation and full run for a single spectrum parquet and predictions CSV.

    Options match [dim]winnow[/dim] (e.g. [bold]--config-dir[/bold] / [bold]-cp[/bold]); logs use Rich.
    """
    _configure_script_logging()
    config_path = config_dir if config_dir is not None else get_config_dir()
    out = output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg_s = _cfg_dir_str(config_path)

    u_spec, u_csv = _resolve_unlabelled_inputs(
        unlabelled_spectrum_parquet,
        unlabelled_predictions_csv,
    )
    labelled_spec = labelled_spectrum_parquet.resolve()
    labelled_csv = labelled_predictions_csv.resolve()
    labelled_pred_path = _labelled_predictions_for_run(
        out,
        labelled_spec,
        labelled_csv,
        no_filter_predictions_to_spectra,
    )

    meta_csv = out / "full_metadata.csv"
    train_matrix = out / "full_training_matrix.parquet"
    seq_for_kfold = out / "_metadata_sequences_for_kfold.parquet"
    assignments = out / "peptide_fold_assignments.csv"

    _compute_features_if_needed(
        skip_compute_features,
        meta_csv,
        train_matrix,
        labelled_spec,
        labelled_pred_path,
        cfg_s,
    )

    assign_df, kfold_seq_col = _run_kfold_split(
        meta_csv,
        seq_for_kfold,
        assignments,
        k,
        seed,
        sequence_col,
    )
    tm = _training_matrix_with_folds(meta_csv, train_matrix, assign_df, kfold_seq_col)

    _run_cv_folds(
        out,
        k,
        tm,
        assign_df,
        labelled_spec,
        labelled_pred_path,
        sequence_col,
        cfg_s,
    )
    full_model = _train_full_model(out, train_matrix, cfg_s)
    _predict_unlabelled_if_configured(
        out,
        u_spec,
        u_csv,
        full_model,
        no_filter_predictions_to_spectra,
        cfg_s,
    )
    _cleanup_seq_parquet(seq_for_kfold)
    logging.info("Done. Outputs under %s", out)


if __name__ == "__main__":
    app()
