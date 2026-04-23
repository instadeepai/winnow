#!/usr/bin/env python3
"""Generic peptide-based train/test split script with optional validation set.

Creates or incrementally updates train/test (and optionally val) splits with no
peptide leakage, saves a CSV manifest of sanitised peptides per split, and
enforces schema compatibility when appending to existing parquet files.
"""

from __future__ import annotations

import logging
import random
import re
from pathlib import Path
from typing import Annotated, Dict, List, Optional, Tuple

import polars as pl
import typer

logger = logging.getLogger(__name__)

app = typer.Typer(
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
    help=(
        "Split peptide datasets into train/test (and optionally val) with no peptide leakage."
    ),
)


def sanitise_sequence(sequence: str) -> str:
    """Sanitise a peptide sequence by stripping modifications and normalising L→I.

    Args:
        sequence: The raw peptide sequence with potential UNIMOD modifications.

    Returns:
        The sanitised sequence with modifications stripped and L replaced by I.
    """
    sanitised = re.sub(r"\[UNIMOD:\d+\]-?", "", sequence)
    sanitised = sanitised.replace("L", "I")
    return sanitised


def add_sanitised_column(df: pl.DataFrame, sequence_col: str) -> pl.DataFrame:
    """Add a sanitised peptide column to the dataframe.

    Args:
        df: Input dataframe with a sequence column.
        sequence_col: Name of the column containing peptide sequences.

    Returns:
        Dataframe with an added '__sanitised_peptide' column.
    """
    return df.with_columns(
        pl.col(sequence_col)
        .str.replace_all(r"\[UNIMOD:\d+\]-?", "")
        .str.replace_all("L", "I")
        .alias("__sanitised_peptide")
    )


def split_by_peptides_balanced(
    df: pl.DataFrame,
    peptide_col: str = "__sanitised_peptide",
    train_frac: float = 0.8,
    test_frac: float = 0.2,
    include_val: bool = False,
    peptide_weight: float = 0.5,
    seed: int = 42,
) -> Tuple[pl.DataFrame, pl.DataFrame, Optional[pl.DataFrame], Dict[str, List[str]]]:
    """Split dataset ensuring no peptide overlap between splits.

    Balances both peptide count and row count targets using deficit-based bin packing.

    Args:
        df: The input dataframe to split.
        peptide_col: The column name containing the sanitised peptide sequences.
        train_frac: The fraction of peptides to assign to the training set.
        test_frac: The fraction of peptides to assign to the test set.
        include_val: Whether to create a validation split from the remainder.
        peptide_weight: The weight of peptide count in the balancing (0-1).
        seed: The random seed for reproducibility.

    Returns:
        A tuple containing:
        - Training dataframe
        - Test dataframe
        - Validation dataframe (None when include_val is False)
        - Dictionary mapping split names to lists of peptides assigned to each
    """
    random.seed(seed)

    peptide_counts = df.group_by(peptide_col).agg(pl.len().alias("count")).to_dicts()

    random.shuffle(peptide_counts)
    peptide_counts.sort(key=lambda x: x["count"], reverse=True)

    total_rows = len(df)
    total_peptides = len(peptide_counts)

    split_names = ["train", "test"]
    fracs = {"train": train_frac, "test": test_frac}
    if include_val:
        val_frac = 1 - train_frac - test_frac
        if val_frac <= 0:
            raise ValueError(
                f"train_frac ({train_frac}) + test_frac ({test_frac}) >= 1.0; "
                "no room for a validation split"
            )
        fracs["val"] = val_frac
        split_names.append("val")

    row_targets = {s: total_rows * fracs[s] for s in split_names}
    peptide_targets = {s: total_peptides * fracs[s] for s in split_names}

    row_current = {s: 0 for s in split_names}
    peptide_current = {s: 0 for s in split_names}
    assignments: Dict[str, List[str]] = {s: [] for s in split_names}

    row_weight = 1 - peptide_weight

    for item in peptide_counts:
        peptide = item[peptide_col]
        count = item["count"]

        scores = {}
        for split in split_names:
            pep_deficit = (peptide_targets[split] - peptide_current[split]) / max(
                peptide_targets[split], 1
            )
            row_deficit = (row_targets[split] - row_current[split]) / max(
                row_targets[split], 1
            )
            scores[split] = peptide_weight * pep_deficit + row_weight * row_deficit

        best_split = max(scores, key=lambda s: scores[s])

        assignments[best_split].append(peptide)
        peptide_current[best_split] += 1
        row_current[best_split] += count

    train_df = df.filter(pl.col(peptide_col).is_in(assignments["train"]))
    test_df = df.filter(pl.col(peptide_col).is_in(assignments["test"]))
    val_df = (
        df.filter(pl.col(peptide_col).is_in(assignments["val"]))
        if include_val
        else None
    )

    logger.info(
        f"{'Split':<8} {'Peptides':>10} {'(target)':>10} {'Rows':>10} {'(target)':>10}"
    )
    logger.info("-" * 50)
    for name in split_names:
        split_df = {"train": train_df, "test": test_df, "val": val_df}.get(name)
        n_pep = len(assignments[name])
        n_rows = len(split_df) if split_df is not None else 0
        logger.info(
            f"{name:<8} {n_pep:>10} {int(peptide_targets[name]):>10} "
            f"{n_rows:>10} {int(row_targets[name]):>10}"
        )

    return train_df, test_df, val_df, assignments


def load_manifest(manifest_path: Path) -> Dict[str, str]:
    """Load an existing peptide manifest CSV.

    Args:
        manifest_path: Path to the manifest CSV file.

    Returns:
        Dictionary mapping peptide sequences to their assigned splits.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    df = pl.read_csv(manifest_path)
    return dict(zip(df["standardised_sequence"].to_list(), df["split"].to_list()))


def save_manifest(
    assignments: Dict[str, List[str]], manifest_path: Path, append: bool = False
) -> None:
    """Save peptide assignments to a manifest CSV.

    Args:
        assignments: Dictionary mapping split names to lists of peptides.
        manifest_path: Path to save the manifest CSV.
        append: If True, append to existing file; otherwise overwrite.
    """
    rows = []
    for split_name, peptides in assignments.items():
        for peptide in peptides:
            rows.append({"standardised_sequence": peptide, "split": split_name})

    df = pl.DataFrame(rows)

    if append and manifest_path.exists():
        existing_df = pl.read_csv(manifest_path)
        existing_peptides = set(existing_df["standardised_sequence"].to_list())
        df = df.filter(~pl.col("standardised_sequence").is_in(list(existing_peptides)))
        if len(df) > 0:
            combined = pl.concat([existing_df, df], how="vertical_relaxed")
            combined.write_csv(manifest_path)
            logger.info(f"Appended {len(df)} new peptides to manifest: {manifest_path}")
        else:
            logger.info("No new peptides to append to manifest.")
    else:
        df.write_csv(manifest_path)
        logger.info(f"Saved manifest with {len(df)} peptides to: {manifest_path}")


def check_schema_compatibility(new_df: pl.DataFrame, existing_df: pl.DataFrame) -> None:
    """Check that new data schema matches existing data schema.

    Args:
        new_df: The new dataframe to validate.
        existing_df: The existing dataframe to compare against.

    Raises:
        ValueError: If schemas do not match, with detailed differences.
    """
    new_schema = new_df.schema
    existing_schema = existing_df.schema

    if new_schema == existing_schema:
        return

    errors = []

    new_cols = set(new_schema.keys())
    existing_cols = set(existing_schema.keys())

    missing_in_new = existing_cols - new_cols
    extra_in_new = new_cols - existing_cols

    if missing_in_new:
        errors.append(f"Missing columns in new data: {sorted(missing_in_new)}")
    if extra_in_new:
        errors.append(f"Extra columns in new data: {sorted(extra_in_new)}")


def load_data(data_paths: List[str]) -> pl.DataFrame:
    """Load parquet data from one or more paths (supports glob patterns).

    Args:
        data_paths: List of paths or glob patterns to parquet files.

    Returns:
        Combined dataframe from all input files.
    """
    all_dfs = []
    for pattern in data_paths:
        path = Path(pattern)
        if "*" in pattern or "?" in pattern:
            matching_files = list(path.parent.glob(path.name))
            if not matching_files:
                raise FileNotFoundError(f"No files matching pattern: {pattern}")
            for f in matching_files:
                all_dfs.append(pl.read_parquet(f))
        else:
            if not path.exists():
                raise FileNotFoundError(f"File not found: {pattern}")
            all_dfs.append(pl.read_parquet(path))

    if not all_dfs:
        raise ValueError("No data files loaded")

    return pl.concat(all_dfs, how="vertical_relaxed")


@app.command("create")
def cmd_create(
    data: Annotated[
        list[str],
        typer.Option(
            ...,
            "--data",
            help="Path(s) to input parquet file(s). Supports glob patterns.",
        ),
    ],
    out_train: Annotated[
        Path,
        typer.Option(..., "--out-train", help="Output path for training parquet."),
    ],
    out_test: Annotated[
        Path,
        typer.Option(..., "--out-test", help="Output path for test parquet."),
    ],
    peptides_csv: Annotated[
        Path,
        typer.Option(
            ..., "--peptides-csv", help="Output path for peptide manifest CSV."
        ),
    ],
    out_val: Annotated[
        Optional[Path],
        typer.Option(
            "--out-val",
            help="Output path for validation parquet (omit for train/test only).",
        ),
    ] = None,
    sequence_col: Annotated[
        str,
        typer.Option(
            "--sequence-col", help="Column name containing peptide sequences."
        ),
    ] = "sequence",
    train_frac: Annotated[
        float,
        typer.Option("--train-frac", help="Fraction of data for training."),
    ] = 0.8,
    test_frac: Annotated[
        float,
        typer.Option(
            "--test-frac",
            help="Fraction for testing (ignored when --out-val is omitted; uses 1 - train_frac).",
        ),
    ] = 0.1,
    peptide_weight: Annotated[
        float,
        typer.Option(
            "--peptide-weight", help="Weight for peptide count vs row count balancing."
        ),
    ] = 0.5,
    seed: Annotated[int, typer.Option("--seed", help="Random seed.")] = 42,
) -> None:
    """Create new train/test (and optionally val) splits from scratch."""
    include_val = out_val is not None
    effective_test_frac = 1.0 - train_frac if not include_val else test_frac

    logger.info("Loading data from: %s", data)
    df = load_data(data)
    logger.info("Loaded %d rows", len(df))

    df = add_sanitised_column(df, sequence_col)

    num_unique = df.select("__sanitised_peptide").n_unique()
    logger.info("Found %s unique peptides", num_unique)

    train_df, test_df, val_df, assignments = split_by_peptides_balanced(
        df,
        peptide_col="__sanitised_peptide",
        train_frac=train_frac,
        test_frac=effective_test_frac,
        include_val=include_val,
        peptide_weight=peptide_weight,
        seed=seed,
    )

    train_df = train_df.drop("__sanitised_peptide")
    test_df = test_df.drop("__sanitised_peptide")

    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_test.parent.mkdir(parents=True, exist_ok=True)
    peptides_csv.parent.mkdir(parents=True, exist_ok=True)

    train_df.write_parquet(out_train)
    test_df.write_parquet(out_test)

    logger.info("Wrote train split: %s (%d rows)", out_train, len(train_df))
    logger.info("Wrote test split: %s (%d rows)", out_test, len(test_df))

    if val_df is not None:
        val_df = val_df.drop("__sanitised_peptide")
        assert out_val is not None
        out_val.parent.mkdir(parents=True, exist_ok=True)
        val_df.write_parquet(out_val)
        logger.info("Wrote val split: %s (%d rows)", out_val, len(val_df))

    save_manifest(assignments, peptides_csv, append=False)


@app.command("kfold")
def cmd_kfold(
    data: Annotated[
        list[str],
        typer.Option(
            ...,
            "--data",
            help="Path(s) to input parquet file(s). Supports glob patterns.",
        ),
    ],
    assignments_out: Annotated[
        Path,
        typer.Option(
            ...,
            "--assignments-out",
            help="Output CSV path (columns: standardised_sequence, fold).",
        ),
    ],
    k: Annotated[int, typer.Option("--k", help="Number of folds.")] = 5,
    sequence_col: Annotated[
        str,
        typer.Option(
            "--sequence-col", help="Column name containing peptide sequences."
        ),
    ] = "sequence",
    seed: Annotated[int, typer.Option("--seed", help="Random seed.")] = 42,
) -> None:
    """Assign each unique sanitised peptide to one of K folds (cross-validation).

    Writes a CSV with columns ``standardised_sequence`` and ``fold`` (0 .. K-1).
    Rows in the same fold are disjoint in peptide space from other folds; each
    peptide appears in exactly one fold.
    """
    logger.info("Loading data from: %s", data)
    df = load_data(data)
    df = add_sanitised_column(df, sequence_col)

    unique_peptides = df["__sanitised_peptide"].unique().to_list()
    rng = random.Random(seed)
    rng.shuffle(unique_peptides)

    rows = []
    for i, pep in enumerate(unique_peptides):
        rows.append({"standardised_sequence": pep, "fold": i % k})

    assignments_out.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(assignments_out)
    logger.info(
        "Wrote K-fold assignments for %d peptides (%d folds) to %s",
        len(unique_peptides),
        k,
        assignments_out,
    )


def _combine_and_drop_peptide_col(dfs: List[pl.DataFrame]) -> pl.DataFrame:
    """Combine dataframes and drop the internal peptide column.

    Args:
        dfs: List of dataframes to combine.

    Returns:
        Combined dataframe with __sanitised_peptide column removed.
    """
    if not dfs:
        return pl.DataFrame()
    combined = pl.concat(dfs, how="vertical_relaxed")
    if "__sanitised_peptide" in combined.columns:
        combined = combined.drop("__sanitised_peptide")
    return combined


def _append_to_parquet(
    existing_path: Path, existing_df: pl.DataFrame, new_df: pl.DataFrame, name: str
) -> None:
    """Append new rows to an existing parquet file.

    Args:
        existing_path: Path to the existing parquet file.
        existing_df: The existing dataframe (already loaded).
        new_df: New rows to append.
        name: Name of the split for logging.
    """
    if len(new_df) > 0:
        updated = pl.concat([existing_df, new_df], how="vertical_relaxed")
        updated.write_parquet(existing_path)
        logger.info(
            f"Updated {name}: {len(existing_df)} + {len(new_df)} = {len(updated)} rows"
        )
    else:
        logger.info(f"No new rows to add to {name} split")


def _log_peptide_summary(new_assignments: Dict[str, List[str]]) -> None:
    """Log a summary of unique peptides added.

    Args:
        new_assignments: Dictionary mapping split names to lists of new peptides.
    """
    total_new_peptides = sum(len(peps) for peps in new_assignments.values())
    logger.info("=" * 50)
    logger.info("SUMMARY: Unique peptides added")
    logger.info("=" * 50)
    logger.info(f"Total new peptides: {total_new_peptides}")
    for name in new_assignments:
        logger.info(f"  - {name}: {len(new_assignments[name])}")
    logger.info("=" * 50)


def _assign_known_peptides(
    known_rows: pl.DataFrame,
    peptide_to_split: Dict[str, str],
    split_names: List[str],
) -> Dict[str, pl.DataFrame]:
    """Assign rows with known peptides to their existing splits.

    Args:
        known_rows: Dataframe containing rows with peptides already in manifest.
        peptide_to_split: Mapping from peptide to split name.
        split_names: List of split names to assign into.

    Returns:
        Dictionary mapping split names to dataframes of assigned rows.
    """
    result: Dict[str, pl.DataFrame] = {}
    for split_name in split_names:
        peptides_in_split = [p for p, s in peptide_to_split.items() if s == split_name]
        result[split_name] = known_rows.filter(
            pl.col("__sanitised_peptide").is_in(peptides_in_split)
        )
    return result


def _split_novel_rows(
    novel_rows: pl.DataFrame,
    split_rows: Dict[str, List[pl.DataFrame]],
    include_val: bool,
    train_frac: float,
    test_frac: float,
    peptide_weight: float,
    seed: int,
) -> Dict[str, List[str]]:
    """Split novel peptide rows and append results to split_rows in place.

    Args:
        novel_rows: Dataframe of rows whose peptides are not yet in the manifest.
        split_rows: Mutable dict accumulating dataframes per split.
        include_val: Whether to include a validation split.
        train_frac: Fraction for training.
        test_frac: Fraction for testing.
        peptide_weight: Weight for peptide vs row balancing.
        seed: Random seed.

    Returns:
        New peptide assignments from the novel rows.
    """
    fracs = [int(train_frac * 100), int(test_frac * 100)]
    if include_val:
        fracs.append(int((1 - train_frac - test_frac) * 100))
    logger.info(f"Splitting rows with novel peptides ({'/'.join(map(str, fracs))})...")

    novel_train, novel_test, novel_val, novel_assignments = split_by_peptides_balanced(
        novel_rows,
        peptide_col="__sanitised_peptide",
        train_frac=train_frac,
        test_frac=test_frac,
        include_val=include_val,
        peptide_weight=peptide_weight,
        seed=seed,
    )
    split_rows["train"].append(novel_train)
    split_rows["test"].append(novel_test)
    if novel_val is not None:
        split_rows["val"].append(novel_val)
    return novel_assignments


@app.command("add")
def cmd_add(
    data: Annotated[
        list[str],
        typer.Option(
            ...,
            "--data",
            help="Path(s) to new input parquet file(s). Supports glob patterns.",
        ),
    ],
    existing_train: Annotated[
        Path,
        typer.Option(
            ...,
            "--existing-train",
            help="Path to existing training parquet (updated in place).",
        ),
    ],
    existing_test: Annotated[
        Path,
        typer.Option(
            ...,
            "--existing-test",
            help="Path to existing test parquet (updated in place).",
        ),
    ],
    peptides_csv: Annotated[
        Path,
        typer.Option(
            ...,
            "--peptides-csv",
            help="Path to existing peptide manifest CSV (appended to).",
        ),
    ],
    existing_val: Annotated[
        Optional[Path],
        typer.Option(
            "--existing-val",
            help="Path to existing validation parquet (omit for train/test only).",
        ),
    ] = None,
    sequence_col: Annotated[
        str,
        typer.Option(
            "--sequence-col", help="Column name containing peptide sequences."
        ),
    ] = "sequence",
    train_frac: Annotated[
        float,
        typer.Option("--train-frac", help="Fraction of new peptides for training."),
    ] = 0.8,
    test_frac: Annotated[
        float,
        typer.Option(
            "--test-frac",
            help="Fraction for testing (ignored when --existing-val omitted).",
        ),
    ] = 0.1,
    peptide_weight: Annotated[
        float,
        typer.Option("--peptide-weight", help="Peptide vs row balance weight."),
    ] = 0.5,
    seed: Annotated[int, typer.Option("--seed", help="Random seed.")] = 42,
) -> None:
    """Append new data to existing train/test (and optionally val) splits."""
    include_val = existing_val is not None
    effective_test_frac = 1.0 - train_frac if not include_val else test_frac

    split_names = ["train", "test"] + (["val"] if include_val else [])

    existing_paths = [
        (existing_train, "train"),
        (existing_test, "test"),
    ]
    if include_val:
        assert existing_val is not None
        existing_paths.append((existing_val, "val"))

    for p, name in existing_paths:
        if not p.exists():
            raise FileNotFoundError(f"Existing {name} parquet not found: {p}")

    logger.info("Loading existing manifest...")
    peptide_to_split = load_manifest(peptides_csv)
    logger.info("Loaded %d existing peptide assignments", len(peptide_to_split))

    logger.info("Loading new data from: %s", data)
    new_df = load_data(data)
    logger.info("Loaded %d new rows", len(new_df))

    new_df = add_sanitised_column(new_df, sequence_col)

    logger.info("Loading existing train parquet for schema validation...")
    existing_train_df = pl.read_parquet(existing_train)

    new_df_for_schema = new_df.drop("__sanitised_peptide")
    check_schema_compatibility(new_df_for_schema, existing_train_df)
    logger.info("Schema validation passed")

    existing_peptides = set(peptide_to_split.keys())
    all_peptides_in_new = set(new_df["__sanitised_peptide"].unique().to_list())
    known_peptides = all_peptides_in_new & existing_peptides
    novel_peptides = all_peptides_in_new - existing_peptides

    logger.info("Peptides in new data: %d", len(all_peptides_in_new))
    logger.info("  - Already assigned: %d", len(known_peptides))
    logger.info("  - New (need splitting): %d", len(novel_peptides))

    known_rows = new_df.filter(
        pl.col("__sanitised_peptide").is_in(list(known_peptides))
    )
    novel_rows = new_df.filter(
        pl.col("__sanitised_peptide").is_in(list(novel_peptides))
    )

    split_rows: Dict[str, List[pl.DataFrame]] = {s: [] for s in split_names}
    new_assignments: Dict[str, List[str]] = {s: [] for s in split_names}

    if len(known_rows) > 0:
        assigned = _assign_known_peptides(known_rows, peptide_to_split, split_names)
        for name in split_names:
            split_rows[name].append(assigned[name])
        parts = ", ".join(f"{n}={len(assigned[n])}" for n in split_names)
        logger.info("Assigned %d rows with known peptides: %s", len(known_rows), parts)

    if len(novel_rows) > 0:
        new_assignments = _split_novel_rows(
            novel_rows,
            split_rows,
            include_val,
            train_frac,
            effective_test_frac,
            peptide_weight,
            seed,
        )

    final_new = {
        name: _combine_and_drop_peptide_col(dfs) for name, dfs in split_rows.items()
    }

    logger.info("Appending to existing parquet files...")
    existing_test_df = pl.read_parquet(existing_test)

    _append_to_parquet(existing_train, existing_train_df, final_new["train"], "train")
    _append_to_parquet(existing_test, existing_test_df, final_new["test"], "test")

    if include_val:
        assert existing_val is not None
        existing_val_df = pl.read_parquet(existing_val)
        _append_to_parquet(existing_val, existing_val_df, final_new["val"], "val")

    if any(len(peps) > 0 for peps in new_assignments.values()):
        save_manifest(new_assignments, peptides_csv, append=True)

    _log_peptide_summary(new_assignments)
    logger.info("Done!")


def main() -> None:
    """Configure logging and run the Typer CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    app()


if __name__ == "__main__":
    main()
