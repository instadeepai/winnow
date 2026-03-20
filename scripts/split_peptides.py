#!/usr/bin/env python3
"""Generic peptide-based train/test/val split script.

Creates or incrementally updates train/test/val splits with no peptide leakage,
saves a CSV manifest of sanitised peptides per split, and enforces schema
compatibility when appending to existing parquet files.
"""

from __future__ import annotations

import argparse
import logging
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import polars as pl

logger = logging.getLogger(__name__)


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
    test_frac: float = 0.1,
    peptide_weight: float = 0.5,
    seed: int = 42,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, Dict[str, List[str]]]:
    """Split dataset ensuring no peptide overlap between splits.

    Balances both peptide count and row count targets using deficit-based bin packing.

    Args:
        df: The input dataframe to split.
        peptide_col: The column name containing the sanitised peptide sequences.
        train_frac: The fraction of peptides to assign to the training set.
        test_frac: The fraction of peptides to assign to the test set.
        peptide_weight: The weight of peptide count in the balancing (0-1).
        seed: The random seed for reproducibility.

    Returns:
        A tuple containing:
        - Training dataframe
        - Test dataframe
        - Validation dataframe
        - Dictionary mapping split names to lists of peptides assigned to each
    """
    random.seed(seed)

    peptide_counts = df.group_by(peptide_col).agg(pl.len().alias("count")).to_dicts()

    random.shuffle(peptide_counts)
    peptide_counts.sort(key=lambda x: x["count"], reverse=True)

    total_rows = len(df)
    total_peptides = len(peptide_counts)
    val_frac = 1 - train_frac - test_frac

    row_targets = {
        "train": total_rows * train_frac,
        "test": total_rows * test_frac,
        "val": total_rows * val_frac,
    }
    peptide_targets = {
        "train": total_peptides * train_frac,
        "test": total_peptides * test_frac,
        "val": total_peptides * val_frac,
    }

    row_current = {"train": 0, "test": 0, "val": 0}
    peptide_current = {"train": 0, "test": 0, "val": 0}
    assignments: Dict[str, List[str]] = {"train": [], "test": [], "val": []}

    row_weight = 1 - peptide_weight

    for item in peptide_counts:
        peptide = item[peptide_col]
        count = item["count"]

        scores = {}
        for split in ["train", "test", "val"]:
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
    val_df = df.filter(pl.col(peptide_col).is_in(assignments["val"]))

    logger.info(
        f"{'Split':<8} {'Peptides':>10} {'(target)':>10} {'Rows':>10} {'(target)':>10}"
    )
    logger.info("-" * 50)
    for name, split_df in [("train", train_df), ("test", test_df), ("val", val_df)]:
        n_pep = len(assignments[name])
        n_rows = len(split_df)
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


def cmd_create(args: argparse.Namespace) -> None:
    """Execute the 'create' command to build new splits from scratch.

    Args:
        args: Parsed command-line arguments.
    """
    logger.info(f"Loading data from: {args.data}")
    df = load_data(args.data)
    logger.info(f"Loaded {len(df)} rows")

    df = add_sanitised_column(df, args.sequence_col)

    num_unique = df.select("__sanitised_peptide").n_unique()
    logger.info(f"Found {num_unique} unique peptides")

    train_df, test_df, val_df, assignments = split_by_peptides_balanced(
        df,
        peptide_col="__sanitised_peptide",
        train_frac=args.train_frac,
        test_frac=args.test_frac,
        peptide_weight=args.peptide_weight,
        seed=args.seed,
    )

    train_df = train_df.drop("__sanitised_peptide")
    test_df = test_df.drop("__sanitised_peptide")
    val_df = val_df.drop("__sanitised_peptide")

    out_train = Path(args.out_train)
    out_test = Path(args.out_test)
    out_val = Path(args.out_val)
    manifest_path = Path(args.peptides_csv)

    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_test.parent.mkdir(parents=True, exist_ok=True)
    out_val.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    train_df.write_parquet(out_train)
    test_df.write_parquet(out_test)
    val_df.write_parquet(out_val)

    logger.info(f"Wrote train split: {out_train} ({len(train_df)} rows)")
    logger.info(f"Wrote test split: {out_test} ({len(test_df)} rows)")
    logger.info(f"Wrote val split: {out_val} ({len(val_df)} rows)")

    save_manifest(assignments, manifest_path, append=False)


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
    logger.info(f"  - train: {len(new_assignments['train'])}")
    logger.info(f"  - test:  {len(new_assignments['test'])}")
    logger.info(f"  - val:   {len(new_assignments['val'])}")
    logger.info("=" * 50)


def _assign_known_peptides(
    known_rows: pl.DataFrame, peptide_to_split: Dict[str, str]
) -> Dict[str, pl.DataFrame]:
    """Assign rows with known peptides to their existing splits.

    Args:
        known_rows: Dataframe containing rows with peptides already in manifest.
        peptide_to_split: Mapping from peptide to split name.

    Returns:
        Dictionary mapping split names to dataframes of assigned rows.
    """
    result: Dict[str, pl.DataFrame] = {}
    for split_name in ["train", "test", "val"]:
        peptides_in_split = [p for p, s in peptide_to_split.items() if s == split_name]
        result[split_name] = known_rows.filter(
            pl.col("__sanitised_peptide").is_in(peptides_in_split)
        )
    return result


def cmd_add(args: argparse.Namespace) -> None:
    """Execute the 'add' command to append new data to existing splits.

    Args:
        args: Parsed command-line arguments.
    """
    existing_train_path = Path(args.existing_train)
    existing_test_path = Path(args.existing_test)
    existing_val_path = Path(args.existing_val)
    manifest_path = Path(args.peptides_csv)

    for p, name in [
        (existing_train_path, "train"),
        (existing_test_path, "test"),
        (existing_val_path, "val"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"Existing {name} parquet not found: {p}")

    logger.info("Loading existing manifest...")
    peptide_to_split = load_manifest(manifest_path)
    logger.info(f"Loaded {len(peptide_to_split)} existing peptide assignments")

    logger.info(f"Loading new data from: {args.data}")
    new_df = load_data(args.data)
    logger.info(f"Loaded {len(new_df)} new rows")

    new_df = add_sanitised_column(new_df, args.sequence_col)

    logger.info("Loading existing train parquet for schema validation...")
    existing_train_df = pl.read_parquet(existing_train_path)

    new_df_for_schema = new_df.drop("__sanitised_peptide")
    check_schema_compatibility(new_df_for_schema, existing_train_df)
    logger.info("Schema validation passed")

    existing_peptides = set(peptide_to_split.keys())
    all_peptides_in_new = set(new_df["__sanitised_peptide"].unique().to_list())
    known_peptides = all_peptides_in_new & existing_peptides
    novel_peptides = all_peptides_in_new - existing_peptides

    logger.info(f"Peptides in new data: {len(all_peptides_in_new)}")
    logger.info(f"  - Already assigned: {len(known_peptides)}")
    logger.info(f"  - New (need splitting): {len(novel_peptides)}")

    known_rows = new_df.filter(
        pl.col("__sanitised_peptide").is_in(list(known_peptides))
    )
    novel_rows = new_df.filter(
        pl.col("__sanitised_peptide").is_in(list(novel_peptides))
    )

    split_rows: Dict[str, List[pl.DataFrame]] = {"train": [], "test": [], "val": []}
    new_assignments: Dict[str, List[str]] = {"train": [], "test": [], "val": []}

    if len(known_rows) > 0:
        assigned = _assign_known_peptides(known_rows, peptide_to_split)
        for name in ["train", "test", "val"]:
            split_rows[name].append(assigned[name])
        logger.info(
            f"Assigned {len(known_rows)} rows with known peptides: "
            f"train={len(assigned['train'])}, test={len(assigned['test'])}, "
            f"val={len(assigned['val'])}"
        )

    if len(novel_rows) > 0:
        logger.info("Splitting rows with novel peptides (80/10/10)...")
        novel_train, novel_test, novel_val, novel_assignments = (
            split_by_peptides_balanced(
                novel_rows,
                peptide_col="__sanitised_peptide",
                train_frac=args.train_frac,
                test_frac=args.test_frac,
                peptide_weight=args.peptide_weight,
                seed=args.seed,
            )
        )
        split_rows["train"].append(novel_train)
        split_rows["test"].append(novel_test)
        split_rows["val"].append(novel_val)
        new_assignments = novel_assignments

    final_new = {
        name: _combine_and_drop_peptide_col(dfs) for name, dfs in split_rows.items()
    }

    logger.info("Appending to existing parquet files...")
    existing_test_df = pl.read_parquet(existing_test_path)
    existing_val_df = pl.read_parquet(existing_val_path)

    _append_to_parquet(
        existing_train_path, existing_train_df, final_new["train"], "train"
    )
    _append_to_parquet(existing_test_path, existing_test_df, final_new["test"], "test")
    _append_to_parquet(existing_val_path, existing_val_df, final_new["val"], "val")

    if any(len(peps) > 0 for peps in new_assignments.values()):
        save_manifest(new_assignments, manifest_path, append=True)

    _log_peptide_summary(new_assignments)
    logger.info("Done!")


def main() -> None:
    """Main entry point for the CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Split peptide datasets into train/test/val with no peptide leakage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser(
        "create", help="Create new train/test/val splits from scratch"
    )
    create_parser.add_argument(
        "--data",
        nargs="+",
        required=True,
        help="Path(s) to input parquet file(s). Supports glob patterns.",
    )
    create_parser.add_argument(
        "--out-train", required=True, help="Output path for training parquet"
    )
    create_parser.add_argument(
        "--out-test", required=True, help="Output path for test parquet"
    )
    create_parser.add_argument(
        "--out-val", required=True, help="Output path for validation parquet"
    )
    create_parser.add_argument(
        "--peptides-csv", required=True, help="Output path for peptide manifest CSV"
    )
    create_parser.add_argument(
        "--sequence-col",
        default="sequence",
        help="Column name containing peptide sequences (default: sequence)",
    )
    create_parser.add_argument(
        "--train-frac",
        type=float,
        default=0.8,
        help="Fraction of data for training (default: 0.8)",
    )
    create_parser.add_argument(
        "--test-frac",
        type=float,
        default=0.1,
        help="Fraction of data for testing (default: 0.1)",
    )
    create_parser.add_argument(
        "--peptide-weight",
        type=float,
        default=0.5,
        help="Weight for peptide count vs row count balancing (default: 0.5)",
    )
    create_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    create_parser.set_defaults(func=cmd_create)

    add_parser = subparsers.add_parser(
        "add", help="Add new data to existing train/test/val splits"
    )
    add_parser.add_argument(
        "--data",
        nargs="+",
        required=True,
        help="Path(s) to new input parquet file(s). Supports glob patterns.",
    )
    add_parser.add_argument(
        "--existing-train",
        required=True,
        help="Path to existing training parquet (will be updated in place)",
    )
    add_parser.add_argument(
        "--existing-test",
        required=True,
        help="Path to existing test parquet (will be updated in place)",
    )
    add_parser.add_argument(
        "--existing-val",
        required=True,
        help="Path to existing validation parquet (will be updated in place)",
    )
    add_parser.add_argument(
        "--peptides-csv",
        required=True,
        help="Path to existing peptide manifest CSV (will be appended to)",
    )
    add_parser.add_argument(
        "--sequence-col",
        default="sequence",
        help="Column name containing peptide sequences (default: sequence)",
    )
    add_parser.add_argument(
        "--train-frac",
        type=float,
        default=0.8,
        help="Fraction of new peptides for training (default: 0.8)",
    )
    add_parser.add_argument(
        "--test-frac",
        type=float,
        default=0.1,
        help="Fraction of new peptides for testing (default: 0.1)",
    )
    add_parser.add_argument(
        "--peptide-weight",
        type=float,
        default=0.5,
        help="Weight for peptide count vs row count balancing (default: 0.5)",
    )
    add_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    add_parser.set_defaults(func=cmd_add)

    args = parser.parse_args()

    try:
        args.func(args)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
