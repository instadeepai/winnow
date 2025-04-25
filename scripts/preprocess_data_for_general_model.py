import logging
from rich.logging import RichHandler
import polars as pl
import argparse
from pathlib import Path
import numpy as np
from typing import Tuple


# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess validation data for general model"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="input_data",
        help="Base directory for input data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_data",
        help="Base directory for output data",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state for reproducibility",
    )
    return parser.parse_args()


def load_spectrum_data(input_dir: str, dataset: str) -> pl.DataFrame:
    """Load and process spectrum data for a single dataset.

    Args:
        input_dir: Base directory containing the validation datasets
        dataset: Name of the dataset to load

    Returns:
        Processed spectrum dataframe
    """
    spectrum_df = pl.read_parquet(
        f"{input_dir}/spectrum_data/labelled/dataset-{dataset}-annotated-0000-0001.parquet"
    )
    spectrum_df = spectrum_df.with_columns(
        [
            pl.col("sequence").str.replace_all("L", "I"),
            pl.lit(dataset).alias("source_dataset"),
        ]
    )
    return spectrum_df.drop("local_index").with_row_index("local_index")


def load_beam_data(input_dir: str, dataset: str) -> pl.DataFrame:
    """Load and process beam predictions for a single dataset.

    Args:
        input_dir: Base directory containing the validation datasets
        dataset: Name of the dataset to load

    Returns:
        Processed beam predictions dataframe
    """
    beam_df = pl.read_csv(
        f"{input_dir}/beam_preds/labelled/{dataset}-annotated_beam_preds.csv"
    )
    # Drop file and index columns if they exist
    if "file" in beam_df.columns:
        beam_df = beam_df.drop("file")
    if "index" in beam_df.columns:
        beam_df = beam_df.drop("index")
    beam_df = beam_df.with_columns([pl.lit(dataset).alias("source_dataset")])
    return beam_df.drop("local_index").with_row_index("local_index")


def add_missing_columns(dfs: list[pl.DataFrame]) -> list[pl.DataFrame]:
    """Add missing columns with null values to each dataframe.

    Args:
        dfs: List of dataframes to process

    Returns:
        List of dataframes with all columns present
    """
    all_columns = set()
    for df in dfs:
        all_columns.update(df.columns)

    for i in range(len(dfs)):
        missing_columns = all_columns - set(dfs[i].columns)
        if missing_columns:
            dfs[i] = dfs[i].with_columns(
                [pl.lit(None).alias(col) for col in missing_columns]
            )
    return dfs


def load_and_process_datasets(input_dir: str) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Load all validation datasets and process them.

    Args:
        input_dir: Base directory containing the validation datasets

    Returns:
        Tuple of (spectrum_df, beam_df)
    """
    logger.info("Loading validation datasets...")

    # List of validation datasets
    datasets = [
        "helaqc",
        "sbrodae",
        "herceptin",
        "immuno",
        "gluc",
        "snakevenoms",
        "woundfluids",
    ]

    # Load each dataset
    spectrum_dfs = []
    beam_dfs = []
    for dataset in datasets:
        try:
            spectrum_df = load_spectrum_data(input_dir, dataset)
            beam_df = load_beam_data(input_dir, dataset)

            spectrum_dfs.append(spectrum_df)
            beam_dfs.append(beam_df)

            logger.info(f"Loaded {dataset} dataset with {len(spectrum_df)} rows")
        except Exception as e:
            logger.warning(f"Failed to load {dataset} dataset: {e}")

    if not spectrum_dfs or not beam_dfs:
        raise ValueError("No datasets were successfully loaded")

    # Add missing columns to ensure consistent schema
    spectrum_dfs = add_missing_columns(spectrum_dfs)
    beam_dfs = add_missing_columns(beam_dfs)

    # Combine all datasets
    combined_spectrum_df = pl.concat(spectrum_dfs, how="vertical_relaxed")
    combined_beam_df = pl.concat(beam_dfs, how="vertical_relaxed")

    # Create new global indices
    combined_spectrum_df = combined_spectrum_df.drop("global_index").with_row_index(
        "global_index"
    )
    combined_beam_df = combined_beam_df.drop("global_index").with_row_index(
        "global_index"
    )

    logger.info(f"Combined dataset has {len(combined_spectrum_df)} rows")

    return combined_spectrum_df, combined_beam_df


def create_splits(
    spectrum_df: pl.DataFrame, beam_df: pl.DataFrame, random_state: int
) -> Tuple[
    Tuple[pl.DataFrame, pl.DataFrame],
    Tuple[pl.DataFrame, pl.DataFrame],
    Tuple[pl.DataFrame, pl.DataFrame],
]:
    """Create train/test/val splits ensuring no peptide appears in multiple splits.

    Args:
        spectrum_df: Input spectrum dataframe
        beam_df: Input beam predictions dataframe
        random_state: Random state for reproducibility

    Returns:
        Tuple of ((train_spectrum, train_beam), (test_spectrum, test_beam), (val_spectrum, val_beam))
    """
    logger.info("Creating train/test/val splits...")

    # Merge spectrum and beam data
    merged_df = spectrum_df.join(beam_df, on="spectrum_id", how="inner")
    logger.info(
        f"Found {len(merged_df)} spectra with both spectrum data and beam predictions"
    )

    # Get unique sequences for splitting
    unique_sequences = merged_df.select("sequence").unique()
    n_peptides = len(unique_sequences)

    # Set random seed
    np.random.seed(random_state)

    # Calculate split sizes (80/10/10)
    train_size = int(0.8 * n_peptides)
    test_size = int(0.1 * n_peptides)

    # Split peptides
    train_peptides = unique_sequences.slice(0, train_size)
    test_peptides = unique_sequences.slice(train_size, test_size)
    val_peptides = unique_sequences.slice(
        train_size + test_size, n_peptides - (train_size + test_size)
    )

    # Create splits from merged data
    train_merged = merged_df.join(train_peptides, on="sequence", how="inner")
    test_merged = merged_df.join(test_peptides, on="sequence", how="inner")
    val_merged = merged_df.join(val_peptides, on="sequence", how="inner")

    # Shuffle each split
    train_merged = train_merged.sample(fraction=1.0, seed=random_state, shuffle=True)
    test_merged = test_merged.sample(fraction=1.0, seed=random_state, shuffle=True)
    val_merged = val_merged.sample(fraction=1.0, seed=random_state, shuffle=True)

    # Split back into spectrum and beam data
    spectrum_columns = spectrum_df.columns
    beam_columns = beam_df.columns

    train_spectrum = train_merged.select(spectrum_columns)
    test_spectrum = test_merged.select(spectrum_columns)
    val_spectrum = val_merged.select(spectrum_columns)

    train_beam = train_merged.select(beam_columns)
    test_beam = test_merged.select(beam_columns)
    val_beam = val_merged.select(beam_columns)

    logger.info(
        f"Created splits: train={len(train_spectrum)}, test={len(test_spectrum)}, val={len(val_spectrum)}"
    )

    return (
        (train_spectrum, train_beam),
        (test_spectrum, test_beam),
        (val_spectrum, val_beam),
    )


def main():
    """Main function to preprocess validation data for general model."""
    args = parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and process datasets
    spectrum_df, beam_df = load_and_process_datasets(args.input_dir)

    # Create splits
    (
        (train_spectrum, train_beam),
        (test_spectrum, test_beam),
        (val_spectrum, val_beam),
    ) = create_splits(spectrum_df, beam_df, args.random_state)

    # Save splits
    train_spectrum.write_parquet(output_dir / "train_spectrum.parquet")
    test_spectrum.write_parquet(output_dir / "test_spectrum.parquet")
    val_spectrum.write_parquet(output_dir / "val_spectrum.parquet")

    train_beam.write_parquet(output_dir / "train_beam.parquet")
    test_beam.write_parquet(output_dir / "test_beam.parquet")
    val_beam.write_parquet(output_dir / "val_beam.parquet")

    logger.info("Successfully saved train/test/val splits")


if __name__ == "__main__":
    main()
