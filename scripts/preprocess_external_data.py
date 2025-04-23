# -- Import
import logging
from rich.logging import RichHandler
import pandas as pd
import ast
import argparse


# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess external data")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data",
        help="Base directory for input data",
    )
    return parser.parse_args()


# --- Utility Functions ---
def check_uniqueness(df: pd.DataFrame, name: str):
    """Check if the dataset has unique scan, header, precursor_mz, precursor_charge, and experiment_name values."""
    if df[["scan", "header", "precursor_mz", "precursor_charge", "experiment_name"]].drop_duplicates().shape[0] != len(df):
        raise ValueError(f"Dataset {name} does not have unique scan.")
    logger.info(f"{name}: scan can uniquely identify rows.")


def try_convert(value: str):
    """Safely convert string representations of lists."""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def get_annotated_beam_preds(
    annotated_data_path: str,
    raw_beam_preds_path: str,
    annotated_beam_preds_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and merge annotated data with raw beam predictions to create annotated beam predictions."""
    logger.info("Loading datasets.")
    annotated_data = pd.read_parquet(annotated_data_path)
    raw_beam_preds = pd.read_csv(raw_beam_preds_path)

    # Apply conversion to object (string) columns
    for col in raw_beam_preds.select_dtypes(include=["object"]).columns:
        raw_beam_preds[col] = raw_beam_preds[col].apply(try_convert)

    check_uniqueness(raw_beam_preds, "raw_beam_preds")
    check_uniqueness(annotated_data, "annotated_data")

    # Check if all rows in annotated_data are present in raw_beam_preds
    merge_cols = ["scan", "header", "precursor_mz", "precursor_charge", "experiment_name"]
    missing_rows = annotated_data[merge_cols].merge(
        raw_beam_preds[merge_cols],
        how="left",
        indicator=True
    ).query('_merge == "left_only"')
    
    if not missing_rows.empty:
        raise ValueError(
            f"{len(missing_rows)} rows in annotated_data are missing from beam_preds."
        )

    logger.info("Merging datasets.")
    annotated_beam_preds = annotated_data.merge(
        raw_beam_preds,
        on=merge_cols,
        how="inner",
        suffixes=("", "_from_raw"),
    )

    # Drop duplicate columns after merge
    for col in raw_beam_preds.columns:
        if col in annotated_data.columns and col not in merge_cols:
            annotated_beam_preds.drop(columns=[col + "_from_raw"], inplace=True)

    # Validate merge result
    if len(annotated_beam_preds) != len(annotated_data):
        raise ValueError(
            f"Merge conflict: Expected {len(annotated_data)} rows, but got {len(annotated_beam_preds)}."
        )

    annotated_beam_preds.to_csv(annotated_beam_preds_path, index=False)
    logger.info(f"Annotated beam predictions saved: {annotated_beam_preds_path}")

    return annotated_data, raw_beam_preds


def get_unseen_de_novo_data(
    raw_data_path: str,
    annotated_data: pd.DataFrame,
    raw_beam_preds: pd.DataFrame,
    de_novo_data_path: str,
    de_novo_beam_preds_path: str,
):
    """Separate out unseen de novo data from seen labelled data in the raw dataset."""
    logger.info("Loading datasets.")
    raw_data = pd.read_parquet(raw_data_path)
    merge_cols = ["scan", "header", "precursor_mz", "precursor_charge", "experiment_name"]

    # Exclude rows that match any row in annotated_data
    raw_data = raw_data.merge(
        annotated_data[merge_cols],
        how="left",
        indicator=True
    ).query('_merge == "left_only"').drop(columns=["_merge"])

    raw_beam_preds = raw_beam_preds.merge(
        annotated_data[merge_cols],
        how="left",
        indicator=True
    ).query('_merge == "left_only"').drop(columns=["_merge"])

    raw_data.to_parquet(de_novo_data_path, index=False)
    raw_beam_preds.to_csv(de_novo_beam_preds_path, index=False)
    logger.info(
        f"Unseen data and beam predictions saved: {de_novo_data_path, de_novo_beam_preds_path}"
    )


def main():
    """Separates out labelled and unlabelled data from a raw dataset."""
    args = parse_args()
    input_dir = args.input_dir

    annotated_data_input_path = f"{input_dir}/spectrum_data/labelled/*.parquet"
    raw_data_input_path = (
        f"{input_dir}/spectrum_data/raw/*.parquet"
    )
    raw_beam_preds_input_path = f"{input_dir}/beam_preds/raw/*.csv"

    # TODO: Might need to save the conglomerated inputs to a single file, respectively.

    annotated_beam_preds_path = (
        f"{input_dir}/beam_preds/labelled/annotated_beam_preds.csv"
    )

    annotated_data, raw_beam_preds = get_annotated_beam_preds(
        annotated_data_input_path,
        raw_beam_preds_input_path,
        annotated_beam_preds_path,
    )

    unseen_de_novo_data_path = (
        f"{input_dir}/spectrum_data/de_novo/raw_filtered.parquet"
    )
    unseen_de_novo_beam_preds_path = (
        f"{input_dir}/beam_preds/de_novo/raw_beam_preds_filtered.csv"
    )

    get_unseen_de_novo_data(
        raw_data_input_path,
        annotated_data,
        raw_beam_preds,
        unseen_de_novo_data_path,
        unseen_de_novo_beam_preds_path,
    )


if __name__ == "__main__":
    main()
