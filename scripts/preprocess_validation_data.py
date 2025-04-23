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
    parser = argparse.ArgumentParser(description="Preprocess validation data")
    parser.add_argument(
        "--species",
        type=str,
        required=True,
        choices=["helaqc", "sbrodae", "herceptin", "immuno", "gluc", "snakevenoms", "tplantibodies", "woundfluids"],
        help="Species to preprocess",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data",
        help="Base directory for input data",
    )
    return parser.parse_args()


# --- Utility Functions ---
def check_uniqueness(df: pd.DataFrame, name: str):
    """Ensure spectrum_id is unique."""
    if df["spectrum_id"].nunique() != len(df):
        raise ValueError(f"Dataset {name} does not have unique spectrum_id.")
    logger.info(f"{name}: spectrum_id can uniquely identify rows.")


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

    # Check if all spectrum_id values in annotated_data are present in raw_beam_preds
    missing_scans = set(annotated_data["spectrum_id"]) - set(
        raw_beam_preds["spectrum_id"]
    )
    if missing_scans:
        raise ValueError(
            f"{len(missing_scans)} spectrum_id values in annotated_data are missing from beam_preds."
        )

    logger.info("Merging datasets.")
    annotated_beam_preds = annotated_data.merge(
        raw_beam_preds,
        on=["spectrum_id"],
        how="inner",
        suffixes=("", "_from_raw"),
    )

    # Drop duplicate columns after merge
    for col in raw_beam_preds.columns:
        if col in annotated_data.columns and col not in ["spectrum_id"]:
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
    labelled_ids = annotated_data["spectrum_id"]

    # Exclude rows with spectrum_id in labelled_ids
    raw_data = raw_data[~raw_data["spectrum_id"].isin(labelled_ids)]
    raw_beam_preds = raw_beam_preds[~raw_beam_preds["spectrum_id"].isin(labelled_ids)]

    raw_data.to_parquet(de_novo_data_path, index=False)
    raw_beam_preds.to_csv(de_novo_beam_preds_path, index=False)
    logger.info(
        f"Unseen data and beam predictions saved: {de_novo_data_path, de_novo_beam_preds_path}"
    )


def main():
    """Separates out labelled and unlabelled data from a raw dataset."""
    args = parse_args()
    species = args.species
    input_dir = args.input_dir
    logger.info(f"Starting data separation process for {species}.")

    annotated_data_input_path = f"{input_dir}/spectrum_data/labelled/dataset-{species}-annotated-0000-0001.parquet"
    raw_data_input_path = (
        f"{input_dir}/spectrum_data/raw/dataset-{species}-raw-0000-0001.parquet"
    )
    raw_beam_preds_input_path = f"{input_dir}/beam_preds/raw/{species}_beam_preds.csv"
    annotated_beam_preds_path = (
        f"{input_dir}/beam_preds/labelled/{species}-annotated_beam_preds.csv"
    )

    annotated_data, raw_beam_preds = get_annotated_beam_preds(
        annotated_data_input_path,
        raw_beam_preds_input_path,
        annotated_beam_preds_path,
    )

    unseen_de_novo_data_path = (
        f"{input_dir}/spectrum_data/de_novo/{species}_raw_filtered.parquet"
    )
    unseen_de_novo_beam_preds_path = (
        f"{input_dir}/beam_preds/de_novo/{species}_raw_beam_preds_filtered.csv"
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
