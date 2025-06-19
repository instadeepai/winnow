"""Add database-grounded FDR control to calibrated metadata.

This script takes the output from winnow's calibration and adds database-grounded FDR control
using the calibrated confidences. It adds database-grounded FDR values to the metadata.
"""

import logging
from pathlib import Path
import ast
from typing import Optional

import typer
from typing_extensions import Annotated
import pandas as pd

from winnow.fdr.database_grounded import DatabaseGroundedFDRControl
from winnow.datasets.calibration_dataset import RESIDUE_MASSES

# --- Logging Setup ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

app = typer.Typer(
    name="add_db_fdr",
    help="Add database-grounded FDR control to calibrated metadata.",
)


def check_if_labelled(metadata: pd.DataFrame) -> None:
    """Check if the dataset contains a ground-truth column."""
    if "sequence" not in metadata.columns:
        raise ValueError(
            "Database-grounded FDR control can only be performed on annotated data."
        )


def check_existing_columns(metadata: pd.DataFrame, column_name: str) -> str:
    """Check if a column exists and return a modified name if needed.

    Args:
        metadata: The DataFrame to check
        column_name: The desired column name

    Returns:
        str: A column name that doesn't exist in the DataFrame
    """
    if column_name not in metadata.columns:
        return column_name

    # If column exists, append _db to indicate database-grounded
    new_name = f"{column_name}_db"
    if new_name in metadata.columns:
        raise ValueError(
            f"Both {column_name} and {new_name} already exist in the metadata. "
            "Please rename or remove one of these columns before proceeding."
        )
    logger.warning(f"Column {column_name} already exists. Using {new_name} instead.")
    return new_name


def _convert_object_columns(metadata: pd.DataFrame) -> pd.DataFrame:
    """Convert object columns that might contain string representations of Python objects."""

    def try_convert(value):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value  # Return original if conversion fails

    # Apply conversion to object (string) columns
    for col in metadata.select_dtypes(include=["object"]).columns:
        metadata[col] = metadata[col].apply(try_convert)

    return metadata


def _map_l_to_i_in_sequences(metadata: pd.DataFrame) -> pd.DataFrame:
    """Map L to I in sequences and predictions."""
    logger.info("Mapping L to I in sequences and predictions")

    def _replace_l_with_i(value):
        """Replace L with I in a value, handling both strings and lists."""
        if isinstance(value, str):
            return value.replace("L", "I")
        elif isinstance(value, list):
            return [
                token.replace("L", "I") if isinstance(token, str) else token
                for token in value
            ]
        return value

    for col in ["sequence", "prediction"]:
        if col in metadata.columns:
            metadata[col] = metadata[col].apply(_replace_l_with_i)

    return metadata


def _save_confidence_cutoff(
    confidence_cutoff: float, confidence_cutoff_path: Optional[Path]
) -> None:
    """Save confidence cutoff to file if path is provided."""
    if confidence_cutoff_path is not None:
        confidence_cutoff_path.parent.mkdir(parents=True, exist_ok=True)
        with open(confidence_cutoff_path, "w") as f:
            f.write(str(confidence_cutoff))
        logger.info(f"Confidence cutoff saved: {confidence_cutoff_path}")


def _apply_fdr_control_and_save(
    metadata: pd.DataFrame,
    confidence_column: str,
    fdr_threshold: float,
    fdr_column: str,
    output_path: Path,
) -> None:
    """Apply FDR control and save results."""
    # Initialize and fit database-grounded FDR control
    logger.info("Applying database-grounded FDR control")
    fdr_control = DatabaseGroundedFDRControl(confidence_feature=confidence_column)
    fdr_control.fit(
        dataset=metadata,
        residue_masses=RESIDUE_MASSES,
    )

    confidence_cutoff = fdr_control.get_confidence_cutoff(threshold=fdr_threshold)
    logger.info(f"Confidence cutoff for FDR {fdr_threshold}: {confidence_cutoff}")

    # Add FDR values with the appropriate column name
    logger.info(f"Adding database-grounded FDR values to metadata as {fdr_column}")
    # Rename the column if needed
    if fdr_column != "psm_fdr":
        metadata = metadata.rename(columns={"psm_fdr": "psm_fdr_winnow"})
    metadata = fdr_control.add_psm_fdr(metadata, confidence_column)
    if fdr_column != "psm_fdr":
        metadata = metadata.rename(columns={"psm_fdr": "psm_fdr_dbg"})

    # Save results
    logger.info(f"Saving results to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output_path, index=False)
    logger.info("Done!")


@app.command()
def add_db_fdr(
    input_path: Annotated[
        Path,
        typer.Option(help="Path to the calibrated metadata CSV file."),
    ],
    output_path: Annotated[
        Path,
        typer.Option(help="Path to save the metadata with database-grounded FDR."),
    ],
    confidence_column: Annotated[
        str,
        typer.Option(
            help="Name of the column containing calibrated confidence scores.",
        ),
    ] = "calibrated_confidence",
    fdr_threshold: Annotated[
        float,
        typer.Option(
            help="The target FDR threshold (e.g. 0.01 for 1%, 0.05 for 5% etc.)",
        ),
    ] = 0.05,
    confidence_cutoff_path: Annotated[
        Optional[Path],
        typer.Option(help="Path to save the confidence cutoff to."),
    ] = None,
) -> None:
    """Add database-grounded FDR control to calibrated metadata.

    Args:
        input_path: Path to the calibrated metadata CSV file.
        output_path: Path to save the metadata with database-grounded FDR.
        confidence_column: Name of the column containing calibrated confidence scores.
        fdr_threshold: The target FDR threshold.
        confidence_cutoff_path: Path to save the confidence cutoff to.
    """
    # Load metadata
    logger.info(f"Loading metadata from {input_path}")
    metadata = pd.read_csv(input_path)

    # Convert object columns
    metadata = _convert_object_columns(metadata)

    # Map L to I in sequences and predictions
    metadata = _map_l_to_i_in_sequences(metadata)

    # Check if dataset is labelled
    check_if_labelled(metadata)

    # Check for existing FDR column and get appropriate name
    fdr_column = check_existing_columns(metadata, "psm_fdr")

    # Apply FDR control and save results
    _apply_fdr_control_and_save(
        metadata, confidence_column, fdr_threshold, fdr_column, output_path
    )

    # Save confidence cutoff if path is provided
    if confidence_cutoff_path is not None:
        # Re-initialize FDR control to get the cutoff
        fdr_control = DatabaseGroundedFDRControl(confidence_feature=confidence_column)
        fdr_control.fit(dataset=metadata, residue_masses=RESIDUE_MASSES)
        confidence_cutoff = fdr_control.get_confidence_cutoff(threshold=fdr_threshold)
        _save_confidence_cutoff(confidence_cutoff, confidence_cutoff_path)


if __name__ == "__main__":
    app()
