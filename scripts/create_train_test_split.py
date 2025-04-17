from winnow.datasets.calibration_dataset import CalibrationDataset

from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
import logging
from rich.logging import RichHandler
import argparse


# --- Logging Setup ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.addHandler(RichHandler())


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create train/test splits for calibration datasets"
    )
    parser.add_argument(
        "--spectrum_path",
        type=str,
        required=True,
        help="Path to the spectrum data parquet file",
    )
    parser.add_argument(
        "--beam_predictions_path",
        type=str,
        required=True,
        help="Path to the beam predictions CSV file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Base directory to save train and test splits",
    )
    parser.add_argument(
        "--test_fraction",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing (default: 0.2)",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def main():
    """Script entrypoint."""
    args = parse_args()

    logger.info("Loading dataset.")
    dataset = CalibrationDataset.from_predictions_csv(
        spectrum_path=args.spectrum_path,
        beam_predictions_path=args.beam_predictions_path,
    )

    logger.info("Creating train/test split.")
    train, test = train_test_split(
        dataset, test_size=args.test_fraction, random_state=args.random_state
    )

    train_metadata, train_predictions = zip(*train)
    train_dataset = CalibrationDataset(
        metadata=pd.DataFrame(train_metadata).reset_index(drop=True),
        predictions=list(train_predictions),
    )

    test_metadata, test_predictions = zip(*test)
    test_dataset = CalibrationDataset(
        metadata=pd.DataFrame(test_metadata).reset_index(drop=True),
        predictions=list(test_predictions),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing outputs to {output_dir}")
    train_dataset.save(output_dir / "train")
    test_dataset.save(output_dir / "test")


if __name__ == "__main__":
    main()
