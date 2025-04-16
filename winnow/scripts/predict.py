# -- Import
from winnow.calibration.calibration_features import (
    PrositFeatures,
    MassErrorFeature,
    RetentionTimeFeature,
    ChimericFeatures,
    BeamFeatures,
)
from winnow.calibration.calibrator import ProbabilityCalibrator
from winnow.datasets.calibration_dataset import RESIDUE_MASSES, CalibrationDataset
from winnow.fdr.bayes import EmpiricalBayesFDRControl

import logging
from rich.logging import RichHandler
from pathlib import Path
import pandas as pd
import ast
import pickle
from sklearn.model_selection import train_test_split


# --- Configuration ---
INPUTS_BASE_DIR = "input_data"
OUTPUTS_BASE_DIR = "calibrated_datasets"
SPECIES = "sbrodae"  # [gluc, helaqc, herceptin, immuno, sbrodae, snakevenoms, tplantibodies, woundfluids]
CONFIDENCE_TYPE = "calibrated_confidence"
TEST_FRACTION = 0.2
RANDOM_STATE = 42
SEED = 42
MZ_TOLERANCE = 0.02
HIDDEN_DIM = 10
TRAIN_FRACTION = 0.1
FDR_LR = 0.005
FDR_NSTEPS = 5000

# --- Logging Setup ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.addHandler(RichHandler())


# --- Utility Functions ---
def try_convert(value: str):
    """Safely convert string representations of lists."""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def check_uniqueness(df: pd.DataFrame, name: str):
    """Ensure spectrum_id is unique."""
    if df["spectrum_id"].nunique() != len(df):
        raise ValueError(f"Dataset {name} does not have unique spectrum_id.")
    logger.info(f"{name}: spectrum_id can uniquely identify rows.")


def filter_dataset(dataset: CalibrationDataset) -> CalibrationDataset:
    """Apply dataset filtering."""
    return (
        dataset.filter_entries(
            metadata_predicate=lambda row: not isinstance(row["prediction"], list)
        )
        .filter_entries(metadata_predicate=lambda row: not row["prediction"])
        .filter_entries(metadata_predicate=lambda row: row["precursor_charge"] > 6)
    )


def split_dataset(dataset: CalibrationDataset):
    """Create a train-test split."""
    return train_test_split(dataset, test_size=TEST_FRACTION, random_state=RANDOM_STATE)


def initialise_calibrator():
    """Set up the probability calibrator with features."""
    calibrator = ProbabilityCalibrator(SEED)
    calibrator.add_feature(MassErrorFeature(residue_masses=RESIDUE_MASSES))
    calibrator.add_feature(PrositFeatures(mz_tolerance=MZ_TOLERANCE))
    calibrator.add_feature(
        RetentionTimeFeature(hidden_dim=HIDDEN_DIM, train_fraction=TRAIN_FRACTION)
    )
    calibrator.add_feature(ChimericFeatures(mz_tolerance=MZ_TOLERANCE))
    calibrator.add_feature(BeamFeatures())
    return calibrator


def apply_fdr_control(dataset: CalibrationDataset) -> CalibrationDataset:
    """Apply empirical Bayes FDR control."""
    fdr_control = EmpiricalBayesFDRControl()
    fdr_control.fit(
        dataset=dataset.metadata[CONFIDENCE_TYPE], lr=FDR_LR, n_steps=FDR_NSTEPS
    )
    dataset.metadata = fdr_control.add_psm_fdr(dataset.metadata, CONFIDENCE_TYPE)
    dataset.metadata = fdr_control.add_psm_pep(dataset.metadata, CONFIDENCE_TYPE)
    dataset.metadata = fdr_control.add_psm_p_value(dataset.metadata, CONFIDENCE_TYPE)
    return dataset


def main():
    """Runs the end-to-end train and predict pipeline."""
    # --- Main Execution ---
    logger.info("Starting calibration pipeline.")

    annotated_spectrum_data_path = Path(
        f"{INPUTS_BASE_DIR}/spectrum_data/labelled/dataset-{SPECIES}-annotated-0000-0001.parquet"
    )
    annotated_beam_preds_path = Path(
        f"{INPUTS_BASE_DIR}/beam_preds/labelled/{SPECIES}-annotated_beam_preds.csv"
    )

    annotated_dataset = CalibrationDataset.from_predictions_csv(
        spectrum_path=annotated_spectrum_data_path,
        beam_predictions_path=annotated_beam_preds_path,
    )

    annotated_dataset = filter_dataset(annotated_dataset)
    train, test = split_dataset(annotated_dataset)
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

    # Train
    calibrator = initialise_calibrator()
    logger.info("Training calibrator.")
    calibrator.fit(train_dataset)
    output_path = Path(f"{OUTPUTS_BASE_DIR}/checkpoints/{SPECIES}_calibrator.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open(mode="wb") as file:
        pickle.dump(calibrator.classifier, file)
    logger.info(
        f"Calibration model saved: {OUTPUTS_BASE_DIR}/checkpoints/calibrator.pkl"
    )
    output_path = f"{OUTPUTS_BASE_DIR}/labelled/{SPECIES}_train_labelled.csv"
    train_dataset.to_csv(output_path)
    logger.info(f"Train dataset predictions saved: {output_path}")

    # Predict on known data
    logger.info("Calibrating on test dataset.")
    input_path = f"{OUTPUTS_BASE_DIR}/checkpoints/{SPECIES}_calibrator.pkl"
    logger.info("Loading calibration model checkpoint.")
    with open(input_path, "rb") as file:
        calibrator.classifier = pickle.load(file)
    calibrator.predict(test_dataset)
    logger.info("Applying FDR control to test dataset.")
    test_dataset = apply_fdr_control(test_dataset)
    output_path = f"{OUTPUTS_BASE_DIR}/labelled/{SPECIES}_test_labelled.csv"
    test_dataset.to_csv(output_path)
    logger.info(f"Test dataset predictions saved: {output_path}")

    # Predict on unknown data
    unseen_de_novo_data_path = (
        f"{INPUTS_BASE_DIR}/spectrum_data/de_novo/{SPECIES}_raw_filtered.parquet"
    )
    unseen_de_novo_beam_preds_path = (
        f"{INPUTS_BASE_DIR}/beam_preds/de_novo/{SPECIES}_raw_beam_preds_filtered.csv"
    )
    annotated_dataset = CalibrationDataset.from_predictions_csv(
        spectrum_path=unseen_de_novo_data_path,
        beam_predictions_path=unseen_de_novo_beam_preds_path,
    )

    annotated_dataset = filter_dataset(annotated_dataset)
    logger.info("Predicting on de novo dataset.")
    calibrator.predict(annotated_dataset)
    logger.info("Applying FDR control to de novo dataset.")
    annotated_dataset = apply_fdr_control(annotated_dataset)
    output_path = f"{OUTPUTS_BASE_DIR}/de_novo/{SPECIES}_de_novo_preds.csv"
    annotated_dataset.to_csv(output_path)
    logger.info(f"de novo dataset predictions saved: {output_path}")


if __name__ == "__main__":
    main()
