"""CLI entry point for winnow.

Note: This module uses lazy imports to minimise CLI startup time.
Heavy dependencies (PyTorch, InstaNovo, etc.) are imported only when
needed, significantly reducing --help and config command times.
"""

from __future__ import annotations

from typing import Union, Tuple, Optional, List, TYPE_CHECKING
import typer
import logging
from rich.logging import RichHandler
from pathlib import Path

# Lazy imports for heavy dependencies - only imported when actually needed
if TYPE_CHECKING:
    import pandas as pd
    from winnow.datasets.calibration_dataset import CalibrationDataset
    from winnow.fdr.nonparametric import NonParametricFDRControl
    from winnow.fdr.database_grounded import DatabaseGroundedFDRControl

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Prevent duplicate messages by disabling propagation and using only RichHandler
logger.propagate = False
if not logger.handlers:
    logger.addHandler(RichHandler())


# Typer CLI setup
app = typer.Typer(
    name="winnow",
    help="""Confidence calibration and FDR estimation for de novo peptide sequencing.""",
    rich_markup_mode="rich",
)

# Config command group
config_app = typer.Typer(
    name="config",
    help="Configuration utilities for inspecting resolved settings.",
    rich_markup_mode="rich",
)
app.add_typer(config_app)


def print_config(cfg) -> None:
    """Print configuration with hierarchical colour-coding based on nesting depth.

    Args:
        cfg: OmegaConf configuration object to print
    """
    from winnow.scripts.config_formatter import ConfigFormatter

    formatter = ConfigFormatter()
    formatter.print_config(cfg)


def filter_dataset(dataset: CalibrationDataset) -> CalibrationDataset:
    """Filter out rows whose predictions are empty or contain unsupported PSMs.

    Args:
        dataset (CalibrationDataset): The dataset to be filtered

    Returns:
        CalibrationDataset: The filtered dataset
    """
    filtered_dataset = (
        dataset.filter_entries(
            # Filter out non-list predictions
            metadata_predicate=lambda row: not isinstance(row["prediction"], list),
        )
        # Filter out empty predictions
        .filter_entries(metadata_predicate=lambda row: not row["prediction"])
    )
    return filtered_dataset


def apply_fdr_control(
    fdr_control: Union[NonParametricFDRControl, DatabaseGroundedFDRControl],
    dataset: CalibrationDataset,
    fdr_threshold: float,
    confidence_column: str,
) -> pd.DataFrame:
    """Apply FDR control to a dataset."""
    from winnow.fdr.nonparametric import NonParametricFDRControl

    if isinstance(fdr_control, NonParametricFDRControl):
        fdr_control.fit(dataset=dataset.metadata[confidence_column])
        dataset.metadata = fdr_control.add_psm_pep(dataset.metadata, confidence_column)
    else:
        fdr_control.fit(dataset=dataset.metadata[confidence_column])

    dataset.metadata = fdr_control.add_psm_fdr(dataset.metadata, confidence_column)
    dataset.metadata = fdr_control.add_psm_q_value(dataset.metadata, confidence_column)
    confidence_cutoff = fdr_control.get_confidence_cutoff(threshold=fdr_threshold)
    output_data = dataset.metadata
    output_data = output_data[output_data[confidence_column] >= confidence_cutoff]
    return output_data


def check_if_labelled(dataset: CalibrationDataset) -> None:
    """Check if the dataset contains a ground-truth column."""
    if "sequence" not in dataset.metadata.columns:
        raise ValueError(
            "Database-grounded FDR control can only be performed on annotated data."
        )


def separate_metadata_and_predictions(
    dataset_metadata: pd.DataFrame,
    fdr_control: Union[NonParametricFDRControl, DatabaseGroundedFDRControl],
    confidence_column: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Separate out metadata from prediction and FDR metrics.

    Args:
        dataset_metadata: The metadata dataframe to separate out prediction and FDR metrics from metadata and computed features.
        fdr_control: The FDR control object used (to determine which columns were added).
        confidence_column: The name of the confidence column.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the metadata dataframe and the prediction and FDR metrics dataframe.
    """
    from winnow.fdr.nonparametric import NonParametricFDRControl

    # Separate out metadata from prediction and FDR metrics
    preds_and_fdr_metrics_cols = [
        "spectrum_id",
        confidence_column,
        "prediction",
        "psm_fdr",
        "psm_q_value",
    ]
    if "sequence" in dataset_metadata.columns:
        preds_and_fdr_metrics_cols.append("sequence")
    # NonParametricFDRControl adds psm_pep column
    if isinstance(fdr_control, NonParametricFDRControl):
        preds_and_fdr_metrics_cols.append("psm_pep")
    dataset_preds_and_fdr_metrics = dataset_metadata[preds_and_fdr_metrics_cols]
    dataset_metadata = dataset_metadata.drop(columns=preds_and_fdr_metrics_cols)
    return dataset_metadata, dataset_preds_and_fdr_metrics


def train_entry_point(
    overrides: Optional[List[str]] = None, execute: bool = True
) -> None:
    """The main training pipeline entry point.

    Args:
        overrides: Optional list of config overrides.
        execute: If False, only print the configuration and return without executing the pipeline.
    """
    from hydra import initialize, compose
    from hydra.utils import instantiate

    with initialize(
        config_path="../../config", version_base="1.3", job_name="winnow_train"
    ):
        cfg = compose(config_name="train", overrides=overrides)

    if not execute:
        print_config(cfg)
        return

    from winnow.calibration.calibrator import ProbabilityCalibrator

    logger.info("Starting training pipeline.")
    logger.info(f"Training configuration: {cfg}")

    # Load dataset - Hydra creates the DatasetLoader object
    logger.info("Loading dataset.")
    data_loader = instantiate(cfg.data_loader)

    # Extract dataset loading parameters and convert to dict for flexible kwargs
    dataset_params = dict(cfg.dataset)
    # Rename config keys to match the Protocol interface
    dataset_params["data_path"] = dataset_params.pop("spectrum_path_or_directory")
    dataset_params["predictions_path"] = dataset_params.pop("predictions_path", None)

    annotated_dataset = data_loader.load(**dataset_params)

    logger.info("Filtering dataset.")
    annotated_dataset = filter_dataset(annotated_dataset)

    # Instantiate the calibrator from the config
    logger.info("Instantiating calibrator from config.")
    calibrator = instantiate(cfg.calibrator)

    # Fit the calibrator to the dataset
    logger.info("Fitting calibrator to dataset.")
    calibrator.fit(annotated_dataset)

    # Save the model
    logger.info(f"Saving model to {cfg.model_output_dir}")
    ProbabilityCalibrator.save(calibrator, cfg.model_output_dir)

    # Save the training dataset results
    logger.info(f"Saving training dataset results to {cfg.dataset_output_path}")
    annotated_dataset.to_csv(cfg.dataset_output_path)

    logger.info("Training pipeline completed successfully.")


def predict_entry_point(
    overrides: Optional[List[str]] = None, execute: bool = True
) -> None:
    """The main prediction pipeline entry point.

    Args:
        overrides: Optional list of config overrides.
        execute: If False, only print the configuration and return without executing the pipeline.
    """
    from hydra import initialize, compose
    from hydra.utils import instantiate

    with initialize(
        config_path="../../config", version_base="1.3", job_name="winnow_predict"
    ):
        cfg = compose(config_name="predict", overrides=overrides)

    if not execute:
        print_config(cfg)
        return

    from winnow.calibration.calibrator import ProbabilityCalibrator
    from winnow.fdr.database_grounded import DatabaseGroundedFDRControl

    logger.info("Starting prediction pipeline.")
    logger.info(f"Prediction configuration: {cfg}")

    # Load dataset - Hydra creates the DatasetLoader object
    logger.info("Loading dataset.")
    data_loader = instantiate(cfg.data_loader)

    # Extract dataset loading parameters and convert to dict for flexible kwargs
    dataset_params = dict(cfg.dataset)
    # Rename config keys to match the Protocol interface
    dataset_params["data_path"] = dataset_params.pop("spectrum_path_or_directory")
    dataset_params["predictions_path"] = dataset_params.pop("predictions_path", None)

    dataset = data_loader.load(**dataset_params)

    logger.info("Filtering dataset.")
    dataset = filter_dataset(dataset)

    # Load trained calibrator
    logger.info("Loading trained calibrator.")
    calibrator = ProbabilityCalibrator.load(
        pretrained_model_name_or_path=cfg.calibrator.pretrained_model_name_or_path,
        cache_dir=cfg.calibrator.cache_dir,
    )

    # Calibrate scores
    logger.info("Calibrating scores.")
    calibrator.predict(dataset)

    # Instantiate FDR control from config - Hydra handles which FDR method to use
    logger.info("Instantiating FDR control from config.")
    fdr_control = instantiate(cfg.fdr_method)

    # Check if dataset is labelled for database-grounded FDR
    if isinstance(fdr_control, DatabaseGroundedFDRControl):
        check_if_labelled(dataset)

    # Apply FDR control
    logger.info(f"Applying {fdr_control.__class__.__name__} FDR control.")
    dataset_metadata = apply_fdr_control(
        fdr_control,
        dataset,
        cfg.fdr_control.fdr_threshold,
        cfg.fdr_control.confidence_column,
    )

    # Write output
    logger.info(f"Writing output to {cfg.output_folder}")
    dataset_metadata, dataset_preds_and_fdr_metrics = separate_metadata_and_predictions(
        dataset_metadata, fdr_control, cfg.fdr_control.confidence_column
    )
    output_folder = Path(cfg.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    dataset_metadata.to_csv(output_folder.joinpath("metadata.csv"))
    dataset_preds_and_fdr_metrics.to_csv(
        output_folder.joinpath("preds_and_fdr_metrics.csv")
    )

    logger.info("Prediction pipeline completed successfully.")


@app.command(
    name="train",
    help=(
        "Train a probability calibration model on annotated peptide sequencing data.\n\n"
        "This command loads your dataset, trains calibration features, and saves the trained model.\n\n"
        "[bold cyan]Quick start:[/bold cyan]\n"
        "  [dim]winnow train[/dim]  # Uses default config from config/train.yaml\n\n"
        "[bold cyan]Override parameters:[/bold cyan]\n"
        "  [dim]winnow train data_loader=mztab[/dim]  # Use MZTab format instead of InstaNovo\n"
        "  [dim]winnow train model_output_dir=models/my_model[/dim]  # Custom output location\n"
        "  [dim]winnow train calibrator.seed=42[/dim]  # Set random seed\n\n"
        "[bold cyan]Configuration files to customise:[/bold cyan]\n"
        "  • config/train.yaml - Main config (data paths, output locations)\n"
        "  • config/calibrator.yaml - Model architecture and features\n"
        "  • config/data_loader/ - Dataset format loaders\n"
        "  • config/residues.yaml - Amino acid masses and modifications"
    ),
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def train(ctx: typer.Context) -> None:
    """Passes control directly to the Hydra training pipeline."""
    # Capture extra arguments as Hydra overrides
    overrides = ctx.args if ctx.args else None
    train_entry_point(overrides)


@app.command(
    name="predict",
    help=(
        "Calibrate confidence scores and filter peptide predictions by false discovery rate (FDR).\n\n"
        "This command loads your dataset, applies a trained calibrator to improve confidence scores, "
        "estimates FDR using your chosen method, and outputs filtered predictions at your target FDR threshold.\n\n"
        "[bold cyan]Quick start:[/bold cyan]\n"
        "  [dim]winnow predict[/dim]  # Uses default config from config/predict.yaml\n\n"
        "[bold cyan]Override parameters:[/bold cyan]\n"
        "  [dim]winnow predict data_loader=mztab[/dim]  # Use MZTab format instead of InstaNovo\n"
        "  [dim]winnow predict fdr_method=database_grounded[/dim]  # Use database-grounded FDR\n"
        "  [dim]winnow predict fdr_threshold=0.01[/dim]  # Target 1% FDR instead of 5%\n"
        "  [dim]winnow predict output_folder=results/my_run[/dim]  # Custom output location\n\n"
        "[bold cyan]Configuration files to customise:[/bold cyan]\n"
        "  • config/predict.yaml - Main config (data paths, FDR settings, output)\n"
        "  • config/fdr_method/ - FDR methods (nonparametric, database_grounded)\n"
        "  • config/data_loader/ - Dataset format loaders\n"
        "  • config/residues.yaml - Amino acid masses and modifications"
    ),
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def predict(ctx: typer.Context) -> None:
    """Passes control directly to the Hydra predict pipeline."""
    # Capture extra arguments as Hydra overrides
    overrides = ctx.args if ctx.args else None
    predict_entry_point(overrides)


@config_app.command(
    name="train",
    help=(
        "Display the resolved training configuration without running the pipeline.\n\n"
        "This is useful for inspecting the final configuration after all defaults "
        "and overrides have been applied.\n\n"
        "[bold cyan]Usage:[/bold cyan]\n"
        "  [dim]winnow config train[/dim]  # Show default config\n"
        "  [dim]winnow config train data_loader=mztab[/dim]  # Show config with overrides\n"
        "  [dim]winnow config train calibrator.seed=42[/dim]  # Check override application"
    ),
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def config_train(ctx: typer.Context) -> None:
    """Display the resolved training configuration."""
    overrides = ctx.args if ctx.args else None
    train_entry_point(overrides, execute=False)


@config_app.command(
    name="predict",
    help=(
        "Display the resolved prediction configuration without running the pipeline.\n\n"
        "This is useful for inspecting the final configuration after all defaults "
        "and overrides have been applied.\n\n"
        "[bold cyan]Usage:[/bold cyan]\n"
        "  [dim]winnow config predict[/dim]  # Show default config\n"
        "  [dim]winnow config predict fdr_method=database_grounded[/dim]  # Show config with overrides\n"
        "  [dim]winnow config predict fdr_control.fdr_threshold=0.01[/dim]  # Check override application"
    ),
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def config_predict(ctx: typer.Context) -> None:
    """Display the resolved prediction configuration."""
    overrides = ctx.args if ctx.args else None
    predict_entry_point(overrides, execute=False)


if __name__ == "__main__":
    app()
