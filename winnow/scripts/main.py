"""CLI entry point for winnow.

Note: This module uses lazy imports to minimise CLI startup time.
Heavy dependencies (PyTorch, InstaNovo, etc.) are imported only when
needed, significantly reducing --help and config command times.
"""

from __future__ import annotations

from typing import Union, Tuple, Optional, List, TYPE_CHECKING, Annotated
import typer
import logging
from rich.logging import RichHandler

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
    from winnow.utils.config_formatter import ConfigFormatter

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
        "prediction",
        confidence_column,
        "psm_fdr",
        "psm_q_value",
    ]
    # NonParametricFDRControl adds psm_pep column
    if isinstance(fdr_control, NonParametricFDRControl):
        preds_and_fdr_metrics_cols.append("psm_pep")
    if "sequence" in dataset_metadata.columns:
        preds_and_fdr_metrics_cols.append("sequence")
        preds_and_fdr_metrics_cols.append("num_matches")
        preds_and_fdr_metrics_cols.append("correct")
    dataset_preds_and_fdr_metrics = dataset_metadata[
        ["spectrum_id"] + preds_and_fdr_metrics_cols
    ]
    dataset_metadata = dataset_metadata.drop(columns=preds_and_fdr_metrics_cols)
    return dataset_metadata, dataset_preds_and_fdr_metrics


def train_entry_point(
    overrides: Optional[List[str]] = None,
    execute: bool = True,
    config_dir: Optional[str] = None,
) -> None:
    """The main training pipeline entry point.

    Args:
        overrides: Optional list of config overrides.
        execute: If False, only print the configuration and return without executing the pipeline.
        config_dir: Optional path to custom config directory. If provided, configs in this
            directory take precedence over package configs. Files not in custom dir will use package defaults (file-by-file resolution).
    """
    from hydra import initialize_config_dir, compose
    from hydra.utils import instantiate
    from winnow.utils.config_path import get_primary_config_dir

    # Get primary config directory (custom if provided, otherwise package/dev)
    primary_config_dir = get_primary_config_dir(config_dir)

    # Initialise Hydra with primary config directory
    with initialize_config_dir(
        config_dir=str(primary_config_dir),
        version_base="1.3",
        job_name="winnow_train",
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

    logger.info(f"Loaded: {len(annotated_dataset.metadata)} spectra")

    logger.info("Filtering dataset for empty predictions.")
    annotated_dataset = filter_dataset(annotated_dataset)

    logger.info(f"After filtering: {len(annotated_dataset.metadata)} spectra")

    # Instantiate the calibrator from the config
    logger.info("Instantiating calibrator from config.")
    calibrator = instantiate(cfg.calibrator)

    # Fit the calibrator to the dataset
    logger.info("Fitting calibrator to dataset.")
    training_history = calibrator.fit(annotated_dataset)

    # Log final training metrics
    logger.info(f"Training completed in {training_history.n_iter} iterations.")
    logger.info(f"Final training loss: {training_history.final_training_loss:.6f}")
    if training_history.final_validation_score is not None:
        logger.info(
            f"Final validation score: {training_history.final_validation_score:.6f}"
        )

    # Save training history
    logger.info(f"Saving training history to {cfg.training_history_path}")
    training_history.save(cfg.training_history_path)

    # Save the model
    logger.info(f"Saving model to {cfg.model_output_dir}")
    ProbabilityCalibrator.save(calibrator, cfg.model_output_dir)

    # Save the training dataset results
    logger.info(f"Final dataset: {len(annotated_dataset)} spectra")
    logger.info(f"Saving training dataset results to {cfg.dataset_output_path}")
    annotated_dataset.save_metadata(cfg.dataset_output_path)

    logger.info("Training pipeline completed successfully.")


def compute_features_entry_point(
    overrides: Optional[List[str]] = None,
    execute: bool = True,
    config_dir: Optional[str] = None,
) -> None:
    """Load a dataset, compute calibration features into metadata, and save CSV.

    Does not fit the MLP or write a calibrator checkpoint; reuses ``calibrator.features`` from Hydra like training.

    Args:
        overrides: Optional list of config overrides.
        execute: If False, only print the configuration and return.
        config_dir: Optional path to custom config directory.
    """
    from hydra import initialize_config_dir, compose
    from hydra.utils import instantiate
    from winnow.utils.config_path import get_primary_config_dir

    primary_config_dir = get_primary_config_dir(config_dir)

    with initialize_config_dir(
        config_dir=str(primary_config_dir),
        version_base="1.3",
        job_name="winnow_compute_features",
    ):
        cfg = compose(config_name="compute_features", overrides=overrides)

    if not execute:
        print_config(cfg)
        return

    logger.info("Starting compute-features pipeline.")
    logger.info(f"Compute-features configuration: {cfg}")

    labelled = bool(cfg.labelled)
    if not labelled and cfg.calibrator.features.retention_time_feature is not None:
        raise ValueError(
            f"Compute-features config setting labelled={labelled}, but standalone feature computation for RetentionTimeFeature is not supported for unlabelled datasets.\n"
            f"Please remove RetentionTimeFeature from the calibration feature set or use a labelled dataset with labelled=True."
        )

    logger.info("Loading dataset.")
    data_loader = instantiate(cfg.data_loader)

    dataset_params = dict(cfg.dataset)
    dataset_params["data_path"] = dataset_params.pop("spectrum_path_or_directory")
    dataset_params["predictions_path"] = dataset_params.pop("predictions_path", None)

    dataset = data_loader.load(**dataset_params)

    logger.info(f"Loaded: {len(dataset.metadata)} spectra")

    if labelled and "sequence" not in dataset.metadata.columns:
        raise ValueError(
            "Labelled dataset must contain a 'sequence' column with ground-truth sequences."
        )

    if cfg.filter_empty_predictions:
        logger.info("Filtering dataset for empty predictions.")
        dataset = filter_dataset(dataset)
        logger.info(f"After filtering: {len(dataset.metadata)} spectra")

    logger.info("Instantiating calibrator from config.")
    calibrator = instantiate(cfg.calibrator)

    logger.info(f"Computing features with labelled={labelled}.")
    calibrator.compute_features(dataset, labelled=labelled)

    logger.info(f"Saving dataset with features to {cfg.dataset_output_path}")
    dataset.save_metadata(cfg.dataset_output_path)

    logger.info("Compute-features pipeline completed successfully.")


def predict_entry_point(
    overrides: Optional[List[str]] = None,
    execute: bool = True,
    config_dir: Optional[str] = None,
) -> None:
    """The main prediction pipeline entry point.

    Args:
        overrides: Optional list of config overrides.
        execute: If False, only print the configuration and return without executing the pipeline.
        config_dir: Optional path to custom config directory. If provided, configs in this
            directory take precedence over package configs. Files not in custom dir will use
            package defaults (file-by-file resolution).
    """
    from hydra import initialize_config_dir, compose
    from hydra.utils import instantiate
    from winnow.utils.config_path import get_primary_config_dir

    # Get primary config directory (custom if provided, otherwise package/dev)
    primary_config_dir = get_primary_config_dir(config_dir)

    # Initialize Hydra with primary config directory
    with initialize_config_dir(
        config_dir=str(primary_config_dir),
        version_base="1.3",
        job_name="winnow_predict",
    ):
        cfg = compose(config_name="predict", overrides=overrides)

    if not execute:
        print_config(cfg)
        return

    from winnow.calibration.calibrator import ProbabilityCalibrator
    from winnow.datasets.calibration_dataset import CalibrationDataset
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

    logger.info(f"Loaded: {len(dataset.metadata)} spectra")

    logger.info("Filtering dataset for empty predictions.")
    dataset = filter_dataset(dataset)

    logger.info(f"After filtering: {len(dataset.metadata)} spectra")

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
    logger.info(f"Final dataset: {len(dataset_metadata)} spectra")
    logger.info(f"Writing output to {cfg.output_folder}")
    dataset_metadata, dataset_preds_and_fdr_metrics = separate_metadata_and_predictions(
        dataset_metadata, fdr_control, cfg.fdr_control.confidence_column
    )

    CalibrationDataset(metadata=dataset_metadata).save_metadata(
        cfg.output_folder + "/" + "metadata.csv"
    )
    CalibrationDataset(metadata=dataset_preds_and_fdr_metrics).save_metadata(
        cfg.output_folder + "/" + "preds_and_fdr_metrics.csv"
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
        "[bold cyan]Custom config directory:[/bold cyan]\n"
        "  [dim]winnow train --config-dir /path/to/configs[/dim]  # Use custom config directory\n"
        "  [dim]winnow train -cp ./my_configs[/dim]  # Short form (relative or absolute path)\n"
        "  See docs for advanced usage.\n\n"
        "[bold cyan]Configuration files to customise:[/bold cyan]\n"
        "  • config/train.yaml - Main config (data paths, output locations)\n"
        "  • config/calibrator.yaml - Model architecture and features\n"
        "  • config/data_loader/ - Dataset format loaders\n"
        "  • config/residues.yaml - Amino acid masses and modifications"
    ),
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def train(
    ctx: typer.Context,
    config_dir: Annotated[
        Optional[str],
        typer.Option(
            "--config-dir",
            "-cp",
            help="Path to custom config directory (relative or absolute). See documentation for advanced usage.",
        ),
    ] = None,
) -> None:
    """Passes control directly to the Hydra training pipeline."""
    # Capture extra arguments as Hydra overrides (--config-dir already parsed out by Typer)
    overrides = ctx.args if ctx.args else None
    train_entry_point(overrides, config_dir=config_dir)


@app.command(
    name="compute-features",
    help=(
        "Compute calibration features and write enriched metadata to CSV.\n\n"
        "Loads the dataset using the same options as training, instantiates the calibrator "
        "feature stack from config, runs feature computation (``prepare`` when labelled=true), "
        "and saves the result without fitting the MLP or saving a model.\n\n"
        "[bold cyan]Quick start:[/bold cyan]\n"
        "  [dim]winnow compute-features[/dim]  # Uses config/compute_features.yaml\n\n"
        "[bold cyan]Override parameters:[/bold cyan]\n"
        "  [dim]winnow compute-features dataset_output_path=results/my_features.csv[/dim]\n"
        "  [dim]winnow compute-features run_prepare=false[/dim]  # Skip feature.prepare()\n"
        "  [dim]winnow compute-features filter_empty_predictions=false[/dim]\n\n"
        "[bold cyan]Custom config directory:[/bold cyan]\n"
        "  [dim]winnow compute-features --config-dir /path/to/configs[/dim]\n"
        "  [dim]winnow compute-features -cp ./my_configs[/dim]\n\n"
        "[bold cyan]Feature set:[/bold cyan]\n"
        "  Reuses [dim]calibrator.features[/dim] from calibrator.yaml; override with Hydra "
        "(e.g. drop a feature with [dim]'~calibrator.features.retention_time_feature'[/dim])."
    ),
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def compute_features(
    ctx: typer.Context,
    config_dir: Annotated[
        Optional[str],
        typer.Option(
            "--config-dir",
            "-cp",
            help="Path to custom config directory (relative or absolute). See documentation for advanced usage.",
        ),
    ] = None,
) -> None:
    """Compute calibration features and save metadata CSV."""
    overrides = ctx.args if ctx.args else None
    compute_features_entry_point(overrides, config_dir=config_dir)


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
        "  [dim]winnow predict fdr_control.fdr_threshold=0.01[/dim]  # Target 1% FDR instead of 5%\n"
        "  [dim]winnow predict output_folder=results/my_run[/dim]  # Custom output location\n\n"
        "[bold cyan]Custom config directory:[/bold cyan]\n"
        "  [dim]winnow predict --config-dir /path/to/configs[/dim]  # Use custom config directory\n"
        "  [dim]winnow predict -cp ./my_configs[/dim]  # Short form (relative or absolute path)\n"
        "  See docs for advanced usage.\n\n"
        "[bold cyan]Configuration files to customise:[/bold cyan]\n"
        "  • config/predict.yaml - Main config (data paths, FDR settings, output)\n"
        "  • config/fdr_method/ - FDR methods (nonparametric, database_grounded)\n"
        "  • config/data_loader/ - Dataset format loaders\n"
        "  • config/residues.yaml - Amino acid masses and modifications"
    ),
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def predict(
    ctx: typer.Context,
    config_dir: Annotated[
        Optional[str],
        typer.Option(
            "--config-dir",
            "-cp",
            help="Path to custom config directory (relative or absolute). See documentation for advanced usage.",
        ),
    ] = None,
) -> None:
    """Passes control directly to the Hydra predict pipeline."""
    # Capture extra arguments as Hydra overrides (--config-dir already parsed out by Typer)
    overrides = ctx.args if ctx.args else None
    predict_entry_point(overrides, config_dir=config_dir)


@config_app.command(
    name="train",
    help=(
        "Display the resolved training configuration without running the pipeline.\n\n"
        "This is useful for inspecting the final configuration after all defaults "
        "and overrides have been applied.\n\n"
        "[bold cyan]Usage:[/bold cyan]\n"
        "  [dim]winnow config train[/dim]  # Show default config\n"
        "  [dim]winnow config train data_loader=mztab[/dim]  # Show config with overrides\n"
        "  [dim]winnow config train calibrator.seed=42[/dim]  # Check override application\n"
        "  [dim]winnow config train --config-dir /path/to/configs[/dim]  # Show config with custom directory\n"
        "  [dim]winnow config train -cp ./my_configs[/dim]  # Short form (relative or absolute path)"
    ),
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def config_train(
    ctx: typer.Context,
    config_dir: Annotated[
        Optional[str],
        typer.Option(
            "--config-dir",
            "-cp",
            help="Path to custom config directory (relative or absolute). See documentation for advanced usage.",
        ),
    ] = None,
) -> None:
    """Display the resolved training configuration."""
    overrides = ctx.args if ctx.args else None
    train_entry_point(overrides, execute=False, config_dir=config_dir)


@config_app.command(
    name="compute-features",
    help=(
        "Display the resolved compute-features configuration without running the pipeline.\n\n"
        "[bold cyan]Usage:[/bold cyan]\n"
        "  [dim]winnow config compute-features[/dim]\n"
        "  [dim]winnow config compute-features dataset_output_path=out.csv[/dim]\n"
        "  [dim]winnow config compute-features --config-dir /path/to/configs[/dim]\n"
        "  [dim]winnow config compute-features -cp ./my_configs[/dim]"
    ),
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def config_compute_features(
    ctx: typer.Context,
    config_dir: Annotated[
        Optional[str],
        typer.Option(
            "--config-dir",
            "-cp",
            help="Path to custom config directory (relative or absolute). See documentation for advanced usage.",
        ),
    ] = None,
) -> None:
    """Display the resolved compute-features configuration."""
    overrides = ctx.args if ctx.args else None
    compute_features_entry_point(overrides, execute=False, config_dir=config_dir)


@config_app.command(
    name="predict",
    help=(
        "Display the resolved prediction configuration without running the pipeline.\n\n"
        "This is useful for inspecting the final configuration after all defaults "
        "and overrides have been applied.\n\n"
        "[bold cyan]Usage:[/bold cyan]\n"
        "  [dim]winnow config predict[/dim]  # Show default config\n"
        "  [dim]winnow config predict fdr_method=database_grounded[/dim]  # Show config with overrides\n"
        "  [dim]winnow config predict fdr_control.fdr_threshold=0.01[/dim]  # Check override application\n"
        "  [dim]winnow config predict --config-dir /path/to/configs[/dim]  # Show config with custom directory\n"
        "  [dim]winnow config predict -cp ./my_configs[/dim]  # Short form (relative or absolute path)"
    ),
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def config_predict(
    ctx: typer.Context,
    config_dir: Annotated[
        Optional[str],
        typer.Option(
            "--config-dir",
            "-cp",
            help="Path to custom config directory (relative or absolute). See documentation for advanced usage.",
        ),
    ] = None,
) -> None:
    """Display the resolved prediction configuration."""
    overrides = ctx.args if ctx.args else None
    predict_entry_point(overrides, execute=False, config_dir=config_dir)


if __name__ == "__main__":
    app()
