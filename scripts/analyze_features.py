"""Analyze feature importance and correlations for a pretrained calibrator.

This script provides comprehensive analysis of feature importance:
    - Permutation importance on test set
    - SHAP values with training background on test set
    - Feature correlation analysis on training data
    - Optional visualization of results
"""

import logging
import pickle
from pathlib import Path
from typing import Annotated, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import torch
import typer
import yaml
from rich.console import Console
from rich.theme import Theme
from sklearn.inspection import permutation_importance

from winnow.calibration.calibrator import ProbabilityCalibrator
from winnow.datasets.data_loaders import InstaNovoDatasetLoader

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
sns.set_theme(style="white", palette="colorblind", context="paper", font_scale=1.5)

_PALETTE = sns.color_palette("colorblind")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("winnow")
logger.setLevel(logging.INFO)

logging.getLogger("shap").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Constants — loaded from the canonical Winnow YAML configs
# ---------------------------------------------------------------------------
SEED = 42

_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "winnow" / "configs"

with open(_CONFIGS_DIR / "residues.yaml") as _f:
    RESIDUE_MASSES: dict[str, float] = yaml.safe_load(_f)["residue_masses"]

with open(_CONFIGS_DIR / "data_loader" / "instanovo.yaml") as _f:
    _instanovo_cfg = yaml.safe_load(_f)
    RESIDUE_REMAPPING: dict[str, str] = _instanovo_cfg.get("residue_remapping", {})
    BEAM_COLUMNS: dict[str, str] | None = _instanovo_cfg.get("beam_columns")

COLUMN_DISPLAY_NAMES = {
    "confidence": "Raw confidence",
    "mass_error": "Mass error",
    "ion_matches": "Ion matches",
    "ion_match_intensity": "Ion match intensity",
    "chimeric_ion_matches": "Chimeric ion matches",
    "chimeric_ion_match_intensity": "Chimeric ion match intensity",
    "irt_error": "iRT error",
    "margin": "Margin",
    "median_margin": "Median margin",
    "entropy": "Entropy",
    "z-score": "Z-score",
    "edit_distance": "Edit distance",
    "xcorr": "XCorr",
    "chimeric_xcorr": "Chimeric XCorr",
    "longest_ion_series": "Longest ion series",
    "complementary_ion_count": "Complementary ion count",
    "max_ion_gap": "Max ion gap",
    "chimeric_longest_ion_series": "Chimeric longest ion series",
    "chimeric_complementary_ion_count": "Chimeric complementary ion count",
    "chimeric_max_ion_gap": "Chimeric max ion gap",
    "is_missing_fragment_match_features": "Missing fragment match",
    "is_missing_chimeric_features": "Missing chimeric",
    "is_missing_irt_error": "Missing iRT",
    "sequence_length": "Sequence length",
    "precursor_charge": "Precursor charge",
    "min_token_probability": "Min token probability",
    "std_token_probability": "Std token probability",
}

error_theme = Theme({"error": "red bold", "error_highlight": "red bold underline"})
console = Console(theme=error_theme)

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


def to_sentence_case(name: str) -> str:
    """Convert a feature display name to sentence case."""
    return name.lower()


# ---------------------------------------------------------------------------
# Model wrapper for sklearn-compatible predict_proba
# ---------------------------------------------------------------------------
class _CalibratorPredictor:
    """Wraps a fitted ProbabilityCalibrator as an sklearn-style estimator.

    Provides ``predict_proba`` and ``predict`` on *pre-normalised* feature
    arrays so that permutation importance and SHAP can treat it like a
    classifier.  The ``classes_`` attribute is set to ``[0, 1]``.
    """

    def __init__(self, calibrator: ProbabilityCalibrator) -> None:
        assert calibrator.network is not None
        assert calibrator.feature_mean is not None
        assert calibrator.feature_std is not None

        self.network = calibrator.network
        self.feature_mean = calibrator.feature_mean.cpu()
        self.feature_std = calibrator.feature_std.cpu()
        self.classes_ = np.array([0, 1])

    def fit(self, x_input: np.ndarray, y: np.ndarray) -> "_CalibratorPredictor":
        """No-op fit to satisfy sklearn estimator interface."""
        return self

    def score(self, x_input: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy to satisfy sklearn estimator interface."""
        return float(np.mean(self.predict(x_input) == y))

    def predict_proba(self, x_input: np.ndarray) -> np.ndarray:  # noqa: N803
        """Return class probabilities for each sample."""
        x = torch.as_tensor(x_input, dtype=torch.float32)
        self.network.eval()
        with torch.no_grad():
            logits = self.network(x)
        probs = torch.sigmoid(logits).numpy().ravel()
        return np.column_stack([1 - probs, probs])

    def predict(self, x_input: np.ndarray) -> np.ndarray:  # noqa: N803
        """Return binary predictions for each sample."""
        return (self.predict_proba(x_input)[:, 1] >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------
def plot_feature_importance(
    importance_scores: Dict[str, float],
    title: str,
    output_path_base: Path,
) -> None:
    """Plot horizontal bar chart of permutation feature importance scores."""
    plt.figure(figsize=(8, 6))
    features = list(importance_scores.keys())
    scores = list(importance_scores.values())

    sorted_idx = np.argsort(scores)
    features = [features[i] for i in sorted_idx]
    scores = [scores[i] for i in sorted_idx]

    plt.barh(range(len(features)), scores, color=_PALETTE[0])
    plt.yticks(range(len(features)), features)
    plt.xlabel("Importance score")
    plt.title(title)

    plt.savefig(f"{output_path_base}.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(f"{output_path_base}.png", bbox_inches="tight", dpi=300)
    plt.close()


def plot_feature_correlations(features: pd.DataFrame, output_path_base: Path) -> None:
    """Plot lower-triangle feature correlation heatmap."""
    plt.figure(figsize=(12, 10))
    corr_matrix = features.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        annot=True,
        fmt=".2f",
        cbar_kws={"shrink": 0.5},
    )
    plt.title("Feature correlation matrix")

    plt.savefig(f"{output_path_base}.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(f"{output_path_base}.png", bbox_inches="tight", dpi=300)
    plt.close()


def plot_shap_summary(shap_values, correct_class_idx: int, output_dir: Path) -> None:
    """Plot SHAP beeswarm summary for the correct class."""
    plt.figure(figsize=(8, 6))
    shap.plots.beeswarm(
        shap_values[:, :, correct_class_idx],
        show=False,
        max_display=12,
        color=plt.cm.viridis,
    )
    plt.title(r"SHAP feature impact on $P(\text{correct})$")

    plt.savefig(output_dir / "shap_summary.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(output_dir / "shap_summary.png", bbox_inches="tight", dpi=300)
    plt.close()


def plot_shap_bar(
    shap_values,
    test_features_scaled,
    test_labels,
    correct_class_idx: int,
    output_dir: Path,
) -> None:
    """Plot SHAP bar chart with hierarchical clustering."""
    plt.figure(figsize=(8, 6))
    clustering = shap.utils.hclust(test_features_scaled, test_labels)
    shap.plots.bar(
        shap_values[:, :, correct_class_idx],
        clustering=clustering,
        show=False,
        clustering_cutoff=0.5,
        max_display=12,
    )
    ax = plt.gca()
    for patch in ax.patches:
        patch.set_facecolor(_PALETTE[1])
    plt.title(r"SHAP feature importance for $P(\text{correct})$")

    plt.savefig(output_dir / "shap_importance.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(output_dir / "shap_importance.png", bbox_inches="tight", dpi=300)
    plt.close()


def plot_shap_dependence(
    shap_values,
    feature_names: list,
    display_feature_names: list,
    correct_class_idx: int,
    output_dir: Path,
    top_n: int = 3,
) -> None:
    """Plot SHAP dependence scatter for the top-N most important features."""
    mean_abs_shap = np.abs(shap_values.values[:, :, correct_class_idx]).mean(axis=0)
    top_features_idx = np.argsort(mean_abs_shap)[-top_n:][::-1]

    for idx in top_features_idx:
        feature_name = display_feature_names[idx]
        original_feature_name = feature_names[idx]
        plt.figure(figsize=(8, 6))
        shap.plots.scatter(
            shap_values[:, idx, correct_class_idx],
            show=False,
            color=_PALETTE[2],
        )
        plt.title(
            "SHAP dependence plot for "
            + to_sentence_case(feature_name)
            + "\n"
            + r"(impact on $P(\text{correct})$)"
        )

        ax = plt.gca()
        ylabel = ax.get_ylabel()
        if "SHAP value for" in ylabel:
            ax.set_ylabel(ylabel.replace(feature_name, to_sentence_case(feature_name)))

        plt.savefig(
            output_dir / f"shap_dependence_{original_feature_name}.pdf",
            bbox_inches="tight",
            dpi=300,
        )
        plt.savefig(
            output_dir / f"shap_dependence_{original_feature_name}.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()


def plot_shap_interactions(
    shap_values,
    feature_names: list,
    display_feature_names: list,
    correct_class_idx: int,
    output_dir: Path,
    top_n: int = 3,
) -> None:
    """Plot pairwise SHAP interaction scatter plots for top-N features."""
    mean_abs_shap = np.abs(shap_values.values[:, :, correct_class_idx]).mean(axis=0)
    top_features_idx = np.argsort(mean_abs_shap)[-top_n:][::-1]

    for i, idx1 in enumerate(top_features_idx):
        f1_display = display_feature_names[idx1]
        f1_orig = feature_names[idx1]
        for j, idx2 in enumerate(top_features_idx):
            if i == j:
                continue
            f2_display = display_feature_names[idx2]
            f2_orig = feature_names[idx2]

            plt.figure(figsize=(8, 6))
            shap.plots.scatter(
                shap_values[:, f1_display, correct_class_idx],
                color=shap_values[:, f2_display, correct_class_idx],
                show=False,
                cmap="viridis",
            )
            plt.title(
                "SHAP interaction plot for "
                + to_sentence_case(f1_display)
                + " vs "
                + to_sentence_case(f2_display)
                + "\n"
                + r" (impact on $P(\text{correct})$)"
            )

            ax = plt.gca()
            ylabel = ax.get_ylabel()
            if "SHAP value for" in ylabel:
                ax.set_ylabel(ylabel.replace(f1_display, to_sentence_case(f1_display)))

            plt.savefig(
                output_dir / f"shap_interaction_{f1_orig}_vs_{f2_orig}.pdf",
                bbox_inches="tight",
                dpi=300,
            )
            plt.savefig(
                output_dir / f"shap_interaction_{f1_orig}_vs_{f2_orig}.png",
                bbox_inches="tight",
                dpi=300,
            )
            plt.close()


def plot_shap_heatmap(shap_values, correct_class_idx: int, output_dir: Path) -> None:
    """Plot SHAP heatmap showing per-sample feature contributions."""
    plt.figure(figsize=(8, 6))
    shap.plots.heatmap(
        shap_values[:, :, correct_class_idx],
        max_display=12,
        show=False,
        cmap="RdBu_r",
    )
    plt.title("SHAP feature impact heatmap\n" + r"(impact on $P(\text{correct})$)")

    plt.savefig(output_dir / "shap_heatmap.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(output_dir / "shap_heatmap.png", bbox_inches="tight", dpi=300)
    plt.close()


def create_all_plots(
    perm_importance_dict: Dict[str, float],
    shap_values,
    train_features_scaled_df: pd.DataFrame,
    test_features_scaled: np.ndarray,
    test_labels: np.ndarray,
    feature_names: list,
    display_feature_names: list,
    correct_class_idx: int,
    output_dir: Path,
) -> None:
    """Generate all analysis plots (importance, correlations, SHAP)."""
    logger.info("Creating plots...")

    plot_feature_importance(
        perm_importance_dict,
        "Permutation feature importance",
        output_dir / "permutation_importance",
    )
    plot_shap_summary(shap_values, correct_class_idx, output_dir)
    plot_shap_bar(
        shap_values, test_features_scaled, test_labels, correct_class_idx, output_dir
    )
    plot_shap_dependence(
        shap_values, feature_names, display_feature_names, correct_class_idx, output_dir
    )
    plot_shap_interactions(
        shap_values, feature_names, display_feature_names, correct_class_idx, output_dir
    )
    plot_shap_heatmap(shap_values, correct_class_idx, output_dir)
    plot_feature_correlations(
        train_features_scaled_df, output_dir / "feature_correlations"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_features_from_parquet(
    path: Path,
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Load feature matrix and labels from a Parquet file or directory."""
    import polars as pl

    p = Path(path)
    if p.is_dir():
        parquet_files = sorted(p.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No .parquet files found in {p}")
        df = pl.concat([pl.read_parquet(f) for f in parquet_files])
    else:
        df = pl.read_parquet(p)

    if "correct" not in df.columns:
        raise ValueError(f"Parquet at {path} must contain a 'correct' column")
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in Parquet: {missing}")

    features = df.select(feature_columns).to_numpy().astype(np.float32)
    labels = df["correct"].to_numpy().astype(np.float32)
    return features, labels


@app.command()
def main(
    model_path: Annotated[
        Path, typer.Option(help="Path to pretrained calibrator model directory.")
    ],
    output_dir: Annotated[
        Path, typer.Option(help="Directory to save analysis results and plots.")
    ],
    data_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="Directory containing train and test data files (raw spectra path)."
        ),
    ] = None,
    train_features_path: Annotated[
        Optional[Path],
        typer.Option(
            help="Path to pre-computed training feature Parquet (alternative to --data-dir)."
        ),
    ] = None,
    test_features_path: Annotated[
        Optional[Path],
        typer.Option(
            help="Path to pre-computed test feature Parquet (alternative to --data-dir)."
        ),
    ] = None,
    train_spectra: Annotated[
        str, typer.Option(help="Filename of training spectra parquet inside data-dir.")
    ] = "general_train.parquet",
    train_preds: Annotated[
        str, typer.Option(help="Filename of training predictions CSV inside data-dir.")
    ] = "general_train_beams.csv",
    test_spectra: Annotated[
        str, typer.Option(help="Filename of test spectra parquet inside data-dir.")
    ] = "general_test.parquet",
    test_preds: Annotated[
        str, typer.Option(help="Filename of test predictions CSV inside data-dir.")
    ] = "general_test_beams.csv",
    n_background_samples: Annotated[
        int, typer.Option(help="Background samples for SHAP.", min=1, max=10000)
    ] = 500,
    n_test_samples: Annotated[
        int, typer.Option(help="Test samples for SHAP.", min=1, max=10000)
    ] = 1000,
    create_plots: Annotated[
        bool, typer.Option("--create-plots/--no-plots", help="Whether to create plots.")
    ] = True,
) -> None:
    """Analyze feature importance and correlations for a pretrained calibrator."""
    use_parquet = train_features_path is not None or test_features_path is not None
    if use_parquet and (train_features_path is None or test_features_path is None):
        raise typer.BadParameter(
            "--train-features-path and --test-features-path must both be provided."
        )
    if not use_parquet and data_dir is None:
        raise typer.BadParameter(
            "Either --data-dir or --train-features-path/--test-features-path must be provided."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load calibrator
    logger.info("Loading pretrained calibrator from %s", model_path)
    calibrator = ProbabilityCalibrator.load(model_path)

    # Build sklearn-compatible predictor wrapper
    predictor = _CalibratorPredictor(calibrator)

    if use_parquet:
        assert train_features_path is not None
        assert test_features_path is not None

        feature_columns = ["confidence"] + calibrator.columns

        logger.info("Loading training features from Parquet: %s", train_features_path)
        train_features, train_labels = _load_features_from_parquet(
            train_features_path,
            feature_columns,
        )
        logger.info(
            "  %d training samples, %d features",
            len(train_labels),
            train_features.shape[1],
        )

        logger.info("Loading test features from Parquet: %s", test_features_path)
        test_features, test_labels = _load_features_from_parquet(
            test_features_path,
            feature_columns,
        )
        logger.info(
            "  %d test samples, %d features", len(test_labels), test_features.shape[1]
        )

        feature_names = feature_columns
    else:
        assert data_dir is not None
        logger.info("Loading training dataset from raw spectra...")
        loader = InstaNovoDatasetLoader(
            residue_masses=RESIDUE_MASSES,
            residue_remapping=RESIDUE_REMAPPING,
            beam_columns=BEAM_COLUMNS,
        )
        train_dataset = loader.load(
            data_path=data_dir / train_spectra,
            predictions_path=data_dir / train_preds,
        )

        logger.info("Loading test dataset from raw spectra...")
        test_dataset = loader.load(
            data_path=data_dir / test_spectra,
            predictions_path=data_dir / test_preds,
        )

        logger.info("Computing features for training set...")
        calibrator.compute_features(train_dataset)
        train_features, train_labels = calibrator._extract_feature_matrix(
            train_dataset, labelled=True
        )

        logger.info("Computing features for test set...")
        calibrator.compute_features(test_dataset)
        test_features, test_labels = calibrator._extract_feature_matrix(
            test_dataset, labelled=True
        )

        feature_names = [train_dataset.confidence_column] + calibrator.columns

    display_feature_names = [
        COLUMN_DISPLAY_NAMES.get(name, name) for name in feature_names
    ]

    assert calibrator.feature_mean is not None
    assert calibrator.feature_std is not None
    feature_mean = calibrator.feature_mean.cpu().numpy()
    feature_std = calibrator.feature_std.cpu().numpy()
    train_features_scaled = (train_features - feature_mean) / feature_std
    test_features_scaled = (test_features - feature_mean) / feature_std

    correct_class_idx = 1  # class 1 = correct

    # 1. Permutation importance on test set
    logger.info("Computing permutation importance on test set...")
    perm_importance = permutation_importance(
        predictor,
        test_features_scaled,
        test_labels,
        n_repeats=10,
        random_state=SEED,
        n_jobs=-1,
    )
    perm_importance_dict = dict(
        zip(display_feature_names, perm_importance.importances_mean)
    )

    # 2. SHAP values
    logger.info("Computing SHAP values...")
    background = shap.sample(
        train_features_scaled,
        min(n_background_samples, len(train_features_scaled)),
        random_state=SEED,
    )

    explainer = shap.KernelExplainer(
        model=predictor.predict_proba,
        data=background,
        seed=SEED,
        link="identity",
    )

    np.random.seed(SEED)
    n_samples = min(n_test_samples, test_features_scaled.shape[0])
    indices = np.random.choice(
        test_features_scaled.shape[0], size=n_samples, replace=False
    )

    shap_values = explainer(test_features_scaled[indices])

    # Switch to original feature space for visualisation
    shap_values.data = test_features[indices]
    shap_values.feature_names = display_feature_names

    # 3. Feature correlations on training data
    logger.info("Computing feature correlations on training data...")
    train_features_scaled_df = pd.DataFrame(
        train_features_scaled, columns=display_feature_names
    )

    if create_plots:
        create_all_plots(
            perm_importance_dict=perm_importance_dict,
            shap_values=shap_values,
            train_features_scaled_df=train_features_scaled_df,
            test_features_scaled=test_features_scaled,
            test_labels=test_labels,
            feature_names=feature_names,
            display_feature_names=display_feature_names,
            correct_class_idx=correct_class_idx,
            output_dir=output_dir,
        )

    # Save raw objects
    logger.info("Saving raw analysis objects...")

    with open(output_dir / "perm_importance.pkl", "wb") as f:
        pickle.dump(perm_importance, f)

    with open(output_dir / "shap_values.pkl", "wb") as f:
        pickle.dump(shap_values, f)

    logger.info("Analysis complete!")
    logger.info("Results saved to %s", output_dir)
    logger.info(
        "Permutation Feature Importance: computed on %d test samples", len(test_labels)
    )
    logger.info(
        "SHAP values: computed on %d test samples with %d training samples as background",
        n_samples,
        len(background),
    )
    logger.info(
        "Correlation matrix: computed on %d training samples", len(train_labels)
    )

    saved_files = ["perm_importance.pkl", "shap_values.pkl"]
    if create_plots:
        saved_files.append("All plots in PDF and PNG formats")
    else:
        logger.info("Plots were skipped (--no-plots flag used)")

    logger.info("Saved files: %s", ", ".join(saved_files))
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    app()
